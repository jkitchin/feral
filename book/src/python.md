# Python bindings

feral ships Python bindings as the [`feral-solver`](https://pypi.org/project/feral-solver/)
package (import name `feral`). They wrap the same Rust solver, so the
certified inertia and the batched multi-RHS performance carry over.

## Install

```bash
pip install feral-solver           # plain
pip install 'feral-solver[scipy]'  # with scipy.sparse adapters
uv add feral-solver                # via uv
```

Wheels cover CPython 3.10+ on Linux x86_64/aarch64, macOS universal2,
and Windows x86_64.

## Quickstart

```python
import numpy as np
import feral

A = feral.CscMatrix.from_dense(np.array([
    [4.0, 1.0, 0.0],
    [1.0, 3.0, 2.0],
    [0.0, 2.0, 5.0],
]))

solver = feral.Solver()
status, inertia = solver.factor(A)
assert status == feral.FactorStatus.SUCCESS
print(inertia)                       # Inertia(n_pos=3, n_neg=0, n_zero=0)

b = np.array([1.0, 2.0, 3.0])
x = solver.solve(b)
print(np.allclose(A.symv(x), b))     # True
```

## Batched multi-RHS solves

Pass `Solver.solve` a 2-D `(n, nrhs)` array and it returns an
`(n, nrhs)` solution, solving every column against the one shared
factorization. For wide `nrhs` this engages the register-blocked panel
kernels automatically — no API change, just a 2-D right-hand side:

```python
B = np.random.default_rng(0).standard_normal((n, nrhs))
X = solver.solve(B)                  # (n, nrhs), one batched call
# X[:, j] == solver.solve(B[:, j]) to machine precision
```

On 2-D Laplacians this is roughly 3–6× faster per RHS than looping the
single-RHS solve. The
[`02_multi_rhs_batched`](https://github.com/jkitchin/feral/tree/main/python/examples/notebooks)
notebook walks through a steady-state heat-conduction example with a
correctness check, a visualization, and a looped-vs-batched timing.

Iterative refinement batches too: pass a 2-D `B` to
`solver.solve_refined(A, B)` and each refinement step runs through the
panel kernel over the still-unconverged columns (issue #58), so the
refined path — the default for KKT back-solves — is ~2–3× faster per RHS
than looping the single-RHS refined solve. The batched path amortizes
the *solves*; the per-column residual `B − A·X` is still un-batched, so
on dense Hessians (where the matrix–vector product rivals the solve) the
gain is smaller. The notebook's final section measures it.

## scipy.sparse interop

```python
import scipy.sparse as sp

A = feral.from_scipy(A_scipy, symmetric="full")   # reads the lower triangle
# ... factor, solve ...
A_back = feral.to_scipy(A)                         # round-trips to scipy
```

## Unsymmetric LU basis engine

`LuFactor` factors a general (unsymmetric) square matrix and solves
`A x = b` (`ftran`) and `Aᵀ y = c` (`btran`), with simplex-style
product-form column `update`s. It auto-routes between a dense and a
sparse engine via the same `should_use_dense_lu` heuristic the Rust core
uses; pass `force_dense=True`/`False` to override the routing.

```python
import numpy as np
import feral

A = np.array([[2.0, 1.0, 0.0],
              [0.0, 3.0, 1.0],
              [1.0, 0.0, 4.0]])

lu = feral.LuFactor(feral.LuMatrix.from_dense(A))
x = lu.ftran(np.array([1.0, 2.0, 3.0]))     # solve A x = b
y = lu.btran(np.array([1.0, 0.0, 0.0]))     # solve Aᵀ y = c

lu.update(1, np.array([0.0, 5.0, 1.0]))     # replace basis column 1
print(lu.updates_since_refactor)            # 1

# Factor identity  P A Q = L U :
#   A[lu.perm][:, lu.qcol] == lu.l_array() @ lu.u_array()
```

A singular basis raises `SingularBasisError` (a subclass of
`FactorError`, so existing `except FactorError` handlers keep working);
an exhausted update budget raises `NeedsRefactorError` — recover by
calling `lu.refactor(new_matrix)`, which resets
`updates_since_refactor` to 0.

## Factor access and introspection

After `Solver.factor`, the assembled factor and its statistics are
available without re-solving. All of the following are **additive** — the
original `factor`/`solve` surface is unchanged.

```python
s = feral.Solver(ordering="amd", profiling=True)
s.factor(A)

fac = s.factors()                       # Factors snapshot (None before factor)
indptr, indices, data = fac.l_csc()     # unit-lower L as CSC (factorization order)
d_diag, d_subdiag = fac.d_blocks()      # block-diagonal D (2×2 where d_subdiag != 0)
L_scipy = fac.to_scipy_l()              # optional scipy.sparse.csc_matrix

# Reconstruction identity (in factorization order):
#   L @ D @ Lᵀ  ==  P (S A S) Pᵀ
# using fac.perm and the per-row fac.scaling vector.

stats = s.last_factor_stats()           # nnz_a, nnz_l, fill_ratio, inertia, pivots
print(s.min_pivot_magnitude, s.max_pivot_magnitude)
print(s.scaling_info.kind)              # "applied" | "mc64_fallback_to_infnorm" | ...
print(s.profile_report())               # populated only when profiling=True
```

`Solver.symbolic()` — and the standalone `feral.analyze(A, ordering=...)`,
which runs **no** numeric factorization — return a `SymbolicAnalysis`
exposing the resolved `ordering`, `perm`/`perm_inv`, `num_supernodes`,
`factor_nnz_estimate` (a slack-inflated upper bound on the realized
`factor_nnz`), `col_counts`, and the elimination-tree `etree_parent`
array (roots marked by `root_sentinel`).

New `Solver(...)` keyword arguments — all optional, each defaulting to
the prior behavior — expose the tuning knobs: `ordering` (`"amd"`,
`"amf"`, `"metis"`, `"scotch"`, `"kahip"`, `"auto"`, `"auto_race"`),
`mc64_cache`, `profiling`, `partial_singular_warning`, and
`auto_cascade_break`. Counters `mc64_fallback_count`,
`mc64_scaling_fallback_count`, `mc64_retry_attempt_count`, and
`mc64_cache_hit_count` round out the introspection surface, alongside
`invalidate_factors()` / `invalidate_symbolic_cache()`.

## Conversion conveniences

```python
A = feral.CscMatrix.from_dense(M, triangle="lower")   # or "upper" / "full"
dense = A.to_dense()                                  # full symmetric 2-D array
indptr, indices = A.symmetric_pattern()               # full structural pattern
```

`from_dense`'s `triangle` keyword defaults to `"lower"` (unchanged
behavior); `"upper"` mirrors the upper triangle and `"full"` reads the
whole array.

## Interior-point methods

`feral.ipm.KktSolver` wraps `Solver` with the Wächter–Biegler 2006 §3.1
perturbation-escalation loop and caches the symbolic analysis across a
Newton run. See the
[Python README](https://github.com/jkitchin/feral/tree/main/python)
and the `01`–`05` example notebooks for the full surface — `05` walks
through the LU engine, factor/symbolic access, and introspection added
in 0.11.0.
