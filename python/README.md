# feral-solver

Python bindings for [feral](https://github.com/jkitchin/feral), a
pure-Rust sparse symmetric indefinite direct solver with certified
inertia counts. Aimed at interior-point methods (the IPM in
[discopt](https://github.com/jkitchin/discopt) is the primary
consumer), but usable for any application that factors symmetric
KKT-shaped systems.

## Install

```bash
pip install feral-solver           # plain
pip install 'feral-solver[scipy]'  # with scipy.sparse adapters
uv add feral-solver                # via uv
```

Wheels are published for CPython 3.10+ on Linux x86_64/aarch64,
macOS universal2, and Windows x86_64. abi3 means one wheel per
platform/arch covers all supported Python minor versions.

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

## IPM use

The `feral.ipm.KktSolver` class wraps `Solver` with the Wächter–Biegler
2006 §3.1 perturbation-escalation loop. Symbolic analysis is cached;
across an entire Newton run `solver.symbolic_call_count` stays at 1.

```python
import feral
import feral.ipm

kkt_pattern = feral.CscMatrix.from_scipy(my_kkt)   # see scipy adapter
kkt = feral.ipm.KktSolver(
    kkt_pattern,
    expected_inertia=feral.Inertia(n_vars, n_equality_constraints),
)
for newton_iter in range(max_iter):
    report = kkt.factor(values_this_iter)          # auto-perturbs if needed
    if report.status != feral.FactorStatus.SUCCESS:
        break
    dx_aff, dx_corr = kkt.solve_pair(b_aff, b_corr)
    ...
```

See `examples/discopt_ipm_kkt.py` for an end-to-end Newton step
against a small NLP.

## Unsymmetric LU basis engine

`LuFactor` factors a general square matrix and solves `A x = b`
(`ftran`) / `Aᵀ y = c` (`btran`), with simplex-style product-form
`update`s. It auto-routes to a dense or sparse engine via the same
`should_use_dense_lu` heuristic the Rust core uses; pass
`force_dense=True`/`False` to override.

```python
import numpy as np
import feral

A = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]])
lu = feral.LuFactor(feral.LuMatrix.from_dense(A))
x = lu.ftran(np.array([1.0, 2.0, 3.0]))     # solve A x = b
y = lu.btran(np.array([1.0, 0.0, 0.0]))     # solve Aᵀ y = c
lu.update(1, np.array([0.0, 5.0, 1.0]))     # replace basis column 1
# P A Q = L U :  A[perm][:, qcol] == l_array() @ u_array()
```

A singular basis raises `SingularBasisError` (a `FactorError`); an
exhausted update budget raises `NeedsRefactorError` — call
`lu.refactor(new_matrix)`.

## Factor access and introspection

After `Solver.factor`, the assembled factor and its statistics are
available without re-solving:

```python
s = feral.Solver(ordering="amd", profiling=True)
s.factor(A_csc)

fac = s.factors()                  # Factors snapshot
indptr, indices, data = fac.l_csc()   # unit-lower L as CSC (factorization order)
d_diag, d_subdiag = fac.d_blocks()    # block-diagonal D (2×2 where d_subdiag != 0)
L_scipy = fac.to_scipy_l()            # optional scipy.sparse.csc_matrix

# Reconstruction identity (factorization order):
#   L @ D @ L.T  ==  P (S A S) Pᵀ
# with fac.perm and the per-row fac.scaling vector.

stats = s.last_factor_stats()      # nnz, fill_ratio, inertia, pivot range, ...
print(s.min_pivot_magnitude, s.max_pivot_magnitude)
print(s.scaling_info.kind)         # "applied" | "mc64_fallback_to_infnorm" | ...
print(s.profile_report())          # populated when profiling=True
```

`Solver.symbolic()` (and the standalone `feral.analyze(A_csc,
ordering=...)`, which runs **no** numeric factorization) return a
`SymbolicAnalysis` with the resolved `ordering`, `perm`/`perm_inv`,
`num_supernodes`, `factor_nnz_estimate`, `col_counts`, and the
elimination-tree `etree_parent` array (roots marked `-1`).

New `Solver(...)` keyword arguments — all optional, defaulting to the
prior behavior — expose the tuning knobs: `ordering` (`"amd"`, `"amf"`,
`"metis"`, `"scotch"`, `"kahip"`, `"auto"`, `"auto_race"`),
`mc64_cache`, `profiling`, `partial_singular_warning`, and
`auto_cascade_break`.

## Conversion conveniences

`CscMatrix.to_dense()` returns the full symmetric matrix as a 2-D numpy
array; `CscMatrix.from_dense(a, triangle="lower"|"upper"|"full")` ingests
either triangle; `CscMatrix.symmetric_pattern()` returns the full
`(indptr, indices)` structural pattern.

## Example notebooks

Runnable notebooks live in `examples/notebooks/`. Regenerate them from
the reviewable `_build_notebooks.py` generator: `python _build_notebooks.py`
re-executes each notebook and commits its cell outputs (the embedded
assertions double as a smoke test), or pass `--no-execute` for
source-only `.ipynb` when `feral` is not installed in the running
interpreter.

- `01_basic_factor_solve` — factor, certified inertia, solve, refine, reuse.
- `02_multi_rhs_batched` — **batched multi-RHS solve**, motivated by a
  steady-state heat-conduction sweep, with a correctness check and a
  looped-vs-batched timing showing the per-RHS speedup (issue #57).
- `03_kkt_saddle_inertia` — indefinite KKT system with certified inertia.
- `04_scipy_numpy_interop` — `scipy.sparse` round-trip vs `spsolve`.
- `05_lu_and_introspection` — the **LU basis engine** (`ftran`/`btran`,
  product-form updates, `P A Q = L U`), **factor access** (`L`/`D`
  reconstruction, `feral.analyze`), and **introspection** (knobs, factor
  stats, pivot range, scaling info) added in 0.11.0.

## scipy.sparse interop

```python
import scipy.sparse as sp
import feral

A_scipy = sp.csc_matrix(...)
A = feral.from_scipy(A_scipy, symmetric="full")    # reads lower triangle
# ... factor, solve ...
A_back = feral.to_scipy(A)                          # round-trips to scipy
```

## Building from source

Requires a stable Rust toolchain (1.75+) and Python 3.10+.

```bash
git clone https://github.com/jkitchin/feral.git
cd feral/python
pip install maturin
maturin develop --release    # builds and installs into current venv
pytest tests/
```

## License

MIT, same as the underlying Rust crate.
