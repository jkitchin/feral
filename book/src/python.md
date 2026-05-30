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

## Interior-point methods

`feral.ipm.KktSolver` wraps `Solver` with the Wächter–Biegler 2006 §3.1
perturbation-escalation loop and caches the symbolic analysis across a
Newton run. See the
[Python README](https://github.com/jkitchin/feral/tree/main/python)
and the `01`–`04` example notebooks for the full surface.
