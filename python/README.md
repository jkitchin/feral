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
