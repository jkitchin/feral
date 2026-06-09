# Introduction

**feral** is a sparse symmetric indefinite direct solver written in pure Rust,
with certified inertia counts. It is a clean-room implementation from published
papers and BSD-licensed references, MIT-licensed, with no BLAS, LAPACK, or
Fortran dependencies in the core solver.

This book is the narrative companion to the [API reference](./api.md). The book
covers concepts, semantics, and design choices; the API reference covers types
and functions.

## What it does

- **Sparse and dense** symmetric indefinite factorization (LDLᵀ with
  Bunch–Kaufman pivoting).
- **Certified inertia** `(n_pos, n_neg, n_zero)` alongside every factor.
- **Single and batched (multi-RHS) solves** — factor once, solve against
  many right-hand sides, with register-blocked panel kernels for wide
  `nrhs`.
- **Iterative refinement** and optional **MC64 scaling** for
  ill-conditioned and near-singular systems.
- **Unsymmetric LU basis engine** (`feral::lu`) — a separate factorization
  family for revised-simplex bases, with cheap rank-1 column-replacement
  updates and warm `ftran`/`btran` solves (no inertia).
- **Python bindings** (`feral-solver` on PyPI) and `scipy.sparse` interop.

## Where to go next

- [Getting started](./getting-started.md) — install, link, and run sparse,
  batched, and dense solves.
- [Inertia semantics](./inertia.md) — what `Inertia` means and what feral
  guarantees on singular matrices.
- [Unsymmetric LU basis engine](./lu.md) — the separate `feral::lu` family
  for simplex bases: `P B Q = L U`, rank-1 updates, and warm solves.
- [Python bindings](./python.md) — install, quickstart, and batched solves
  from Python.
- [API reference](./api.md) — the rustdoc output for the `feral` crate and
  workspace members.

## Project links

- Repository: <https://github.com/jkitchin/feral>
- Crate: <https://crates.io/crates/feral>
- Issue tracker: <https://github.com/jkitchin/feral/issues>
