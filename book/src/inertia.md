# Inertia semantics

`Inertia` is the triple `(n_pos, n_neg, n_zero)` — the number of positive,
negative, and zero eigenvalues of a symmetric matrix. feral reports an inertia
alongside every factorization.

## Guarantees

- **Non-singular matrices**: inertia is exact.
- **Singular matrices**: on matrices where the canonical Fortran direct solvers
  (MUMPS 5.8.2 and SPRAL SSIDS) disagree on inertia, feral agrees with at least
  one of them.

The corpus consensus framework lives under
[`external_benchmarks/consensus/`](https://github.com/jkitchin/feral/tree/main/external_benchmarks/consensus)
and tags matrices with no 3-of-4-oracle agreement as `excluded`; those matrices
are not part of the inertia gate.

See [`Inertia`](./api.md) in the API reference for the type itself.
