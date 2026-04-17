# feral-amd

Approximate Minimum Degree (AMD) fill-reducing ordering for sparse
symmetric matrices, implemented in pure Rust using the in-place
quotient-graph algorithm of Amestoy, Davis & Duff (1996, 2004).

- **Status:** pre-implementation (scaffolding only)
- **License:** MIT
- **Reference:** SuiteSparse AMD `amd_2.c` (BSD-3-Clause) via the
  faer-rs in-tree Rust port.
- **Dependencies:** none in the runtime library.

## Scope

This crate is standalone. It defines its own sparsity-pattern type and
returns a permutation without taking a dependency on any particular
sparse-matrix library. Downstream consumers (solvers, analysis
pipelines) convert at the boundary.

## Plan

See `../../dev/plans/ordering-amd-upgrade.md` for the full design,
audit history, oracle-precondition gate, and slice-by-slice commit
plan.
