# Implementation Plan: External / user-supplied ordering (issue #107)

**Date:** 2026-07-02
**Research note:** dev/research/issue-107-external-ordering.md
**Spec sections:** 5.1

## Files to Create/Modify
- `src/symbolic/mod.rs` — add `External(Vec<usize>)` variant, drop `Copy`,
  manual `Debug`, `validate_external_perm`, wire `run_external_ordering` and
  `symbolic_factorize_with_method`, unit tests.
- `src/numeric/solver.rs` — clone `self.ordering` at the two factor sites.
- `src/numeric/factorize.rs` — clone `symbolic.resolved_method` at the three
  numeric constructors.
- `python/src/common.rs` — `ordering_to_str` gains an `External` arm.
- `crates/feral-diagnostics/**` — `.clone()` wherever a `method` binding is
  reused by value (compiler-driven).
- `tests/issue107_external_ordering.rs` — behavioral tests (oracle = hand
  permutation).
- `CHANGELOG.md`, `dev/decisions.md`, session + journal.

## Implementation Steps
1. Enum: drop `Copy`, add `External(Vec<usize>)`, hand-written `Debug`.
2. `validate_external_perm(perm, n) -> Result<(), FeralError>`.
3. `run_external_ordering(&CscPattern, &OrderingMethod)`: early-return the
   user perm for `External`; `.clone()` the resolved method for other arms.
4. `symbolic_factorize_with_method`: validate `External` up front; force
   `resolved_preprocess = None`; skip the preprocess-`Auto` delegation.
5. Fix `AutoRace` race loop + preprocess-`Auto` `run` closure to clone method.
6. Fix solver / numeric / python / diagnostics fallout from `Copy` removal.

## Tests (write first)
- `tests/issue107_external_ordering.rs`: identity perm solve+inertia; fixed
  non-trivial perm solve+inertia parity; `resolved_method == External` and
  `resolved_preprocess == None`; validation rejects (len / range / dup);
  `Solver::with_ordering(External)` end-to-end parity.
- `src/symbolic/mod.rs` unit: `External` yields a valid bijection perm and
  forces `None` preprocess.

## Success Criteria
- `cargo test` (feral) green; `cargo test -p feral-diagnostics` green.
- `cargo clippy --all-targets -- -D warnings` and
  `cargo clippy -p feral-diagnostics --all-targets -- -D warnings` clean.
- `cargo fmt --check` clean.
- No `unwrap`/`expect`/`unsafe` added in `src/`.
- Byte-exact: default path (`Auto`, no External) unchanged.
