# Plan — issue #190: residual target + outcome for iterative refinement

Research note: `dev/research/issue-190-refine-target.md`. Read it first;
this file is the execution order only.

Decision taken with the user 2026-08-20: ship **both** criteria
(normwise target as #190 asks, plus componentwise backward error), via a
`StopCriterion` enum. Release shape deferred until #190 is measured.

## Order of work

1. **`abs_symv` on `CscMatrix`** — `y = |A|·|x|`, same lower-triangle
   loop as `symv` with `.abs()` on both factors. Test against a dense
   reference on a matrix with mixed signs, and assert `abs_symv` of a
   non-negative matrix equals `symv` of it.
2. **`backward_error(matrix, x, rhs, r) -> f64`** — LAPACK `dgerfs`
   guarded form, formula quoted in the research note. External oracle:
   values computed independently in Python from the published formula.
   Tests written *before* the Rust body.
3. **`StopCriterion` + `RefineOptions { max_steps, stop }`** — builders
   `with_max_steps`, `with_target`, `with_backward_error`, `and_*`.
   Drop the `Eq` derive (an `f64` payload forbids it), keep `PartialEq`.
4. **`RefineOutcome` / `RefineStop`** — `Copy`, four exit variants.
5. **Wire the single-RHS core** (`solve_sparse_refined_core`): replace
   the `relative_reached` closure with a criterion dispatch, track which
   exit fired, return the outcome.
6. **Wire the multi-RHS refiner** (`solve_sparse_many_refined_into`):
   same criterion per column, aggregate the outcome as max/worst.
7. **`Solver::solve_refined_into` / `solve_many_refined_into`** — return
   the outcome instead of `()`.
8. **Docs**: `RefineOptions` type docs, CHANGELOG, and the `#[non_exhaustive]`
   question answered in the negative (public fields kept, per the API
   sketch in #190).

## Test obligations (each must be able to fail)

- `EpsSqrtN` default is **bit-identical** to `7de1a93` on the existing
  refinement tests — this is the no-regression gate, and it is what
  makes the change safe to ship without tolerance sign-off.
- A loose `RelativeResidual` target stops in strictly fewer steps than
  the default on a matrix where the default runs to the cap, and
  `outcome.stop` reads `Converged` rather than `MaxSteps`.
- `RefineStop::MaxSteps` is reported when, and only when, the cap binds.
- `backward_error` matches the Python oracle to `1e-15` relative on a
  well-scaled case, a badly-scaled case, and a case with a zero row in
  the denominator (the `safe1`/`safe2` branch).
- Cap still beats target: a met target at step 1 returns at step 1 under
  `max_steps = 10`; raising `max_steps` never adds work to a converged
  system (#178's invariant, re-asserted under each criterion).
- Multi-RHS under each criterion agrees column-for-column with the
  single-RHS refiner on the same columns.

## Out of scope, recorded

`max_steps = 0` remains unusable for the host because
`ZeroPivotAction::ForceAccept` can leave a residual against the
factorized system. That needs threshold pivoting with delayed pivots,
not a stopping rule. Not attempted here.
