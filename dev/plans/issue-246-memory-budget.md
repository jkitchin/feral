# Plan — issue #246 step 3: `max_factor_bytes`

Research: `dev/research/issue-246-memory-budget.md`.

## Tests first (oracle: hand arithmetic on a path graph)

Tridiagonal n = 10. Any of the orderings has no fill on a path, so
Σ colcount = 2n − 1 = 19, `factor_nnz_estimate = floor(19 · 1.2) = 22`,
and factor bytes = 22 · 8 = **176**.

1. `max_factor_bytes = Some(175)`: the result is
   `MemoryBudgetExceeded { predicted_bytes: 176, budget_bytes: 175,
   stage: "factor" }`. This is check 1, and it must fire *before* the
   finish. The existing `FINISH_RUNS` test probe must read 0.
2. `Some(176)` passes check 1. Then the result is either `Ok` (if
   `peak_bytes ≤ 176`) or a `"peak"` error. Assert that it matches whatever
   `peak_bytes` an unbudgeted run reports.
3. `None` and `Some(usize::MAX)` return the same analysis as today.
4. `AutoRace` with a tiny budget returns `MemoryBudgetExceeded`, and the
   winner is never finished.
5. `Solver::new().with_max_factor_bytes(Some(1)).factor(..)` returns
   `FactorStatus::FatalError(MemoryBudgetExceeded{..})` and does not panic.
   `Solver` with `None` still factors.
6. A check-2 case: a matrix with a non-empty contribution block (an arrow
   matrix, whose hub is a separate parent supernode). Set the budget to
   `factor_bytes` so check 1 passes and check 2 fails with stage `"peak"`.
   The budget value comes from an unbudgeted run's `work_estimate()`;
   the assertion is on the stage and the bytes.

## Implementation

- `SupernodeParams::max_factor_bytes` plus `Default`.
- `FeralError::MemoryBudgetExceeded` plus `Display`, and the Python
  mapping.
- `finish_checked(prefix, params, profiler)` replaces the two
  `finish_and_copy` call sites: check 1, then finish, then check 2.
- `Solver::with_max_factor_bytes`.
- CHANGELOG and book docs if the Solver options page lists builders.
