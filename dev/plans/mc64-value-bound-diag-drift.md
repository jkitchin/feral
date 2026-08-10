# Plan — make MC64 value-bound condition 3 a drift measure

Research: `dev/research/mc64-value-bound-diag-drift-2026-08-09.md`.

## Change

`src/scaling/value_bound.rs` only. No API surface moves; everything here is
`pub(crate)`.

1. Add `min_diag_0: f64` to `Mc64CacheValidity`, populated in
   `precompute_mc64_validity` from `stats.min_diag`. Keep `mean_diag_0`.
2. Add `const DIAG_SHRINK: f64 = 0.5;` documented as `1.0 / GROWTH_FACTOR`.
3. `cond_diag` becomes the disjunction of the existing absolute floor and the
   new drift clause.
4. Update the doc comment on `mc64_value_bound_passes` (condition 3) and the
   defensive length-mismatch comment in `precompute_mc64_validity`, which
   currently reasons about `mean_diag_0 = 0` alone.

## Tests, written first

New, in `value_bound.rs`:

* `value_bound_passes_on_identical_matrix_with_wide_diagonal_range` — the
  regression. `diag(1e-14, 1.0)`, identity scaling, validity from
  `precompute_mc64_validity` on that same matrix. Hand oracle: no
  off-diagonals so `max_ratio = 0`, `n_off_dominant = 0`, `min_diag = 1e-14`,
  `mean_diag = (1e-14 + 1)/2 = 0.5`. Today `cond_diag` is
  `1e-14 >= 1e-12 * 0.5 = 5e-13` -> false, so the gate rejects a matrix
  against its own fingerprint. **This test must fail before the change.**
* `value_bound_rejects_collapse_from_tiny_baseline` — the `arki0003` control,
  numbers taken from the instrumented corpus run, not from the code:
  `min_diag_0 = 9.372359e-6`, `mean_diag_0 = 9.925188e-1`, current
  `min_diag = 1.984196e-13`. Both clauses must fail.
* `value_bound_vacuous_when_no_qualifying_rows` — all-zero diagonal gives
  `min_diag_0 = mean_diag_0 = 0`; both clauses are `>= 0` and condition 3 is
  vacuous, as it already is today. Locks the edge case against the new field.

Existing tests keep their assertions. The three hand-built `Mc64CacheValidity`
literals gain `min_diag_0`; `value_bound_rejects_on_diagonal_collapse` gets
`min_diag_0: 1.0` so its `1e-20` current diagonal fails the drift clause too
and the test still rejects for the reason its doc comment states.

## Oracle provenance

Hand calculation for the synthetic cases (arithmetic shown in each doc
comment), and the instrumented corpus run for the `arki0003` numbers. Neither
is read off the implementation.

## Gates

* `cargo test` green, including `tests/issue_38_*`, `tests/mc64_*`,
  `tests/n2_static_pivot_scaling.rs`, `tests/n4_mc64_retry_latch.rs`,
  `tests/issue65_mc64_fallback.rs`, `tests/lu_scaling.rs`.
* `diag_trajectory_scaling` on the seven MC64-routed families: exactly two
  additional hits, both `robot_1600`, no other family's hit pattern changes.
* Inertia unchanged on those families.
