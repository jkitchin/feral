# FERAL Context (auto-generated)

Generated: 2026-07-10T12:42:23Z

## Latest Session
File: dev/sessions/2026-07-10-04.md
```
# Session 2026-07-10-04

## Goal
Fix the four correctness-edge issues #120–#123 (from the 2026-07-10 six-agent
audit), one at a time, each as its own tested commit with an empirical
reproduction before encoding the fix. Then one PR for the batch.

## Accomplished
All four fixed, committed, full-suite (75 binaries) + clippy
`--all-targets -D warnings` + fmt green after each. Every fix reproduced or
guard-verified empirically before/after encoding (issue-112 discipline).

- **#120** (`d6f8ebd`): `apply_post_scaling_overrides` now scales
  `cascade_break_eps` by `‖D·A·D‖∞` (like the N2 `static_pivot_floor` fix), so
  the cascade-break `PerturbToEps` floor is RELATIVE to the scaled matrix the
  BK kernel operates on. Root cause: the default-armed `1e-10` was an ABSOLUTE
  per-pivot floor; under Identity scaling on `γ·K` (γ=1e-12) every pivot was
  bent to ±1e-10 (~1e2·‖A‖ perturbation), silently returning the inertia of
  `A+Δ` with `‖Δ‖ ≫ ‖A‖`. Test `cascade_break_eps_scaled_by_matrix_norm`.
- **#121** (`f8d5f8a`): symbolic-cache hit was gated on the `PatternFingerprint`
  (a u64 hash) alone. Added a `last_pattern: Option<(Vec<usize>, Vec<usize>)>`
  field kept in lockstep with the fingerprint; `cache_valid` now requires the
  fingerprint match AND an exact O(n+nnz) `(col_ptr, row_idx)` compare, so a
  2⁻⁶⁴ hash collision can never reuse a stale symbolic (which would scatter new
  values through the old `perm` and corrupt factor/inertia). Test forges a
  collision deterministically; verified it fails pre-fix (call_count 1 vs 2).
- **#122** (`cb96b6e`): three-part guard-hardening bundle.
  - **A** — ordering results (`run_external_ordering`, `schur::run_amd`) were
    range-checked but never bijectivity-checked; a dup/missing perm corrupts
    `permute_pattern`'s per-column count assignment. `validate_external_perm`
    promoted to `pub(crate)` and called on both internal ordering results.
  - **B** — `classify_2x2_inertia` mapped a NaN block to certified `(0,0,2)`.
    Now returns `Result<Inertia, FeralError>` with a release-mode `!is_finite`
    guard; Result threaded through `count_2x2_inertia_val`, `finish_1x1_outcome`
    (now `Result<PivotStepResult>`), and all six call sites via `?`. Backstops
    the debug-only finite entry-scan at `factor.rs:1749` (the only such gate on
    the inertia path — grep confirmed).
  - **C** — `LuParams::validate` now rejects `max_growth` NaN/≤1.0 (a NaN
    silently disabled the growth guard in the update paths) and non-finite/≤0
    `refine_tol`. Both dense and sparse factor entries call `validate()`.
- **#123** (`876cf72`): the MAXFROMM short-circuit gate `akk >= alpha * mf` is
  unconditionally true at `mf == 0.0` (`capture_maxfromm_col` returns `Some(0.0)`
  for an all-zero trailing column), routing a zero column through rook-rescue —
  where Plain's full scan hits `gamma0 == 0.0` and takes the dedicated
  zero-column branch. Broke the documented Plain/Maxfromm bit-parity (delay/
  inertia under `may_delay`, D under ForceAccept). Fix: `mf != 0.0 && …` treats
  `Some(0.0)` as a cache miss. New `maxfromm_zero_tail_cache_miss_parity` (both
  `may_delay` values, scalar + blocked) verified failing pre-fix; unit test
  pins `capture_maxfromm_col`'s `Some(0.0)` precondition.

```

## Git Status
```
cafa023 issue #129: measure panel fragmentation — not justified, close with data
fc4e9b8 issue #128: skip the bpack1 alloc+zero-fill on all-1x1 dense Schur panels
4124faf issue #126: fuse the D-block solve into the forward pass (single-RHS)
164d201 issue #124: thread the permute cache through the parallel driver
d9d5e86 session 2026-07-10-04: checkpoint, CHANGELOG, decisions for #120–#123
```

## Test Status
```
test symbolic::tests::schur_symbolic_tail_invariant_reversed_user_order ... ok
test symbolic::tests::schur_symbolic_tail_invariant_user_order ... ok
test symbolic::tests::symbolic_factorize_amf_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_auto_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
test symbolic::tests::symbolic_factorize_external_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 405 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.49s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-07-10-04.md)

No regression vs 2026-07-10-03 (inertia 100%, same worst residuals).
--- Dense solver validation ---
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
--- Sparse solver validation ---
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
Dense/Sparse failure analysis: no failures

```

## Recent Decisions
replacement (not a summation artifact), strengthening the existing
`NeedsRefactor` semantics. No tolerances changed.

## 2026-07-10 — Issue #122: propagate `Result` from `classify_2x2_inertia`; `max_growth` semantics

Two small design choices made while closing the #122 guard-hardening bundle.

**2×2 non-finite guard is a `Result`, not an inline per-site check.**
`classify_2x2_inertia` previously returned `Inertia` and a NaN `det`/`tr` fell
through every ordered comparison to the `(0,0,2)` arm — certifying a NaN block
as two *zero* eigenvalues. The fix changes the signature to
`Result<Inertia, FeralError>` (release-mode `!is_finite` → `InvalidInput`) and
threads the `Result` through `count_2x2_inertia_val`, `finish_1x1_outcome`
(now `Result<PivotStepResult>`), and all six call sites via `?`. Chosen over a
per-call-site guard because it is DRY (one guard, impossible to forget at a
future site) and every caller already sits in a `Result`-returning frame, so
the ripple is mechanical. This backstops the debug-only finite entry-scan at
`factor.rs:1749` in release without a full-column scan on the hot path (the
guard is O(1) at the pivot-block level).

**`LuParams::max_growth` is `> 1.0` with `+∞` an explicit disable.**
`validate()` now rejects `max_growth` that is `NaN` or `≤ 1.0` and accepts any
finite `> 1.0` plus `+∞`. `+∞` is the documented "never trigger a refactor on
growth" opt-out (`growth > +∞` is always false); `NaN` is rejected because it
silently disabled the growth guard in the update paths (which — unlike
`should_refactor_growth` — do not defend with `is_finite`). `refine_tol` must
be finite and `> 0`. No tolerances changed; all existing `LuParams` in the
tree already satisfy the bounds (min `max_growth` `1.0 + 1e-9`, all
`refine_tol > 0`), so this is pure input validation with no behavior change on
valid inputs.

## Recent Tried-and-Rejected
sweep replays across four hand constructions (journal 2026-07-10-01,
research note §UPDATE).

Also rejected en route: classic **Kahan** compensation for the sweep
accumulator (its `y = v − c` pre-subtraction re-absorbs the compensation
into the next 2²⁰-scale addend — computed `0.0` again; verified
numerically); the **Neumaier** two-sum variant works and shipped. And three
regression-matrix constructions whose base or replacement was numerically
singular for every path (±1 cascade to 2³⁴: `σ_min(B') = 1.5e-16`; diag-4
cascade: rescue-true `4.5e-13 <` ztol; spike-poison m=6: fresh LU burns the
4e6 spike entry and deflates its tail pivot to 0) — any single-shot
absorption reproducer necessarily has `σ_min(B') ⪅ δ·∏retained`, so the
"fresh factor succeeds" oracle is unsatisfiable without a multi-update
imbalance history.

**Shipped instead.** Always-on Neumaier-compensated scatter (recovers the
true pivot bit-for-bit on the regression basis) + `update_pivot_search` as an
always-on opt-in trajectory variant (bounded multipliers across chains),
default false. See `dev/research/issue-112-bg-update.md` §UPDATE and
`dev/decisions.md` 2026-07-10.

## Source Files
```
src/bin/bench.rs
src/bin/perf_probe.rs
src/bin/probe_panel_frag.rs
src/capi.rs
src/dense/block_ldlt32.rs
src/dense/equilibrate.rs
src/dense/factor.rs
src/dense/matrix.rs
src/dense/mod.rs
src/dense/rook.rs
src/dense/schur_kernel.rs
src/dense/solve.rs
src/error.rs
src/inertia.rs
src/io/mod.rs
src/io/mtx.rs
src/io/sidecar.rs
src/lib.rs
src/lu/condition.rs
src/lu/dense_factor.rs
src/lu/dense_matrix.rs
src/lu/dense_solve.rs
src/lu/dense_update.rs
src/lu/mod.rs
src/lu/scaling.rs
src/lu/sparse_factor.rs
src/lu/sparse_matrix.rs
src/lu/sparse_solve.rs
src/lu/sparse_symbolic.rs
src/lu/sparse_update.rs
src/numeric/condition.rs
src/numeric/factorize.rs
src/numeric/mod.rs
src/numeric/solve.rs
src/numeric/solver.rs
src/ordering/amd.rs
src/ordering/elimination_tree.rs
src/ordering/mod.rs
src/ordering/postorder.rs
src/ordering/schur.rs
src/scaling/hungarian.rs
src/scaling/infnorm.rs
src/scaling/mc64.rs
src/scaling/mod.rs
src/scaling/value_bound.rs
src/sparse/csc.rs
src/sparse/mod.rs
src/symbolic/column_counts.rs
src/symbolic/ldlt_compress.rs
src/symbolic/mod.rs
src/symbolic/profiler.rs
src/symbolic/small_leaf.rs
src/symbolic/supernode.rs
```

## Test Files
```
tests/amf_corpus_oracle.rs
tests/auto_strategy.rs
tests/blocked_ldlt.rs
tests/build_row_indices_trailing_invariant.rs
tests/column_renumbering.rs
tests/column_renumbering_parity.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
tests/factors_ld_export.rs
tests/fine_grained_delay.rs
tests/fma_opt_in_roundtrip.rs
tests/growth_flag.rs
tests/issue102_intrafront_deadlock.rs
tests/issue102_ordering_escalation.rs
tests/issue107_external_ordering.rs
tests/issue112_bg_update.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/issue91_preprocess_misfire.rs
tests/issue99_fma_front_gate.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/lu_adversarial_inputs.rs
tests/lu_dense.rs
tests/lu_dense_update_bg.rs
tests/lu_ft_widebump.rs
tests/lu_scaling.rs
tests/lu_sparse.rs
tests/lu_update_alloc_probe.rs
tests/lu_update_casctanks.rs
tests/maxfromm_parity.rs
tests/mc64_end_to_end.rs
tests/mc64_scaling.rs
tests/multi_rhs.rs
tests/n2_static_pivot_scaling.rs
tests/n3_parallel_profiler.rs
tests/n4_mc64_retry_latch.rs
tests/parallel_parity.rs
tests/parity.rs
tests/pivot_rejection.rs
tests/pounce_interface.rs
tests/profiler_smoke.rs
tests/property_tests.rs
tests/rook_rescue.rs
tests/rook_rescue_kkt.rs
tests/small_leaf_parity.rs
tests/solver_with_ordering.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/sqd_fast_path.rs
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
