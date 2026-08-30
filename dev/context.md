# FERAL Context (auto-generated)

Generated: 2026-08-30T02:51:11Z

## Latest Session
File: dev/sessions/2026-08-29-01.md
```
# Session 2026-08-29-01

## Goal

Fix issue #192: `Solver::increase_quality`'s escalation is permanent and
unscopable, so an escalation chosen for one hard factorization governs
every factorization for the remaining life of the `Solver`.

## Benchmark Results

**The exit partition was not measured this session.** The 153k-matrix
corpus is not present in this container, so both partitions report
`N/A` and there is no comparison against session 2026-08-19-05's
1.58 / 1.58 to report — favourable or otherwise.

```
--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A

--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
```

What did run is clean, and the diff has no mechanism by which it could
regress performance: it is a pure API addition whose new code path runs
only when a caller calls `reset_quality()`. No numeric kernel, ordering,
scaling, or pivoting code was touched, and the forward-ladder tests
U1–U5 pass unchanged. That is an argument, not a measurement, and the
next session with corpus access should confirm it.

## Accomplished

### `Solver::reset_quality()` (issue #192, commit `0952f99`)

New `Solver::reset_quality() -> bool` and its Python binding. Reverts
```

## Git Status
```
be1e234 feat(refine): report the achieved componentwise omega in RefineOutcome
9c64660 test(refine): stop two #190 tests from claiming more than they prove
429dfca docs: describe the refinement default that actually ships
9b5323c perf(solve): size the multi-RHS workspace on the dispatch threshold
47f332e fix(refine): a non-finite iterate must not certify as backward stable
```

## Test Status
```
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test numeric::solve::tests::cb_coarsening_threshold_is_arithmetically_inert ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 466 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 1.90s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-29-01.md)


**The exit partition was not measured this session.** The 153k-matrix
corpus is not present in this container, so both partitions report
`N/A` and there is no comparison against session 2026-08-19-05's
1.58 / 1.58 to report — favourable or otherwise.

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A

--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)

What did run is clean, and the diff has no mechanism by which it could
regress performance: it is a pure API addition whose new code path runs
only when a caller calls `reset_quality()`. No numeric kernel, ordering,
scaling, or pivoting code was touched, and the forward-ladder tests
U1–U5 pass unchanged. That is an argument, not a measurement, and the
next session with corpus access should confirm it.

```

## Recent Decisions

**Scope of the change.** The escalation ladder is untouched — its rungs,
the `0.75` exponent, `pivtol_max` — and unit tests U1–U5 pass unchanged,
so a caller that never calls `reset_quality` sees byte-identical
behaviour. The reset touches only the two escalated parameters and the
level, mirroring what `increase_quality` leaves alone: the cached
symbolic factorization survives (scaling-invariant since the β refactor
moved scaling to the numeric phase), so re-baselining costs no
re-analysis exactly as escalating costs none. Pinned by integration test
`i9_reset_quality_rebaselines_without_invalidating_symbolic`.

## 2026-08-29 — the escalation baseline is snapshotted lazily, not at construction (issue #192)

**Decision.** `reset_quality` restores a `QualityBaseline { scaling,
pivot_threshold }` captured on the transition *out of*
`QualityLevel::Baseline` — i.e. at the instant the ladder starts — held
in `Option<QualityBaseline>` and cleared by every reset. Not captured in
`with_params`.

**Why.** The `with_*` builders are consuming and run *after*
`with_params`, so `Solver::with_params(np, sn).with_scaling(Identity)`
would have a construction-time snapshot recording `np`'s strategy, and a
reset would silently discard the caller's builder configuration. The
lazy snapshot also makes the round trip exact by construction: it
records a state the solver demonstrably occupied, so `reset` →
`increase` retraces the same rungs a freshly constructed `Solver` would
— the property downstream needs when re-baselining at a loop boundary.
Pinned by `r6_reset_quality_preserves_builder_configured_scaling` (would
fail under a construction-time snapshot) and
`r4_reset_quality_from_exhausted_restarts_identical_ladder`.

## Recent Tried-and-Rejected
semantically identical, and giving the compiler `&[[f64; 4]]` instead of
`&[f64]` may well be neutral or better for codegen. That is a
*hypothesis*. This is the hottest loop in the factorization; its unroll
depth and `into_remainder()` cleanup are a measured design
(`dev/research/dense-kernel-*.md`), and the container this was found in
has no corpus, so the change could not be benchmarked — the exit
partition reports N/A here. Landing an unmeasured edit to that loop to
satisfy a style lint inverts the project's order of operations.

**What was done instead.** A file-scoped
`#![allow(clippy::chunks_exact_to_as_chunks)]` in `schur_kernel.rs` with
the reasoning in a comment beside it. The three other sites the lint
flagged — `diag_schur_parity.rs` (x2) and `diag_acopr14.rs` — are
byte-decoding loops in diagnostic binaries, not hot paths, so those took
the real rewrite (and lost a `copy_from_slice` each).

**Still open.** Whether `as_chunks` in the kernel is neutral, a win, or
a loss is unmeasured and unclaimed. A session with corpus access should
sweep it and either land the rewrite with numbers or record the
regression here. Until then the `allow` is a deferral, not a verdict.

## Source Files
```
src/bin/bench.rs
src/bin/perf_probe.rs
src/bin/probe_ft_eta.rs
src/bin/probe_lu_phases.rs
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
src/env.rs
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
src/lu/markowitz.rs
src/lu/mod.rs
src/lu/scaling.rs
src/lu/sparse_factor.rs
src/lu/sparse_hyper.rs
src/lu/sparse_matrix.rs
src/lu/sparse_solve.rs
src/lu/sparse_symbolic.rs
src/lu/sparse_triangular.rs
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
tests/cb_core_choice_ignores_env.rs
tests/cb_solve_parity.rs
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/env_knob_parsing.rs
tests/env_knob_scan.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
tests/factors_ld_export.rs
tests/fine_grained_delay.rs
tests/fma_opt_in_roundtrip.rs
tests/golden_bits.rs
tests/growth_flag.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/issue102_intrafront_deadlock.rs
tests/issue102_ordering_escalation.rs
tests/issue107_external_ordering.rs
tests/issue112_bg_update.rs
tests/issue127_pipeline_split.rs
tests/issue128_supernode_nrow.rs
tests/issue177_parallel_entry_point_core.rs
tests/issue178_refine_cap.rs
tests/issue178_solve_into.rs
tests/issue190_componentwise_default.rs
tests/issue190_refine_target.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/issue91_preprocess_misfire.rs
tests/issue99_fma_front_gate.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/lu_adversarial_inputs.rs
tests/lu_default_ordering.rs
tests/lu_dense_bump.rs
tests/lu_dense_update_bg.rs
tests/lu_dense.rs
tests/lu_ft_widebump.rs
tests/lu_hyper_sparse.rs
tests/lu_markowitz.rs
tests/lu_real_bases.rs
tests/lu_scaling.rs
tests/lu_sparse_rhs.rs
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
tests/pounce710_refine_cap_nrhs2.rs
tests/profiler_smoke.rs
tests/property_tests.rs
tests/refined_solve_core_stability.rs
tests/rook_rescue_kkt.rs
tests/rook_rescue.rs
tests/small_leaf_parity.rs
tests/solver_with_ordering.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/sqd_fast_path.rs
tests/static_assembly_maps.rs
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/task_plan_parity.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
