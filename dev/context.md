# FERAL Context (auto-generated)

Generated: 2026-08-29T22:25:22Z

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
0952f99 feat: add Solver::reset_quality() to bound escalation lifetime (#192)
9b9e882 Merge pull request #188 from jkitchin/ci/codecov-coverage
1292984 ci: measure coverage with cargo-llvm-cov and report it to Codecov
ad0d96d Merge pull request #187 from jkitchin/docs/session-2026-08-19-05
b224ed1 docs: session checkpoint 2026-08-19-05 (pre-release review, 0.17.0 tagged)
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
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 448 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.35s

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
total factor time** on a matrix whose paying bucket is 91% of that time. Moving
three quarters of the panel share into BLAS-3 buys 1.3%: the two kernels cost
nearly the same per flop at this front shape, so the 53.5% panel share is not
recoverable time.

It also does not generalize. `bs = 48` and `bs = 64` are identical for any front
with `ncol ≤ 48`, and that is every other matrix sampled — `ncol` p90 is 1-19
across clnlbeam, dtoc2, marine_1600, rocket, steering, gasoil_3200, pinene_3200,
robot_1600, svanberg, nql180, qcqp1500-1c, cont5_2_4_l; only dtoc1nd is at 63. On
the two with any wide fronts at all the paired sweep finds nothing: nql180 0.990
(5/12 wins, tied with the default), qcqp1500-1c 0.994 (3/12). A 1.3% win on one
corpus matrix and a no-op elsewhere is below the bar for changing a global default.

**Kept from this attempt:** `block_size` is bit-neutral on all three matrices
swept — identical inertia, zero delayed pivots, identical residual, and an
identical hash over every `L`/`D` bit in storage order across
`bs ∈ {8,16,24,32,48,62,64}` (`dtoc1nd_0010` 9cb93f568423e6c0, `nql180_0000`
4f588093d6bac8c7, `qcqp1500-1c_0000` cfec17df1a4f8d38). So future retuning of it
is a performance-only change. Not yet established on a matrix that actually
delays a pivot — all three report `d0`.

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
tests/column_renumbering.rs
tests/column_renumbering_parity.rs
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
tests/issue102_intrafront_deadlock.rs
tests/issue102_ordering_escalation.rs
tests/issue107_external_ordering.rs
tests/issue112_bg_update.rs
tests/issue127_pipeline_split.rs
tests/issue128_supernode_nrow.rs
tests/issue177_parallel_entry_point_core.rs
tests/issue178_refine_cap.rs
tests/issue178_solve_into.rs
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
tests/lu_default_ordering.rs
tests/lu_dense.rs
tests/lu_dense_bump.rs
tests/lu_dense_update_bg.rs
tests/lu_ft_widebump.rs
tests/lu_hyper_sparse.rs
tests/lu_markowitz.rs
tests/lu_real_bases.rs
tests/lu_scaling.rs
tests/lu_sparse.rs
tests/lu_sparse_rhs.rs
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
tests/pounce710_refine_cap_nrhs2.rs
tests/pounce_interface.rs
tests/profiler_smoke.rs
tests/property_tests.rs
tests/refined_solve_core_stability.rs
tests/rook_rescue.rs
tests/rook_rescue_kkt.rs
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
