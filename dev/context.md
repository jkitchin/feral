# FERAL Context (auto-generated)

Generated: 2026-10-20T21:44:12Z

## Latest Session
File: dev/sessions/2026-10-19-01.md
```
# Session 2026-10-19-01

## Benchmark note (read first): local corpus is the 2 synthetic fixtures

`cargo run --bin bench --release` found only the 2 generated
synthetic-regression fixtures, as in 2026-10-16-01. Both pass inertia and
residual, and no oracle timings are available, so no factor-ratio p90 can
be compared. The comparable evidence is the HEAD-vs-fix sweep over
`data/matrices/kkt-pounce` (370 matrices) below.

## Unfavourable results (read second)

- **brainpc1 and brainpc2 still time out (300 s) in pounce with the fix.**
  The factorization is no longer the cost. A mid-run `sample` puts almost
  all busy time in MC64 `hungarian_match_instrumented`, while the IPM keeps
  cycling through restoration (brainpc1 reached 1397 iterations).
- brainpc0 now solves (68.8 s, 1512 iterations, 709 restoration), but the
  issue reports about 1 s on pounce 0.12/feral 0.18. Baseline 0.20 timed
  out at 600 s.
- brainpc3 i38/1: the fix counts 6901 negative, eigvalsh gives 6900, and
  HEAD gave 6900. The matrix has 129 eigenvalues below n·eps·‖A‖, so it
  is numerically singular and falls under the singular-matrix exemption.
  It is still a data point against.

## Addendum: brainpc1/2 follow-up (same session)

- pounce 9a97f997 (the reporter's 0.18 commit), `--no-options-file`:
  - feral 0.18: brainpc1 34.7 s, brainpc3 51.9 s, brainpc0 7.4 s
  - the fix: brainpc1 and brainpc3 time out at 300 s; brainpc0 62.6 s
- The trajectories split at iter 42. On that shared KKT:
  - eigvalsh: 6902 negative
  - the fix: 6902, n_tiny 0, refined berr 1.5e-27
  - 0.18: 6901, n_tiny 665
  So 0.18 converged on a wrong count.
- On the fix's trajectory, MC64 costs ~630 ms per call (0.18's: ~100 ms).
  This is inherent hub-column Hungarian cost on those iterates; the
  scaling code is unchanged since v0.18.0.
- POUNCE_FERAL_SCALING=infnorm: brainpc1 18.8 s but brainpc0 228 s.
  Scaling only reshuffles the trajectory.
- Decision (user): open the PR and test whether the remaining stall can
  be fixed on the pounce side with this feral.
- New probe: `diag_221_mc64_cost`.

## Goal

Fix issue #221: brainpc3 KKT factor 0.038 s (0.18) → 9.4 s (0.19/0.20).

## Accomplished

- Root cause, from the reporter's own matrices dumped with pounce:
```

## Git Status
```
80a8094 fix(solver): a rung that forces more pivots is undone (#221)
f849a14 dev: journal verification of the #221 review fixes
1e5795f docs(changelog): two-phase factorization of fronts that cannot delay (#221)
b0845ba test: delay-chain backstop, max_abs_l on a 2x2, floored forced fronts (#221)
140e395 fix(solver): zero pivots make a factor inadmissible for a rung (#221)
```

## Test Status
```
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test numeric::solve::tests::cb_coarsening_threshold_is_arithmetically_inert ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test numeric::solver::tests::warm_refactor_recycles_output_buffers_issue_153 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 460 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.32s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-10-19-01.md)

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)

--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

```

## Recent Decisions
it is not judged worse. Admissibility is stricter: such a factor never
declines a rung, which is the pre-#221 behaviour for it.

**Evidence.** tests/issue_221_quality_rung.rs: `[[1,1],[1,1]]` (inertia
(1,0,1), max |L| = 1, by hand) and `diag(1,0,1)` now proceed; more zero
pivots at the raised rung undo the ladder; same count climbs; a rung
survives an `invalidate_factors` call and is judged by the next factor.
All four fail on ed292fc.

## 2026-10-21 — a rung that forces more pivots is a failed rung (issue #221)

**Decision.** `FactorStability::worse_than` counts every pivot the factor
gave up on: floored (`n_tiny`), zeroed (`n_zero`) and force-accepted below
the threshold (`n_forced`). If the factor at the raised threshold gave up
on more of them than the factor that prompted the raise, the rung is
undone. This replaces the 2026-10-20 choice of floored pivots only.

**Why.** That choice dropped forced accepts as "relative to the
threshold". Relative to the *raised* threshold, a forced accept is a
pivot the rung could not place within its own bound, which is what the
rung exists to do. After the parity fix (1f16980), brainpc1's first rung
forced 2 pivots where it had floored 1. The floored-only test kept the
ladder, which climbed three more rungs to 2.9e-3 with the count unchanged
(165 s against 74 s; pounce verified this is deterministic and bisected
it). Summing the three counts also means a pivot that moves between them
is not a change.

**Evidence.** pounce sweep-fixtures.sh, 220 runs: identical to the
previous rule on every line. brainpc1 74.5 s, 1 escalation; nql180
54.9 s; brainpc2 67.2 s. Unit test Q1 in `src/numeric/solver.rs`.

## Recent Tried-and-Rejected
- **Undo one rung, not to the ladder start.** nql180 stopped at
  u = 3.2e-5 (rungs 1e-8 → 1e-6 → 3.2e-5 → 4.2e-4, the last worse with
  5548 violations). It kept ~70k delays per factor: 224 s, 33 iterations.
  The earlier rungs had never cured the one-short count.
- **A persistent rung record.** pooling_rt2stp compared a later iterate
  (2 floored pivots) with the factor that prompted a rung three calls
  earlier. A clean factor in between had declined the next rung, so the
  stale comparison undid the whole ladder.
- **Lexicographic (violations, then max|L|).** It judged brainpc1's first
  rung worse on max|L| alone and undid a rung that was not the problem.
- **Floored pivots only (n_tiny), forced accepts ignored.** After the
  parity fix 1f16980, brainpc1's first rung forced 2 pivots where it had
  floored 1. The rule kept the ladder, which climbed three more rungs to
  2.9e-3 with the inertia count unchanged: 165 s against 74 s with the
  shipped rule. pounce confirmed it is deterministic and bisected it.

An a-posteriori-pivoting (APP) blocked kernel for cheaper delays was also
considered and set aside, not tried: it would break the blocked/scalar
bit-parity contract (`tests/blocked_ldlt.rs`, `tests/golden_bits.rs`),
which the maintainer chose to keep.

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
tests/factor_output_recycling.rs
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
tests/issue_215_delay_growth_cap.rs
tests/issue_221_delay_chain_backstop.rs
tests/issue_221_forced_front_parity.rs
tests/issue_221_fully_summed_2x2_partner.rs
tests/issue_221_quality_rung.rs
tests/issue_221_tpp_tail.rs
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
tests/issue203_ordering_race.rs
tests/issue204_work_estimate.rs
tests/issue205_ordering_info.rs
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
