# FERAL Context (auto-generated)

Generated: 2026-11-23T17:23:33Z

## Latest Session
File: dev/sessions/2026-11-19-06.md
```
# Session 2026-11-19-06

## Goal
Fix `issue_215_delay_growth_cap::qcqp1500_1nc_factors_without_cascade_break`,
which fails locally on main. The fixture is gitignored, so CI skips it.

## Accomplished
- Diagnosis: bisect gives e702ac9 (#221) as first bad. All 4 CB nodes fire on
  the #221 delay-chain backstop (chain/subtree ratio 10.4–20.0 vs
  `DELAY_CHAIN_GROWTH` 10); the #215 budget is never exceeded. Inertia is
  exact, `n_tiny` 0, `inertia_uncertain()` 0. Research note
  `dev/research/qcqp1500-1nc-chain-cb-2026-11-19.md`; plan
  `dev/plans/qcqp1500-1nc-chain-cb.md` (aedbb79).
- 0d630ff: new `n_cascade_break_chain_nodes` (`NodeFactors::cascade_break_chain`,
  `SparseFactors`, `FactorStats`, Python, book). The test asserts zero
  non-chain CB. No solver behavior change.
- Both issue_215 tests pass (including the ignored column25 seeds 1/4/8).
  The mutation (flag forced false) is caught. `cargo test --release`:
  1060 passed, 0 failed.

## Test assertion relaxed (human-approved)
The assertion `n_cascade_break_nodes == 0` became
`n_cascade_break_nodes - n_cascade_break_chain_nodes == 0`.
Justification: the #221 backstop is an intentional carry-cost bound added
after this #215 test was written. On this matrix it produces an exact
factor with no floored pivots, and the test still checks both of those
properties. The user chose this option over (a) dropping the assertion
and (b) retuning `DELAY_CHAIN_GROWTH`.

## Benchmark Results
Full local corpus. Identical to 2026-11-19-05 except that sparse p90 is
1.58 vs 1.59 and dense small p90 is 1.43 vs 1.44 (noise).
```
KKT summary: 156929 matrices (154482 dense-eligible n <= 1000, 108 skipped n > 1000, 2339 parse-skipped)
  Inertia match: 154429/154482 (100.0%)
  Residual pass: 154207/154482 (99.8%)
  Worst residual: 2.46e-1 (POLAK6_0021)
Sparse solver: 154590/156929 total
  Inertia match vs MUMPS: 154531/154590 (100.0%)
  Residual pass: 154258/154590 (99.8%)
  Worst residual: 2.94e-4 (ERRINBAR_0824)
Dense Phase 2.8.1:  small p90 1.43 (<=2.0 PASS), medium p90 1.96 (<=3.0 PASS)
Sparse Phase 2.8.1: small p90 1.58 (<=2.0 PASS), medium p90 1.58 (<=3.0 PASS)
```

## Decisions Made
- Count chain-triggered cascade-break separately (dev/decisions.md 2026-11-19).

## Abandoned Approaches
- None.
```

## Git Status
```
3d4d64c Merge pull request #244 from jkitchin/release-0.24.0
dcd8fe6 release: feral 0.24.0
e8f47fb Merge pull request #245 from jkitchin/fix-qcqp1500-cascade-break
e2fd07c docs: session 2026-11-19-06 checkpoint, journal, decision
0d630ff fix(test): split cascade-break count by trigger; #215 test checks budget CB only
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

test result: ok. 486 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.48s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-11-19-06.md)

Full local corpus. Identical to 2026-11-19-05 except that sparse p90 is
1.58 vs 1.59 and dense small p90 is 1.43 vs 1.44 (noise).
KKT summary: 156929 matrices (154482 dense-eligible n <= 1000, 108 skipped n > 1000, 2339 parse-skipped)
  Inertia match: 154429/154482 (100.0%)
  Residual pass: 154207/154482 (99.8%)
  Worst residual: 2.46e-1 (POLAK6_0021)
Sparse solver: 154590/156929 total
  Inertia match vs MUMPS: 154531/154590 (100.0%)
  Residual pass: 154258/154590 (99.8%)
  Worst residual: 2.94e-4 (ERRINBAR_0824)
Dense Phase 2.8.1:  small p90 1.43 (<=2.0 PASS), medium p90 1.96 (<=3.0 PASS)
Sparse Phase 2.8.1: small p90 1.58 (<=2.0 PASS), medium p90 1.58 (<=3.0 PASS)

```

## Recent Decisions

**Evidence.** Exported `feral_*` symbols of `libferal.a` identical before
and after (9). Harness solve digests identical (2b7436a25c646d27,
e321bec7612e0d4f). `cargo test --release` 1058 passed, plus every
ordering crate and feral-diagnostics green; clippy `--workspace
--all-targets -D warnings` clean (with MSRV declared, clippy's
`incompatible_msrv` now guards new code). Six post-1.85 API uses
(`is_multiple_of`, `as_chunks`), all in tests and diagnostics tools,
replaced; the published libraries already compiled on 1.85. Five
`tail_expr_drop_order` notes reviewed: profiler mutex guards, a `Result`
temporary and a diagnostics thread pool now drop earlier, which is
harmless. Lockfiles unchanged. Journal 2026-11-19-04.

## 2026-11-19 — Count chain-triggered cascade-break separately; #215 test checks only budget-triggered CB

**Context.** `qcqp1500_1nc_factors_without_cascade_break` (#215) asserted
`n_cascade_break_nodes == 0`. Since e702ac9 (#221) the delay-chain
backstop trips at 4 fronts on that matrix (chain/subtree ratios 10.4–20.0
against `DELAY_CHAIN_GROWTH` = 10). The #215 budget is never exceeded there.
Inertia is exact; `n_tiny` and `inertia_uncertain()` are both 0.
**Decision.** Add `n_cascade_break_chain_nodes` (`NodeFactors::cascade_break_chain`,
`SparseFactors`, `FactorStats`, Python). The #215 test asserts zero
*non-chain* cascade-break, and keeps its exact-inertia and no-floored-pivot
checks. This relaxes the assertion, and the user approved it in session
2026-11-19-06.
**Rejected.** Dropping the count assertion: it would no longer say which
trigger fired. Retuning `DELAY_CHAIN_GROWTH`: there is no wrong answer to
justify moving #221's tuning.
**Evidence.** dev/research/qcqp1500-1nc-chain-cb-2026-11-19.md. Mutation-checked.
`cargo test --release`: 1060/0.

## Recent Tried-and-Rejected
the contract for callers that use `SparseLu` as a rank-reveal probe. With
PR #234 at dd98cb8, pounce's
`crates/pounce-cli/tests/issue_1076_qgfrdxpn_active_set_prune_cap.rs`
(`convex_qp_qgfrdxpn`, `solver_selection=qp-active-set`) went from
SolveSucceeded, 3 553 iterations, 6.0 s on feral 0.22 to
MaximumIterationsExceeded at 27 811 iterations, 50.6 s (objective 1.00569e11
at the cap vs 1.00791e11 optimal); it reproduced on re-run. The other 2 156
pounce tests passed, and pounce's default fixture sweep diffed empty
because its default routing never reaches the rank probe. The only signal
I had measured was fill (+0.27 % geomean over 40 synthetic bases, 36/40
worse), which says nothing about which rows a rank reveal declares
dependent.

**Replaced with.** Exact reproduction of the 0.22 heap contents: peel-phase
pushes are logged and heapified when the singleton stacks first run dry,
minus entries for eliminated rows/columns. Digests are bit-identical to 0.22
on 3 real and 40 synthetic bases, pinned by `tests/lu_markowitz_golden.rs`.
Lesson: on a factorization other code uses for rank decisions, a change to
pivot order is a behaviour change even when fill and residuals are fine;
gate it on a golden pivot test, not on fill.

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
src/numeric/contrib.rs
src/numeric/factorize.rs
src/numeric/mod.rs
src/numeric/reanalysis.rs
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
tests/common227/mod.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/early_assembly_parity.rs
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
tests/issue227_flops.rs
tests/issue227_never_golden.rs
tests/issue227_reanalysis.rs
tests/issue237_infnorm_reuse.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/issue91_preprocess_misfire.rs
tests/issue99_fma_front_gate.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_215_delay_growth_cap.rs
tests/issue_221_delay_chain_backstop.rs
tests/issue_221_forced_front_parity.rs
tests/issue_221_fully_summed_2x2_partner.rs
tests/issue_221_quality_rung.rs
tests/issue_221_tpp_tail.rs
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
tests/lu_markowitz_golden.rs
tests/lu_real_bases.rs
tests/lu_refactor_inplace.rs
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
tests/wide_star/mod.rs
tests/wide_tree_memory.rs
```
