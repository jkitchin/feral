# FERAL Context (auto-generated)

Generated: 2026-11-19T18:24:54Z

## Latest Session
File: dev/sessions/2026-11-19-03.md
```
# Session 2026-11-19-03

## Goal
Triage the open issues for quick wins; close #213; do #134 item F.

## Accomplished
- Triage of 13 open issues (journal 2026-11-19-03 17:30). Quick: #213,
  #227 (user's call), #134 F. Session-sized: #147 (needs MSRV decision),
  #211 (reproducible here). Rest large or waiting on others.
- #213 closed with a summary; remaining exact-inertia work split into
  #240.
- f437ae4: #134 F — BK 1x1 test order (swap-in before LAPACK extension)
  documented at the three kernel sites and pinned by
  `issue_134f_pivot_order_tests` (hand oracle; mutation-checked). No
  logic change. decisions.md entry.
- 4ddae78: clippy `--all-targets` fix in `examples/issue237_scope_probe.rs`
  (from #239; CI and the hook do not lint examples).
- `cargo test --release`: 1058 passed, 0 failed.

## Benchmark Results
```
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
Identical to 2026-11-19-02.

## Decisions Made
- Keep feral's BK 1x1 test order; documented and pinned
  (dev/decisions.md 2026-11-19).

## Abandoned Approaches
- None.

## Next Session Should
1. #134 F lands with this branch's PR; tick item F on #134.
2. #147 Rust 2024 edition once the MSRV policy is decided.
3. #211 wasm vs native factor values: reproducible in this container
   (`rustup target add wasm32-wasip1`, node 22). Narrow which reduction
   breaks the cross-lane-width bit-exactness claim.
4. CI does not lint examples/tests with clippy; consider adding
   `--all-targets` for the feral crate to the pre-commit/CI hook.
```

## Git Status
```
4ddae78 fix(examples): clippy field_reassign_with_default in scope probe
f437ae4 test+docs(#134 F): pin and document the BK 1x1 test order
f04ecce Merge pull request #239 from jkitchin/claude/blissful-ramanujan-w2nhfw
3bfc393 test(#237): InfNorm reuse through a parallel Solver's pool
62034af docs(session): 2026-11-19-02 checkpoint — #237 scope A/B, InfNorm reuse
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

test result: ok. 486 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.96s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-11-19-03.md)

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
Identical to 2026-11-19-02.

```

## Recent Decisions
254-258 -> 183-197 ms, refined residual unchanged (3.02e-13), analytic
inertia. tests/issue237_infnorm_reuse.rs. Journal 2026-11-19-02.
Revisit (default on) after a corpus run and the #237 reporter's in-loop
A/B.

## 2026-11-19 — Keep feral's 1×1 test order (swap-in before LAPACK extension); document and pin it (issue #134 F)

**Decision.** In Bunch–Kaufman pivot selection, when `|a_kk| < α·γ_0`,
feral tests the swap-in 1×1 (`|a_rr| >= α·γ_r`) before LAPACK's no-swap
extension (`|a_kk|·γ_r >= α·γ_0²`). `dsytf2` and Bunch & Kaufman (1977,
`bunch1977stable`) test them the other way round. The order is kept, and
is now documented at all three sites: dense `factor`, frontal
`scalar_pivot_step`, and the blocked panel path, whose bail-outs hand the
same decision to the scalar path. It is pinned by
`issue_134f_pivot_order_tests` in `src/dense/factor.rs`.

**Why not reorder for LAPACK parity.** Both branches are bounded-growth
1×1 pivots, and inertia is the same either way (any admissible pivot
sequence is a congruence; Sylvester). Only L and perm differ, and only
when both tests hold. Reordering would change pivot choices, and so
factors, on every matrix that hits that case. That needs the corpus gate
(inertia 100 %, residual partitions), which this container cannot run,
and buys only bit-parity with a LAPACK oracle, which no test relies on.

**Evidence.** Hand-oracle test on
`[[0.5, 1, 0], [1, 1.5, 2], [0, 2, 3]]`: both 1×1 tests hold at step 1;
feral pivots on 1.5 (perm `[1, 0, 2]`), then a 2×2 on the Schur
complement `[[-1/6, -4/3], [-4/3, 1/3]]`; inertia (2, 1, 0); scalar and
blocked kernels agree. With the swap test disabled (LAPACK order) the
test fails with perm `[0, 1, 2]`. Journal 2026-11-19-03.

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
