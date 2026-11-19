# FERAL Context (auto-generated)

Generated: 2026-11-19T12:52:17Z

## Latest Session
File: dev/sessions/2026-11-19-02.md
```
# Session 2026-11-19-02

## Goal
#237 follow-up after the reporter confirmed PR #238 (warm factor
-25..-28 %, FERAL level with / ahead of MA97): (1) check their 8-thread
"scope" slowdown, (2) opt-in InfNorm reuse, (3) begin_recycle if simple.

## Accomplished
- 5684b6b: `examples/issue237_scope_probe.rs` (parallel-driver phase
  telemetry; builds against 0.23.0 too). A/B vs 0.23.0, MetisND,
  alternating builds: scope 124-134 vs 121-134 ms (4 thr), 218-225 vs
  217-224 ms (2 thr) — no regression. Warm factor 522-564 -> 254-279 ms
  (4 thr). 8 threads not testable here.
- 6094066 / 0761a18: `Solver::with_infnorm_reuse`, `InfNormReuse`,
  `infnorm_reuse_hit_count`, `scaling::infnorm::rows_within_ratio`.
  Default off (digests unchanged). With the default policy: warm factor
  254-258 -> 183-197 ms (MetisND, 4 thr), refined residual unchanged,
  analytic inertia. 8 new tests; mutation-checked.
- `cargo test --release`: 1056 passed, 0 failed.

**Side finding (not acted on):** on the chain KKT the default ordering
(Auto -> Amf, 80 490 supernodes) is ~4x slower than MetisND (1081 vs
283 ms per warm factor, 4 thr). Belongs to #203 / #209 ordering choice.

**Not done:** begin_recycle lazy take (journal 12:45: needs reworking
4 consumers + the parallel driver's slot mutexes for ~4 ms).

**Not pushed:** PR #238 is unmerged; these commits are on local branch
`issue237-followup` and go to `claude/blissful-ramanujan-w2nhfw` once
#238 merges (the branch is restarted from main then).

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
Identical to 2026-11-19-01.

## Decisions Made
- Opt-in guarded InfNorm reuse, default off (dev/decisions.md 2026-11-19).

## Abandoned Approaches
- None.
```

## Git Status
```
0761a18 feat(#237): opt-in InfNorm scaling reuse across same-pattern factors
6094066 docs(#237): research note and plan for opt-in InfNorm reuse
5684b6b bench(#237): parallel-driver phase probe for build-to-build A/B
441ff73 docs(session): 2026-11-19-01 checkpoint — #237 step 3 parallel KR sweeps
ff7f8bb perf(#237): parallel Knight-Ruiz sweeps on a rayon worker
```

## Test Status
```
test symbolic::tests::schur_symbolic_single_schur_index ... ok
test symbolic::tests::schur_symbolic_tail_invariant_reversed_user_order ... ok
test symbolic::tests::schur_symbolic_tail_invariant_user_order ... ok
test symbolic::tests::symbolic_factorize_amf_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_auto_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_external_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 485 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.72s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-11-19-02.md)

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
Identical to 2026-11-19-01.

```

## Recent Decisions
public API whose misuse returns a silently wrong factor (the REG-1 / #121
failure class). Decided with the user, 2026-11-18.

**Evidence.** Journal 2026-11-18-02: outside-driver work per warm call
went from ~65-80 ms to ~20 ms with no API change (hash 31-44 ms -> 0,
finite scan 10-13 -> 5-8 ms, escalation check 6-10 -> 0.7 ms); the
compare (6-9 ms) and begin_recycle (4-7 ms) remain.

## 2026-11-19 — Opt-in, guarded InfNorm scaling reuse (issue #237 step 4, #202 Defect 1)

**Decision.** `Solver::with_infnorm_reuse(Option<InfNormReuse>)` lets a
host reuse a computed `InfNorm` vector across same-pattern factorizations
(age limit + one-pass row-norm ratio guard; default policy max_age 5,
ratio 2). The default stays `None`: per-call recomputation, bit-identical
to before.

**Why.** The scaling is cap-bound (warm starts and lower caps rejected
2026-08-09 / 2026-08-15), so only skipping it saves time. Making it the
default changes numerics and needs the corpus gate, unavailable here.
This is not the #49 defect recorded on 2026-05-2x: that was the MC64
cache *unintentionally* replaying an InfNorm vector with no guard of its
own. Here reuse is explicit, InfNorm-only, age-bounded and checked
against the current values; inertia cannot change (D·A·D has the
inertia of A for any positive diagonal D).

**Evidence.** Chain KKT n = 155 970 (MetisND, 4 threads): warm factor
254-258 -> 183-197 ms, refined residual unchanged (3.02e-13), analytic
inertia. tests/issue237_infnorm_reuse.rs. Journal 2026-11-19-02.
Revisit (default on) after a corpus run and the #237 reporter's in-loop
A/B.

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
