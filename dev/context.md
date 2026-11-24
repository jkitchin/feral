# FERAL Context (auto-generated)

Generated: 2026-11-24T03:14:39Z

## Latest Session
File: dev/sessions/2026-11-24-02.md
```
# Session 2026-11-24-02

**Benchmark caveat:** no KKT corpus in this container; the bench covers
only the 2 committed fixtures (unchanged from 2026-11-24-01).

## Goal
Issue #248: AutoRace recomputed the MC64 matching in every arm (fhnw-binschedule1:
1294 s for the four-arm race). Share it, keep selection bit-identical, and
re-measure fhnw-binschedule1 with a per-stage profile.

## Accomplished
- 01003e7: `Mc64Memo` in src/symbolic/mod.rs, one per race, threaded
  through resolve_prefix → symbolic_prefix_preprocess_auto →
  symbolic_prefix_concrete. Stores only an uninterrupted Ok matching (the
  #246 interrupt check moved into `get_or_compute`).
  `symbolic_factorize_race_over` takes the candidate list for tests.
  examples/issue248_race_mc64.rs times a race on a .mtx file.
  Plan: dev/plans/issue-248-shared-mc64.md.
- Tests: `autorace_computes_mc64_once_per_race` (mutation with a per-arm
  memo fails it, 2 vs 1) and
  `autorace_shared_mc64_selection_matches_independent_arms`. Its oracle is
  each candidate run alone, then min factor_nnz_estimate, first wins on
  ties. It gives the same perm on 3 matrices, for both candidate sets.
  cargo test --workspace --release: 1370 passed, 0 failed.
- fhnw-binschedule1 (rebuilt from plato.asu.edu MPS, n=1 914 525, nnz 10 525 851):
  four-arm AutoRace **1007.3 s** (was 1294.4 s). MC64 ran once (51.8 s).
  Same winner, Amf/LdltCompress, factor_nnz_estimate 3.2588e8.
  VmHWM 2.27 GB. The winner's AMF ordering stage alone took 334 s. The ~955 s
  that is not MC64 is ordering work across 8 prefixes, so **the orderings
  dominate**, not MC64.

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
Identical to 2026-11-24-01.

## Decisions Made
- None architectural (an internal memo; no selection change).

## Abandoned Approaches
- None.
```

## Git Status
```
67ed188 dev: journal for session 2026-11-24-02 (issue #248) and refreshed context
01003e7 perf(symbolic): compute MC64 once per AutoRace, not once per arm (#248)
4051231 Merge pull request #247 from jkitchin/claude/zen-hamilton-tra4mp
41f809e docs: accept issue #246 step 2 with its measured PDE2 cost
698eaa1 test(interrupt): guard the single-thread assumption with a poll census
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

test result: ok. 495 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.13s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-11-24-02.md)

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
Identical to 2026-11-24-01.

```

## Recent Decisions
nnz(L) 2.716e8, predicted peak 4.27 GB (the issue's two-arm race picked
AMD at 6.79 GB). Cost: 1294 s symbolic in this container vs the issue's
476 s for the two-arm race; each arm recomputes MC64 for its LdltCompress
pass (follow-up). A slow race can now be bounded with `Interrupt`.
Unvalidated on pounce's L1_sixm* / L2CTA3D.

## 2026-11-24 — `Auto`: `n > 100_000 && avg_deg < 5` → AMF (was AMD, #50)

**Context.** Issue #246: on Primal2_1000 (n = 3.86M), `Auto` → AMD predicts
an 8.30 GB peak and refactors in 999 s; AMF predicts 3.53 GB and refactors
in 84 s with identical inertia. #50 chose AMD for this branch on powerflow22,
when AMF still had the large-n quantization inversion (fixed in 0f6eb9c)
and was not measured.
**Decision.** Route the branch to `Amf` (commit 1de3977). With this, `Auto`
resolves to AMF for every n > 0. The now-redundant rules are kept for a
separate simplification decision.
**Rejected.** Racing AMD against AMF in this branch: twice the symbolic cost
on every large low-degree matrix, for a choice where AMF's fill was never
worse on the five public LPs measured.
**Evidence.** Maintainer-machine gate (PR #247 comment 7119267924):
- powerflow22: refactor 0.69 s AMD vs 0.68–0.71 s AMF, peak RSS 3.94 vs
  3.92 GB, inertia identical.
- **PDE2_0000: AMF is ~15% slower on the first factor (1.47 s vs
  1.28–1.38 s) and ~17% slower per refactor (1.11–1.12 s vs 0.95–0.96 s),
  i.e. +0.16 s, with inertia identical.** This is a measured cost of this
  decision, accepted against Primal2's 12× refactor speed-up.
- laptime (#208) is unreachable (average degree > 7, n < 1M).

The corpus subset (n > 100k, average degree < 5) and `amf_corpus_oracle.rs`
were not run. See dev/research/issue-246-low-degree-routing.md "Gate results".

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
tests/issue246_amf_quantization.rs
tests/issue246_interrupt.rs
tests/issue246_interrupt_threads.rs
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
