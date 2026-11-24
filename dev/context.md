# FERAL Context (auto-generated)

Generated: 2026-11-24T02:50:39Z

## Latest Session
File: dev/sessions/2026-11-24-01.md
```
# Session 2026-11-24-01

**Benchmark caveat (read first):** no KKT corpus in this container
(`data/matrices/` absent). The bench covered only the 2 committed fixtures,
so it is not comparable to corpus runs.

## Goal
Address the pounce-side review on PR #247 (comment 7116780548): a time limit
for the symbolic phase, wider AutoRace candidates at large n, and the memory
caveat in the book. Continues session 2026-11-23-01 on the same branch.

## Accomplished
- 176c423: book table for real vs predicted memory (1.4–1.8× sequential,
  ≈2.6× parallel) next to `max_factor_bytes`.
- 0729754: research note and plan
  (dev/research/issue-246-symbolic-deadline.md).
- c69f7fe: **symbolic-phase interrupt.** `feral::Interrupt` (a deadline
  and/or a cancel flag) is a scoped thread-local in feral-ordering-core,
  polled at safe points:
  - AMD/AMF every 64 pivots;
  - METIS/SCOTCH per recursion node, level, trial and FM pass;
  - KaHIP per recursion node;
  - MC64 every 64 columns;
  - between pipeline stages and race arms.

  It returns `FeralError::Interrupted { stage }`, which `Solver::factor`
  reports as `FatalError`. Wired through `SupernodeParams::interrupt`,
  `Solver::with_interrupt` / `set_interrupt`, and
  `*_order_interruptible` in each ordering crate.
  - Both races used to swallow arm errors; they now propagate the interrupt.
  - Tests first. cargo test --workspace --release: 1361 passed, 0 failed.
  - fhnw `Auto` with a 60 s deadline stopped at 60.08 s, and `AutoRace`
    with 30 s at 30.04 s. Primal2 MetisND with 10 s stopped at 10.02 s.
- b9ad7cc: **AutoRace at n ≥ 1M** races {Amd, MetisND, ScotchND, Amf}.
  - On fhnw it now picks Amf/LdltCompress (2.716e8 nnz, 4.27 GB peak). The
    two-arm race picked AMD at 6.79 GB.
  - The race took 1294 s vs 476 s (each arm recomputes MC64).

- Second PR #247 review (comment 7119267924):
  - **Step 2 gate passed on the maintainer's machine.** powerflow22 shows no
    regression. PDE2_0000 shows AMF ~15–17% slower (+0.16 s), with identical
    inertia. laptime is unreachable.
  - Step 2 is accepted; decisions.md, CHANGELOG and the research note now
    state the PDE2 cost.
  - 698eaa1: documented the interrupt's single-thread assumption, added
    `interrupt::current()` and an opt-in poll census, plus per-crate and
    feral-level census tests (a mutation check caught a worker thread:
    `unscoped: 169`). 1368 passed, 0 failed.

## Benchmark Results
```

## Git Status
```
4051231 Merge pull request #247 from jkitchin/claude/zen-hamilton-tra4mp
41f809e docs: accept issue #246 step 2 with its measured PDE2 cost
698eaa1 test(interrupt): guard the single-thread assumption with a poll census
8d6ee65 session: 2026-11-24-01 checkpoint (PR #247 review: interrupt, race)
b9ad7cc feat(ordering): AutoRace races four orderings at n >= 1M (#246)
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

test result: ok. 493 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.11s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-11-24-01.md)

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
Identical to 2026-11-23-01 (same 2 fixtures).

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
