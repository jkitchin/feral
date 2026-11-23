# FERAL Context (auto-generated)

Generated: 2026-11-23T18:04:26Z

## Latest Session
File: dev/sessions/2026-11-23-01.md
```
# Session 2026-11-23-01

**Benchmark caveat (read first):** this container has no KKT corpus
(`data/matrices/` absent), so the bench covered only the 2 committed
fixtures. The numbers below are not comparable to 2026-11-19-06's
156 929-matrix run. No regression is shown, and none is ruled out on the
corpus. The next session with the corpus must re-run the bench.

## Goal
Review issue #246 (Auto at n ≳ 1M: AMF symbolic OOM on fhnw-binschedule1;
AMD 4× fill on Primal2_1000), diagnose it, and land step 1 of the fix.

## Accomplished
- Reproduced both matrices without pounce: `dev/scripts/mps_to_kkt.py`
  converts the Mittelmann MPS files to `[[I, Aᵀ], [A, −1e-8 I]]`. fhnw is
  within 0.2% of the issue's KKT. Primal2 is n = 3.86M (pounce: 5.10M) and
  takes the same routing branch.
- Diagnosis. The fhnw abort is a bad AMF ordering: prefix nnz(L) 1.39e10
  (AMD 4.37e8, MetisND 4.03e8), prefix RSS 1.3 GB. The cause is HAMF4's
  non-monotone `RMF / N` quantization, which is verbatim MUMPS 5.8.2
  (ana_orderings.F:4985-4994). Two inferences in the issue were wrong
  (AutoRace never runs AMF; prefixes are O(n + nnz(A))).
- 0f6eb9c: monotone `amf_quantize_score` (i64, clamp(round(rmf), 1, 1e18)).
  Tests first: 2 of 3 hand-arithmetic unit tests failed on the old code
  (3e9 → 1567) and pass after. The fixture-gated
  `tests/issue246_amf_quantization.rs` gives AMF 183 467 409 ≤ AMD
  436 861 025. `cargo test --workspace --release`: 1336 passed, 0 failed.
- End to end, fhnw `Auto` now completes under 13 GB: Amf/LdltCompress,
  nnz(L) 2.72e8, peak 4.27 GB, VmHWM 2.1 GB, 595 s.
- Primal2: `Auto` → AMD reproduces the issue (3.69e8, 8.30 GB). Explicit
  Amf gives 2.08e8 and 3.53 GB. AMF is unaffected by the fix.
- Posted the diagnosis and the end-to-end result on issue #246. Installed
  the pre-commit hooks (they were missing in this clone).

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

## Decisions Made
- AMF score quantization is monotone, a deliberate deviation from MUMPS
  (dev/decisions.md 2026-11-23).
```

## Git Status
```
0f6eb9c fix(amf): monotone RMF score quantization (issue #246)
ed366f4 docs: research note and plan for issue #246 AMF quantization
6bea9f9 journal: issue #246 diagnosis — AMF RMF/N quantization inversion
3d4d64c Merge pull request #244 from jkitchin/release-0.24.0
dcd8fe6 release: feral 0.24.0
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

test result: ok. 486 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.56s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-11-23-01.md)

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
trigger fired. Retuning `DELAY_CHAIN_GROWTH`: there is no wrong answer to
justify moving #221's tuning.
**Evidence.** dev/research/qcqp1500-1nc-chain-cb-2026-11-19.md. Mutation-checked.
`cargo test --release`: 1060/0.

## 2026-11-23 — AMF score quantization is monotone; drop MUMPS's `RMF / N` fall-through

**Context.** Issue #246: `Auto` → AMF on the fhnw-binschedule1 LP KKT
(n = 1.9M) exhausts 11–13 GB in the symbolic phase. The AMF ordering's
nnz(L) is 1.39e10, 32× AMD's. MUMPS 5.8.2 HAMF4 (`ana_orderings.F:4985-4994`)
stores the fill score in a 32-bit `INTEGER`, and for `RMF >= huge-1` it
stores `round(RMF / N)`. That map is not monotone: at n = 1.9M a score of
3e9 becomes ~1.6e3, so near-dense variables are filed as cheap. feral's
port copied this verbatim.
**Decision.** Quantize to `clamp(round(RMF), 1, 1e18)` in `i64`
(`amf_quantize_score`, `crates/feral-ordering-core/src/quotient_graph/algo.rs`).
This is a deliberate deviation from the MUMPS reference. Below `2^31` it
is bit-identical to MUMPS. Above that, feral's AMF no longer reproduces
MUMPS orderings.
**Rejected.** Keeping the `/N` branch but routing large n away from AMF:
that hides an ordering defect that also hits AMF users who pick it
explicitly. Clamping overflowed scores to `i32::MAX - 1` (last bucket):
monotone, but it throws away the order among saturated scores, which
`select_pivot_amf` can compare exactly for free with `i64` keys.
**Evidence.** `dev/research/issue-246-amf-quantization.md`. fhnw AMF
nnz(L) 1.388e10 → 1.835e8 (AMD 4.369e8, MetisND 4.031e8), Σc² 1.49e15 →
5.11e11. Primal2_1000 AMF is unchanged (1.802e8). Ordering time on fhnw
goes from 39 s to 128 s (accepted). The AMF corpus oracle bounds feral's
nnz_L from above, so it still holds where MUMPS itself inverts.
Journal 2026-11-23-01.

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
