# FERAL Context (auto-generated)

Generated: 2026-11-13T21:24:16Z

## Latest Session
File: dev/sessions/2026-11-13-02.md
```
# Session 2026-11-13-02

## Goal
Issue #232: `SparseLu::factor_markowitz` ~6.5x slower than HiGHS HFactor
on a near-triangular LP basis (heap churn + per-column allocation).

## Accomplished
- f3d99e0: `examples/lu_markowitz_triangular.rs` — timing + bitwise digest
  harness on a synthetic basis shaped like the issue's (m = 17 470, ~13k
  one-entry columns, 1 459-col kernel, little fill) and on the committed
  `.mtx` bases. The issue's actual basis is in discopt and could not be
  attached to this session.
- dce0dc6: Markowitz count heaps built only when the singleton stacks first
  run dry (over the kernel); active submatrix in packed `Lists<T>`;
  per-step dedup of the candidate search.
- 6a5afd3: `remap_and_sort_l` without per-column Vecs; `u_above` pre-sized.
- Timings (median, default `LuParams`):

  | basis | before | after |
  |---|---|---|
  | piperout-shape seed 1 | 12.4 ms | 5.4 ms |
  | piperout-shape seed 2 | 12.9 ms | 4.9 ms |
  | piperout-shape seed 3 | 13.4 ms | 5.8 ms |
  | QPLIB_3852 | 0.78 ms | 0.35 ms |
  | bchoco06_illcond | 0.47 ms | 0.26 ms |
  | QPLIB_1157 (fill-bound) | 12.9 ms | 12.7 ms |

- `cargo test --release --no-fail-fast`: 1022 passed, 0 failed.

**Unfavourable:** the dedup changes kernel pivots and fill goes up slightly
— geomean +0.27 % over 40 synthetic bases, max +1.61 %, 36 of 40 worse;
QPLIB_1157 +0.54 %. See decisions.md 2026-11-13.

**Not reached:** the issue's HiGHS parity (~2.7 ms on its basis). About 2.4x
was recovered on the synthetic shape; the remaining cost is mostly
`SparseLu`'s own storage (one Vec per U row, `u_above` Vec<Vec>, ~12
m-length scratch arrays allocated, zeroed and dropped per factorization).
The issue's real basis was not measured.

## Continuation: in-place refactor (same session)
- 132c16e: `SparseLu::refactor_markowitz` + `assemble_into` shared fill
  path; `SparseLu::refactor` routed through it under Markowitz.
- c25e89b: example times fresh-replacing-held vs in-place, checks digests.
- **Correction to the table above:** its "after" times excluded dropping
  the factor. Including the drop (what a caller replacing its factor
  pays), fresh is 6.36 / 6.06 / 6.81 ms; in place 4.97 / 4.61 / 5.32 ms.
  QPLIB_3852 0.371 -> 0.246, bchoco06 0.261 -> 0.202, QPLIB_1157 12.75 ->
  12.15 ms. Overall on the synthetic shape: ~12.9 ms -> ~5.0 ms.
- Only callers that keep one `SparseLu` get the in-place gain. discopt
  currently builds a new factor per refactorization.
```

## Git Status
```
615e8ad lu_markowitz_triangular: read discopt's basis dump (#232)
10be31b dev logs: record the pounce regression and the 0.22 pivot-order fix (#232)
e8a1d59 Markowitz LU: restore the 0.22 pivot order exactly (PR #234 regression)
dd98cb8 Merge origin/main into claude/gallant-cray-pjv1tt (PR #234)
41e7c2b Merge pull request #233 from jkitchin/claude/eloquent-galileo-zssfkc
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

test result: ok. 475 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.88s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-11-13-02.md)

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
Identical to 2026-10-23-01.

```

## Recent Decisions
Markowitz pivoting. `assemble_into` destructures `SparseLu` exhaustively.

**Why.** The in-place path is only safe if it leaves exactly the state a
fresh factor would, and two parallel builders drift. One code path makes
equality hold by construction. The exhaustive destructure makes a newly
added field a compile error until someone decides how a refactor resets
it, rather than a stale value carried into the next factorization.

**Evidence.** tests/lu_refactor_inplace.rs compares in-place against
fresh bitwise, including under the same script of solves and FT updates
afterwards; deleting any of three resets fails it. ~22 % faster on the
#232 basis shape (journal 2026-11-13-02).

## 2026-11-13 — Markowitz pivot order is held bit-identical to 0.22 (reverses the distinct-columns entry above, PR #234)

**Decision.** The threshold-Markowitz search keeps its 0.22 budget: each
live heap entry it examines counts against `markowitz_max_search`,
duplicates included. The kernel-only heap build reproduces the eager heaps'
multiset exactly (peel-phase pushes logged, heapified at the first
non-singleton step, dead rows/columns left out). This reverses the earlier
2026-11-13 entry "Markowitz candidate search counts distinct columns".

**Why.** pounce's active-set QP uses this factorization as a rank-reveal
probe. The distinct-columns rule made QGFRDXPN cycle to the iteration cap
(details in `dev/tried-and-rejected.md`, 2026-11-13). Pivot order is
therefore treated as part of the contract.

**Evidence.** Digests identical to the unmodified 0.22 code on QPLIB_1157,
QPLIB_3852, bchoco06 and 40 synthetic bases; `tests/lu_markowitz_golden.rs`
fails if the dedup is re-added. Journal 2026-11-13-02.

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
