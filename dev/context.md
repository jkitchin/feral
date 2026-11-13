# FERAL Context (auto-generated)

Generated: 2026-11-13T18:47:34Z

## Latest Session
File: dev/sessions/2026-11-13-01.md
```
# Session 2026-11-13-01

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

## Benchmark Results
```
KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
```
```

## Git Status
```
6a5afd3 Sparse LU assemble: no per-column Vecs in remap_and_sort_l, sized u_above (#232)
dce0dc6 Markowitz LU: kernel-only count heaps, packed active submatrix (#232)
f3d99e0 Add lu_markowitz_triangular example: #232 timing and digest harness
d3f7eb6 Merge pull request #230 from jkitchin/ci/windows-wheel-interpreters
31b64e9 ci(wheels): manual runs can build a given tag and publish it to PyPI
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

test result: ok. 468 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.44s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-11-13-01.md)

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
5/10/30/75 needs 20k/23k/24k/34k/41k/39k/36k delays), so "fresh" is not
"better". The review's rollback compared two different matrices and
misfired (adopt at iter_078, roll back at iter_079 on the sequence's own
1.66× drift).

**Evidence.** `dev/research/issue-227-stale-analysis-reanalysis.md` §4.4
and §7. Implementation replays (journal 2026-10-23-01): qcqp1000-2c
init-first 0.751× `Never` flops, kkt-first 0.980×; qcqp1000-2nc 1.000× /
0.546×; clnlbeam, NARX_CFy, pooling 1.000×. Inertia is identical to `Never`
on every factor of all seven sequences. `Never` digests match main on
every factor.

## 2026-11-13 — Markowitz candidate search counts distinct columns (issue #232)

**Decision.** The threshold-Markowitz pivot search examines each column at
most once per step; `max_search` counts distinct columns. Previously a
column with several live `(count, j)` copies in the lazy heap was scanned
once per copy, each counting toward `max_search`.

**Why.** Building the count heaps only over the kernel (the #232 speed fix)
changes the heap's push history, and the copy count was a function of that
history, not of the matrix. Without the dedup the pivot order would depend
on an implementation detail of when the heap was built. With it, the
bookkeeping refactor is bit-identical to "old code + dedup" on every basis
measured, so the only behaviour change is this one, isolated and measured.

**Evidence.** Fill on 40 synthetic near-triangular bases: geomean +0.27 %
(min -0.46 %, max +1.61 %, 36 worse / 4 better). QPLIB_1157 +0.54 %,
QPLIB_3852 and bchoco06 unchanged. All 1022 tests pass. Journal
2026-11-13-01.

## Recent Tried-and-Rejected
  floored 1. The rule kept the ladder, which climbed three more rungs to
  2.9e-3 with the inertia count unchanged: 165 s against 74 s with the
  shipped rule. pounce confirmed it is deterministic and bisected it.

An a-posteriori-pivoting (APP) blocked kernel for cheaper delays was also
considered and set aside, not tried: it would break the blocked/scalar
bit-parity contract (`tests/blocked_ldlt.rs`, `tests/golden_bits.rs`),
which the maintainer chose to keep.

## 2026-10-22 — #224 Step 1: memoising the failed 2×2 gate pair in the rook walk

Tried on top of the per-call row-max memo (`src/dense/rook.rs`): remember
the last `(p, q)` whose `passes_2x2_gates` failed and skip re-testing it
when an oscillating walk proposes the same pair again. Bit-safe (the gate
is a pure function of the pair within one search) but no gain: harness
medians (iter_000 symbolic, 5 reps, qcqp1000-2c 0.21 trajectory) went
iter_083 0.479 → 0.485 s, iter_120 0.477 → 0.483 s, iter_135 0.493 →
0.493 s, i.e. noise to slightly slower. The walk reaches the 2×2 test
less often than it rescans rows. Dropped; the shipped Step 1 is the row
memo plus BK-scan hints. Journal: 2026-10-22-01.org.

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
