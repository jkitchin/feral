# FERAL Context (auto-generated)

Generated: 2026-11-24T04:47:16Z

## Latest Session
File: dev/sessions/2026-11-24-02.md
```
# Session 2026-11-24-02

**Benchmark caveat (read first):** no KKT corpus in this container
(`data/matrices/` absent). The bench covered only the 2 committed fixtures
(unchanged from 2026-11-24-01). The evidence for this session is Primal2_1000
(n = 3.86M), rebuilt here from plato.asu.edu with `dev/scripts/mps_to_kkt.py`.
No timing regressed on it; the parallel and sequential AMF factors got faster.

## Goal
Issue #249: real memory high-water was 1.4–2.6× `WorkEstimate::peak_bytes`,
so `max_factor_bytes` bounded the prediction, not the process. Break the gap
down, fix the estimate or the allocation, tighten the book's advice.

## Accomplished
- Probe `examples/issue249_memory_breakdown.rs`: counting allocator, heap
  timeline, census of retained structures, `predict` mode.
- **Breakdown** (Primal2 AMF; `dev/research/issue-249-memory-gap.md`):
  - amalgamation fill: stored `Σ nrow·ncol` is 1.50× the column counts;
  - per-node index data 0.78 GB;
  - the analysis 0.77 GB;
  - workspace and caches ~0.4 GB;
  - under the parallel driver, square blocks of finished subtree tasks
    waiting for the top chain, plus per-worker workspaces and pools.
- 5895c4b: new `WorkEstimate` terms:
  - `factor_stored_bytes`, `factor_index_bytes`, `analysis_bytes`,
    `workspace_bytes`, `numeric_peak_bytes`;
  - `peak_bytes` is now their sum (sequential);
  - `peak_bytes_parallel(threads)` is a schedule-free bound (task-plan
    frontier plus running tasks).
- b2ce4b1: the parallel driver holds **packed** blocks. The recycled
  contribution buffer is used only on a close fit
  (`len ≤ cap ≤ len + max(len/8, 512)`). The model follows.
- 1e51c36: a parallel `Solver` whose budget admits `peak_bytes` but not the
  parallel bound runs the sequential driver for that call (bit-exact).
  `ran_sequential_for_budget()` reports it. This replaces 5895c4b's refusal.
- 5a2dd0a: an unfit pooled buffer is freed *before* the new block is
  allocated (it was a live match temporary: a 0.35 GB spike per top-chain
  front on AMD). The model counts the kernel's panel scratch.
- Final, heap peak above input (GB):

  | case | old pred | new pred | measured before → after | wall before → after |
  |---|---|---|---|---|
  | AMF seq | 3.53 | 4.92 | 4.87 → 4.86 | 104–106 → 98 s |
  | AMD seq | 8.30 | ~8.58 | 9.36 → 8.56 | 982 → 978 s |
  | AMF par 4t | 3.53 | 10.15 (bound) | 7.63 → 5.89 | 69 → 60 s |
  | AMD par 4t | 8.30 | 34.87 (bound) | n/a → 11.74 | n/a → 429 s |

- Tests:
  - `tests/issue249_memory_model.rs`: a counting allocator shows the
    prediction bounds the measured peak (seq 1.06–1.09×, par 1.20–1.25×;
```

## Git Status
```
5a2dd0a fix(memory): free an unfit pooled block before allocating; model kernel scratch (#249)
1e51c36 feat(solver): over the parallel bound, a budgeted Solver runs sequential (#249)
b2ce4b1 perf(memory): parallel driver holds packed blocks; pool recycles close fits (#249)
5895c4b fix(estimate): peak_bytes models what Solver::factor really holds (#249)
4051231 Merge pull request #247 from jkitchin/claude/zen-hamilton-tra4mp
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

test result: ok. 494 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.60s

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
`grow_pooled_front`. `peak_bytes_parallel(threads)` bounds the parallel
driver over every schedule: the task-plan frontier (max-weight antichain
of task-root blocks) plus `threads + 1` running tasks. A parallel `Solver`
whose budget admits `peak_bytes` but not the parallel bound runs the
sequential driver for that factorization; it does not refuse. Two
allocation changes make the model hold:
- the parallel driver holds packed blocks;
- `extract_contrib` recycles a pooled buffer only when
  `len ≤ cap ≤ len + max(len/8, 512)`, so every block's capacity is
  bounded. An unfit buffer is freed before the new block is allocated.

**Why.** The old `factor_bytes + peak_contrib_bytes` missed four things:
amalgamation fill (1.50× the column counts on Primal2 AMF), per-node index
data (0.78 GB), the analysis (0.77 GB) and the workspace. It ran
1.4–2.6× below the measured memory, so `max_factor_bytes` did not protect
the process. Square blocks in the parallel driver were a scoping leftover
of the 2026-11-12 packing decision, not a requirement. The pool's
`reserve` reuse let a block keep a larger buffer's capacity, which no
schedule-free bound could cover. The parallel bound is loose (1.7× on AMF,
3.0× on AMD), so refusing on it would reject jobs that fit. Falling back
to the bit-exact sequential driver costs only time.

**Evidence.** `dev/research/issue-249-memory-gap.md` and journal
2026-11-24-02. Primal2, sequential: AMF predicted 4.92 vs measured
4.86 GB (old 3.53); AMD 8.58 vs 8.56 (old 8.30). AMF parallel, 4
threads: 7.63 → 5.89 GB, wall 69 → 60 s. `tests/issue249_memory_model.rs`: the prediction bounds the
measured heap peak on grids and a KKT, within 1.03–1.08× sequential and
1.20–1.25× parallel. Stored terms equal the factors' real buffer lengths.
`peak_contrib_bytes` keeps its classic square meaning (tried-and-rejected
2026-11-12).

## Recent Tried-and-Rejected

## 2026-11-24 — issue #249 memory model

- **Refusing a parallel factorization on `peak_bytes_parallel`** (stage
  `"peak_parallel"`, in commit 5895c4b). The parallel figure is a
  schedule-free bound: 1.65× the measured heap peak on Primal2 AMF with 4
  threads (12.61 vs 7.63 GB with square blocks) and 2.6× on AMD (31.7 vs
  12.27 GB). A 16 GB budget would have refused AMD, which fits in
  12.3 GB. Replaced by running the sequential driver for that call.
- **Bounding recycled block capacity by 2× the block length.** With the
  old `clear()+reserve(len)` reuse, a block's capacity is
  `max(pool capacity, 2·len)` by induction. That is bounded by twice the
  largest block ever seen, not by twice its own length. The 2× allowance
  also doubled the frontier term (Primal2 AMD parallel bound 45.5 GB).
  Replaced by changing the recycle rule (close fit only), which bounds
  every block by `len + max(len/8, 512)`.
- **A postorder contribution model that ignored the pool's capacities**
  (lengths only). It under-predicted Primal2 AMD sequential by 0.5%
  (9.317 vs 9.364 GB measured), because recycled buffers kept larger
  capacities.

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
src/numeric/memory_model.rs
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
tests/issue249_memory_model.rs
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
