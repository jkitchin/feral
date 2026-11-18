# FERAL Context (auto-generated)

Generated: 2026-11-18T17:45:45Z

## Latest Session
File: dev/sessions/2026-11-18-01.md
```
# Session 2026-11-18-01

## Goal
Issue #237 (block-banded IPM KKT: serial per-call work is about half of
each refactorization at 16 threads). Review the issue against the code,
then step 1 of the review's order: the permute warm path.

## Accomplished
- Reviewed #237 against 0.23.0. All code claims checked hold (InfNorm
  recomputed every call, permute warm path compares + allocates + clones,
  Solver pattern checks per call, refinement plateau reset by tiny gains).
  #237 item 1 overlaps #202 (still open). Suggested order: permute warm
  path, trust_pattern opt-in, parallel KR sweeps, scaling reuse with a
  drift guard, refinement contraction exit.
- 0431a10: research note + plan (`dev/research/issue-237-permute-warm-path-2026-11-18.md`,
  `dev/plans/issue-237-step1-permute-warm.md`).
- ae1ec31: `examples/issue237_permute_warm.rs`, chain KKT n = 155 970,
  nnz = 5 965 170, analytic inertia oracle (78000, 77970, 0).
  Baseline warm permute 74.8 ms: compare 7, alloc 2, scatter 37
  (incl. page faults), pattern clone 28 ms.
- 8d01094: PermuteCache owns the permuted matrix, values overwritten in
  place, zero fill skipped for bijective maps, Solver-verified pattern
  compare not repeated. Permute 92.2 -> 19.8 ms (same-run comparison).
- 6e894b0: bijective maps gather; parallel on a rayon worker.
  Permute 22.5-24.6 ms serial -> 6.3-9.2 ms on 4 threads.
- Solve digests identical to the 0.23.0 build at two sizes in every
  driver mode. `cargo test --release`: 1044 passed, 0 failed.

**Not shown here:** a factor-wall improvement. On this 4-core container
the tree phase dominates (~1.2 s) and run-to-run noise is ±15 %, so the
~70 ms saved does not show reliably in factor wall. The issue's 16-thread
setting (tree 123 ms vs permute 23 ms) was not reproduced.

**Finding:** the cold permute keeps a `-0.0` input as `-0.0`; 0.23.0's
warm path produced `+0.0`. Pre-existing; the warm path keeps `+0.0`.

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
Identical to 2026-11-13-02.

```

## Git Status
```
6e894b0 perf(#237): permute warm path gathers, in parallel on a rayon worker
8d01094 perf(#237): permute warm path reuses the permuted matrix in place
ae1ec31 bench(#237): warm-refactorization permute harness on a chain KKT
0431a10 docs(#237): research note and plan for the permute warm path
0e7cc5f Merge pull request #236 from jkitchin/claude/gallant-cray-pjv1tt
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

test result: ok. 480 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.23s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-11-18-01.md)

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
Identical to 2026-11-13-02.

```

## Recent Decisions
probe. The distinct-columns rule made QGFRDXPN cycle to the iteration cap
(details in `dev/tried-and-rejected.md`, 2026-11-13). Pivot order is
therefore treated as part of the contract.

**Evidence.** Digests identical to the unmodified 0.22 code on QPLIB_1157,
QPLIB_3852, bchoco06 and 40 synthetic bases; `tests/lu_markowitz_golden.rs`
fails if the dedup is re-added. Journal 2026-11-13-02.

## 2026-11-18 — The permute cache owns the permuted matrix; the Solver vouches for the pattern (issue #237)

**Decision.** `PermuteCache` holds the permuted `CscMatrix` itself. A warm
hit moves it to the driver, which overwrites its values and hands it back
(`FactorWorkspace::restore_permuted`) before returning; a driver that
returns early with an error loses it and the next warm call rebuilds via
the cold path. When the value map is a bijection the cache stores its
inverse and the warm path gathers (in parallel on a rayon worker).
`Solver::factor_impl` sets a crate-private, single-use
`FactorWorkspace::pattern_verified` after its exact pattern comparison,
and the warm path then skips its own `O(nnz)` re-comparison (the `O(n)`
`perm_inv` check stays). `NumericParams` is not used for this signal: it
is public with all-public fields, and its `pattern_reused_hint` can be set
by direct-API callers, so it cannot carry a "pattern already checked"
guarantee.

**Why.** The warm permute is serial work outside the parallel tree; it
cloned the cached pattern and allocated a fresh values array every call.

**Evidence.** Chain KKT n = 155 970, nnz = 5.97 M: warm permute 74.8 ms
-> 6-9 ms on 4 threads, solve digests identical to 0.23.0. Journal
2026-11-18-01; `dev/research/issue-237-permute-warm-path-2026-11-18.md`.

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
