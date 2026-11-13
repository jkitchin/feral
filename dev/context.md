# FERAL Context (auto-generated)

Generated: 2026-11-12T23:22:05Z

## Latest Session
File: dev/sessions/2026-11-12-01.md
```
# Session 2026-11-12-01

## Unfavourable results (read first)

Timings are from a shared machine (load average 16–80 from other work).

- **The first `WideParents` (then `Planned`) version broke its own memory
  claim** (review finding B1): adopting the exact-size accumulator as the
  frontal pool and recycling absorbed blocks left larger pooled buffers
  than `Off` — a hand-built tree 1.117× `Off`, 19 of 2,000 random trees
  more than 2% above, retained factor larger on 1,126. Fixed (buffers
  follow the classic history; an empty block takes no pooled buffer).
- **`WideParents` is 2.6–4.1% slower than `Off`** on corpus matrices
  with one small planned front under 1,000–2,600 tiny children (21 runs
  × 9 rounds, before the buffer fixes: CRESC132 1.041×, VESUVIO_0021
  1.031×, sawpath 1.026×). After the fixes its geomean against `Off` on
  the 22 corpus matrices ≥ 0.5 ms rose 1.004 → 1.019 (multistar:8
  1.093×, saddle_rankdef_100 1.053×, multistar:64 1.050×); freed blocks
  are no longer recycled. Opt-in.
- **`WideParents` adds 96 B of peak on MUONSINE_0019/0027** over `Off`
  (+64 B over 0.22.0; plan bookkeeping, the plan saves nothing). On
  generated trees it is up to 7,208 B above `Off` (within the test's
  bookkeeping allowance); review harness worst +2,168 B.
- **The empty-block rule reaches the parallel driver** (shared kernel
  extract), against the rescope's "parallel driver as in d3f7eb6": its
  digests and 1-thread peaks are unchanged, only the retained factor
  shrinks.
- **POUNCE end to end under `Off`: live heap 805.8 → 806.2 MiB** against
  the previous round (the pooled buffer now stays in the solver's
  workspace instead of the replaced factor).
- **Default (`Off`) vs the parent:** MUONSINE 0.997× in a focused run,
  1.02× in an earlier 6 × 401-run comparison; a 16-leaf star 1.038× with
  range 0.85–1.30.
- **Parallel, 4 threads:** peaks depend on scheduling; grid3:32 reached
  188.6 MB against the parent's 175.2 in one corpus round; 20 runs each
  give the same distribution (parent 175–193, this 175–190 MB).
- **The parent-digest pin covers aarch64 only**; it is ignored on x86_64
  (CI), since the digests need an x86 machine.
- **`Always`** (hidden, tests only) raises the peak 1.6–2.0× on 3D grids
  and 5.5× on nuffield2.

## Goal
Bound the transient contribution-block memory of the sequential
multifrontal driver on wide elimination trees, bit-identically, with
the parallel task-graph driver as in `d3f7eb6`, every default justified
on general grounds and FERAL's own corpus as evidence; then fix the
review of the staged change (`Planned` peak above `Off`).

## Accomplished
- Change 1, packed blocks in the sequential and Schur drivers
```

## Git Status
```
d3f7eb6 Merge pull request #230 from jkitchin/ci/windows-wheel-interpreters
31b64e9 ci(wheels): manual runs can build a given tag and publish it to PyPI
a9ee74c ci(wheels): install all four Python interpreters before maturin
4bfd444 Merge pull request #229 from jkitchin/release/0.22.0
b207999 release: feral 0.22.0
```

## Test Status
```
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test numeric::solve::tests::cb_coarsening_threshold_is_arithmetically_inert ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test numeric::solver::tests::warm_refactor_recycles_output_buffers_issue_153 ... ok
test dense::rook::tests::rook_rescue_matches_uncached_walk ... ok
test dense::factor::row_offdiag_tests::rook_row_argmax_matches_bk_row_scans ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test dense::factor::contrib_layout_tests::packed_contribution_is_the_square_lower_triangle ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 477 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 1.76s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-11-12-01.md)

name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10           37            9     (10, 0, 0)
spd_50             50           20            1     (50, 0, 0)
spd_100           100           79            6    (100, 0, 0)
spd_200           200          409           16    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           20            1    (30, 10, 0)
kkt_50_15          65           50            2    (50, 15, 0)
kkt_100_30        130          207            7   (100, 30, 0)

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
Identical inertia, residual and validation lines to the parent
(`d3f7eb6`); factor times are single runs of microsecond problems.

```

## Recent Decisions
there the parent's frontal is an accumulator into which each child is
extend-added on deposit, in `children` order. Buffers follow the classic
scheme's history: absorbed blocks and remainders are freed, not pooled;
an adopted accumulator is freed after its front is factored and the
frontal pool grows by the one rule (`fit_pooled_front`); and an empty
block (a root) takes no pooled buffer, in every driver, so no returned
factor holds contribution storage. `WorkEstimate::peak_contrib_bytes`
stays the classic square model.

**Why.** Holding every child's block until the parent is assembled
makes a wide parent's assembly cost the sum of its children's blocks.
Packing halves that at no measurable cost. Under `WideParents` the
peak live heap is `Off`'s plus bookkeeping (a few words per planned
parent, one block's row map, `O(order)` words while a planned front is
factored) and the retained factor is the same. The first version
claimed this by construction but recycled buffers differently and broke
it (review finding B1; `dev/tried-and-rejected.md`). It is not the
default because its bookkeeping costs 2.6–4.1% of factorization time on
matrices with one small planned front under thousands of tiny children
(CRESC132, VESUVIO, sawpath). The parallel driver is left out:
out-of-order deposits need per-parent synchronisation and a backlog
limit, a separate change. The empty-block rule does reach the parallel
driver, because the kernel's extract is shared; it only removes a
buffer from the returned factor. Factors are bit-identical under every
scheme and driver.

**Evidence.** `dev/research/wide-tree-contribution-memory.md`;
`tests/wide_tree_memory.rs` (generated trees, `WideParents` peak at most
`Off`'s plus a bookkeeping allowance, retained factor no larger, no
factor owns contribution storage; fails before the fix).

## Recent Tried-and-Rejected
  (`dense::matrix::fit_pooled_front`).
- **Recycling absorbed blocks' buffers through the contribution pool.**
  The kernel reuses a pooled buffer with `reserve`, so a buffer grows
  along a chain of siblings up to twice the largest block, and the
  parent's small block then pins it until the grandparent assembles: 19
  of 2,000 random trees more than 2% above `Off` (worst 1.128×), and the
  retained factor larger in 1,126 (review finding). Pooling only
  delayed-pivot remainders still grew one delaying tree's retained
  factor by 32 B. Absorbed blocks and remainders are now freed; `Off`
  already pays one allocation per leaf block.
- **Counting packed blocks in `WorkEstimate::peak_contrib_bytes`.** The
  default `Solver` runs the parallel task graph when the tree has task
  parallelism, which holds square blocks, so a packed figure understated
  that path's contribution memory by about half (review finding). The
  estimate stays the classic square one, an upper bound for every driver
  and scheme.
- **A `ContribLayout` argument on the public dense kernels.** Only the
  crate's own drivers pass anything but `Square`. The public signatures
  stay as in 0.22.0; the drivers call crate-internal `_with_layout`
  variants.

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
tests/wide_star/mod.rs
tests/wide_tree_memory.rs

(truncated from      351 lines to 350 line budget)
