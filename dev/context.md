# FERAL Context (auto-generated)

Generated: 2026-11-13T18:31:50Z

## Latest Session
File: dev/sessions/2026-11-13-01.md
```
# Session 2026-11-13-01

## Unfavourable results (read first)

- **Bench factor times are 2–3× the previous checkpoint's** (spd_200
  409 → 1253 µs, kkt_100_30 207 → 475 µs). That checkpoint ran on
  Apple Silicon; this one ran in a shared x86_64 cloud container, and
  no source change in this session touches the dense kernel paths
  timed here (loop rewrites only). Treat this as a machine difference,
  not a regression; it was not re-measured on the earlier machine.

## Goal
Review PR #231 (packed contribution blocks, opt-in
`EarlyAssembly::WideParents`), merge it, and fix the CI clippy
failure under Rust 1.100.

## Accomplished
- Reviewed PR #231 and posted 8 inline findings. The author fixed all
  of them in 14d9bc6, and the PR was merged as 675a468. Verified on
  x86_64: `golden_digests` on d3f7eb6 and on the PR give the same 16
  digests, equal to the aarch64-pinned `PARENT_GOLDEN` (so the
  parent-release test now runs on x86 too); `cargo test --release` on
  14d9bc6 gave 1,030 passed / 0 failed; wide_tree_memory 3/3.
- CI's `dtolnay/rust-toolchain@stable` is now 1.100.0, whose clippy
  flags `needless_range_loop`: main (d3f7eb6) fails
  `cargo clippy -- -D warnings` in feral-metis. Rewrote 46 index
  loops as slice iterations across the workspace (same reads, same
  order). Rust 1.100: fmt and clippy (workspace + feral-diagnostics)
  clean; cargo test, examples, feral-diagnostics and ordering-crate
  tests: 1,288 passed, 0 failed, 26 ignored.

## Benchmark Results
```
name                n   factor(μs)    solve(μs)        inertia
spd_10             10           18            3     (10, 0, 0)
spd_50             50           62            4     (50, 0, 0)
spd_100           100          271            8    (100, 0, 0)
spd_200           200         1253           26    (200, 0, 0)
kkt_10_3           13            8            1     (10, 3, 0)
kkt_30_10          40           45            2    (30, 10, 0)
kkt_50_15          65          101            3    (50, 15, 0)
kkt_100_30        130          475           11   (100, 30, 0)
KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
```

## Git Status
```
39ea596 Fix clippy needless_range_loop under Rust 1.100
675a468 Merge pull request #231 from firstprinciples/wide-parent-contrib-memory
14d9bc6 Address review: per-factorization plan, out-of-order fallback, gated Always
0d29668 Add Solver::with_early_assembly
34494d5 Opt-in accumulators for wide parents (sequential drivers)
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

test result: ok. 474 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.98s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-11-13-01.md)

name                n   factor(μs)    solve(μs)        inertia
spd_10             10           18            3     (10, 0, 0)
spd_50             50           62            4     (50, 0, 0)
spd_100           100          271            8    (100, 0, 0)
spd_200           200         1253           26    (200, 0, 0)
kkt_10_3           13            8            1     (10, 3, 0)
kkt_30_10          40           45            2    (30, 10, 0)
kkt_50_15          65          101            3    (50, 15, 0)
kkt_100_30        130          475           11   (100, 30, 0)
KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
Inertia, residual and validation lines are identical to 2026-11-12-01.

```

## Recent Decisions
buffer from the returned factor. Factors are bit-identical under every
scheme and driver.

**Evidence.** `dev/research/wide-tree-contribution-memory.md`;
`tests/wide_tree_memory.rs` (generated trees, `WideParents` peak at most
`Off`'s plus a bookkeeping allowance, retained factor no larger, no
factor owns contribution storage; fails before the fix).

## 2026-11-13 — `WideParents` review follow-ups

**Decision.** The accumulator plan is derived by each factorization that
uses it (`Accumulators::new`, from `SymbolicFactorization::
wide_parent_plan`), not cached on the analysis: no private field on the
public struct, and no stale plan after a caller edits `supernodes`. A
deposit out of ascending order abandons the plans it passes over instead
of failing at the parent (debug builds assert): the accumulator keeps the
prefix of children it absorbed, the rest are assembled whole after it in
children order, so factors are unchanged. `EarlyAssembly::Always` sits
behind the internal `__early-assembly-always` feature, which the crate's
tests enable through a self dev-dependency. The frontal pool's growth rule
is explicit (`grow_pooled_front`: capacity `max(2·cap, n²)`), and an
adopted accumulator grows the idle pool's capacity without writing it.

**Why.** Maintainer review of PR #231. Deriving the plan costs one pass
over supernodes and children (about 6 µs on the 1,501-supernode star).

**Evidence.** `tests/wide_tree_memory.rs` (now `harness = false`)
unchanged figures: star `Off` 65.73 MiB, `WideParents` 12.12 MiB; 192
matrices, at most 7,208 B above `Off`. `tests/early_assembly_parity.rs`
digests unchanged; the parent-release pin now runs on every target.

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
```
