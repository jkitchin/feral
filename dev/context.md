# FERAL Context (auto-generated)

Generated: 2026-08-10T11:44:47Z

## Latest Session
File: dev/sessions/2026-08-10-01.md
```
# Session 2026-08-10-01

## Goal

Fix the post-amalgamation `Supernode.nrow` underestimate.

This was item E of issue #128, a five-part allocation-churn bundle. That issue
has been **closed as not planned**; this item was extracted onto its own
branch because it is the only part that was a correctness bug rather than a
micro-optimization.

Why the rest was dropped, recorded so nobody re-opens the question:

- **Item A** (dense Schur pack buffers) had already landed in `fc4e9b8` and
  `484bda7`.
- **Item C** (`DenseLu::update` clones) is bound by the issue's own
  instruction to the #115 dense-update rework, "not before."
- **Items B and D** were implemented and measured, then dropped as not worth
  the review surface. B (FT update allocation pooling) took 11.2 -> 2.8
  allocations per update, which at the probe's ~60 ns/alloc convention is
  ~0.6 us against an 85.8 us/update budget — under 1%, and the large win on
  that path had already landed in earlier sessions. D (postorder child arena)
  cut symbolic time 12-19%, but symbolic runs once per sparsity pattern, so
  for an IPM consumer that refactorizes the same pattern every iteration it
  amortizes to nothing. Both were bit-exact and correct; neither was
  load-bearing.

## Benchmark Results

**No usable corpus perf signal** — the benchmark corpus is not present in this
container, so the harness found 2 matrices. Reported as-is rather than
omitted; identical to session 2026-08-09-04's numbers.

```
--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

Dense failure analysis: no failures
Sparse failure analysis: no failures
```

This change is not expected to move those numbers — it corrects estimates, not
kernels — and a 2-matrix sample could not show it either way.

## Accomplished

`find_supernodes` set `nrow = col_counts[first_col].max(ncol)`. Exact for a
```

## Git Status
```
509f0ce perf(mc64): store the key inline in the Hungarian heap; bit-identical
e8a7283 research(mc64): condition 1 rejects on a barrier-trajectory metric; cost share predicts the outcome
795dcd2 docs: session checkpoint 2026-08-09-09 (#125 sizing + MC64 value-bound fix)
bd1bc26 fix(scaling): make MC64 value-bound condition 3 a drift measure
8f8aa8c feat(diag): trajectory-level scaling profile and a scaling-reuse safety check
```

## Test Status
```
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 412 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.40s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-10-01.md)


**No usable corpus perf signal** — the benchmark corpus is not present in this
container, so the harness found 2 matrices. Reported as-is rather than
omitted; identical to session 2026-08-09-04's numbers.

--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

Dense failure analysis: no failures
Sparse failure analysis: no failures

This change is not expected to move those numbers — it corrects estimates, not
kernels — and a 2-matrix sample could not show it either way.

```

## Recent Decisions

    merged_nrow = child_group_ncol + parent_group_nrow

maintained as a running per-group value. This is the union *cardinality*, not
a bound, and it composes for chains under both amalgamation iteration orders.

Verified rather than assumed: compared against
`SymbolicFactorization::static_rows(i).len()` (the issue #125 static frontal
layout, an independent computation already pinned to both a from-scratch
`BTreeSet` recompute and `build_row_indices`) across 7 matrix families x 3
`nemin` values — **zero error on every supernode**. The pre-change proxy was
wrong on up to 40% of summed `nrow`.

**Accepted consequence, with a caveat.** `nrow` feeds
`estimate_assembly_flops`, so the `PAR_MIN_FLOPS` gate now sees true costs
and borderline matrices can flip from sequential to parallel (one flip
recorded: a 60x60 grid Laplacian at `nemin = 32`, 4.3M -> 12.2M estimated
flops). Numeric factors and inertia are byte-identical. The caveat is that
`PAR_MIN_FLOPS` was calibrated against the *understated* estimate, so the
constant itself may now be mis-placed; the flip is unverified on the real
corpus (absent from the container this landed in). Re-deriving the threshold
against corrected flops is open work, not something this change did.

The `merge_flop_budget` guard's merged-height model was corrected in lockstep
at both of its sites. It shared the understatement, which made merges look
cheaper than they are — the wrong direction for a guard meant to reject
expensive merges. The knob defaults to `None`, so the default path is
unaffected, but the sweep recorded in
`dev/research/amalgamation-cost-model-2026-08-09.md` was taken under the old
model and its numbers do not transfer.

## Recent Tried-and-Rejected
Swept `RAYON_NUM_THREADS` = 1, 2, 4, 8, 10 against the default on six
real chain KKTs, 15 paired runs each, on an M4 Pro (10P + 4E — *more*
efficiency cores than the 4P+4E M2 the hypothesis came from, so the
predicted effect should be larger). Every ratio vs the default landed
within 6% of 1.0; the only significant one was `marine_1600` at t1
(0.961, 0/15, p = 0.0001), in the wrong direction to support the
hypothesis. `steering_12800` at one thread: 1.003 (9/15, p = 0.6072).
`dtoc1nd`, the matrix that actually regressed, at t4: 1.005 (7/15,
p = 1.0000).

The knob was verified live, not assumed: feral reads the global rayon
pool (`rayon::current_num_threads()`, `src/numeric/factorize.rs:3262`),
and the same variable moved the proxy matrices by up to 65%.

Consequence worth carrying forward: single-threaded main matches
all-threads main on every one of these matrices, so **#150's 1.20x to
2.05x gains on the large chains are not parallelism gains**. Do not
build on the assumption that they are.

Full data: `dev/research/chain-kkt-corpus-2026-08-09.md`, Result 3.

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
src/lu/mod.rs
src/lu/scaling.rs
src/lu/sparse_factor.rs
src/lu/sparse_matrix.rs
src/lu/sparse_solve.rs
src/lu/sparse_symbolic.rs
src/lu/sparse_update.rs
src/numeric/condition.rs
src/numeric/factorize.rs
src/numeric/mod.rs
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
tests/cb_solve_parity.rs
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
tests/factors_ld_export.rs
tests/fine_grained_delay.rs
tests/fma_opt_in_roundtrip.rs
tests/golden_bits.rs
tests/growth_flag.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/issue102_intrafront_deadlock.rs
tests/issue102_ordering_escalation.rs
tests/issue107_external_ordering.rs
tests/issue112_bg_update.rs
tests/issue127_pipeline_split.rs
tests/issue128_supernode_nrow.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/issue91_preprocess_misfire.rs
tests/issue99_fma_front_gate.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/lu_adversarial_inputs.rs
tests/lu_dense_update_bg.rs
tests/lu_dense.rs
tests/lu_ft_widebump.rs
tests/lu_scaling.rs
tests/lu_sparse.rs
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
tests/pounce_interface.rs
tests/profiler_smoke.rs
tests/property_tests.rs
tests/rook_rescue_kkt.rs
tests/rook_rescue.rs
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
