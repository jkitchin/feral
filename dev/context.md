# FERAL Context (auto-generated)

Generated: 2026-08-10T11:03:03Z

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
fc84eb3 fix(symbolic): correct post-amalgamation Supernode.nrow (#128 item E)
f7a152a Merge pull request #156 from jkitchin/claude/review-issue-154-ukpt7t
af73f63 Merge origin/main into claude/review-issue-154-ukpt7t
6c87a0e docs: session checkpoint 2026-08-09-03 (issue #154 review + implementation)
4f2fad6 fix(solver): derive use_parallel from the platform; fall back to sequential when the pool fails
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
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 407 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.54s

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
`nemin=8`, MEYER3NE 83× at `nemin=4`), which is what makes it a property
of the direction rather than of this rule.

**Why rejected.** "Correctness before performance, always" is a hard
constraint. 2–7% of factor time and 11–45% of fill does not buy seven
digits of residual. Neither my pre-registered criterion nor the queue
item thought to check the axis that decided it — recorded here because
the next person to have this idea will not think to check it either.

The knob stays in-tree defaulting to `None` (bit-identical default path)
as the reproduction apparatus, with the accuracy result in its doc
comment. Research note:
`dev/research/amalgamation-cost-model-2026-08-09.md`.

**Also redirects the target.** pounce#552's re-measurement against a
released 0.15.0 (comment 5232409020) shows clnlbeam more than halved
(8.05× → 3.54× vs MA57) and **no longer the worst case** — `dtoc1nd` is,
at 3.77×, and it is a dense-front matrix (nnz/dim 23.0, fronts of 33–64
columns). Amalgamation is a chain-KKT lever aimed at a problem that has
largely receded.

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
tests/column_renumbering.rs
tests/column_renumbering_parity.rs
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
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
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
tests/lu_dense.rs
tests/lu_dense_update_bg.rs
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
