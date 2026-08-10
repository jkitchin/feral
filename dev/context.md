# FERAL Context (auto-generated)

Generated: 2026-08-10T01:17:08Z

## Latest Session
File: dev/sessions/2026-08-10-01.md
```
# Session 2026-08-10-01

## Goal

Fix issue #128 — the five-part allocation-churn bundle (A dense Schur pack
buffers, B `SparseLu::update_sparse` residual allocations, C `DenseLu::update`
clones, D postorder per-node `Vec`s, E post-amalgamation `Supernode.nrow`
underestimate).

## Benchmark Results

**No usable corpus perf signal this session** — the benchmark corpus is not
present in this container, so the harness found 2 matrices. Reported as-is
rather than omitted, and identical to session 2026-08-09-04's numbers.

```
--- Dense solver validation ---
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)

--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

Dense failure analysis: no failures
Sparse failure analysis: no failures

--- Dense perf vs oracles: no matrices have oracle timings ---
--- Sparse perf vs oracles: no matrices have oracle timings ---
```

Nothing regressed against the prior session (both residuals identical to the
digit, inertia 100%), but that is a 2-matrix sample and not evidence of a
perf win either. The real evidence for this session is the targeted
instrumentation below — allocation probes and the symbolic stage profiler —
which does not depend on the corpus.

## Accomplished

### Scope triage

Checked every item against the tree at `f7a152a` rather than trusting the
issue body (written against `660224d`):

- **A — already landed.** `bpack1` laziness in `fc4e9b8`, `apack`/`bpack0`
  pooling via `PackPool` on `FactorScratch` in `484bda7`. The intra-front
  rayon path keeps per-range fresh allocations by design, recorded in the
  `pack_pool` doc comment.
- **C — out of scope by the issue's own instruction**: "do this as part of
```

## Git Status
```
1a84b31 perf(ordering): CSR child arena for the postorder traversals (#128 item D)
2484c27 fix(symbolic): correct post-amalgamation Supernode.nrow (#128 item E)
c83bd0b perf(lu): pool the residual per-update allocations in the FT update (#128 item B)
f7a152a Merge pull request #156 from jkitchin/claude/review-issue-154-ukpt7t
af73f63 Merge origin/main into claude/review-issue-154-ukpt7t
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

test result: ok. 407 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.99s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-10-01.md)


**No usable corpus perf signal this session** — the benchmark corpus is not
present in this container, so the harness found 2 matrices. Reported as-is
rather than omitted, and identical to session 2026-08-09-04's numbers.

--- Dense solver validation ---
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)

--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

Dense failure analysis: no failures
Sparse failure analysis: no failures

--- Dense perf vs oracles: no matrices have oracle timings ---
--- Sparse perf vs oracles: no matrices have oracle timings ---

Nothing regressed against the prior session (both residuals identical to the
digit, inertia 100%), but that is a 2-matrix sample and not evidence of a
perf win either. The real evidence for this session is the targeted
instrumentation below — allocation probes and the symbolic stage profiler —
which does not depend on the corpus.

```

## Recent Decisions

The `merge_flop_budget` guard's merged-height model was corrected in lockstep
at both of its sites. It shared the understatement, which made merges look
cheaper than they are — the wrong direction for a guard meant to reject
expensive merges. The knob defaults to `None`, so the default path is
unaffected, but the sweep recorded in
`dev/research/amalgamation-cost-model-2026-08-09.md` was taken under the old
model and its numbers do not transfer.

## 2026-08-10 — postorder child ordering must partition *stably* (issue #128 item D)

The three postorder variants now order each node's children in place inside a
CSR arena rather than building a fresh sorted `Vec` per node. The two
partitioning rules (`merge_bias_partition`, `schur_partition_children`) use a
stable partition with a reused scratch buffer, not the cheaper two-pointer
swap.

Reason: the code being replaced built each group with
`iter().copied().filter(..).collect()`, which preserves input order, and the
subsequent `sort_unstable_by_key` is only deterministic given a fixed input
sequence. An unstable partition would feed the sorts a different permutation
and could silently change the emitted postorder — and therefore the
fill-reducing ordering and every downstream numeric result. The extra scratch
buffer is one allocation per traversal, against ~2n removed.

The hoist itself is safe because all three ordering rules are pure functions
of `(slice, sizes, bias/is_schur)` with no dependence on traversal state, so
ordering every slice before the walk cannot change the result. Pinned by a
symbolic-output digest over 12 matrices x 3 orderings x 3 amalgamation
strategies x 3 `nemin` values plus the Schur-constrained variant.

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
tests/issue128_postorder_alloc.rs
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
