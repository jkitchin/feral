# FERAL Context (auto-generated)

Generated: 2026-08-19T20:08:07Z

## Latest Session
File: dev/sessions/2026-08-19-03.md
```
# Session 2026-08-19-03

## BENCHMARK NUMBERS ARE NOT COMPARABLE TO LAST SESSION

Reported first, per the hard rule in CLAUDE.md — and for the same reason
as 2026-08-19-02: `cargo run --bin bench --release` in this container
finds only the 8 synthetic matrices. The external corpus and the
MUMPS/SPRAL oracle timings are absent, so both Phase 2.8.1 exit
partitions report `N/A` and **no comparison against 2026-08-15-02's
1.61 / 2.00 / 1.67 / 1.67 is possible from this run**. I am not claiming
those numbers held; they were not measured.

What the run does confirm, identical to last session: inertia 2/2 vs
MUMPS, residual 2/2, worst residual 1.26e-16.

It is also the wrong benchmark for this change, which touches solve
*scheduling* and not the factor path. The measurements that matter are
in "Benchmark Results" below.

## Goal

Fix issue #175 — the tree-parallel solve added for #131 Gap A is a net
15% loss on the wide-sparse Mittelmann KKT `NARX_CFy`, and
`CbTaskPlan::worthwhile` has no per-call-overhead term.

## Accomplished

### The report is real, and it is purely a scheduling bug

The reporter serialized the tree-parallel solve with `FERAL_CB_THRESH`
and recovered 7.35 s of 49.41 s (15%) plus ~3.0M involuntary context
switches, on 14 cores over 100 IPM iterations. Two things follow that
the issue does not state:

1. `FERAL_CB_THRESH` does not choose the solve *core* — since #177 that
   is `cb_core_profitable`, which reads neither the worker count nor the
   environment. A huge threshold collapses the plan to one task, so the
   **same** CB core runs serially. The whole 7.35 s is scheduling
   overhead, and fixing it cannot move a bit of any solve.
2. Rows 3 and 4 of the reporter's table are within noise, so the CB core
   running serially already matches switching parallelism off entirely
   on this problem. Nothing in the report argues for changing which core
   runs.

### Mechanism: the overhead is per front, not per task

`cb_run_parallel` takes the shared `contribs` mutex inside its per-front
loop — once per child drained, once to store the front's own block. That
cost scales with the supernode count; `MIN_TOTAL_COST` is a floor on
*total* work. `NARX_CFy` has 45,736 supernodes and a Lagrangian Hessian
```

## Git Status
```
ffb7599 Merge pull request #180 from jkitchin/claude/quirky-bardeen-c3ynyz
cb16458 Merge origin/main into claude/quirky-bardeen-c3ynyz (#177 + #178)
5018575 Merge pull request #179 from jkitchin/claude/issue-178-ycwr81
c154c92 docs: session checkpoint 2026-08-19-01 (#177 fixed)
b75da82 test(solve): pin the refined solve's arithmetic against the host (#177)
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
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 431 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 3.04s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-19-03.md)


`issue175_cb_gate_calibration` (`#[ignore]`d, in-crate): pooled CB core,
serial vs tree-parallel, best of 30, arms interleaved, 4-core container.
par/ser < 1 means tree-parallel wins.

| fixture | total/n_nodes | run 1 (w=2,4,8) | run 2 (w=2,4,8) | geo. mean |
|---|---:|---|---|---:|
| narx_w1 | 25 | 1.07 1.10 1.42 | 0.91 0.98 1.04 | 1.08 |
| narx_w2 | 28 | 0.97 1.08 1.04 | 0.94 1.14 0.98 | 1.02 |
| narx_w4 | 53 | 1.01 1.11 0.93 | 0.89 1.02 0.95 | 0.98 |
| narx_w6 | 74 | 1.15 0.95 0.88 | 0.74 0.88 0.87 | 0.91 |
| narx_w8 | 103 | 0.74 0.94 0.70 | 0.81 0.84 0.83 | 0.81 |
| poisson_96 | 202 | 0.70 0.76 0.70 | 0.74 0.80 0.69 | 0.73 |
| poisson_160 | 235 | 0.74 0.78 0.73 | 0.80 0.75 0.68 | 0.75 |
| narx_w3 | 305 | 0.87 0.64 0.64 | 0.63 0.54 0.51 | 0.63 |

Monotone in work per front; break-even between 53 and 74. `narx_w1`
(47,228 fronts, 22 seeds, 25 units/front) is the local analogue of
`NARX_CFy` and never pays. The local losses are milder than the reported
15% because this container has 4 cores and the dominant cost is mutex
contention, which worsens with worker count.

`cargo run --bin bench --release` (unchanged from last session, see the
top of this file):

8 matrices benchmarked
KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
--- Dense/Sparse Phase 2.8.1 exit partition: N/A (no corpus, no oracles)

```

## Recent Decisions
  `MAX_LOCAL_SHARE` of the work);
- `cb_sync_amortized(total, n_nodes)` — new: `total ≥ 64 · n_nodes`,
  i.e. a front must average ≥64 `nrow·(nelim+1)` units before
  `cb_run_parallel`'s per-front synchronization is worth paying.

`worthwhile = shape ∧ amortized`; `cb_core_profitable` — the predicate
that chooses between the two numerically distinct solve cores (#177) —
applies **the shape half only**.

Why the split rather than one gate: the two predicates answer different
questions. `worthwhile` picks between two byte-identical executions of
one core, so it may model machine overhead freely. `cb_core_profitable`
picks between two different reassociations, so it must stay a function
of the factor alone — folding an overhead term into it would silently
change which arithmetic wide-sparse factors solve with, which is exactly
the failure #177 fixed. `cb_core_profitable_matches_the_plan_gate` pins
the shared half so the two implementations cannot drift.

Evidence: issue #175 (15% of an IPM run and ~3.0M involuntary context
switches on `NARX_CFy`, 14 cores);
`dev/research/issue-175-cb-solve-gate-overhead.md` (break-even between
53 and 74 units/front over eight fixtures × two runs × three worker
counts); `dev/journal/2026-08-19-03.org`.

Accepted cost: a bushy factor whose fronts average under 64 units now
runs the CB core serially even where tree-parallelism would have won a
few percent. The floor sits at the measured break-even, so the expected
cost of a false negative is ~0 and its worst observed case is ~1.1x,
against a false positive's measured 1.42x locally and 15% end-to-end on
the reporting host.

## Recent Tried-and-Rejected
The winner has the *least* work per seed of the three. A floor high
enough to reject `NARX_CFy` would reject `poisson_160` — the factor
whose 25-37% win is the whole point of #131 Gap A — six times over.

Work per **front** (`total / n_nodes`) does separate them (25 vs 235
units, monotone across eight fixtures), and it is what the mechanism
predicts: `cb_run_parallel` takes the shared `contribs` mutex inside the
per-front loop, so its overhead scales with the supernode count, not
with the task count.

### Rejected: calibrating through `solve_sparse_refined_cb`

First calibration harness was an `examples/` probe timing the public
refined-solve entry point, on the theory that it is what an IPM host
calls. It reported par/ser 1.00-1.02 on fixtures that the in-crate
harness later showed at 1.16-2.11 — the refined solve's residual sweeps
and its per-call workspace construction are `O(n)` work in *both* arms,
which dilutes a per-front effect until it disappears. Replaced by an
in-crate `#[ignore]`d test that times `CbSolveWorkspace::solve_into`
against a pooled workspace: the exact call `worthwhile` decides.

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
tests/issue178_refine_cap.rs
tests/issue178_solve_into.rs
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
