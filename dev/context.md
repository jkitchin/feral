# FERAL Context (auto-generated)

Generated: 2026-08-09T15:24:26Z

## Latest Session
File: dev/sessions/2026-08-09-02.md
```
# Session 2026-08-09-02

## Goal

Issue #148: the default parallel multifrontal driver is slower than
serial on 3 of 4 POUNCE problems (~1.8M `scope.spawn` boxed-closure
allocations per solve, glibc arena contention growing with thread
count). Fix via the issue's suggested directions 1+2: work-based task
coarsening and a serial fallback. Environment: x86_64 4-core container
(issue reproduces at 2-4 threads); POUNCE `.nl` problems unavailable —
synthetic proxies (chain / grid / banded-QP KKT) built and measured
with interleaved old-vs-new binaries (git worktree at e8e1c5a).

## Accomplished

1. **Reproduction** (research note): chain12000 parallel never beats
   serial; sparseqp proxy degrades monotonically with threads; grid250
   is the one winner but pays +19% single-thread driver overhead.
2. **TaskPlan coarsening (7814bfe)**: one spawn per subtree task —
   boundaries at tree roots and at children of nodes with >= 2 sibling
   subtrees each >= `FERAL_PAR_TASK_MIN_FLOPS` (default 1e6); lone big
   children continue inline (chains collapse to one task; the naive
   subtree>=cutoff rule produced 6319 tasks of 6564 supernodes on the
   banded-QP proxy, the sibling rule 1). Owned nodes factored serially
   in postorder inside their task; parent-task trampoline via
   task-children pending counters; `seeds < 2` delegates to the
   sequential driver (intrafront parallelism kept on).
   `FERAL_DEBUG_TASK_PLAN=1` dumps plan shapes.
3. **Byte-exactness**: scheduling-only; new `tests/task_plan_parity.rs`
   pins fine (cutoff=1: 153 tasks/132 seeds), default, and fallback
   configurations bit-identical to the sequential driver; 84/84 test
   binaries green.
4. **chainW anomaly investigated and documented**: the old per-node
   driver oddly beat both sequential drivers ~20% on one wide-block
   chain proxy; AtomicLockStats telemetry located the difference
   *inside* `factor_one_supernode` (912 vs 1276 ms per 9 factors) —
   not spawn/lock overhead, not intrafront, not tree parallelism.
   Unexplainable further without perf/heaptrack (absent here);
   accepted as a proxy quirk since the issue's real chains lose under
   per-node spawning. Full analysis in the research note.

## Benchmark Results

Session bench (corpus absent): synthetic set + regression fixtures all
inertia/residual pass; oracle partitions N/A in container.

Interleaved old-vs-new (warm medians, default parallel config):

| proxy | old par@4 | new par@4 | old serial | new serial |
|---|---:|---:|---:|---:|
```

## Git Status
```
808babb release: feral v0.15.0 (#151)
fad5670 perf(parallel): task-per-subtree coarsening + profiler nanoseconds (#150)
e8e1c5a perf(kernel): explicit SIMD packed trailing update + x86 pulp dispatch fix (#149)
6589570 docs: session checkpoint 2026-07-11-02 (issue triage, #127, release 0.14.0) (#146)
c05eb77 release: feral v0.14.0 (#145)
```

## Test Status
```
test symbolic::tests::symbolic_factorize_auto_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
test symbolic::tests::symbolic_factorize_external_produces_valid_perm ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 409 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 5.72s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-09-02.md)


Session bench (corpus absent): synthetic set + regression fixtures all
inertia/residual pass; oracle partitions N/A in container.

Interleaved old-vs-new (warm medians, default parallel config):

| proxy | old par@4 | new par@4 | old serial | new serial |
|---|---:|---:|---:|---:|
| sparseqpL (issue signature) | 82.4-88.0 ms | 71.7-76.3 ms | 69.9-74.6 | 70.7-72.2 |
| grid250 | 78.8-92.7 ms | 71.4-73.4 ms | 123.0-128.7 | 122.0-128.0 |
| chainW | 186.7-216.6 ms | 221.1-223.9 ms | 228.1-229.5 | 214.9-215.7 |
| chain12000 | 12.0-12.5 ms | 11.9-12.5 ms | 11.3-11.7 | 10.9-11.0 |

Spawn reduction: grid250 11171 → 51 tasks; chains → 1 (sequential
path). Small fixtures in the noise band (−6%..+4%). The chainW par@4
old-vs-new gap is the documented anomaly above (old-par was beating
serial there; new ≈ serial).

```

## Recent Decisions
run medians. `min_us` per invocation is the preferred per-sample
statistic (least interfered). Cross-time comparison of numbers taken in
different sessions is not evidence at all.

**Why.** Measured on the issue-#148 chainW proxy (session 2026-08-09-03):
three `FERAL_PAR_TASK_MIN_FLOPS` settings that produce *identical* task
plans — same code path, byte-identical work — measured 139.6 / 259.1 /
155.5 ms, a 1.9x spread. Eight invocations of one fixed config spanned
min_us 124.7-163.2 ms (31%) and median_us 149.2-183.7 ms (23%). Two
conclusions had already been drawn from inside that band and were both
wrong: a claimed "chainW anomaly" (per-node spawning 20% faster than
sequential) and a claimed 5-18% regression from PR #150. Paired
re-measurement reversed both — 9/12 pairs favour the new code (median
ratio 0.961) and 9/10 favour coarse over fine-grained tasks (median
1.045, sign-test p~0.02).

**Relationship to the existing rule.** The 2026-04-14 entry ("any
bench-p90 delta smaller than ~5% must be confirmed with a 3-run
median") is necessary but NOT sufficient: three consecutive medians can
all land inside one drift excursion, which is exactly how both wrong
conclusions above were reached. Paired A/B supersedes it for container
measurement; the 3-run rule still applies to the corpus bench on a
quiet machine.

**Consequence for prior sessions.** Numbers in
dev/sessions/2026-08-09-01.md and -02.md were collected unpaired.
Those with large effects (dense-front kernel 2.7-7x, grid250, sparseqpL
- since re-confirmed paired at 10/10 and 9/10) stand; sub-10% fixture
deltas in those checkpoints should be treated as unresolved rather than
as measured wins until re-run paired.

## Recent Tried-and-Rejected
plain `for i in j..n { a[j*n+i] -= a[k*n+i]*alpha }` loops are
textbook-autovectorizable, and the eager path's remaining time is
pivot search + memory traffic, not multiply-subtract throughput.
Explicit lanes duplicated what LLVM already did. This matches the
2026-05-16 finding (pulp == scalar == manual unroll at lengths 3..128)
at the whole-front scale.

**What was kept.** The de-duplication refactor (shared scalar
`rank1_scale_update_argmax`, byte-identical, golden digests unchanged)
stays; the pulp kernel, its gate/env var, the dedicated parity test,
and the A/B example were removed.

**Lesson.** The small-front/MA57 gap is NOT lane width in the eager
update. Remaining suspects, in evidence order: per-front fixed
overhead (assembly/scatter/build-row, 8.8-14.8% on the small
fixtures), pivot-search scans, `scalar_pivot_step` in blocked fronts,
and the delayed-pivot cascade (per-factor-cost-cluster mechanism A).
Any retry of eager-path SIMD must first show a front-level profile
where the update loops are >30% of eager time AND not already
vectorized in the disassembly.

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
