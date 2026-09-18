# FERAL Context (auto-generated)

Generated: 2026-09-18T13:36:43Z

## Latest Session
File: dev/sessions/2026-09-18-01.md
```
# Session 2026-09-18-01

Continuation of `2026-09-17-01.md` (same unbroken session; that checkpoint
covers the diagnosis and the `feral-metis` fix and stops there). Read this one
for everything after.

## Benchmark note (read first)

**`cargo run --bin bench --release` still has not been run against the new
corpus, and no factor-ratio-vs-MUMPS number is quoted anywhere in this
session.** That gate needs `*.mumps.json` oracle sidecars, which need a MUMPS
5.8.2 build that does not exist on this machine (`ref/` is empty). It was
judged unnecessary for what this branch changes — the ordering permutation is
bit-identical on 7 of 9 corpus families, so a ratio-vs-MUMPS comparison would
be measuring unchanged code — but that is a judgement, not a measurement, and
the next session should say so out loud rather than let it pass silently.

What *was* measured: steady-state numeric factor time, paired, on 135 real KKT
matrices across 9 families, gated on cross-arm inertia agreement and a refined
residual. Those numbers are throughout.

## Goal

Finish issue #203: fix the remaining backends, get a real corpus, decide
whether the work is release-ready.

## Accomplished

### `ScotchND` and `KahipND` had the identical defect, now fixed

All three ND backends refined a 2-way edge bisection through uncoarsening and
built the node separator once at the finest level. Elimination flops on the
collocation KKT at n=224,646:

| backend | before | after | gain |
|---|---|---|---|
| `MetisND` | 2.095e10 | 6.737e9 | 3.11x *(2026-09-17)* |
| `ScotchND` | 2.264e10 | 1.017e10 | 2.23x |
| `KahipND` | 2.577e10 | 7.184e9 | 3.59x |

KaHIP's symbolic analysis roughly halves (5,263 -> 2,274 ms) because its
max-flow lift now runs on the coarsest graph. Corpus wall-clock is neutral for
both (scotch 1.060 -> 1.045, kahip 1.118 -> 1.102 against `Amf`).

### A real corpus, without AMPL

`scripts/build-pounce-corpus.sh`. pounce's `generate_nl.py` builds six large
NLPs in Pyomo and writes `.nl` directly, so the AMPL Community Edition licence
is not needed. 135 matrices, 9 families, covering the shapes the argument
turns on: `bratu` (#67's `bratu3d`), `poisson` (#73's `cont5_1_l`),
```

## Git Status
```
9bcecac test(bench): laptime anomaly resolved — no blocker, and two of my results corrected
f25cf3d feat(scotch,kahip): refine the node separator through uncoarsening
58d8cf4 test(bench): scotch and kahip have the same defect metis had, measured
3e358af test(bench): corpus A/B settles both routing questions — and retracts my Amd claim
605536f test(bench): policy break-even data — and Amd beats Auto on all six families
```

## Test Status
```
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test numeric::solve::tests::issue175_wide_thin_tree_is_not_scheduled_in_parallel ... ok
test numeric::solve::tests::issue175_overhead_term_is_scheduling_only ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test numeric::solve::tests::cb_coarsening_threshold_is_arithmetically_inert ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 452 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.65s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; no session checkpoint with bench)
```

## Recent Decisions
bisection and converting once at the finest level.

**Why, given a narrow audience.** Neither backend is reachable from `Auto`,
so nobody gets this by default. It was done anyway because leaving a known,
measured, 2-3.6x deficiency in two of three backends through a release is
worse than the audience is small — a later user who reaches for `ScotchND`
has no way to know it is the un-fixed one.

**Results** (gaslib nfe=48, elimination flops): scotch 2.264e10 -> 1.017e10
(2.23x), kahip 2.577e10 -> 7.184e9 (3.59x). On the pounce corpus both are
neutral in wall-clock (scotch geomean vs `Amf` 1.060 -> 1.045, kahip 1.118 ->
1.102) — the same "large win on collocation, inert elsewhere" profile metis
had.

**KaHIP keeps its flow lift at the coarsest level only**, rather than
re-running it per level. `flow_node_separator` is a max-flow vertex-cover
reduction and per-level use would cost more than the refinement is worth; the
FM pass down the hierarchy is what fixes the objective. A side effect is that
kahip's analysis roughly halves, since the max-flow now runs on the smallest
graph rather than the largest.

**Corroboration worth keeping.** Scotch's finest-level separator step was
already *better* than pre-fix metis's — a two-sided FM optimising separator
weight directly, against a König cover plus a positive-gain-only greedy pass
— and scotch still measured slightly worse than pre-fix metis (3.36x vs
3.11x behind). That is independent evidence for the original diagnosis: the
final polish is not what decides this, the hierarchy is.

**Evidence.** `dev/research/scotch-kahip-node-separator-2026-09-18.md`;
`cargo test --workspace` 1212 passed, 0 failed; fmt and clippy clean.

## Recent Tried-and-Rejected

**Why they fail, as far as it was chased.** `dtoc_wide`'s `max_front` is 31.
At that width the numeric phase is all per-front overhead and no arithmetic,
so the comparison measures supernode count rather than flops, and MetisND's
fewer-but-wider fronts win for reasons that have nothing to do with dtoc2.
Widening the chain from `(nx,nu) = (4,2)` to `(8,4)` moved `max_front` only
24 -> 31; the real dtoc2's avg_deg of 17.5 is not reachable by this
construction at the right `n`.

**This was already on the record.**
`external_benchmarks/chain_proxy/README.md` documents geometry-matched
proxies producing the wrong answer and says to prefer the real corpus. That
warning was read during this session and the proxies were built anyway.

**What survives.** Nothing about routing. The non-proxy measurements in the
same session stand on their own and are recorded in
`dev/research/issue-203-auto-routing-2026-09-17.md`: on the issue #203
patterns MetisND is 1.62x faster sequentially and 3.5x in parallel, and
`nnz_L`, `flop_proxy` and `max_front` were each checked as routing guards
and each mispredicts.

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
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/env_knob_parsing.rs
tests/env_knob_scan.rs
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
tests/issue177_parallel_entry_point_core.rs
tests/issue178_refine_cap.rs
tests/issue178_solve_into.rs
tests/issue203_ordering_race.rs
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
tests/lu_default_ordering.rs
tests/lu_dense_bump.rs
tests/lu_dense_update_bg.rs
tests/lu_dense.rs
tests/lu_ft_widebump.rs
tests/lu_hyper_sparse.rs
tests/lu_markowitz.rs
tests/lu_real_bases.rs
tests/lu_scaling.rs
tests/lu_sparse_rhs.rs
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
tests/pounce710_refine_cap_nrhs2.rs
tests/profiler_smoke.rs
tests/property_tests.rs
tests/refined_solve_core_stability.rs
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
