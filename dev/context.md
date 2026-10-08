# FERAL Context (auto-generated)

Generated: 2026-10-08T10:28:30Z

## Latest Session
File: dev/sessions/2026-10-08-01.md
```
# Session 2026-10-08-01

## Benchmark note (read first)

`cargo run --bin bench --release` ran on the in-repo fixtures only
(`data/benchmark-config.toml` not found, 8 matrices, no oracle timings).
KKT inertia match 2/2 vs MUMPS, residual pass 2/2, no dense or sparse
failures. No timing comparison against the previous session is possible
from this output. The change touches only the threshold-reject branch, which
the fixtures do not reach.

## Goal

Fix issue #213: cascade-break supernodes force-accept threshold-failing
pivots (2.3e-16), so |L| grows to 1e16, inertia is off by 2, the backward
error is 8e3, and nothing shows up in `FactorStats`.

## Accomplished

- **Root cause.** `ZeroPivotAction::PerturbToEps` (the CB policy, floor
  `1e-10·‖DAD‖`) only fired under `|d| <= zero_tol = EPS`. The issue's
  pivots of 2.35e-16 sit just above EPS and fell into case (b), accepted
  unperturbed. The same structure exists in `try_reject_1x1_frontal` and
  `do_1x1_pivot`.
- **Fix.** `below_perturb_floor` widens the gate at both sites: any
  threshold-failing pivot below `abs_floor` is floored and counted in
  `n_tiny`. Root and `ForceAccept` behaviour is unchanged.
- **Diagnostics.** `NodeFactors::cascade_break`,
  `SparseFactors::n_cascade_break_nodes()`,
  `FactorStats::n_cascade_break_nodes`, and the Python binding.
- **Tests.** `issue_213_perturb_band_tests` (3 tests, hand oracle; the key
  test fails with the gate disabled) and
  `issue_55_delay_budget::cascade_break_nodes_are_counted`.
  `cargo test`: 952 passed, 0 failed. fmt and clippy are clean.

## Benchmark Results
```
8 matrices benchmarked
KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
Dense failure analysis: no failures
Sparse failure analysis: no failures
```

## Decisions Made
- `PerturbToEps` floors every threshold-failing pivot below `abs_floor`
  (`dev/decisions.md` 2026-10-08).
```

## Git Status
```
32c936b Merge pull request #210 from jkitchin/fix/issue-208-autorace
d3b7c9a api: export the new 0.18.0 types, add PartialEq, expose OrderingInfo to Python
807ef57 docs(ordering): workload guidance for #209, measured per family
7cd4e95 fix(autorace): drop Amf from RACE_CANDIDATES — it regressed pounce laptime (#208)
d699886 Merge pull request #207 from jkitchin/release/0.18.0
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

test result: ok. 453 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 3.42s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-10-08-01.md)

8 matrices benchmarked
KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
Dense failure analysis: no failures
Sparse failure analysis: no failures

```

## Recent Decisions
abs_floor }` now perturbs any pivot that failed `|d| <= u·col_max` and has
`|d| < abs_floor`. Previously the floor applied only under `|d| <= zero_tol`,
so a pivot in `(zero_tol, abs_floor)` fell through to case (b) and was
accepted unperturbed. Only cascade-break fronts run with `PerturbToEps` by
default, so root fronts and `ForceAccept` callers do not change.

**Why.** The `cascade_break_eps` doc already promised `sign(d) · max(|d|,
floor)` for tiny pivots. Under the default `cascade_break_eps = 1e-10` the
band `(2.2e-16, 1e-10·‖A‖)` broke that promise silently. Issue #213 is a
151,392² KKT where 2.3e-16 pivots in 7 CB nodes produced |L| ≈ 9.6e15,
inertia off by +2, unrefined backward error 7.8e3, and `n_tiny = 0`.
Flooring at 1e-10 bounds |L| by `col_max / 1e-10` instead of `col_max /
2e-16`, and makes every such event visible in `n_tiny`.

**What it does not do.** CB fronts still do not obey the threshold-pivoting
bound `|L| <= 1/u` (1e8 at u = 1e-8). A pivot in `[abs_floor, u·col_max]`
is still case-(b) accepted, with growth up to `col_max / abs_floor`. A
structural fix (swap-to-boundary delayed pivoting,
`dev/research/kkt-cascade-amplifier-2026-05-21.md`) would remove CB's
inertia risk entirely and remains open. To make the remaining risk visible
now, `FactorStats::n_cascade_break_nodes` reports how many fronts ran
without delaying.

**Evidence.** New `issue_213_perturb_band_tests` in `src/dense/factor.rs`
check d → ±1e-10, L, and the Schur complement against a hand calculation on
both the scalar and blocked kernels. The ±3e-16 test fails with the gate
disabled. `tests/issue_55_delay_budget.rs::cascade_break_nodes_are_counted`
shows 3 CB nodes on a budget-zeroed fixture and 0 in budget. `cargo test`:
952 passed, 0 failed. Not run on the reporter's matrix, which was not
available.

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
tests/column_renumbering.rs
tests/column_renumbering_parity.rs
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
```
