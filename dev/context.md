# FERAL Context (auto-generated)

Generated: 2026-08-15T22:04:50Z

## Latest Session
File: dev/sessions/2026-08-15-01.md
```
# Session 2026-08-15-01

## Benchmark note (mandatory unfavorable-comparison section)

The first bench run of this session **regressed both dense buckets** against
the 2026-08-14-03 baseline. It was chased to root cause and fixed, not filed
as noise. Final numbers beat the baseline:

| bucket                       | 08-13-04 | 08-14-02 | 08-14-03 | this session (first) | **this session (final)** | target | verdict |
|------------------------------|----------|----------|----------|----------------------|--------------------------|--------|---------|
| dense small-frontal (<200) p90 | 1.61   | 1.57     | 1.58     | **1.66**             | **1.54**                 | ≤ 2.0  | PASS    |
| dense medium (<500) p90        | 2.09   | 1.96     | 2.00     | **2.13**             | **1.91**                 | ≤ 3.0  | PASS    |
| sparse small-frontal (<200) p90| 1.54   | 1.50     | 1.54     | 1.50                 | 1.50                     | ≤ 2.0  | PASS    |
| sparse medium (<500) p90       | 1.54   | 1.50     | 1.54     | 1.51                 | 1.51                     | ≤ 3.0  | PASS    |

Cause of the regression: the first cut of the #134B fix allocated an
`n`-length degree accumulator on every `pick_scaling_strategy` call. The
router runs per factor and the dense partition is ~148k sub-millisecond
factorizations, so an unconditional per-call allocation is visible (+5%
small-frontal, +6.5% medium) even though it is O(n+nnz) against an
O(n^1.5+) factorization. Fixed in `45c80f3` by ordering the gates so the
allocation stays off the common path.

## Goal

Pick up issue #153 (KR scaling warm-start) and issue #134 item B (the
scaling router's lower-triangle-blind gates). Both were carried in from
triage with a stated premise; measure both before implementing either.

## Accomplished

### #134 item B — router permutation-invariance (shipped)

`pick_scaling_strategy`'s dense-head gate now counts **symmetric degree**
instead of stored lower-triangle column length.

- **Bug confirmed and resized.** Under the pure relabeling `P(i) = n-1-i`,
  VESUVIO's head reports stored max degree 1026 one way and 11 the other and
  the route flips `Mc64Symmetric` → `InfNorm`. Over the full 1004-family
  corpus (`kkt` + `kkt-mittelmann` + `kkt-expansion`) the old router was
  permutation-invariant on only **841**.
- **Fix measured against the shipped router**, not a reimplementation:
  invariance **841 → 890**, **15 route changes, 15 gains, 0 losses**. The
  change is monotone — symmetric degree is never below stored degree — so no
  family can lose MC64.
- **Movers priced** over every iterate: inertia identical under both routes
  on all 15; factor-time ratio median 0.98 (range 0.95–1.19); accuracy
  neutral or better on 14 (CHAIN 5.02e-6 → 1.42e-8, SOSQP1 8.49e-6 →
  1.95e-6). MSS1 is the one unfavorable mover (+19% time, 1.43e-1 → 6.15e-1
  forward error) but neither route solves it and the Policy 4 fallback test
```

## Git Status
```
fbb1a9d docs: session checkpoint 2026-08-15-01 (#134B shipped, #153 falsified)
45c80f3 perf(scaling): keep the router's symmetric pass off the common path
e9470ca fix(scaling): count symmetric degree in the router's head gate (#134B)
8acb1be docs(scaling): research + plan for router permutation-invariance (#134B)
14b3865 diag: size the KR warm-start lever against steady-state routes
```

## Test Status
```
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 427 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.57s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-15-01.md)


--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.54     <= 2.0     PASS
medium (<500)            152145     1.91     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.50     <= 2.0     PASS
medium (<500)            153560     1.51     <= 3.0     PASS

```

## Recent Decisions
route is gated on a bump that was actually peeled (21f5e74), so it is
unreachable unless triangularization is on. It is one lever, not two.

**AMF stays off** because no downstream measurement was taken this session. That
is an absence of evidence, not a finding against it.

**Known cost of this decision.** `Markowitz` ignores `factor`'s `symbolic`
argument, because it does not use a precomputed column order. That is a silent
semantic change for any caller that carefully chose an ordering and passed it in
— exactly the silent-fallback shape #168 warned about. Three mitigations, and no
claim that they eliminate it: the selector is an explicit enum rather than a
bool, `SparseLu::used_markowitz()` makes the executed route observable so a
measurement can assert instead of infer, and every in-repo ordering comparison
(`examples/lu_fill_orderings.rs`, `src/bin/probe_lu_phases.rs`,
`src/bin/probe_ft_eta.rs`) is pinned to `GilbertPeierls` in this change. An
out-of-tree caller comparing orderings against `LuParams::default()` will
silently compare nothing until it pins the rule. Seven in-repo test sites
across five files failed on this change — `sparse_lu_honors_pivot_threshold`,
`dense_bump_route_needs_the_peel_and_the_cap_together`, the whole
`lu_dense_bump` suite, `reach_route_composes_with_the_dense_bump_route`,
`perturb_chooses_largest_magnitude_row_matching_dense`,
`factor_traversal_is_subquadratic`, and
`sparse_solves_compose_with_the_dense_bump_route` — and every one was fixed by
pinning `GilbertPeierls`, never by weakening an assertion. That seven
independent tests failed is corroboration that the hazard is real, not
hypothetical, and it is a fair estimate of what a downstream suite should
expect to have to pin.

Evidence: issue #171; `dev/research/markowitz-fill-measurement.md`;
`dev/journal/2026-08-14-01.org`; the #166 and #168 arm harnesses.

## Recent Tried-and-Rejected

**Also:** this hypothesis had already been falsified six days earlier in
`dev/research/scaling-warm-start-2026-08-09.md` (zero iteration reduction on
6 fixtures; "lower the cap" ranked worst on risk/benefit, cap 5 giving
3.7e-1 vs 1.4e-2 — 26x worse). That note was not read before recommending
the work, which is the protocol step — read `dev/research/` and this file
*before* writing code — that exists to prevent exactly this.

**Sizing was also wrong.** The lever was first sized off what
`pick_scaling_strategy` returns, but the sticky-`Auto` pin (#51/#65) means
the steady-state route can differ: `mc64_cache_hit_count()` shows dtoc1nd at
14/20 hits and marine at 12/18, i.e. both run MC64, not InfNorm. Only 2 of
the 6 #153 fixtures (clnlbeam, steering_12800) actually run KR, so the lever
was ~7-12%, not the 10-20% first reported. Size scaling levers off the
observed route, not the picker.

**Not rejected:** warm@10 — the same sweep budget, better conditioning
(geomean 3x-100x lower final deviation). That is free quality, not a
speedup, and would need downstream iterative-refinement counts to justify.
Recorded as option 2 in the 2026-08-09 note; still open.

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
