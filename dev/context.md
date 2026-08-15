# FERAL Context (auto-generated)

Generated: 2026-08-15T22:22:29Z

## Latest Session
File: dev/sessions/2026-08-15-02.md
```
# Session 2026-08-15-02

## BENCHMARK NUMBERS ARE WORSE THAN LAST SESSION

Reported first, per the hard rule in CLAUDE.md.

    partition                  2026-08-15-01   this session   delta
    dense  small-frontal (<200)     1.54           1.61        +0.07
    dense  medium (<500)            1.91           2.00        +0.09
    sparse small-frontal (<200)     1.50           1.67        +0.17
    sparse medium (<500)            1.51           1.67        +0.16

All four partitions regressed. All four still PASS their targets.

**No Rust source changed this session.** `git diff --stat fbb1a9d HEAD
-- src/ crates/ tests/` is empty; the only changes are `dev/`
documents. So this is not a code regression — it is run-to-run
variance on the same binary.

That conclusion is itself worth recording, because it bears on a claim
made last session. 2026-08-15-01 reported a dense small-frontal
sequence of 1.58 (baseline) -> 1.66 (regression I introduced) -> 1.54
(after the gate reordering), and I described the final number as
"beating the baseline". Today's run puts the same unchanged code at
1.61. A +0.07 swing with zero code change means the bench's noise
floor on this metric is at least as large as the 1.58 -> 1.54
"improvement" I claimed. **The gate-reordering fix should be regarded
as having removed the 1.66 regression, not as having beaten the
baseline.** The 1.66 measurement is still meaningful (it exceeded the
noise band), but the final 0.04 is not.

Action for a future session: establish the bench's noise floor by
running it N times on an unchanged binary, and record a
minimum-detectable-difference so per-session comparisons stop
over-reading sub-0.1 movements. Until then, treat p90 deltas under
~0.15 as noise.

Also note the worst-ratio table is now **6 of 10 KIRBY2 iterates**
(worst 9.22, up from 8.95), plus GROUPING_0205 — i.e. the outlier
family this session diagnosed dominates the tail more clearly than
before.

## Goal

Investigate the two items carried out of 2026-08-15-01, both approved
by the user:

1. **#153 remainder** — MC64 warm-cache miss cost. marine_1600 spends
   ~19% of a 1784 ms solve in cache-missed MC64 recomputes. Decide
   whether `GROWTH_FACTOR`/`GROWTH_COUNT` can be tightened without
```

## Git Status
```
3029905 docs(compress): localize the KIRBY2 factor-ratio outlier to LdltCompress
fbb1a9d docs: session checkpoint 2026-08-15-01 (#134B shipped, #153 falsified)
45c80f3 perf(scaling): keep the router's symmetric pass off the common path
e9470ca fix(scaling): count symmetric degree in the router's head gate (#134B)
8acb1be docs(scaling): research + plan for router permutation-invariance (#134B)
```

## Test Status
```
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 427 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.85s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-15-02.md)


**No Rust source changed this session** (`git diff --stat fbb1a9d HEAD
-- src/ crates/ tests/` is empty; the only changes are `dev/`
documents). The run below is therefore a re-measurement of unchanged
code and is expected to reproduce 2026-08-15-01.

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(us)    mumps(us)      ratio
KIRBY2_0007                    458         1097          119       9.22
KIRBY2_0006                    458         1075          127       8.46
KIRBY2_0008                    458          971          122       7.96
KIRBY2_0010                    458          992          133       7.46
LAKES_0144                     168          372           54       6.89
KIRBY2_0009                    458          879          128       6.87
KIRBY2_0011                    458          820          120       6.83
LAKES_0146                     168          350           54       6.48
GROUPING_0205                  225          700          111       6.31
QPCBLEND_0030                  157          362           60       6.03

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.61     <= 2.0     PASS
medium (<500)            152145     2.00     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.67     <= 2.0     PASS
medium (<500)            153560     1.67     <= 3.0     PASS

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
**Refuted by measurement.** `diag_symbolic_stages_argv` on
KIRBY2_0007:

    TOTAL 1182 us
      ldlt_compress   972   82.2%
      renumber         57    4.8%
      ordering         32    2.7%

Ordering is 32 us — 2.7% of symbolic and ~3% of the reported
`factor_us`. Eliminating AMD cost entirely could not move the ratio.
The cost is `ldlt_compress` (the MC64 matching feeding Duff-Pralet
compression), which is a different subsystem from the one the
hypothesis named.

A second prediction in the same hypothesis — that feral was producing
more fill than MUMPS — is also refuted: the numeric driver is 127 us
and `num_c ~ num_n` (149 vs 143 us), so the factorization is not the
problem in either time or fill.

Superseded by `dev/research/ldlt-compress-cost-benefit-2026-08-15.md`.

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
