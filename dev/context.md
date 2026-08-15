# FERAL Context (auto-generated)

Generated: 2026-08-15T19:31:04Z

## Latest Session
File: dev/sessions/2026-08-14-03.md
```
# Session 2026-08-14-03

## Benchmark numbers are slightly worse than last session — reported first, per protocol

All four exit-partition p90 buckets moved up relative to session
2026-08-14-02, run four hours earlier on the same machine:

| bucket | 08-13-04 | 08-14-02 | **this session** | target | verdict |
|---|---|---|---|---|---|
| dense small-frontal (<200) p90 | 1.61 | 1.57 | **1.58** | <= 2.0 | PASS |
| dense medium (<500) p90 | 2.09 | 1.96 | **2.00** | <= 3.0 | PASS |
| sparse small-frontal (<200) p90 | 1.54 | 1.50 | **1.54** | <= 2.0 | PASS |
| sparse medium (<500) p90 | 1.54 | 1.50 | **1.54** | <= 3.0 | PASS |

All four still PASS, and every one of them sits inside the band the last
three sessions have bounced around in. This session is the strongest
available evidence that the band *is* noise rather than drift: the only
non-version, non-changelog edits were **doc comments**. The compiled code
is byte-for-byte the same solver session 08-14-02 measured at 1.57 / 1.96
/ 1.50 / 1.50. A harness that reports a 0.04 spread on identical codegen
is telling you its own resolution, not the solver's.

Sparse factor tail this run: geomean 0.44, p50 0.30, p90 1.54, p99 3.25,
max 9.14. Against the tail recorded in the 0.15.1 release commit (geomean
0.44, p90 1.57, p99 3.45, max 12.40) the tail is *better*; against the
0.15.0-era baseline it quotes (geomean 0.43, p90 1.54, p99 3.30, max
8.70) the max is still worse. Unchanged conclusion from that release: the
n=225-458 CUTEst matrices set these tails and nothing in the LU work
touches them.

## Goal

Release 0.16.0. The #161-#168 + #171 arc was merged but undelivered —
crates.io still served 0.15.1, so none of it had reached discopt.

## Accomplished

**0.16.0 shipped, verified live on both registries.**

- `c681c2a` release commit -> PR #173 (all checks pass) -> merge `6fc92d6`
  -> tag `v0.16.0` -> GitHub release.
- crates.io `feral` max_version **0.16.0**, newest_version 0.16.0. Queried
  with a `User-Agent` and sanity-checked against `serde` (1.0.229), per the
  method note in the global agent instructions.
- PyPI `feral-solver` **0.16.0**, 5 files: macOS universal2, manylinux
  x86_64, manylinux aarch64, win_amd64, sdist.
- The six ordering crates correctly stayed at **0.2.1**. None changed since
  v0.15.1, so `release.yml`'s "already exists on crates.io index" path
  treated them as success. The checklist's staleness guard confirmed this
  was intentional and not a silent skip of changed code.
```

## Git Status
```
810080d docs: session checkpoint 2026-08-14-03 (0.16.0 released)
6fc92d6 Merge pull request #173 from jkitchin/release/0.16.0
c681c2a release: feral v0.16.0
c9c3adc docs: session checkpoint 2026-08-14-02 (161-168 closed, #171 landed)
ec67e85 Merge pull request #172 from jkitchin/feat/171-lu-defaults
```

## Test Status
```
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 423 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.43s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-14-03.md)


=== Dense perf vs canonical oracles (154481 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153472       0.24       0.10       2.15      38.21      88.39
solve/MUMPS        153472       0.10       0.08       0.40       4.82      37.54
factor/SSIDS       154393       0.02       0.01       0.46      10.36      25.77
solve/SSIDS        154393       1.32       1.00       7.00      60.50     432.28
nnzL/MUMPS         153472       1.53       1.00       5.67      35.32      99.41
nnzL/SSIDS         154393       2.23       1.78       5.27      61.86     103.86


=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.44       0.30       1.54       3.25       9.14
solve/MUMPS        153560       0.07       0.08       0.14       0.66       2.31
factor/SSIDS       154500       0.04       0.03       0.32       0.95       1.97
solve/SSIDS        154500       0.93       1.00       2.44       8.00      30.00
nnzL/MUMPS         153560       0.61       0.58       0.75       4.50      23.11
nnzL/SSIDS         154500       0.88       1.00       1.00       4.50       5.00


--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.58     <= 2.0     PASS
medium (<500)            152145     2.00     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.54     <= 2.0     PASS
medium (<500)            153560     1.54     <= 3.0     PASS
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
two discopt worktrees pinned at `bce881ff` via `[patch.crates-io]`, two
extension `.so`s with distinct md5s loaded by `PYTHONPATH`, each arm
asserting its own module path before solving. Slower to build, but it
measures the thing the issue is about.

## 2026-08-14 — #171: plain `cargo test` as the verification gate

**Tried.** Verifying the Markowitz-default change with `cargo test`.

**Why it failed.** `cargo test` stops after the first failing test *target*.
Four consecutive runs each reported exactly one failing binary, so the same
"the suite is green except X" conclusion was drawn — and was wrong — three
times. Run 5's log contains zero occurrences of `lu_sparse_rhs`: that binary
never executed. The true blast radius was seven test sites across five
files.

**Replaced with.** `cargo test --no-fail-fast`, which surfaced all of them
in one pass (864 passed, 0 failed). For any change to a default that every
test inherits, fail-fast turns one measurement into N sequential ones and
hides the scope.

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
