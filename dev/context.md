# FERAL Context (auto-generated)

Generated: 2026-08-14T18:06:07Z

## Latest Session
File: dev/sessions/2026-08-14-02.md
```
# Session 2026-08-14-02

## Goal
Fix the open issues in the 161-168 range rather than commenting on them:
land #167 (threshold-Markowitz LU), resolve #161 (kernel vs SuperLU), and
settle #166 (QPLIB_3225 under the triangularized ordering).

## Accomplished

### #167 threshold-Markowitz LU -- implemented, merged, closed
`SparseLu::factor_markowitz` picks each pivot to minimise `(r_i-1)(c_j-1)`
subject to `|a_ij| >= u*max_k|a_kj|`. On 16 preserved LP bases,
`factor_nnz()/nnz(B)` geomean **2.77x -> 1.06x** vs AMD-full, zero fill on
10 of 16. Wall geomean vs AMD-full 4.92x (was 3.50x before the in-place
rank-1 update, which was the real lever -- the original per-column Vec
rebuild cost one allocation per (pivot, column) pair).
Reported honestly, and repeated here: the Suhl-Suhl singleton fast path
**failed** at its stated purpose (fill 1.07x -> 1.06x, wall 3.43x -> 3.50x,
i.e. slightly worse) and was kept only because Markowitz subsumes it at
cost 0. Markowitz also **loses** to the cheap peel on near-triangular bases
(QPLIB_0911_rlt0, bump 29: peel 1.10ms vs Markowitz 4.32ms) and wins on
bump-heavy ones (QPLIB_1143_rlt1, bump 624: 4.87ms vs peel 31.77ms).
Commits 04756ac, 48382c7, 0e8e976; PR #170 merged as bd78bad.

### #161 kernel vs SuperLU -- re-measured on merged main, closed
SuperLU reproduced its original numbers exactly (213,132 factor-nnz, 7.26x
fill, 11.13ms on QPLIB_1157), so old and new numbers are comparable.
Both named defects are fixed:
- factorization: dense-bump 10.91ms = parity with SuperLU's 11.13ms;
  Markowitz 6.73ms at fill 1.74x = **1.65x faster at 4.2x less fill**.
- solve: at the SHIPPED `hyper_sparse_max_density=0.10`, ftran p50
  88.42us -> 7.92us = **11.17x**; btran 0.98x (neutral).
Two reframings recorded:
- SuperLU's own solve is **not** work-proportional either: same factor,
  unit rhs 107.2us vs dense rhs 106.0us. #161 measured feral against
  itself on that axis; against the reference it was never a feral-specific
  defect, and at 7.92us it is now a feral advantage.
- #161's premise "the ordering is not the problem, our fill matches
  COLAMD" was a true observation with a false conclusion. AMD-on-A^T A and
  COLAMD are the same algorithm class; matching the reference *inside* a
  class cannot detect that the class leaves 4x on the table. That premise
  is what made this expensive to find.
Opened **#171** for what remains: all three measured levers (dense bump,
Markowitz, AMF) are off by default and the shipped config is still 4.03x
SuperLU on QPLIB_1157. That is a defaults decision, not a kernel defect.

### #166 QPLIB_3225 under the triangularized ordering -- closed, not reproducible
Two feral worktrees off `3209fad` differing in **exactly one line**
(`analyze` -> `analyze_with(default)` vs `analyze` -> `analyze_triangularized`),
verified by `diff -r --brief`; two discopt worktrees pinned at `bce881ff`
```

## Git Status
```
6fc92d6 Merge pull request #173 from jkitchin/release/0.16.0
c681c2a release: feral v0.16.0
c9c3adc docs: session checkpoint 2026-08-14-02 (161-168 closed, #171 landed)
ec67e85 Merge pull request #172 from jkitchin/feat/171-lu-defaults
d5048b9 feat(lu)!: default SparseLu::factor to threshold-Markowitz pivoting
```

## Test Status
```
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 423 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.45s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-14-02.md)


No regression to report. All four exit-partition buckets are equal or
better than session 2026-08-13-04, the last session that actually ran the
bench:

| bucket | 08-13-04 | this session |
|---|---|---|
| dense small-frontal (<200) p90 | 1.61 | **1.57** |
| dense medium (<500) p90 | 2.09 | **1.96** |
| sparse small-frontal (<200) p90 | 1.54 | **1.50** |
| sparse medium (<500) p90 | 1.54 | **1.50** |

Read this as run-to-run variation, not as a benefit of this session's
work: `bin/bench` measures the LDL^T/KKT path, and #171 changed only the
LU basis factorization, which that harness does not exercise. The
practical content is that session 04's reported dense p90 regression
(1.57 -> 1.61, 1.96 -> 2.09) did not persist.

MCONCON                  3000       0.85       0.89       1.62
HATFLDH                  3000       0.42       0.45       0.55
CONCON                   3000       0.82       0.88       1.66
PALMER7A                 3000       0.27       0.30       0.33
DJTL                     3000       0.09       0.10       0.22
PALMER5A                 3000       0.29       0.30       0.33
HS90                     3000       0.20       0.20       0.30
SSI                      3000       0.20       0.22       0.27
HATFLDBNE                3000       0.38       0.40       0.82
MGH10LS                  3000       0.20       0.22       0.25
HS92                     3000       0.35       0.40       0.44
AVION2                   2682       1.41       1.46       1.92
CERI651ALS               2331       0.27       0.27       0.40
PFIT4                    2286       0.23       0.25       0.30
CERI651C                 2233       0.28       0.30       0.40
CERI651CLS               2227       0.26       0.27       0.40
BATCH                    2054       1.28       1.34       1.66

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
KIRBY2_0007                    458         1065          119       8.95
KIRBY2_0006                    458         1003          127       7.90
KIRBY2_0008                    458          930          122       7.62
KIRBY2_0009                    458          847          128       6.62
KIRBY2_0010                    458          776          133       5.83
KIRBY2_0011                    458          670          120       5.58
GROUPING_0243                  225          591          111       5.32
GROUPING_0031                  225          564          109       5.17
GROUPING_0045                  225          563          113       4.98
GROUPING_0231                  225          556          113       4.92

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.57     <= 2.0     PASS
medium (<500)            152145     1.96     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.50     <= 2.0     PASS
medium (<500)            153560     1.50     <= 3.0     PASS

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

(truncated from      360 lines to 350 line budget)
