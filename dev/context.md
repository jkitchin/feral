# FERAL Context (auto-generated)

Generated: 2026-06-06T18:06:39Z

## Latest Session
File: dev/sessions/2026-06-06-04.md
```
# Session 2026-06-06-04

## Goal

Implement the dense-column follow-up "safe win" (scaling audit §8.1 option
(b)): skip the speculative MC64 in symbolic `LdltCompress` when the resolved
numeric scaling will not reuse the cache.

## Accomplished

**Investigated the proposed scaling-aware gate and rejected it on empirical
evidence — the premise is refuted. No `src/` change landed.**

Two design facts established first (by reading solver.rs + symbolic/mod.rs +
ldlt_compress.rs):

1. Symbolic runs at `factor()` Step 3 (solver.rs:805), **on a cache miss
   only**, and is cached per pattern fingerprint — so the `LdltCompress` MC64
   runs **once per pattern**. Scaling resolves later (Steps 3.75/3.8). Making
   symbolic scaling-aware is feasible (compute `pick_scaling_strategy` before
   the symbolic call), but —
2. **The MC64 in `LdltCompress` is load-bearing for compression, not
   speculative.** `build_supermap` (ldlt_compress.rs:39-77) walks the MC64
   matching permutation's cycle structure to form super-variables (MUMPS
   `ICNTL(12)=2` / Duff-Pralet). Skipping MC64 = skipping compression =
   **changing the ordering**. So the win is real only if compression is not
   worth its MC64 cost — an empirical question.

Empirical findings (two new diagnostic probes):

- `probe_compress_scaling_bucket` (3 roots, 1006 families): 376 `LdltCompress`,
  of which 118 reuse MC64 (keep) and **258 do not** (target bucket). The target
  bucket is **not vacuous and not all small**: it includes large dense-column
  matrices — INDEFM (n=100000, deg=100000), SINQUAD2 (5000/5000),
  ex8_2_3 (18791/3132), ex8_2_2 (9453/1894), ORTHREGF (6405/1601),
  ROSEPETAL (3000/2001), all `InfNorm`.
- `probe_compress_costbenefit_argv` (symbolic+numeric, None vs LdltCompress,
  5-run median, release) refutes the premise. Crux = **ROSEPETAL vs ORTHREGF**,
  both large near-dense-column `InfNorm` (won't-reuse), opposite verdicts:
  - **ROSEPETAL** −75.7% (5.97 s → 1.45 s): pays 0.68 s MC64 but the compressed
    ordering gives an **8× numeric speedup** (5.72 s → 0.77 s). Reproducible.
  - **ORTHREGF** +91.8% (6.2 ms → 11.9 ms): MC64 pure overhead, zero numeric
    benefit. Reproducible.

**Conclusion:** the value of `LdltCompress` is its numeric fill reduction,
which is independent of the scaling choice and unpredicted by max_col_deg /
MC64 cost / n (ROSEPETAL's MC64 is 68× ORTHREGF's, yet ROSEPETAL is the win).
The scaling-reuse signal carries no information about whether compression pays
off. A gate keyed on it would regress the fill-reduction wins (ROSEPETAL ~4×,
ex8_2_2) to save milliseconds on the overhead-only losses. **Not a safe win.**
```

## Git Status
```
bb74821 docs(research): MC64 dense-column fast path — definitive negative result
98f85e0 fix(diagnostics): clippy needless_range_loop in scaling_sweep generators
2ef38d5 docs(research): scaling audit report — MC64 is the sole super-linear phase
402630e feat(scaling): localize MC64 dense-column cost; resolve debug/release (#80)
503af96 docs(session): checkpoint 2026-06-06-02 — scaling audit Steps 0-2 (#80)
```

## Test Status
```
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_resolves_to_amd_when_bisection_degenerates ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 317 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.48s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-06-04.md)


`bench --release` NOT re-run: no `src/` (solver) code changed this session.
The only additions are two diagnostic binaries in the non-default
`feral-diagnostics` crate plus docs. No performance delta to track.

Targeted measurement (the session's actual result):
probe_compress_costbenefit_argv (None vs LdltCompress, 5-run median, release)
matrix       n      deg   tot_None     tot_Compress   delta%   verdict
ROSEPETAL   3000   2001   5 965 926us  1 452 187us    -75.7%   compress WINS
ex8_2_2     9453   1894     215 758us    214 777us     -0.5%   compress
ex8_2_3    18791   3132     957 786us    968 299us     +1.1%   neutral
INDEFM    100000 100000   5 030 355us  5 147 508us     +2.3%   neutral
SINQUAD2    5000   5000      17 232us     18 210us     +5.7%   None
ORTHREGF    6405   1601       6 224us     11 936us    +91.8%   None wins
  → scaling-reuse does NOT predict the verdict; gate is unsafe

```

## Recent Decisions
`n > 100_000 && avg_deg < 5 → Amd` (#50 powerflow) and arrow → AMF (#64)
catches fire first and are untouched, so the powerflow-class guardrail and
the dense-border catch still hold; only the uniformly-thin would-be-MetisND
population is rerouted.

Rejected alternative — **fill-guarded reroute** (route above 100k to AMF
only when AMF `factor_nnz_estimate ≤ MetisND's`): this was the design
proposed in the #73 research note *before* the real A/B and the one
originally requested. It is wrong: Finding 3 shows nql180's fill predicate
is *anti-correlated* with real speed (MetisND smaller fill, AMF 2× faster),
so the guard would have kept nql180 on MetisND and forfeited the 2× win.
Fill is not a speed proxy here; a guard keyed on it adds a per-solve
symbolic-race cost to make the *wrong* call. Logged in
`dev/tried-and-rejected.md`.

Scope / generalization: the mechanism is the same as #67 (AMF's cheaper
symbolic + competitive-or-better numeric on uniformly-thin patterns), now
shown to hold above 100k too. The `n>100k && avg_deg<5 → Amd` powerflow
guardrail (#50) is the one place broad thin-matrix reroutes were shown to
regress, and it is preserved by firing first. RDW2D51U + QUADCOPTER did not
finish the real A/B on the loaded test machine; their symbolic predictors
(AMF 1.55× cheaper / tie) and Finding 1 already favor AMF and do not change
the conclusion.

Evidence: dev/research/issue-73-n100k-thin-regime.md (Findings 1–3 +
Decision), dev/journal/2026-06-03-06.org (:issue-73:ab:factor-solve:),
src/symbolic/mod.rs choose_adaptive (AMF_BAND_MAX removed) +
choose_adaptive_rules / choose_adaptive_routes_arrow_to_amf tests,
crates/feral-diagnostics/src/bin/probe_issue73_symbolic.rs,
crates/feral-diagnostics/src/bin/probe_issue67_thin.rs.

## Recent Tried-and-Rejected
   the scaling choice and unpredicted by max_col_deg / MC64 cost / n
   (ROSEPETAL's MC64 is 68x ORTHREGF's, yet ROSEPETAL is the win).

A gate keyed on "scaling won't reuse MC64" would regress the fill-reduction
wins (ROSEPETAL, ex8_2_2) to save milliseconds on the overhead-only losses
(ORTHREGF, SINQUAD2, sub-ms small matrices). Not a safe win.

Bucket size for the record (`probe_compress_scaling_bucket`, 3 roots, 1006
families): 376 LdltCompress, of which 118 reuse MC64 (keep) and 258 do not
(the target bucket — heterogeneous, contains both ROSEPETAL-type wins and
ORTHREGF-type losses).

Future sessions: do NOT gate `LdltCompress` on the scaling strategy. The real
(separate, harder) lever is an orthogonal **compression cost/benefit gate**
that estimates fill reduction vs MC64+ordering cost; the current cheap proxy is
`pick_ordering_preprocess`'s low-degree fraction, and no cheap structural
feature yet separates ROSEPETAL (win) from ORTHREGF (loss). Data:
dev/research/mc64-symbolic-skip-2026-06-06.md, dev/journal/2026-06-06-04.org.
This closes the dense-column follow-up: both option (a) (inner-loop fast path)
and option (b) (scaling-aware skip) are now closed with negative results.

## Source Files
```
src/bin/bench.rs
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
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
tests/fine_grained_delay.rs
tests/fma_opt_in_roundtrip.rs
tests/growth_flag.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/maxfromm_parity.rs
tests/mc64_end_to_end.rs
tests/mc64_scaling.rs
tests/multi_rhs.rs
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
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
