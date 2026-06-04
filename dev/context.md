# FERAL Context (auto-generated)

Generated: 2026-06-04T11:56:25Z

## Latest Session
File: dev/sessions/2026-06-04-02.md
```
# Session 2026-06-04-02

## Goal
Check for outstanding issues and, if clear, cut the next release.

## Accomplished

### Pre-flight — issue tracker clean
- `gh issue list --state open` → **0 open issues**. The three most-recently
  triaged are all closed: #76 (not-a-bug, near-singular LP KKT with no
  well-defined inertia, excluded by consensus framework), #77 (not
  reproducible), #78 (not-a-bug, IPM trajectory sensitivity — see session
  2026-06-04-01 journal). 20 commits since v0.9.0.

### Release 0.10.0 (minor)
- **Version decision: 0.10.0, not 0.9.1.** Two of the five `[Unreleased]`
  CHANGELOG entries are `### Changed` (deliberate default-behavior changes),
  not just fixes. The headline #73/#67 changes default `Auto` ordering for
  every user (AMF over MetisND at every size; `AMF_BAND_MAX` removed) — fails
  the patch test. Repo precedent: no patch release has ever existed (every tag
  is `v0.x.0`); 0.6.0 shipped the analogous #50 `Auto`-dispatcher change as a
  minor. User confirmed.
- **Version bumps** (functional change confined to root `feral` package;
  ordering crates unchanged at 0.2.0):
  - `Cargo.toml`           `feral`        `0.9.0 → 0.10.0`
  - `python/pyproject.toml` `feral-solver` `0.9.0 → 0.10.0`
  - `Cargo.lock`           `feral`        `0.9.0 → 0.10.0`
  - `CHANGELOG.md`         `[Unreleased] → [0.10.0] - 2026-06-04` + fresh
    empty `[Unreleased]`
- **Commit** `23ecfaf` `release: 0.10.0` (pre-commit fmt/clippy skipped — no
  `.rs` staged). **Annotated tag** `v0.10.0` (matches prior `feral vX.Y.Z`
  style).
- **Published.** Pushed `main` + tag; created GitHub Release v0.10.0
  (`--notes-file`). Release event triggered the `Python wheels` workflow whose
  `publish` job (trusted publishing) pushes `feral-solver 0.10.0` to PyPI.
  `Release` workflow: success. `Python wheels`: building/publishing at
  checkpoint time — verify it reached PyPI.

### CI evidence
Code was green at `d144ade` (CI + Pages success). `2f8d2f1` (docs triage) and
`23ecfaf` (version strings) on top are non-source, so the tested-code bar holds
without a re-run.

## Benchmark Results
Not re-run this session. No solver source changed (version strings + CHANGELOG
+ release artifacts only), so the bench would re-measure unchanged code. Last
recorded run is session **2026-06-03-06**: both Phase 2.8.1 partition gates
**PASS**; worst factor-ratios are tiny-n fixed-cost-dominated matrices
(KIRBY2 n=458, CRESC132 n=5314), consistent with prior sessions. Those numbers
stand for 0.10.0.
```

## Git Status
```
23ecfaf release: 0.10.0
2f8d2f1 docs(triage): investigate #78 — trajectory sensitivity, not a feral bug
d144ade docs(triage): close #76 (not-a-bug), investigate #77 (not reproducible)
51f0472 feat(ordering): route Auto to AMF at every size, drop AMF_BAND_MAX (closes #73) (#75)
caf4120 docs(readme): document the feral-diagnostics crate invocation (#74)
```

## Test Status
```
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_resolves_to_amd_when_bisection_degenerates ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok

test result: ok. 322 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 0.41s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-04-02.md)

Not re-run this session. No solver source changed (version strings + CHANGELOG
+ release artifacts only), so the bench would re-measure unchanged code. Last
recorded run is session **2026-06-03-06**: both Phase 2.8.1 partition gates
**PASS**; worst factor-ratios are tiny-n fixed-cost-dominated matrices
(KIRBY2 n=458, CRESC132 n=5314), consistent with prior sessions. Those numbers
stand for 0.10.0.

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

Rejected by the real factor+solve A/B (`probe_issue67_thin --reps 1`). The
guard's predicate is **anti-correlated with real speed on nql180**: MetisND has
2% *smaller* fill (fill_r 0.98) yet AMF is **2.05× faster** on the actual
factor+solve (fac_amf 1903 ms vs fac_met 3949 ms). The fill guard would have
read "MetisND fill is smaller → keep MetisND" and **forfeited a 2× speedup**.
nnz_L and the Σ ncol·nrow² flop_proxy do not predict factor+solve wall-time at
this scale (the numeric phase's cache/critical-path behavior dominates), so any
routing guard keyed on symbolic fill makes the wrong call exactly where it
matters — and adds a per-solve symbolic-race cost to do it.

Symptoms / evidence of the failure: on real factor+solve AMF wins ALL 5
measured n>100k families (dtoc2 2.49×, pinene 1.18×, cont5_1_l 2.75×, nql180
2.05×, YATP1NE 2.13×), including the matrix the guard would have demoted.
Superseded by the **unconditional** AMF extension (drop `AMF_BAND_MAX`; route
every would-be-MetisND decision to AMF), recorded in `dev/decisions.md`
(2026-06-03, issue #73). Future sessions: do NOT reintroduce a fill / nnz_L /
flop_proxy guard on the n>100k AMF reroute — nql180 is the standing
counterexample. Data: dev/research/issue-73-n100k-thin-regime.md (Finding 3),
dev/journal/2026-06-03-06.org (:issue-73:ab:factor-solve:surprise:).

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
