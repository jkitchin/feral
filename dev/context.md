# FERAL Context (auto-generated)

Generated: 2026-06-06T14:22:21Z

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
48bb238 docs(research): issue #80 is MC64 preprocessor cost, not AMD (#80)
36d847a fix(symbolic): give LdltCompress/MC64 its own profiler stage (#80)
471b9e9 docs(session): checkpoint 2026-06-04-02 — 0.10.0 release
23ecfaf release: 0.10.0
2f8d2f1 docs(triage): investigate #78 — trajectory sensitivity, not a feral bug
```

## Test Status
```
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::is_arrow_bordered_rejects_low_nnz_share_border ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_resolves_to_amd_when_bisection_degenerates ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok

test result: ok. 322 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 0.40s

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
   `schur.rs:200`). Only `permute_pattern` from that file is still used. The
   real `feral_amd` is already a bucketed quotient-graph AMD and orders pf22 in
   **0.276s**. Implementing bucketed min-degree there would have fixed a
   function nobody calls.
2. The real ~53s is the **`LdltCompress` preprocessor's MC64 matching**
   (`mc64::compute_matching`, ~O(n^1.9)), which the per-stage profiler folded
   into the `ordering` stage timer. `preprocess=None` drops total symbolic
   from 54.5s to 1.23s.

Symptoms that revealed the false start: on real pf22 values
`feral_amd::amd_order` = 0.276s while the full symbolic = 54.5s with `ordering`
stage 53.6s; forcing `preprocess=None` collapsed it to 1.23s. With `vals=1.0`
(MC64 trivial) symbolic was only 1.5s — the value-dependence is the tell that
the cost is in MC64, not the structure-only ordering.

Future sessions: do NOT "optimize" `src/ordering/amd.rs` for performance — it
is not in the factorization path. The production AMD is `feral_amd`. For
issue #80 the lever is MC64 (gate it on large arrow-signature KKTs), not AMD.
Data: dev/research/issue-80-mc64-preprocessor-cost.md,
dev/journal/2026-06-06-01.org.

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
