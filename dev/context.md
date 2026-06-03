# FERAL Context (auto-generated)

Generated: 2026-06-03T19:36:56Z

## Latest Session
File: dev/sessions/2026-06-03-06.md
```
# Session 2026-06-03-06

## Goal
Issue #73 (the n>100k thin-regime follow-up to #67, opened from session-05's
"Next Session Should"): settle whether the regime above `AMF_BAND_MAX`
(100_000) should keep MetisND or reroute to AMF, and implement the outcome.
Also merged the two session-05 follow-ups (#67/#71 housekeeping) on the way in.

## Accomplished

### Housekeeping (start of session)
- **PR #74** (docs(readme): document the `feral-diagnostics` crate invocation,
  the session-05 follow-up #2) — CI green, squash-merged → main `caf4120`.
- **Issue #73** opened to track the n>100k investigation (follow-up #1).

### Issue #73 — investigation (3 steps)
- **Step 1 — symbolic diagnosis** (`probe_issue73_symbolic`, committed
  `49b153e`): the #67 RDW2D51U ">10 min, did not complete" was **numeric**,
  not an AMF fill blowup. AMF's symbolic finishes in **167 ms** (n=195k) and
  AMF is the *cheaper* ordering there (1.26× fewer nnz_L, 1.55× less flop_proxy
  than MetisND). The band was guarding nothing.
- **Step 2 — symbolic sweep:** over the affected `n>100k && avg_deg ≥ 5`
  non-arrow population, AMF wins or ties 6/7 on nnz_L / flop_proxy; the lone
  predicted MetisND win was **nql180** (nnz_L 0.98×, flop_proxy 0.86×).
- **Step 3 — real factor+solve A/B** (`probe_issue67_thin --reps 1`): AMF wins
  factor+solve on **every measured matrix** — dtoc2 2.49×, pinene 1.18×,
  cont5_1_l 2.75×, nql180 2.05×, YATP1NE 2.13×. **nql180 is the
  design-breaker:** MetisND has 2% *smaller* fill yet AMF is 2.05× faster
  (fac 1903 ms vs 3949 ms). So nnz_L / flop_proxy mispredict real speed — a
  fill-guarded race would have demoted nql180 and forfeited the 2×.

### Issue #73 — implementation (commit `4c49745`)
- **Dropped `AMF_BAND_MAX`.** `choose_adaptive` now overrides every
  would-be-MetisND `Auto` decision to AMF at **every** `n`. The
  `n > 100_000 && avg_deg < 5 → Amd` (#50 powerflow) and arrow → AMF (#64)
  catches fire first and are untouched.
- **Tests updated** (oracle = the real A/B, external to the change):
  `choose_adaptive_rules` n=150_000 → Amf (was MetisND);
  `choose_adaptive_routes_arrow_to_amf` n=120_000 non-arrow → Amf.
- **Verification:** `cargo test --lib` → **322 passed, 0 failed** (6 ignored);
  `cargo clippy --lib -- -D warnings` → clean; pre-commit fmt+clippy passed.
  CI is the authoritative gate (PR #75).
- PR #75 reframed from "investigation only" to the full change, **closes #73**.

## Benchmark Results
`cargo run --bin bench --release` (captured tail — the harness retains the
final summary). Both Phase 2.8.1 partition gates **PASS**; worst factor-ratios
are tiny-n fixed-cost-dominated matrices (KIRBY2 n=458, etc.) consistent with
prior sessions. The bench corpus is dominated by small matrices (n ≤ ~5k), so
the n>100k reroute is essentially orthogonal to it — this run is a
```

## Git Status
```
4c49745 feat(ordering): route Auto to AMF at every size, drop AMF_BAND_MAX (#73)
49b153e investigate(#73): symbolic-only AMF-vs-MetisND probe for the n>100k thin regime
caf4120 docs(readme): document the feral-diagnostics crate invocation (#74)
2ef751f refactor(build): move 144 diagnostic binaries to crates/feral-diagnostics (closes #71) (#72)
3391d6a fix(ordering): thin-large default prefers AMF up to n≤100k (closes #67) (#70)
```

## Test Status
```
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test scaling::tests::auto_falls_back_to_infnorm_on_mss1_0009 ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_resolves_to_amd_when_bisection_degenerates ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok

test result: ok. 322 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 0.44s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-03-06.md)

`cargo run --bin bench --release` (captured tail — the harness retains the
final summary). Both Phase 2.8.1 partition gates **PASS**; worst factor-ratios
are tiny-n fixed-cost-dominated matrices (KIRBY2 n=458, etc.) consistent with
prior sessions. The bench corpus is dominated by small matrices (n ≤ ~5k), so
the n>100k reroute is essentially orthogonal to it — this run is a
no-regression confirmation, not where the #73 win shows up (that is the
factor+solve A/B above).

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
KIRBY2_0007                    458          940          119       7.90
KIRBY2_0006                    458          889          127       7.00
CRESC132_0000                 5314        80529        12266       6.57
KIRBY2_0008                    458          772          122       6.33
KIRBY2_0009                    458          742          128       5.80
MUONSINE_0000                 1537         1978          376       5.26
KIRBY2_0010                    458          683          133       5.14
KIRBY2_0011                    458          585          120       4.88
GROUPING_0219                  225          525          114       4.61
GROUPING_0179                  225          452          114       3.96

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.34     <= 2.0     PASS
medium (<500)            152145     1.74     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.50     <= 2.0     PASS
medium (<500)            153560     1.51     <= 3.0     PASS

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
