# FERAL Context (auto-generated)

Generated: 2026-06-06T17:40:17Z

## Latest Session
File: dev/sessions/2026-06-06-03.md
```
# Session 2026-06-06-03

## Goal

Take on the MC64 dense-column fast path (scaling audit §8.1 option (a), the
one real super-linear lead remaining from session -02). Mid-session, fix a
reported CI failure.

## Accomplished

**MC64 dense-column fast path — investigated, definitively closed with a
negative result.**

- Re-grounded the cost in the production kernel: `O(searches × dense_deg)` —
  the degree-38401 coupling column is matched in the main loop and rescanned
  whenever its matched row is popped (hungarian.rs:585).
- Traced the LdltCompress/scaling coupling (subagent). Key facts:
  `pick_ordering_preprocess` (mod.rs:485) never inspects max_col_degree; the
  symbolic `Mc64Cache` is reused for `Mc64Symmetric` scaling purely as an
  optimization (not load-bearing for inertia/residual — numeric recomputes the
  identical matching if absent). BUT `pick_scaling_strategy` (mod.rs:653)
  routes rocket's dense column to MC64, so the symbolic-side skip (audit option
  (b)) saves nothing for rocket. The only correctness-safe lever for rocket is
  a behavior-preserving speedup of the matching itself.
- **Implemented** a behavior-preserving column lower-bound skip (`u` is
  monotone non-increasing ⇒ `lb[j]=min_k(cost[k]-u_init[i])` is a permanent
  lower bound; `vj+lb[j2]≥csp` ⇒ skip the scan bit-identically). All 7
  Hungarian tests + full 317-test lib suite green (behavior preserved).
- **Measured: it fires 0 times** on rocket_12800_0000 (`skips=0`,
  `edges_saved=0`, edge_scans 3.71e8 and wall 3958 ms both unchanged).
- **Proved impossibility:** for the matched column of a popped row `q0`,
  complementary slackness + dual feasibility make the tightest column bound
  `lb_tight = cost[jperm[j2]]-u[q0]`, and `vj+lb_tight = dq0`; the skip fires
  iff `dq0≥csp`, but `q0` was popped only because `dq0<csp`. So a column-level
  reduced-cost bound can NEVER prune, at any tightness.
- **SPRAL confirms** (spral-expert read scaling.f90:938-1171): the inner scan
  walks the full matched column every settle, no range cut, no dense-column
  special case, `dualv` computed once at the end. SPRAL has the identical
  `O(searches × dense_deg)` cost; feral's port is faithful. No trick to adopt.
- Reverted the lb-skip (dead weight). Wrote
  `dev/research/mc64-dense-column-2026-06-06.md` (mechanism, impossibility
  proof, SPRAL confirmation, recommendation).

**CI hotfix (commit 98f85e0, pushed).** User reported "gh actions failed". CI
red since 1b2b21c: the `diagnostics lint + test` job runs
`cargo clippy -p feral-diagnostics --all-targets -- -D warnings`, which the
local pre-commit clippy hook does not cover (diagnostics crate is outside the
root build set, ci.yml:34-39). Two diagonal-fill loops in scaling_sweep.rs
tripped `needless_range_loop`. Fixed to `iter().enumerate()`; clippy clean,
`cargo test -p feral-diagnostics` green.
```

## Git Status
```
98f85e0 fix(diagnostics): clippy needless_range_loop in scaling_sweep generators
2ef38d5 docs(research): scaling audit report — MC64 is the sole super-linear phase
402630e feat(scaling): localize MC64 dense-column cost; resolve debug/release (#80)
503af96 docs(session): checkpoint 2026-06-06-02 — scaling audit Steps 0-2 (#80)
1b2b21c feat(diagnostics): scaling_sweep binary; verify #80 + find ldlt_compress O(n^2)
```

## Test Status
```
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
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
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 317 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.45s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-06-03.md)


`bench --release` NOT re-run: no numeric hot-path code changed. The lb-skip
experiment was reverted; the only landed Rust change is a clippy fix in the
non-default `feral-diagnostics` crate (a diagnostic binary, not the solver).
No performance delta to track.

Targeted measurement (the session's actual result):
rocket_12800_0000  n=89601  max_col_degree=38401  (MC64, release)
  baseline:     edge_scans=3.71e8  wall=3.69–3.96 s
  with lb-skip: edge_scans=3.71e8  inner_scan_skips=0  edges_saved=0  wall=3.96 s
  → behavior-preserving column-bound skip is inert (proven impossible)

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
3. **SPRAL confirms the cost is inherent.** `ref/spral/src/scaling.f90::
   hungarian_match` (938-1171) walks the full matched column every settle with
   only per-entry filters (no range cut, no dense-column special case),
   computes `dualv` once at the end, and its only dense-aware logic is the
   greedy-init claim guard (line 857) — which feral already mirrors. SPRAL has
   the identical O(searches × dense_deg) cost; feral's port is faithful.

Symptoms: the lb-skip compiled, kept all 7 Hungarian unit tests and the full
317-test lib suite green (behavior preserved), but the counters showed it was
inert. Reverted (dead weight: never fires, adds an O(nnz) pass + a branch per
pop).

Future sessions: do NOT attempt to prune the MC64 inner column scan with any
per-column reduced-cost bound — it is provably impossible. The dense-column
MC64 cost is inherent to the sparse shortest-augmenting-path algorithm and
matches the SPRAL reference. The only remaining lever is to AVOID MC64 scaling
on single-dense-column KKTs (a scaling-policy change that alters the scaling
vector → needs a corpus inertia/residual study + human approval per the
constraints). Data: dev/research/mc64-dense-column-2026-06-06.md,
dev/journal/2026-06-06-03.org.

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
