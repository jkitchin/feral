# FERAL Context (auto-generated)

Generated: 2026-06-06T16:33:38Z

## Latest Session
File: dev/sessions/2026-06-06-02.md
```
# Session 2026-06-06-02

## Goal

Invest in a systematic super-linear (O(n²)) scaling audit across ordering,
scaling, and the numeric/symbolic prologue — including verifying the recent
MC64 heap-reuse fix (issue #80, commit 6699f09). Plan approved: full
systematic sweep + a deterministic (CI-noise-immune) MC64 regression guard.

## Accomplished

**Step 0 — MC64 Hungarian regression guard (committed 767b0d9).**
- Added `HungarianStats {heap_init_slots, augment_searches, touched_total,
  phase3_inner_iters}` to `src/scaling/hungarian.rs`, threaded `&mut stats`
  through `IndexHeap::new`/`reset` (counting `pos[]` slots zeroed) and the
  phase-3 length-2 augmentation loop. `hungarian_match_instrumented()` returns
  the counters; `hungarian_match()` is a thin wrapper (off the Dijkstra inner
  loop → negligible overhead, bit-identical matchings).
- Test `mc64_hungarian_no_quadratic_heap_realloc_regression`. Calibration
  (gen_random_sparse deg=3, n=1k..8k) showed the legitimate `touched_total` is
  itself ~quadratic on a hard matching (long augmenting paths), so a growth
  ratio on total heap work is the wrong guard. The correct, threshold-free
  guard is the exact structural invariant `heap_init_slots == n + touched_total`
  (heap allocated once + incremental resets), which fires on the realloc revert
  at any n. `phase3_inner_iters` IS linear (563→4609, 8.2× over 8× n) → a <16×
  ratio guards the O(nnz²) phase-3 suspect.
- Teeth verified: injecting a per-search `IndexHeap::new` → `heap_init` 254727
  vs expected 53727 → test FAILS as designed; reverted. Full feral lib suite
  317 passed/0 failed; clippy/fmt clean.

**Step 1 — `scaling_sweep` diagnostic binary (committed 1b2b21c).**
- `crates/feral-diagnostics/src/bin/scaling_sweep.rs`. Modes `--family` /
  `--manifest` / `--generated {spd,kkt} --sizes`. Per matrix: profiling-on
  Solver, `invalidate_symbolic_cache()` before each of K factors (forces a
  symbolic miss so the normally-cached symbolic phase is timed), per-field
  median over K, CSV with the full prologue breakdown + all 17 symbolic stages
  + `max_col_degree`/`sum_d_logd` control variates. `--scaling` pins the
  strategy. Rust = data collection only; α-fits run in Python over the CSV.
- Generators are constant-bandwidth (banded) so fill stays near-linear; on the
  banded SPD/KKT ladders all phases scale α≈1.0 (good fittable baseline).

**Step 2 — rocket_12800 localization + #80 (committed in 1b2b21c; CORRECTED
in journal 16:35).**
- rocket_12800_0000 (n=89601, nnz=332793, **max_col_degree=38401**) with MC64:
  `pb_scaling_us` = 4.3 ms (numeric), symbolic = 38.8 s of which
  `sym_ldlt_compress` = 38.3 s (98.9%). `permute_us` = 64 ms (exonerated, as the
  research note predicted).
- **Correction:** `ldlt_compress` = `compute_mc64_cache` → `compute_matching`
  → `hungarian_match`. So the 38.3 s IS the MC64 Hungarian, run in symbolic for
  the LdltCompress ordering-compression preprocessor. The cheap 4.3 ms numeric
```

## Git Status
```
1b2b21c feat(diagnostics): scaling_sweep binary; verify #80 + find ldlt_compress O(n^2)
767b0d9 test(scaling): deterministic MC64 Hungarian O(n^2) regression guard (#80)
bc8496a docs(session): addendum — MC64 heap fix + dead-code removal (#80)
10a3a1a refactor(ordering): remove dead amd_order, keep permute_pattern (#80)
6699f09 perf(scaling): reuse MC64 Hungarian heap across columns (#80)
```

## Test Status
```
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
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

test result: ok. 317 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.47s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-06-02.md)


Full corpus `bench --release` was **not** re-run this session. Rationale: no
numeric hot-path code changed — Step 0 adds counters that live off the Dijkstra
inner loop (in `IndexHeap::new`/`reset` and phase-3), and the new `scaling_sweep`
binary is in the non-default `feral-diagnostics` crate. There is no performance
delta to track. (CI on the most recent code commit covers correctness.)

Targeted scaling_sweep numbers instead:
banded SPD  (n 300→10000, 33×): loop 40× prologue 33× scaling 40× sym 32×  (all α≈1.0)
banded KKT  (n 400→13333, 33×): loop 28× prologue 32× scaling 29× sym 30×  (all α≈1.0)
rocket_12800_0000 (n=89601, max_col_deg=38401, MC64):
  pb_scaling=4.3ms  sym_ldlt_compress=38.33s (98.9% of 38.8s symbolic)  loop=399ms

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
