# FERAL Context (auto-generated)

Generated: 2026-07-11T16:23:37Z

## Latest Session
File: dev/sessions/2026-07-11-01.md
```
# Session 2026-07-11-01

## Goal

Review all open branches for up-to-dateness (user request). Two follow-on
tasks emerged: (1) merge dev/ records stranded on closed-issue branches and
delete the branches; (2) diagnose and fix a locally-failing test surfaced by
the first task, and close the fixture-gating blind spot that hid it from CI.

## Benchmark comparison to previous session

**No regression.** This session changed **no solver code** — only a test
assertion, CI config, `.gitignore`, committed fixtures, and docs — so the
bench reflects main's unchanged code (`dfae3c5` → `7c56427`). Inertia gate
holds at **100.0%** on both paths. The last full-corpus run was 2026-05-21-04
(the intervening container sessions had no corpus); vs that baseline the
corpus grew by 1–2 matrices, inertia stayed 100%, residual-pass stayed 99.8%
on both paths. Any May→July residual drift spans dozens of intervening solver
PRs and is not attributable to this session.

## Accomplished

- **Branch review + cleanup (PR #140, `805b85d`).** Verified branch-by-branch
  that all *code* on every open branch had already landed on main (issue-112
  tip byte-identical; #107 via PR #108; #99 packed-kernel via PR #103; four
  branches 0 ahead). Three branches held dev/ records existing nowhere else;
  merged them (issue-110's 545-line research note + 3 diagnostics study bins;
  issue-102's stall-verify note + a journal entry; issue-99's container
  checkpoint/journal renumbered `2026-07-01-03`→`-07` to avoid colliding with
  main's different session -03). CI green, bins compile against main. Deleted
  all nine stale branches; fast-forwarded local main `9596472`→`dfae3c5`.

- **Bisected the issue-65 test regression.** `cargo test` failed locally on
  `issue65_mc64_fallback::explicit_infnorm_is_respected_no_fallback`
  (expected `(789,670,116)`, got `(789,785,1)`) while CI was green on the
  same SHA. Cause: the fixtures are gitignored + non-regenerable on CI, so
  the guard SKIP-passed everywhere but this Mac. Bisect (fixtures unchanged
  since Jun 3): `9596472` ok → `660224d`/#113 ok → **`8a980e4`/#135 FAILED**.
  #135's rook fixes (#116 solve skips only exactly-zeroed pivots; #117 blocked
  panel defers rook-eligible 1×1s to scalar) legitimately rescued 115 of 116
  formerly force-zeroed pivots — moving the InfNorm signature *closer* to the
  oracle `(789,786,0)`. An improvement that invalidated a pinned constant, not
  a correctness bug.

- **Fixed it + closed the blind spot (PR #141, `7c56427`).**
  - Test now asserts the contract (`mc64_scaling_fallback_count()==0`,
    `inertia.zero>0`, components sum to n=1575) instead of a pivot-policy
    signature that has now drifted once. Human-approved.
  - Committed the two generated fixtures (~280 KB) via `.gitignore` negation
    (`tests/data/large/*` + `!sawpath`/`!twirism1`); large fetchable
```

## Git Status
```
c05eb77 release: feral v0.14.0 (#145)
8a6992e perf(symbolic): split pipeline so ordering-race losers skip the tail (#127) (#144)
683933a docs(changelog): mark #125 and #128 as partial in Unreleased (#143)
a0bf2db docs: session checkpoint 2026-07-11-01 (branch cleanup + issue-65 fix) (#142)
7c56427 test(issue65): semantic assertion + commit fixtures, surface CI skips (#141)
```

## Test Status
```
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 409 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.57s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-07-11-01.md)


cargo run --bin bench --release  (full local corpus, aarch64 M-series)

--- Dense solver validation ---
  Inertia match:  154429/154482 (100.0%)   [53 consensus-excluded]
  Residual pass:  154149/154482 (99.8%)
  Worst residual: 2.46e-1 (POLAK6_0021)     # known residual-hard case

--- Sparse solver validation ---
  Inertia match vs MUMPS: 154531/154590 (100.0%)
  Residual pass:          154258/154590 (99.8%)
  Worst residual:         2.94e-4 (ERRINBAR_0824)

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
  small-frontal (<200)  count 147982  p90 1.72  <= 2.0  PASS
  medium       (<500)   count 152145  p90 2.13  <= 3.0  PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
  small-frontal (<200)  count 153455  p90 1.61  <= 2.0  PASS
  medium       (<500)   count 153560  p90 1.61  <= 3.0  PASS

Worst factor-ratio vs MUMPS (dense): KIRBY2_0007 9.13×, HAHN1_0078 8.58×,
CRESC132_0000 7.30× (n=5314) — the persistent small-n dense-front tail.

```

## Recent Decisions
left on the critical path is the root/near-root fronts' O(nrow²), behind the
root's O(nrow³) dense factor that intra-front parallelism (Lever 1.1) already
targets — so column-partitioned parallel assembly would chase <1–3% of the
factor. #125 already captured the tractable, bit-exact assembly win
(`build_row_indices`). Not built. Evidence:
`dev/research/issue-131-gapb-assembly-measure-2026-07-10.md`.

## 2026-07-11 — issue-65 guard: semantic assertion + committed fixtures (fixture-gating blind spot)

Two decisions from the post-#135 breakage of
`tests/issue65_mc64_fallback.rs::explicit_infnorm_is_respected_no_fallback`:

1. **The explicit-InfNorm test asserts the contract, not a pinned inertia.**
   The pinned `(789,670,116)` was InfNorm's misfactoring signature under the
   pre-#135 pivot policy; #135's rook fixes (#116/#117) legitimately changed
   it to `(789,785,1)` (closer to the oracle `(789,786,0)`). The signature is
   a pivot-policy artifact and will drift again; the invariant the test
   guards is "explicit strategy respected": `mc64_scaling_fallback_count()
   == 0`, `inertia.zero > 0` (zeros kept, not rescued), components sum to n.
   Human-approved (session 2026-07-11).

2. **The two issue-65 fixtures are committed, not gitignored.** They are
   small (~280 KB total) *generated* matrices that CI can never fetch or
   regenerate (`regen_issue65_kkts.sh` needs pounce + a local .nl set), so
   the SKIP-when-absent design made the guard local-only: PR #135 shipped
   "full suite green" from a fixture-less container while breaking it.
   `.gitignore` now uses `tests/data/large/*` with explicit negations;
   large fetchable SuiteSparse matrices stay ignored. CI additionally
   surfaces every remaining "SKIP:" line in the job summary
   (`.github/workflows/ci.yml`) so skipped guards are visible, not silent.

## Recent Tried-and-Rejected
sweep replays across four hand constructions (journal 2026-07-10-01,
research note §UPDATE).

Also rejected en route: classic **Kahan** compensation for the sweep
accumulator (its `y = v − c` pre-subtraction re-absorbs the compensation
into the next 2²⁰-scale addend — computed `0.0` again; verified
numerically); the **Neumaier** two-sum variant works and shipped. And three
regression-matrix constructions whose base or replacement was numerically
singular for every path (±1 cascade to 2³⁴: `σ_min(B') = 1.5e-16`; diag-4
cascade: rescue-true `4.5e-13 <` ztol; spike-poison m=6: fresh LU burns the
4e6 spike entry and deflates its tail pivot to 0) — any single-shot
absorption reproducer necessarily has `σ_min(B') ⪅ δ·∏retained`, so the
"fresh factor succeeds" oracle is unsatisfiable without a multi-update
imbalance history.

**Shipped instead.** Always-on Neumaier-compensated scatter (recovers the
true pivot bit-for-bit on the regression basis) + `update_pivot_search` as an
always-on opt-in trajectory variant (bounded multipliers across chains),
default false. See `dev/research/issue-112-bg-update.md` §UPDATE and
`dev/decisions.md` 2026-07-10.

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
src/lu/mod.rs
src/lu/scaling.rs
src/lu/sparse_factor.rs
src/lu/sparse_matrix.rs
src/lu/sparse_solve.rs
src/lu/sparse_symbolic.rs
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
tests/lu_dense_update_bg.rs
tests/lu_dense.rs
tests/lu_ft_widebump.rs
tests/lu_scaling.rs
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
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
