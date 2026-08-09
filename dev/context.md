# FERAL Context (auto-generated)

Generated: 2026-08-09T01:16:25Z

## Latest Session
File: dev/sessions/2026-07-11-02.md
```
# Session 2026-07-11-02

## Goal

Continuation of 2026-07-11-01. Review the open issues for release-worthiness,
do the pre-release housekeeping, implement the one issue worth doing first
(#127), then cut and publish release 0.14.0.

## Benchmark comparison to previous session

**No regression.** The only code change this session (#127) is a bit-identical
refactor; the release commit changes no code. Full-corpus bench after #127 is
identical to the 2026-07-11-01 baseline on both inertia (100%) and residual
counts — see Benchmark Results.

## Accomplished

- **Issue triage → recommendation.** Reviewed all five open issues (#125,
  #127, #128, #131, #134). Conclusion: all are deferred perf/robustness work,
  none a correctness blocker; the strongest release argument is the ten
  already-merged-but-unreleased correctness fixes (#114–#123). Recommended
  releasing now and doing #127 first (highest value / lowest risk for the IPM
  host workload). User agreed.

- **Housekeeping (PR #143, merged).** The Unreleased CHANGELOG credited #125
  and #128 as done though only a slice of each had landed. Marked both as
  partial and posted landed-vs-remaining status comments (with commit refs) on
  issues #125 and #128; verified both new public surfaces (`solve_sparse_cb`,
  `LuParams::update_pivot_search` + its Python keyword) are documented.

- **Issue #127 (PR #144, merged; issue closed).** Split
  `symbolic_factorize_with_method` into a cheap *prefix* (ordering → column
  counts → `factor_nnz`) and a *finish* (supernodes, small-leaf, peak-contrib,
  static rows, struct assembly). Both race dispatchers (preprocess-`Auto`,
  `AutoRace`) now race prefixes and finish only the winner — previously each
  candidate ran the full pipeline (up to ~8× symbolic when both races nest).
  Chose prefix/finish over "estimate then recompute winner" (the latter would
  double the LdltCompress MC64 matching when compression wins). Winner
  selection bit-identical; produced `SymbolicFactorization` unchanged;
  Schur-tail variant untouched; profiler "one run" behaviour preserved. New
  self-consistency parity tests (`tests/issue127_pipeline_split.rs`) + a
  thread-local `#[cfg(test)]` FINISH_RUNS counter proving losers never reach
  the tail. Research/plan: `dev/research/issue-127-symbolic-pipeline-split.md`,
  `dev/plans/issue-127-pipeline-split.md`.

- **Release 0.14.0 (PR #145, merged; published).** Bumped all six version
  strings 0.13.0 → 0.14.0 (root + python `Cargo.toml`/`Cargo.lock`,
  `pyproject.toml`), cut the CHANGELOG `[0.14.0] - 2026-07-11` section, tagged
  the `v0.14.0` GitHub Release. Both publish workflows succeeded:
  `release.yml` → crates.io (cargo publish in dependency order),
```

## Git Status
```
6589570 docs: session checkpoint 2026-07-11-02 (issue triage, #127, release 0.14.0) (#146)
c05eb77 release: feral v0.14.0 (#145)
8a6992e perf(symbolic): split pipeline so ordering-race losers skip the tail (#127) (#144)
683933a docs(changelog): mark #125 and #128 as partial in Unreleased (#143)
a0bf2db docs: session checkpoint 2026-07-11-01 (branch cleanup + issue-65 fix) (#142)
```

## Test Status
```
test symbolic::tests::schur_symbolic_tail_invariant_reversed_user_order ... ok
test symbolic::tests::schur_symbolic_tail_invariant_user_order ... ok
test symbolic::tests::symbolic_factorize_amf_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_auto_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
test symbolic::tests::symbolic_factorize_external_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 407 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.93s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-07-11-02.md)


cargo run --bin bench --release  (full local corpus, aarch64 M-series; after #127)

--- Dense solver validation ---
  Inertia match:  154429/154482 (100.0%)
  Residual pass:  154149/154482 (99.8%)   worst 2.46e-1 (POLAK6_0021)

--- Sparse solver validation ---
  Inertia match vs MUMPS: 154531/154590 (100.0%)
  Residual pass:          154258/154590 (99.8%)   worst 2.94e-4 (ERRINBAR_0824)

--- exit partitions ---
  Dense  small-frontal p90 1.67 (<=2.0) PASS ; medium p90 2.08 (<=3.0) PASS
  Sparse small-frontal p90 1.56 (<=2.0) PASS ; medium p90 1.57 (<=3.0) PASS

Inertia AND residual counts are identical to the pre-#127 baseline
(2026-07-11-01), confirming winner selection is unchanged (same factors).

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
tests/column_renumbering.rs
tests/column_renumbering_parity.rs
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
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/lu_adversarial_inputs.rs
tests/lu_dense.rs
tests/lu_dense_update_bg.rs
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
tests/rook_rescue.rs
tests/rook_rescue_kkt.rs
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
