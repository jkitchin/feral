# FERAL Context (auto-generated)

Generated: 2026-07-10T11:05:54Z

## Latest Session
File: dev/sessions/2026-07-10-03.md
```
# Session 2026-07-10-03

## Goal
Fix the six audit-confirmed correctness bugs #114–#119 (from the 2026-07-10
six-agent audit), one at a time, each as its own tested commit.

## Accomplished
All six fixed, committed, full-suite + clippy green after each. Empirical
reproduction confirmed before encoding every fix (issue-112 discipline).

- **#114** (`90cb4fe`): finiteness validation in `update_sparse` +
  `DenseLu::update`. NaN was silently committed → `Ok(NaN)` solve.
- **#118** (`373264f`): store factor `a_max`; anchor update ztol to
  `zero_pivot_tol·a_max`, not `u_max0`. Fixes spurious `TinyPivot` +
  update→refactor **livelock** on high-growth bases.
- **#115** (`ae13f27`): reworked `DenseLu::update` to an eta-based
  Bartels–Golub update. **The issue's suggested "swap rows in U + fix L" does
  not work** — a row swap of U forces a column swap of L that breaks its
  unit-lower-triangular structure (worked the algebra). Correct fix mirrors
  the sparse path: base `L` fixed, eliminations+interchanges recorded as
  `DenseFtOp` etas replayed between the L- and U-solves; `ftran_partial` now
  returns `G⁻¹L⁻¹Pa`. Partial pivoting bounds multipliers by 1 (growth `O(m)`
  on the Hessenberg), superseding the sparse path's Neumaier need. New
  `tests/lu_dense_update_bg.rs` (bump chains, slack bases, ftran+btran parity
  vs fresh factor); all 10 adversarial tests now pass (0 ignored).
- **#119** (`d7aafea`): `scaling::kr_guarded_update` applied to all 7
  Knight–Ruiz update sites. Bit-identical on healthy matrices (KR
  dense/sparse parity preserved); guards the `Inf→NaN→0` cascade on subnormal
  couplings (confirmed the unguarded loop yields `[NaN, 0.0]`).
- **#116** (`e5b6d4b`): chose the **solve-side** fix (skip iff
  `d_diag == 0.0` exactly) over the audit's separate mask — safer for the hard
  inertia rule because it changes **no** factorization (the force-accept-zero
  path already sets `d = 0.0` exactly and is the only skip outcome). The
  skip-iff-exactly-0.0 invariant was verified **empirically**: the whole
  corpus (inertia oracles + threshold suites) stays green with the gate
  change. Rook sub-floor accepts now set `needs_refinement`/`n_tiny`.
- **#117** (`4a1d163`): blocked panel bails a rook-eligible below-threshold
  1×1 pivot to the scalar (rook) path (the panel can't rook itself — its
  trailing submatrix isn't Schur-updated yet). The 1×1-no-swap gate
  (`akk ≥ α·gamma0`) means the bail only fires for near-zero columns (the
  audit's case). `remaining==1` scalar site also routed through the rook
  wrapper. Parity test: EPS-scale near-zero panel column → scalar==blocked
  byte-identical with `n_rook_rescues > 0` (impossible pre-fix). Rook-disabled
  default path byte-unchanged.

## Benchmark Results
```
Dense KKT:  inertia match 1/1, residual pass 1/1, worst 1.14e-15
Sparse KKT: inertia vs MUMPS 2/2, residual pass 2/2, worst 1.26e-16
```
```

## Git Status
```
8a980e4 Audit correctness fixes: LU update/solve + dense LDLᵀ rook (#135)
660224d issue #112: compensated FT sweep + opt-in Bartels-Golub pivot search (#113)
9596472 docs: session checkpoint 2026-07-02-02 (v0.13.0 release)
36da100 release: feral v0.13.0 (#109)
9d05f75 issue #107: add OrderingMethod::External for user-supplied orderings (#108)
```

## Test Status
```
test symbolic::tests::schur_symbolic_supernodes_cover_n ... ok
test symbolic::tests::schur_symbolic_tail_invariant_reversed_user_order ... ok
test symbolic::tests::schur_symbolic_tail_invariant_user_order ... ok
test symbolic::tests::symbolic_factorize_amf_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_auto_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
test symbolic::tests::symbolic_factorize_external_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 398 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.51s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-07-10-03.md)

Dense KKT:  inertia match 1/1, residual pass 1/1, worst 1.14e-15
Sparse KKT: inertia vs MUMPS 2/2, residual pass 2/2, worst 1.26e-16
No inertia or residual regression. (No perf-partition corpus in-container.)
Full suite: **769 passed / 0 failed**; clippy `--all-targets -D warnings`
clean; `cargo fmt --check` clean.

```

## Recent Decisions
**Decision.** Fix the exact-`0.0` `TinyPivot` failures of `SparseLu`'s
Forrest–Tomlin update (issue #112) with **Neumaier (Kahan–Babuška)
compensated accumulation** in the bump elimination's working-row scatter,
always on; and ship Bartels–Golub row interchanges as an **opt-in, always-on
variant** (`LuParams::update_pivot_search`, default `false`) rather than the
issue's requested rescue-after-failure.

**Why.** The exact-`0.0` on a nonsingular basis is summation absorption: the
fixed-order sweep grows an intermediate past `|true pivot|/ε` and one rounded
add destroys the pivot's bits. Re-selecting pivots afterwards provably cannot
recover them (any interchange order's working row is exactly proportional to
the fixed order's — see `dev/research/issue-112-bg-update.md` §UPDATE and the
tried-and-rejected entry), while the compensated sum retains them: on the
regression basis (`tests/issue112_bg_update.rs`) the committed diagonal
equals the hand-computed true value `2⁻³⁵` bit-for-bit where the plain sweep
returned `0.0` and scipy's fresh LU returns ε-noise. Cost: ~4 flops per
scatter add + one pooled length-m `f64` buffer; no allocation, no API change.
Pivot search remains valuable as a *trajectory* choice (multipliers bounded
by 1 keep U balanced over long update chains, preventing the imbalance that
makes absorption possible) — but it changes committed factors/etas wherever a
working-row entry dominates a retained diagonal, so it defaults off pending
discopt A/B on the captured corpus. New machinery: `FtOp::Swap` (physical
row-content swap preserves the symmetric `uperm` invariant, diagonal-first
storage, prior etas), `pivot_search_swaps()` telemetry, wholesale `u_above`
rebuild on swap commits.

**Contract note.** A compensated final diagonal at/below
`zero_pivot_tol·u_max0` is now trustworthy evidence of a genuinely dependent
replacement (not a summation artifact), strengthening the existing
`NeedsRefactor` semantics. No tolerances changed.

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
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
