# FERAL Context (auto-generated)

Generated: 2026-07-10T03:55:16Z

## Latest Session
File: dev/sessions/2026-07-10-01.md
```
# Session 2026-07-10-01

## Goal
Issue #112: the sparse LU Forrest–Tomlin update fails with
`RefactorCause::TinyPivot` at magnitude exactly `0.0` on provably nonsingular,
well-conditioned bases (100% of FT failures in discopt's refactor storms).
Requested: a pivot-searching (Bartels–Golub/Suhl–Suhl) update variant.

## Accomplished
- Research note `dev/research/issue-112-bg-update.md` + plan
  `dev/plans/issue-112-bg-update.md` (branch `claude/issue-112-0885lr`).
- **Disproof of the requested rescue design** (implemented first, then
  removed on proof): any within-bump row-interchange order's working row is
  exactly proportional to the fixed order's (`W'_k = λ_k·W_k`, λ resets to
  `−piv/vrc` per swap), so the true final pivot is `λ·t_FT` (determinant
  identity), skip patterns coincide, and FP absorption is scale-invariant —
  a pivot the fixed order computed as exactly 0.0 by summation absorption is
  unrecoverable by ANY pivot re-ordering. Verified numerically (float +
  exact-Fraction sweep replays; on the regression basis the swap path's
  final diag is `1.39e-17 = λ·2⁻³⁵`, correctly sub-ztol).
- **Shipped the actual fix: Neumaier-compensated accumulation** in the bump
  elimination's working-row scatter (always on; pooled `ft_rw_comp`;
  branch-free two-sum). On the hand-traced m=4 regression basis the plain
  sweep commits `0.0` exactly, the compensated sweep commits the true
  diagonal `2⁻³⁵` **bit-for-bit** (oracle: hand calculation verified offline
  in exact rational arithmetic; scipy's fresh LU sees only ε-noise). Classic
  Kahan verified insufficient (its `y = v − c` re-absorbs the compensation).
- **Shipped the pivot search as an opt-in always-on variant**:
  `LuParams::update_pivot_search` (default `false`), Bartels–Golub
  interchanges as physical row-content swaps (`FtOp::Swap` etas — symmetric
  `uperm` invariant, diagonal-first storage, and prior etas untouched;
  forward/transpose/spike replays; wholesale `u_above` rebuild and
  swapped-row rollback snapshots on that path; `pivot_search_swaps()`
  telemetry). Python bindings expose the keyword.
- Tests: `tests/issue112_bg_update.rs` (5 tests) — bit-exact recovered
  diagonal, backward-stable ftran/btran residuals (2.9e-11 vs bound
  3.7e-10), genuine-singular rejection + clean rollback in both modes,
  swap-mode factor validity incl. chained updates replaying Swap etas,
  default path never swaps. Full suite green (395 lib + all integration),
  clippy `--all-targets -D warnings` clean, fmt clean.
- Test-design finding (documented): any single-shot absorption reproducer
  has `σ_min(B') ⪅ δ·∏(retained diags)` — numerically singular to a
  from-scratch factorization — so the issue's "residual ≤ refactor's"
  acceptance can only be validated on discopt's captured corpus (whose
  imbalance lives in chain-built factorization state, not in B').

## Benchmark Results
```
(this container has no oracle-timed corpus; the bench harness ran its
in-tree matrices only — correctness gates, no perf partition)
```

## Git Status
```
82a81bd issue #112: compensated FT sweep + opt-in Bartels-Golub pivot search
5794f0e issue #112: research note + plan for Bartels-Golub pivot-searching update
9596472 docs: session checkpoint 2026-07-02-02 (v0.13.0 release)
36da100 release: feral v0.13.0 (#109)
9d05f75 issue #107: add OrderingMethod::External for user-supplied orderings (#108)
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

test result: ok. 395 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.89s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-07-10-01.md)

(this container has no oracle-timed corpus; the bench harness ran its
in-tree matrices only — correctness gates, no perf partition)
KKT summary: 2 matrices — inertia match 1/1, residual pass 1/1,
  worst residual 1.14e-15
Sparse solver: 2/2 — inertia vs MUMPS 2/2, residual pass 2/2,
  worst residual 1.26e-16
Dense/Sparse exit partition: 0 matrices (corpus absent) — N/A
No solver-perf-relevant kernels changed on the default path except the
compensated scatter (~4 flops per scatter add in the update only; factor
and solve paths untouched).

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
