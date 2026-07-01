# FERAL Context (auto-generated)

Generated: 2026-07-01T01:38:26Z

## Latest Session
File: dev/sessions/2026-07-01-02.md
```
# Session 2026-07-01-02

## Goal

Issue #95: enrich the LU rank-1 `update()` failure signal so a caller can tell an
ill-conditioning failure (refine and retry) from a bookkeeping-budget trip (just
refactor), and close the `should_refactor()` sparse/dense parity gap. discopt#364
needs the *why* behind a `NeedsRefactor`, and discopt forces the dense LU for
small bases (`m ≤ 256`).

## Accomplished

Chose the issue's **preferred additive (non-breaking)** design: keep the
payload-free `Err(NeedsRefactor)`, record cause + magnitude alongside.

- New public enum `RefactorCause { Growth, UpdateBudget, TinyPivot, Singular }`
  (`src/lu/mod.rs`, re-exported from the crate root via `src/lib.rs`).
- `last_refactor() -> Option<(RefactorCause, f64)>` on both `SparseLu` and
  `DenseLu`. Set on **every** `NeedsRefactor` return path (update-count budget,
  singular/empty-support, growth, tiny/zero/non-finite pivot). `None` after a
  fresh factor/refactor; untouched by a successful update (read only after `Err`).
- `growth()` getter on both (exposes the element-growth high-water monitor).
- `should_refactor_growth()` on both: growth-aware recommendation firing once
  `growth >= sqrt(max_growth)` (finite, > 1) — the geometric midpoint in log
  space — so a caller can pre-empt a growth trip.
- `should_refactor()` parity on `DenseLu` (cost-based: `updates_since_refactor >= m`;
  dense update is O(m²), fresh factor O(m³)), mirroring the sparse cost-based
  `should_refactor()`. `updates_since_refactor()` already existed on both.
- Docs: `src/error.rs` `NeedsRefactor` variant now points at the accessor;
  `CHANGELOG.md` Unreleased; design note `dev/research/refactor-signal-2026-07-01.md`.

**Magnitude semantics** — Growth: the growth ratio that tripped (> max_growth);
UpdateBudget: the update count that hit the cap (= max_updates); TinyPivot:
|offending pivot| (≤ zero_pivot_tol·u_max0); Singular: 0.0.

**Dense/sparse asymmetry (inherent, documented):** the dense path has no distinct
`Singular` branch — a linearly dependent replacement drives the final `U` diagonal
to ~0 and reports `TinyPivot`. Only the sparse path can cheaply detect the
empty-support case (`h_rank < r_rank`) *before* eliminating, so `Singular` is
sparse-only.

### Evidence

- 9 new unit tests (5 sparse, 4 dense): each cause reached via a deterministically
  constructed input, plus the two recommendation getters. Assertions are
  behavioral (which cause) and self-consistent (the magnitude relation the path
  itself guarantees) — no external numerical oracle required.
  - Note: the naive identity-basis update `update(0, e0/[1,1])` trips a
    `TinyPivot` (after the cyclic shift `u[0,0]` = the identity's off-diagonal 0),
    *not* a clean commit — so the budget/recommendation tests use the tridiagonal
```

## Git Status
```
daa058f issue #95: richer update() instability signal + refactor recommendations
f9398fd issue #94: one-norm condition estimate for the unsymmetric LU factor (#98)
f114b5d issue #93: expose the LU element-growth factor via public getters (#96)
f037285 release: feral v0.11.3
a5c789f issue #89: FT update u_above reindex O(m³)→O(m²) + true per-update cost counter (#90)
```

## Test Status
```
test symbolic::tests::schur_symbolic_supernodes_cover_n ... ok
test symbolic::tests::schur_symbolic_tail_invariant_reversed_user_order ... ok
test symbolic::tests::schur_symbolic_tail_invariant_user_order ... ok
test symbolic::tests::symbolic_factorize_amf_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_auto_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
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

test result: ok. 391 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.40s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-07-01-02.md)


Residual pass: 2/2 (100.0%)
Worst residual: 1.26e-16 (densecol_kkt_300_0000)
Dense failure analysis: no failures
Sparse failure analysis: no failures
Dense ∩ Sparse failure overlap: 0 / 0 / 0
(no oracle timings loaded in this environment; perf partitions N/A)

Purely additive API + metadata on the error path (a single `Option` write only on
a `NeedsRefactor` return); no factorization or solve arithmetic changed.

```

## Recent Decisions
accessor, not a breaking error-variant change. `SparseLu::update`/`DenseLu::update`
keep returning the payload-free `Err(FeralError::NeedsRefactor)`; the cause +
magnitude are recorded on `self` and read back via
`last_refactor() -> Option<(RefactorCause, f64)>`. New public enum
`RefactorCause { Growth, UpdateBudget, TinyPivot, Singular }`.

**Why.** discopt#364 needs to distinguish an ill-conditioning failure
(Growth/TinyPivot/Singular → refine-and-retry) from a mere update-count budget
trip (UpdateBudget → refactor). The additive route (the issue's stated
preference) leaves every existing caller compiling unchanged.

**Magnitude semantics.** Growth = growth ratio that tripped; UpdateBudget =
update count that hit the cap (= max_updates); TinyPivot = |offending pivot|;
Singular = 0.0. `last_refactor()` is `None` after factor/refactor, untouched by a
successful update.

**Dense/sparse asymmetry (accepted, not a gap).** The dense path has no distinct
`Singular` cause — a dependent replacement drives the final `U` diagonal to ~0 and
reports `TinyPivot`. Only the sparse path detects the empty-support case
(`h_rank < r_rank`) before eliminating, so `Singular` is sparse-only.

**Refactor recommendations.** `should_refactor_growth()` (both types) fires at
`growth >= sqrt(max_growth)` — the log-space midpoint between the floor 1 and the
cap — to pre-empt a growth trip. Dense `should_refactor()` (cost-based parity) =
`updates_since_refactor() >= m` (O(m²) update vs O(m³) factor), the dense analogue
of sparse's `update_work_total >= factor_nnz()`.

**Evidence.** +9 unit tests (each cause + both recommendation getters), 381 lib
tests green, fmt/clippy clean, no numerical change (bench: no failures). Design
note: `dev/research/refactor-signal-2026-07-01.md`.

## Recent Tried-and-Rejected
more work.** Step 2's premise (dense-spike bump, contiguous SAXPY) does not hold
for this workload; implementing it would **regress** casctanks, not speed it up.

This also explains Step 1's 15.8×: the old O(bump²) **scan** probed ~731²/2 ≈ 267k
cells per update to find the ~24 that needed work. Removing the scan (Step 1) was
the correct and sufficient fix for the sparse-wide-bump regime; the residual
elimination work is already near-minimal.

**Not generally rejected.** A dense path could still help a genuinely *dense*-spike
basis (the journal's 2026-06-08 tridiagonal/`L⁻¹`-dense worst case). If such a
workload appears, Step 2 should be width-AND-density gated (dense path only when
block density exceeds a threshold), never width-only. For the McCormick LP regime
that motivated discopt#229, Step 1 stands alone.

**Evidence.** `FERAL_BUMP_STATS` aggregate over
`FERAL_LU_TRACE=.../casctanks_trace.txt` (full trace): avg_width=731.3,
max_width=2157, avg_density=0.1346 (all bumps) / 0.0023 (width>500),
avg_axpy=23.9, avg_merge_work=233.4. Step 1 end-to-end: casctanks LP solve
82.4 s → 5.2 s debug (15.8×), optimum −167.751 unchanged. Journal:
dev/journal/2026-06-18-01.org.

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
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
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
