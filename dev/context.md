# FERAL Context (auto-generated)

Generated: 2026-06-08T01:07:37Z

## Latest Session
File: dev/sessions/2026-06-08-01.md
```
# Session 2026-06-08-01

## Goal

Implement issue #81: a new **unsymmetric LU factorization family** for feral,
designed as a revised-simplex basis engine — dense + sparse LU with first-class
rank-1 column-replacement updates, `ftran`/`btran`, auto routing, and the full
robustness layer (equilibration + unsymmetric MC64 + iterative refinement).

## Accomplished

All six in-scope phases landed, each with green oracle tests and clippy clean.
The symmetric LDLᵀ solver is untouched (additive `feral::lu` module).

- **Docs first (mandatory).** Research note `dev/research/unsymmetric-lu.md` and
  epic plan `dev/plans/unsymmetric-lu-epic.md`; 6 LU references added to
  `dev/references.bib`.
- **Dense path** (`src/lu/dense_*`). `GeneralMatrix`; `DenseLu` right-looking
  LU with threshold partial pivoting (`P B Q = L U`, explicit `L`/`U`);
  `ftran`/`btran`/`ftran_partial`; refinement; rank-1 Bartels–Golub `update()`
  maintaining `P B Q = L U` via an explicit column permutation.
  `tests/lu_dense.rs` (11 tests).
- **Sparse path** (`src/lu/sparse_*`). `SparseColMatrix` (general CSC) +
  `ata_pattern`; `SparseLuSymbolic` (AMD-on-AᵀA via `feral_amd`); `SparseLu`
  Gilbert–Peierls factor; sparse solves + refinement; product-form `U`-update
  rank-1 column replacement. `tests/lu_sparse.rs` (10 tests).
- **Router.** `should_use_dense_lu` mirroring `should_use_dense_fast_path`.
- **Scaling** (`src/lu/scaling.rs`). Two-sided ∞-norm equilibration +
  unsymmetric MC64 (over the existing `hungarian_match`); `params.scaling`
  drives `factor()`; solves wrap scaling around a core solve.
  `tests/lu_scaling.rs` (5 tests).
- **Errors.** `FeralError::SingularBasis { column }` and `NeedsRefactor`.

Test evidence (all green): equation residuals `‖Bx−a‖/‖a‖` < 1e-8 after
single/chained updates (dense and sparse), update-with-row-swap (perm
composition), reconstruction `‖PAQ−LU‖` < 1e-10, dense↔sparse agreement < 1e-9
(factor + post-update), budget→`NeedsRefactor` with `self` unchanged,
ill-conditioned refinement < 1e-12, and ill-scaled (16 orders) correctness
under InfNorm/Mc64 with the matrix equilibrated into [0.1,10] and MC64 placing
order-1 entries on the diagonal. `cargo clippy --all-targets -- -D warnings`
clean throughout; `pre-commit` installed and enforcing fmt/clippy on every
commit.

## Benchmark Results

The LU module is additive; the symmetric LDLᵀ corpus is unaffected. `cargo run
--bin bench --release` (no external corpus present — synthetic SPD/KKT only):

```
spd_10..spd_200, kkt_10_3..kkt_100_30 — 8 matrices, all inertia exact.
```

## Git Status
```
a781ad4 feat(lu): scaling robustness layer — equilibration + unsymmetric MC64 (#81)
258db84 feat(lu): sparse rank-1 update (product-form U update) + unified update API (#81)
2128e1a feat(lu): sparse Gilbert–Peierls LU factor, solves, AMD-on-AᵀA ordering (#81)
7d914e9 feat(lu): dense LU factor, solves, and rank-1 update (#81)
f4bf2c2 feat(lu): LU module scaffold — GeneralMatrix, LuParams, router, errors (#81)
```

## Test Status
```
test symbolic::tests::schur_symbolic_supernodes_cover_n ... ok
test symbolic::tests::schur_symbolic_single_schur_index ... ok
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
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 324 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.51s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-08-01.md)


The LU module is additive; the symmetric LDLᵀ corpus is unaffected. `cargo run
--bin bench --release` (no external corpus present — synthetic SPD/KKT only):

spd_10..spd_200, kkt_10_3..kkt_100_30 — 8 matrices, all inertia exact.
KKT summary: Inertia match 1/1, Residual pass 1/1, worst 1.14e-15.
Sparse solver: 2/2 inertia match vs MUMPS, 2/2 residual pass, worst 1.26e-16.
Dense/Sparse failure analysis: no failures.

(No prior-session comparison applies — this session added a disjoint module and
changed no symmetric code path.)

```

## Recent Decisions
  `U' = U·F`, `F = I + (τ−e_q)e_qᵀ`, so `U'⁻¹ = F⁻¹U⁻¹`; one eta `(q, τ)` per
  update, applied after the `U`-solve (transposed-in-reverse in `btran`).
  `τ[q]` is the stability pivot. This is correct and genuinely sparse with a
  clean refactor budget; a full Forrest–Tomlin row-eta file (keeping the eta
  sparser than the dense `τ`) is deferred as an optimization, not a correctness
  gap.

- **Sparse factor.** Gilbert–Peierls left-looking LU. The forward-substitution
  variant used is correct but not yet output-sensitive (no DFS reach); the
  depth-first symbolic reach that makes it O(flops) is deferred.

- **Column ordering.** Reuse `feral_amd::amd_order` on the explicitly-formed
  `AᵀA` (column-intersection) pattern as a stand-in for COLAMD. The `AᵀA`
  pattern is invariant under the row permutation/scaling, so the ordering is
  also valid for the scaled matrix `Ã`.

- **Scaling.** Unsymmetric MC64 is a thin driver over the existing
  `crate::scaling::hungarian_match` (already an unsymmetric bipartite matcher),
  not a new algorithm; ∞-norm equilibration adapts the two-sided Knight–Ruiz
  idea. `params.scaling` drives `factor()`, which factors
  `Ã = D_row Π B D_col`; solves wrap the scaling around a core solve.

- **API.** `update(leaving_slot, entering_col)` takes the raw entering column
  (computes the spike internally) on both paths, matching the simplex
  "swap column" operation and the `BasisEngine` seam shape.

- **Out of scope (deferred).** The `pounce-simplex` `BasisEngine` integration
  and GLOBALLib/netlib end-to-end benchmarks cannot be done here (pounce is not
  in this environment); reference (UMFPACK/KLU) benchmarks and the GP
  reach / full FT optimizations are Phase 7 in `dev/plans/unsymmetric-lu-epic.md`.

## Recent Tried-and-Rejected
   the scaling choice and unpredicted by max_col_deg / MC64 cost / n
   (ROSEPETAL's MC64 is 68x ORTHREGF's, yet ROSEPETAL is the win).

A gate keyed on "scaling won't reuse MC64" would regress the fill-reduction
wins (ROSEPETAL, ex8_2_2) to save milliseconds on the overhead-only losses
(ORTHREGF, SINQUAD2, sub-ms small matrices). Not a safe win.

Bucket size for the record (`probe_compress_scaling_bucket`, 3 roots, 1006
families): 376 LdltCompress, of which 118 reuse MC64 (keep) and 258 do not
(the target bucket — heterogeneous, contains both ROSEPETAL-type wins and
ORTHREGF-type losses).

Future sessions: do NOT gate `LdltCompress` on the scaling strategy. The real
(separate, harder) lever is an orthogonal **compression cost/benefit gate**
that estimates fill reduction vs MC64+ordering cost; the current cheap proxy is
`pick_ordering_preprocess`'s low-degree fraction, and no cheap structural
feature yet separates ROSEPETAL (win) from ORTHREGF (loss). Data:
dev/research/mc64-symbolic-skip-2026-06-06.md, dev/journal/2026-06-06-04.org.
This closes the dense-column follow-up: both option (a) (inner-loop fast path)
and option (b) (scaling-aware skip) are now closed with negative results.

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
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
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
tests/lu_scaling.rs
tests/lu_sparse.rs
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
