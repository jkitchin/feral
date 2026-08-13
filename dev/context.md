# FERAL Context (auto-generated)

Generated: 2026-08-13T16:04:18Z

## Latest Session
File: dev/sessions/2026-08-13-01.md
```
# Session 2026-08-13-01

## Goal

Review the open PRs on jkitchin/feral, then fix the findings the user selected.
One PR was open: **#160**, "feat(lu): Suhl-Suhl basis triangularization +
optional dense-bump route" (3 commits, 11 files, +1376/-40). The user scoped the
fix to findings 1 and 2.

## Accomplished

### Review of #160

The PR's core math holds up. Verified that the Suhl–Suhl peel yields the claimed
block structure, and that the dense splice is exact given the two structural
checks the caller makes: every bump-column entry lands either in `above`
(already-pivoted front row, copied straight through to `U` because `L` is the
identity there) or in `packed`, so no entry can be dropped. Worst dense-route
residual over randomized differential probes: **7.9e-16**. The `zero_pivot_tol`
rescale by `a_max / bump_a_max` is correct — it exactly cancels
`factorize_packed`'s own block-local scaling, so both routes reach the same
singularity verdict.

Four findings. Two fixed this session, two left open at the user's scoping.

### Fixed — finding 1: dense route named the wrong singular column (a91519d)

`factor_bump_dense` propagated `factorize_packed`'s error with `?`, and that
error carries the **block-local** column index (`dense_factor.rs:411`). Every
site on the sparse path emits `qcol[k]`, the original basis column.

Evidence on the PR's own fixture: sparse reports `SingularBasis { column: 11 }`,
dense reports `column: 7`. Columns 10 and 11 are the identical (singular) pair;
7 is unrelated. This is exactly the defect
`tests/lu_sparse.rs::singular_basis_reports_original_column_not_factorization_position`
already pins for the sparse path, and for the same reason — a simplex driver
knows original basis columns, not internal positions.

The PR's own `singular_bump_is_reported_on_both_routes` matched
`Err(SingularBasis { .. })` only, so it passed against the wrong column.
Strengthened to compare the two routes' columns and pin the answer to the
duplicated pair.

### Fixed — finding 2: dense route fired on bumps that were never peeled (21f5e74)

`want_dense_bump` keyed only on `bump_hi - bump_lo` fitting the cap. A bump equal
to the whole basis satisfies that trivially, so a whole sparse basis was packed
into an `m²` f64 buffer and factored densely.

Measured on tridiagonal `m = 3000`, `cap = 4096`, release, this machine:
```

## Git Status
```
21f5e74 fix(lu): gate the dense-bump route on a bump that was actually peeled
a91519d fix(lu): report the original basis column when the dense bump is singular
bdbdec7 fix(python): thread dense_bump_max_dim through the LuFactor binding
5246727 docs(changelog): triangularization + dense-bump route
1217992 feat(lu): Suhl-Suhl basis triangularization + optional dense-bump route
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

test result: ok. 420 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.96s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-13-01.md)


8 matrices benchmarked
2 KKT matrices total

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)

--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

Dense failure analysis: no failures
Sparse failure analysis: no failures

--- Dense perf vs oracles: no matrices have oracle timings ---
--- Sparse perf vs oracles: no matrices have oracle timings ---

**Not a meaningful comparison to prior sessions.** This container carries only
the 2 synthetic KKT fixtures the bench generates itself; the real corpus is
absent, so the Phase 2.8.1 partitions are all `N/A` and there are no oracle
timings. It is reported for completeness, not as a regression check. These
changes touch only the LU family (a separate factorization from the LDLᵀ solver
the bench exercises), so no movement was expected there either way.

The numbers that matter for this session are the tridiagonal timings in the
table above, taken directly.

```

## Recent Decisions
   sets it; `natural`, `with_order` and `analyze_amd_only` do not. Those three
   report `(bump_lo, bump_hi) = (0, m)` because they never looked for structure,
   which is not the same claim as having looked and found the basis irreducible.
   The indices cannot distinguish the two and they warrant opposite answers, so
   the provenance is recorded explicitly rather than inferred.

2. When the bump *is* the whole basis (a measured `(0, m)` from `analyze` on a
   basis with nothing to peel), `dense_bump_max_dim` does not apply; such a basis
   is bounded by `dense_threshold` instead.

The rationale for (2) is that the empirical case for the dense kernel — a bump at
2.2% input density whose *factor* is 42% dense — depends on the peel having
already stripped the easy structure and left the irreducible core. That is why
the PR is right that input density is a bad predictor *for a peeled bump*. With
nothing stripped, the premise is absent and ordinary sparsity still governs, so
the decision belongs to `dense_threshold`, which weighs density rather than
dimension alone.

**Both guards are load-bearing.** Provenance alone leaves the `analyze`
-on-unpeelable case at 199x (the 297 ms row above); the `dense_threshold`
allowance alone would still admit the `natural` constructors. Guard 2 is also
what keeps the legitimate small-dense case working: the `(0, 16, 0)` no-border
basis in `tests/lu_dense_bump.rs` peels to nothing but is genuinely dense at
`m = 16 <= 128`, and stays on the dense route.

**Cost.** A new public field on `SparseLuSymbolic`, and callers constructing that
struct by literal must now supply it. Consistent with `bump_lo`/`bump_hi`, added
by the same unreleased change. `analyze` on a large sparse basis that peels to
nothing can no longer opt into the dense route at all; if that case ever matters,
the right lever is a density test on the bump, not a dimension cap.

## Recent Tried-and-Rejected

```rust
let peeled = bump_lo > 0 || bump_hi < m;
```

This broke the PR's own `(0, 16, 0)` differential case in
`tests/lu_dense_bump.rs` — a genuinely dense 16x16 basis with no triangular
border, which peels to nothing and so reports `(0, m)`, but for which the dense
kernel is exactly right. Failure was
`the dense arm fell back to sparse (seed 5) - test is vacuous`.

The distinction the index test cannot make is *why* the bump is the whole basis:
a default from a constructor that never looked, or a measurement from a peel
that found nothing. Those want opposite answers. Replaced with a
`SparseLuSymbolic::triangularized` provenance flag plus a `bump_dim <=
dense_threshold` allowance for the unpeelable case.

Note the provenance flag *alone* was also insufficient — `analyze` on a
tridiagonal m=3000 peels nothing, sets `triangularized = true`, and hit the same
cliff at 297 ms vs 1.5 ms sparse. Both guards are load-bearing.

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
src/lu/sparse_triangular.rs
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
tests/golden_bits.rs
tests/growth_flag.rs
tests/issue102_intrafront_deadlock.rs
tests/issue102_ordering_escalation.rs
tests/issue107_external_ordering.rs
tests/issue112_bg_update.rs
tests/issue127_pipeline_split.rs
tests/issue128_supernode_nrow.rs
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
tests/lu_dense_bump.rs
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
tests/task_plan_parity.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
