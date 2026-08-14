# FERAL Context (auto-generated)

Generated: 2026-08-14T01:28:45Z

## Latest Session
File: dev/sessions/2026-08-13-05.md
```
# Session 2026-08-13-05

## Goal

Answer the one question discopt #1008 left open to feral: is the 19.1x LU fill on
its LP bases intrinsic to those matrices, or an artifact of choosing the column
order statically and then pivoting for stability?

## Accomplished

**Answer: artifact.** Built a threshold-Markowitz LU as a fill oracle
(`dev/probes/markowitz-fill/`), gated on a per-factorization check of
`||PBQ - LU||inf / ||B||inf < 1e-10`, and ran it against feral's two orderings and
SuperLU COLAMD on 16 real discopt simplex bases.

```
n=16  geomean fill:  AMDfull 3.00x  peel 1.52x  COLAMD 3.24x  MARKOWITZ 1.11x
geomean Markowitz advantage over feral's best ordering: 1.37x (min 1.00x, max 4.30x)
```

Markowitz never exceeds 1.89x on any basis in the corpus; feral's shipped ordering
tails to 24.90x. Full table and reasoning in
`dev/research/lu-fill-markowitz-2026-08-13.md`.

Two findings beyond the headline:

- **discopt's premise is wrong on both halves.** It states feral's `analyze`
  already triangularizes, but it pins crates.io `feral 0.15.1` (predates #160,
  which is merged and unreleased) *and* `LuOrderingParams::default()` is
  `triangularize: false`, so `analyze()` is `analyze_amd_only`. The 19.1x was
  measured on the un-peeled path.
- **#1008's "fill theory dead" measurement could not have detected the headroom.**
  SuperLU is the same algorithm class as feral — static column order plus partial
  pivoting — so agreement between them says nothing about the dynamic alternative.
  On this corpus SuperLU is *worse* than feral (3.24x vs 3.00x) and both sit
  2.7–2.9x above the achievable. Corroborating: inside the SuperLU arm,
  `diag_pivot_thresh` 0.1 vs 1.0 moves fill under 2% on every basis. The
  pivoting threshold is not what sets the fill.

Costs measured, not assumed: Markowitz at `u = 0.1` on QPLIB_1157 gives element
growth 81.8x and `max|L|` 9.70 (at its 1/u bound) against partial pivoting's 2.56x
and 1.00. It also does not fit the `analyze`-then-`factor` split at all, since
pivots come from numeric values.

Filed as **feral #167**. Correction posted to **discopt #1008**.

### Self-correction during the session

The first pass concluded the peel closes `QPLIB_1451_rlt0` outright (7.86x → 1.01x)
and that #1008's 19.1x was not reproduced. Both were wrong, from sampling the
```

## Git Status
```
14bfd1d research: correct the fill note against a deep-trajectory basis (#167)
f7a29cf research: the LP-basis fill is an artifact of static ordering (discopt #1008)
b071d54 Merge pull request #160 from jkitchin/feat/lp-basis-triangularization
9ed8907 docs: session checkpoint 2026-08-13-01 (review of #160 + two fixes)
21f5e74 fix(lu): gate the dense-bump route on a bump that was actually peeled
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
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 422 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.44s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-13-05.md)


**Not re-run this session, deliberately.** This session touched only
`dev/research/`, `dev/probes/`, `dev/journal/` and `dev/sessions/` — no library
code. `cargo run --bin bench --release` would measure the same code session 04
measured, and its results stand as recorded in `dev/sessions/2026-08-13-04.md`
(including the dense-KKT p90 regression reported there: small-frontal 1.57 → 1.61,
medium 1.96 → 2.09, both still PASS).

The measurements this session *did* produce are fill ratios from
`examples/basis_refactor` plus the Python oracle, not wall-clock, and no timing
claim is made from them.

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
tests/golden_bits.rs
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
tests/issue128_supernode_nrow.rs
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
tests/lu_dense_bump.rs
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
tests/task_plan_parity.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
