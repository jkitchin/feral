# FERAL Context (auto-generated)

Generated: 2026-11-19T18:57:45Z

## Latest Session
File: dev/sessions/2026-11-19-04.md
```
# Session 2026-11-19-04

## Goal
Issue #147: Rust 2024 edition, resolver 3, declared MSRV (user chose
1.85). Also landed: PR #241 (#134 F) merged at session start.

## Accomplished
- 3510a0c: research note + plan
  (`dev/research/rust-2024-edition-2026-11-19.md`).
- 7c13b4e: 8 workspace crates on edition 2024 + rust-version 1.85,
  resolver 3; python/ rust-version 1.85 + resolver 3 but edition 2021
  (pyo3 0.22 macros). capi.rs: #[unsafe(no_mangle)] + 11 narrow unsafe
  blocks; allocators: 13 narrow wraps; set_var FIXMEs replaced with
  SAFETY reasoning; 6 post-1.85 API uses in tests/tools replaced;
  rustfmt style_edition 2021.
- 0152aca: CI `msrv` job (cargo check on 1.85, --locked).
- Verification: C ABI symbols identical (9), harness digests identical,
  cargo test --release 1058 passed + all crate suites green, 1.85 check
  clean, clippy/fmt clean, lockfiles unchanged.

## Benchmark Results
```
KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
```
Identical to 2026-11-19-03.

## Decisions Made
- Edition 2024 / resolver 3 / MSRV 1.85; python/ stays 2021; rustfmt
  2021 style; next release 0.24.0 (dev/decisions.md 2026-11-19).

## Abandoned Approaches
- cargo fix's whole-body `unsafe {}` wraps (replaced by narrow blocks).
- Edition 2024 for python/ (pyo3 0.22 macro warnings) — deferred to a
  pyo3 upgrade, not abandoned.

## Next Session Should
1. Release 0.24.0 (not 0.23.x): the MSRV rises to 1.85.
2. Optional: a separate 2024-style rustfmt commit (drop rustfmt.toml).
3. pyo3 upgrade, then python/ to edition 2024.
4. #211 (wasm vs native factor values) — reproducible in the container.
```

## Git Status
```
0152aca ci(#147): msrv job checks the workspace on Rust 1.85
7c13b4e build(#147): Rust 2024 edition, resolver 3, rust-version 1.85
3510a0c docs(#147): research note and plan for Rust 2024 edition + MSRV 1.85
1ac83be Merge pull request #241 from jkitchin/claude/blissful-ramanujan-w2nhfw
5f06f7a docs(session): 2026-11-19-03 checkpoint — triage, #213 closed, #134 F
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
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 486 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.87s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-11-19-04.md)

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)
--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
Identical to 2026-11-19-03.

```

## Recent Decisions
A CI job (`msrv`) runs `cargo check --workspace --all-targets --locked` on
toolchain 1.85. rustfmt is pinned to `style_edition = "2021"`
(`rustfmt.toml`). The next release carrying this is 0.24.0, not a 0.23.x
patch. MSRV 1.85 chosen by the user.

**Why.** Edition 2024 makes the C ABI's unsafety explicit
(`#[unsafe(no_mangle)]`) and requires `unsafe {}` around unsafe operations
inside `unsafe fn` (each kept narrow, under the SAFETY comment that was
already there). Editions are per crate, so 2021-edition dependents such as
pounce are unaffected; the only downstream effect is the compiler floor,
and releasing as 0.24.0 keeps older pounce releases (pinned to 0.23) off
it. Resolver 3 only does something with a declared `rust-version`, hence
the MSRV, and the CI job makes the declaration checked rather than
aspirational. `python/` stays 2021 because pyo3 0.22's `#[pymethods]` /
`#[pyfunction]` expansions produce ~190 `unsafe_op_in_unsafe_fn` warnings
under 2024; revisit with a pyo3 upgrade. The 2021 rustfmt style keeps the
diff to the edition change: the 2024 style would have reformatted ~320
files (import ordering).

**Evidence.** Exported `feral_*` symbols of `libferal.a` identical before
and after (9). Harness solve digests identical (2b7436a25c646d27,
e321bec7612e0d4f). `cargo test --release` 1058 passed, plus every
ordering crate and feral-diagnostics green; clippy `--workspace
--all-targets -D warnings` clean (with MSRV declared, clippy's
`incompatible_msrv` now guards new code). Six post-1.85 API uses
(`is_multiple_of`, `as_chunks`), all in tests and diagnostics tools,
replaced; the published libraries already compiled on 1.85. Five
`tail_expr_drop_order` notes reviewed: profiler mutex guards, a `Result`
temporary and a diagnostics thread pool now drop earlier, which is
harmless. Lockfiles unchanged. Journal 2026-11-19-04.

## Recent Tried-and-Rejected
the contract for callers that use `SparseLu` as a rank-reveal probe. With
PR #234 at dd98cb8, pounce's
`crates/pounce-cli/tests/issue_1076_qgfrdxpn_active_set_prune_cap.rs`
(`convex_qp_qgfrdxpn`, `solver_selection=qp-active-set`) went from
SolveSucceeded, 3 553 iterations, 6.0 s on feral 0.22 to
MaximumIterationsExceeded at 27 811 iterations, 50.6 s (objective 1.00569e11
at the cap vs 1.00791e11 optimal); it reproduced on re-run. The other 2 156
pounce tests passed, and pounce's default fixture sweep diffed empty
because its default routing never reaches the rank probe. The only signal
I had measured was fill (+0.27 % geomean over 40 synthetic bases, 36/40
worse), which says nothing about which rows a rank reveal declares
dependent.

**Replaced with.** Exact reproduction of the 0.22 heap contents: peel-phase
pushes are logged and heapified when the singleton stacks first run dry,
minus entries for eliminated rows/columns. Digests are bit-identical to 0.22
on 3 real and 40 synthetic bases, pinned by `tests/lu_markowitz_golden.rs`.
Lesson: on a factorization other code uses for rank decisions, a change to
pivot order is a behaviour change even when fill and residuals are fine;
gate it on a golden pivot test, not on fill.

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
src/env.rs
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
src/lu/markowitz.rs
src/lu/mod.rs
src/lu/scaling.rs
src/lu/sparse_factor.rs
src/lu/sparse_hyper.rs
src/lu/sparse_matrix.rs
src/lu/sparse_solve.rs
src/lu/sparse_symbolic.rs
src/lu/sparse_triangular.rs
src/lu/sparse_update.rs
src/numeric/condition.rs
src/numeric/contrib.rs
src/numeric/factorize.rs
src/numeric/mod.rs
src/numeric/reanalysis.rs
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
tests/cb_core_choice_ignores_env.rs
tests/cb_solve_parity.rs
tests/column_renumbering.rs
tests/column_renumbering_parity.rs
tests/common227/mod.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/early_assembly_parity.rs
tests/env_knob_parsing.rs
tests/env_knob_scan.rs
tests/factor_output_recycling.rs
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
tests/issue177_parallel_entry_point_core.rs
tests/issue178_refine_cap.rs
tests/issue178_solve_into.rs
tests/issue203_ordering_race.rs
tests/issue204_work_estimate.rs
tests/issue205_ordering_info.rs
tests/issue227_flops.rs
tests/issue227_never_golden.rs
tests/issue227_reanalysis.rs
tests/issue237_infnorm_reuse.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/issue91_preprocess_misfire.rs
tests/issue99_fma_front_gate.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_215_delay_growth_cap.rs
tests/issue_221_delay_chain_backstop.rs
tests/issue_221_forced_front_parity.rs
tests/issue_221_fully_summed_2x2_partner.rs
tests/issue_221_quality_rung.rs
tests/issue_221_tpp_tail.rs
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
tests/lu_default_ordering.rs
tests/lu_dense.rs
tests/lu_dense_bump.rs
tests/lu_dense_update_bg.rs
tests/lu_ft_widebump.rs
tests/lu_hyper_sparse.rs
tests/lu_markowitz.rs
tests/lu_markowitz_golden.rs
tests/lu_real_bases.rs
tests/lu_refactor_inplace.rs
tests/lu_scaling.rs
tests/lu_sparse.rs
tests/lu_sparse_rhs.rs
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
tests/pounce710_refine_cap_nrhs2.rs
tests/pounce_interface.rs
tests/profiler_smoke.rs
tests/property_tests.rs
tests/refined_solve_core_stability.rs
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
tests/wide_star/mod.rs
tests/wide_tree_memory.rs
```
