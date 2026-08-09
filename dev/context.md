# FERAL Context (auto-generated)

Generated: 2026-08-09T16:26:16Z

## Latest Session
File: dev/sessions/2026-08-09-04.md
```
# Session 2026-08-09-04

## Goal

Review issue #154 ("default `use_parallel` from the platform instead of
hardcoding `true`"), then implement it fully so the issue can be closed.

## Accomplished

### Review

Verified every claim in #154 against `808babb` (v0.15.0). The diagnosis is
correct and all line references are exact: `solver.rs:430` (`use_parallel:
true`), `:972` (unconditional `ensure_parallel_pool()`), `:1148`, `:1216`,
`:1540`, and the quoted `ensure_parallel_pool` body. Three gaps in the
proposal, all confirmed by measurement rather than reading:

1. **The test inventory was incomplete, and understated.** The issue names
   one test and calls it a latent flake. Applying *only* the issue's
   one-liner to a pristine tree and running under `taskset -c 0`:

   ```
   test result: FAILED. 404 passed; 3 failed; 6 ignored
     solver_parallel_default_is_on             solver.rs:2825
     solver_parallel_factor_matches_sequential solver.rs:3852
       assertion failed: par.parallel()
     solver_reuses_thread_pool_across_factors  solver.rs:2887
       .expect("pool must be built after first parallel factor")
   ```

   `solver_parallel_factor_matches_sequential` is the #7 bit-exactness
   regression test. Repairing only its assertion would leave it comparing
   the sequential driver against itself — passing vacuously, with the
   contract silently unchecked.

2. **The "filed separately" fallback bug is not separable.** After the
   default flips, `with_parallel(true)` — the escape hatch #154 recommends
   for wasm — still reaches `ensure_parallel_pool()` and, on build failure,
   falls to a `None` arm that runs the *parallel* driver bare, i.e. on
   rayon's global pool. That is the failure being avoided, reintroduced
   through the workaround prescribed for it. Shipping the default flip alone
   would release a version whose documented wasm answer is a trap.

3. **A fourth dispatch site the issue misses:** `solve_many_refined`
   (`solver.rs:1601`) has the same `None`-arm shape as `solve_refined`.

Adjacent sweep over every rayon entry point reachable from `Solver`:
`intrafront_parallel` (Lever 1.1) is set only inside the parallel driver
(`factorize.rs:3127`), so the sequential path never spawns nested rayon work
— the pounce#79 oversubscription guarantee holds, not a bug. `solve.rs`
```

## Git Status
```
6c87a0e docs: session checkpoint 2026-08-09-03 (issue #154 review + implementation)
4f2fad6 fix(solver): derive use_parallel from the platform; fall back to sequential when the pool fails
808babb release: feral v0.15.0 (#151)
fad5670 perf(parallel): task-per-subtree coarsening + profiler nanoseconds (#150)
e8e1c5a perf(kernel): explicit SIMD packed trailing update + x86 pulp dispatch fix (#149)
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

test result: ok. 411 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.12s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-09-04.md)


No usable perf signal this session: the benchmark corpus is not present in
this container, so the harness found 2 matrices. Reported as-is rather than
omitted.

--- Dense solver validation ---
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

There is no reason to expect a perf delta on a multi-core host: the derived
default is `true` there, and the pool-fallback change is unreachable unless
`ThreadPoolBuilder::build()` fails. The one measurable change would be on a
single-CPU host, where the sequential driver is now selected — that is the
intended effect, not a regression.

```

## Recent Decisions
`solver_parallel_factor_matches_sequential` is the #7 bit-exactness
regression test; repairing only its assertion would leave it comparing
the sequential driver against itself and passing vacuously. The rule
adopted: any test that means "the parallel driver" constructs with an
explicit `with_parallel(true)`, and only the default test asserts the
derived value — against the same probe the constructor uses, so it
cannot silently become environment-dependent again.

**New coverage.** `solver_parallel_default_follows_platform`,
`pool_num_threads_precedence` (via a pure
`pool_num_threads_from(env, hardware)` helper, so no test mutates
process-global environment state), and
`solver_parallel_without_pool_falls_back_to_serial_refine`, which
reproduces the post-build-failure field state and asserts the refine
output is bit-identical to the sequential solver.

**Also changed.** `FERAL_PARALLEL` in the C ABI (`src/capi.rs`) was
off-only; with a derived default it needs a force-on arm
(`1`/`on`/`true`/`yes`), otherwise a wasm-bindgen-rayon embedder has
no way to opt in without a rebuild. Unset or unrecognized values leave
the derived default alone.

**Validation.** `taskset -c 0 cargo test --lib` → 409 passed, 0
failed. Full `cargo test` on 4 cores → 0 failed across all binaries.
`cargo clippy --all-targets -- -D warnings` → clean.

**Out of scope.** This does not address the wasm hang in
jkitchin/pounce#482. That reproduces only under `nightly-2026-08-02`,
is a CPU spin, and occurs inside `pounce_load` — parsing, upstream of
feral entirely.

## Recent Tried-and-Rejected
`nemin=8`, MEYER3NE 83× at `nemin=4`), which is what makes it a property
of the direction rather than of this rule.

**Why rejected.** "Correctness before performance, always" is a hard
constraint. 2–7% of factor time and 11–45% of fill does not buy seven
digits of residual. Neither my pre-registered criterion nor the queue
item thought to check the axis that decided it — recorded here because
the next person to have this idea will not think to check it either.

The knob stays in-tree defaulting to `None` (bit-identical default path)
as the reproduction apparatus, with the accuracy result in its doc
comment. Research note:
`dev/research/amalgamation-cost-model-2026-08-09.md`.

**Also redirects the target.** pounce#552's re-measurement against a
released 0.15.0 (comment 5232409020) shows clnlbeam more than halved
(8.05× → 3.54× vs MA57) and **no longer the worst case** — `dtoc1nd` is,
at 3.77×, and it is a dense-front matrix (nnz/dim 23.0, fronts of 33–64
columns). Amalgamation is a chain-KKT lever aimed at a problem that has
largely receded.

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
tests/golden_bits.rs
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
tests/task_plan_parity.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
