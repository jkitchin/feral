# FERAL Context (auto-generated)

Generated: 2026-08-09T21:06:51Z

## Latest Session
File: dev/sessions/2026-08-09-07.md
```
# Session 2026-08-09-07

## Goal

Two halves. First: take PR #151 from green to shipped (merge, tag,
publish, notify pounce). Second, once that was done: take the
measurement the release was cut for, which is the factorization gap
against Harwell on chain-structured KKTs.

## Unfavorable result, stated first

**On the two largest chain-structured proxies, `main` is significantly
slower than 0.14.0.** `prommis_sx_like` 0.833x (2/15 wins, p = 0.0074),
`double_column_like` 0.914x (4/15, p = 0.1185). This contradicts the
0.15.0 release note's "wins all twelve arms", which was measured on real
matrices on a 4-core homogeneous x86_64 container. It also independently
reproduces the `chainW` anomaly that session 2026-08-09-02 recorded and
accepted as a proxy quirk, and that the pounce-side reviewer on PR #150
flagged as a probable regression on exactly this geometry.

Caveats that keep this from being a verdict: these are synthetic
proxies, the "new" arm is `main` at `7a31ff6` rather than the `v0.15.0`
tag, and the bisect that would separate the two did not run. Full
numbers, method and limits in
`dev/research/chain-kkt-ma57-gap-2026-08-09.md`.

## Accomplished — release

1. Verified #151 green, merged as `808babb`, waited for main CI on the
   squash commit before tagging (a different SHA than the PR head).
2. Tagged `v0.15.0` and published the GitHub Release. Release job: tag /
   `Cargo.toml` check PASS, `cargo test` PASS, `cargo publish` PASS for
   all seven crates. Wheels job: sdist + four wheels, PyPI publish, `uv
   pip` smoke test PASS.
3. Verified against the registries, not workflow exit codes: crates.io
   `feral` 0.15.0; PyPI `feral-solver` 0.15.0 with macosx universal2,
   manylinux x86_64, manylinux aarch64, win_amd64 and the sdist.
4. Notified pounce ([pounce#552 comment 5232312068](https://github.com/jkitchin/pounce/issues/552#issuecomment-5232312068)).
   Release checklist §3 complete.

## Accomplished — measurement

5. Established what this machine can and cannot measure. It has the
   CoinHSL v2023.11.17 bundle (so MA57 is available, oracle built and
   linked) and Ipopt 3.13.2 with MA27/MA57/MUMPS all confirmed live. It
   does **not** have `data/matrices/` at all, nor the Pyomo NMPC stack
   for the five #552 models, nor CUTEst to regenerate the corpus.
6. Built `external_benchmarks/chain_proxy/`: block-tridiagonal KKT
   proxies at the reported #552 geometries, plus the paired A/B runner
   and a mechanism probe. Portable (env-driven paths) so it runs on the
```

## Git Status
```
1833768 docs: note that #156 changed the parallel default after the measurement
14bbaa2 docs: ship 0.15.0, then measure the gap vs MA57 (chain proxies)
f7a152a Merge pull request #156 from jkitchin/claude/review-issue-154-ukpt7t
af73f63 Merge origin/main into claude/review-issue-154-ukpt7t
6c87a0e docs: session checkpoint 2026-08-09-03 (issue #154 review + implementation)
```

## Test Status
```
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
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

test result: ok. 413 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.53s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-09-07.md)


No corpus bench: this machine has no corpus, and the release half of the
session changed zero lines of source. Numbers for the shipped code stand
from session 03 and the aarch64 revalidation at `6fd12d4`. `cargo test`
for the published SHA ran green inside the release job. The proxy
measurements are tabulated in the research note rather than duplicated
here.

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
Swept `RAYON_NUM_THREADS` = 1, 2, 4, 8, 10 against the default on six
real chain KKTs, 15 paired runs each, on an M4 Pro (10P + 4E — *more*
efficiency cores than the 4P+4E M2 the hypothesis came from, so the
predicted effect should be larger). Every ratio vs the default landed
within 6% of 1.0; the only significant one was `marine_1600` at t1
(0.961, 0/15, p = 0.0001), in the wrong direction to support the
hypothesis. `steering_12800` at one thread: 1.003 (9/15, p = 0.6072).
`dtoc1nd`, the matrix that actually regressed, at t4: 1.005 (7/15,
p = 1.0000).

The knob was verified live, not assumed: feral reads the global rayon
pool (`rayon::current_num_threads()`, `src/numeric/factorize.rs:3262`),
and the same variable moved the proxy matrices by up to 65%.

Consequence worth carrying forward: single-threaded main matches
all-threads main on every one of these matrices, so **#150's 1.20x to
2.05x gains on the large chains are not parallelism gains**. Do not
build on the assumption that they are.

Full data: `dev/research/chain-kkt-corpus-2026-08-09.md`, Result 3.

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
tests/task_plan_parity.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
