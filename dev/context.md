# FERAL Context (auto-generated)

Generated: 2026-08-30T00:59:34Z

## Latest Session
File: dev/sessions/2026-08-30-01.md
```
# Session 2026-08-30-01

## Goal

Fix issue #194: `Solver::factor` cannot be cancelled once running, and
offers no surface through which a caller could ask, which makes a host's
wall-clock budget unenforceable whenever a single factorization is larger
than the whole budget.

## Benchmark Results

**No comparison against the previous session's numbers is possible in
this container, and that is a gap in this checkpoint rather than a pass.**
The corpus (154k matrices with oracle timings) is not present here, so
`cargo run --bin bench --release` ran only the 8 synthetic matrices and
both exit-partition tables read `N/A`:

```
--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

Dense failure analysis: no failures
Sparse failure analysis: no failures

--- Dense perf vs oracles: no matrices have oracle timings ---
--- Sparse perf vs oracles: no matrices have oracle timings ---

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A
```

Session 2026-08-19-05's p90s (dense 1.58 / 2.00, sparse 1.58 / 1.58)
therefore stand unchallenged and unconfirmed. A session with the corpus
mounted should re-run before this is treated as regression-free on the
corpus.

In place of that, the poll overhead was measured directly: a paired A/B
probe (same source, built against this branch and against a worktree at
`origin/main`, best-of-10 warm refactors), alternated to control for
container drift.
```

## Git Status
```
4c7691a feat: cooperative cancellation for Solver::factor (#194)
9edce95 docs: research note for issue #194 cooperative factor cancellation
9b9e882 Merge pull request #188 from jkitchin/ci/codecov-coverage
1292984 ci: measure coverage with cargo-llvm-cov and report it to Codecov
ad0d96d Merge pull request #187 from jkitchin/docs/session-2026-08-19-05
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

test result: ok. 442 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.79s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-30-01.md)


**No comparison against the previous session's numbers is possible in
this container, and that is a gap in this checkpoint rather than a pass.**
The corpus (154k matrices with oracle timings) is not present here, so
`cargo run --bin bench --release` ran only the 8 synthetic matrices and
both exit-partition tables read `N/A`:

--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

Dense failure analysis: no failures
Sparse failure analysis: no failures

--- Dense perf vs oracles: no matrices have oracle timings ---
--- Sparse perf vs oracles: no matrices have oracle timings ---

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A

Session 2026-08-19-05's p90s (dense 1.58 / 2.00, sparse 1.58 / 1.58)
therefore stand unchallenged and unconfirmed. A session with the corpus
mounted should re-run before this is treated as regression-free on the
corpus.

In place of that, the poll overhead was measured directly: a paired A/B
probe (same source, built against this branch and against a worktree at
`origin/main`, best-of-10 warm refactors), alternated to control for
container drift.

                     branch (this PR)      main (9b9e882)
grid_laplacian_350   203.8 / 197.8 ms      222.8 / 203.5 ms   sequential
grid_laplacian_350    94.2 /  92.3 ms       94.4 /  92.2 ms   parallel
tridiag_200k          57.7 /  57.3 ms       59.4 /  59.2 ms   sequential
tridiag_200k          56.3 /  56.9 ms       56.8 /  54.6 ms   parallel

No measurable overhead; every difference is inside this host's noise
floor. That floor is wide — see the artifact recorded under *Abandoned*
below — so the honest claim is "no regression detectable here", not "zero
cost proven". A regression would in any case be surprising: every poll
site short-circuits on an `Option::is_some` branch and touches no atomic
when unarmed.

```

## Recent Decisions

**The flag lives on `BunchKaufmanParams`, not `NumericParams`.** It is
not a Bunch-Kaufman parameter in spirit, and `NumericParams` is where a
reader would look first. But the multifrontal drivers hold
`params.bk` and the dense frontal factor is handed *only*
`&BunchKaufmanParams`, so one field there is readable from every poll
site with no sync step, while a field on `NumericParams` would need
copying into `bk` — exactly the shape that left `NumericParams::fma` a
silent no-op until finding N1 (`dev/research/repo-review-2026-06-09.md`).
`BunchKaufmanParams` already carries execution knobs of this kind
(`intrafront_parallel`, `fma`), so this is consistent with what the
struct had become. Cost of the choice: discoverability, mitigated by
`Solver::with_interrupt` / `set_interrupt` / `interrupt` being the
documented entry points.

**`Interrupted` is a `FeralError` variant, so both drivers cancel for
free.** The sequential driver already propagates errors out of its
supernode loop with `?`; the parallel driver already funnels them into
`first_error`, whose fast-exit at the top of `run_parallel_task` drains
the scope without starting further work. Raising the interrupt as an
ordinary error reuses both, so cancellation added no new control flow to
either driver — including the several-tasks-in-flight case. The
alternative, a distinct early-return channel, would have duplicated the
parallel driver's unwind for no benefit.

**Consequence for the API.** `FactorStatus` and `FeralError` each gain a
variant, so every exhaustive `match` over them in the tree, in
`feral-diagnostics` and in the Python bindings needed a new arm. All were
given explicit arms rather than wildcards, deliberately: the next variant
should break the same builds rather than be silently absorbed.

## Recent Tried-and-Rejected
total factor time** on a matrix whose paying bucket is 91% of that time. Moving
three quarters of the panel share into BLAS-3 buys 1.3%: the two kernels cost
nearly the same per flop at this front shape, so the 53.5% panel share is not
recoverable time.

It also does not generalize. `bs = 48` and `bs = 64` are identical for any front
with `ncol ≤ 48`, and that is every other matrix sampled — `ncol` p90 is 1-19
across clnlbeam, dtoc2, marine_1600, rocket, steering, gasoil_3200, pinene_3200,
robot_1600, svanberg, nql180, qcqp1500-1c, cont5_2_4_l; only dtoc1nd is at 63. On
the two with any wide fronts at all the paired sweep finds nothing: nql180 0.990
(5/12 wins, tied with the default), qcqp1500-1c 0.994 (3/12). A 1.3% win on one
corpus matrix and a no-op elsewhere is below the bar for changing a global default.

**Kept from this attempt:** `block_size` is bit-neutral on all three matrices
swept — identical inertia, zero delayed pivots, identical residual, and an
identical hash over every `L`/`D` bit in storage order across
`bs ∈ {8,16,24,32,48,62,64}` (`dtoc1nd_0010` 9cb93f568423e6c0, `nql180_0000`
4f588093d6bac8c7, `qcqp1500-1c_0000` cfec17df1a4f8d38). So future retuning of it
is a performance-only change. Not yet established on a matrix that actually
delays a pivot — all three report `d0`.

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
tests/cb_core_choice_ignores_env.rs
tests/cb_solve_parity.rs
tests/column_renumbering.rs
tests/column_renumbering_parity.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/env_knob_parsing.rs
tests/env_knob_scan.rs
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
tests/issue194_factor_interrupt.rs
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
tests/lu_default_ordering.rs
tests/lu_dense.rs
tests/lu_dense_bump.rs
tests/lu_dense_update_bg.rs
tests/lu_ft_widebump.rs
tests/lu_hyper_sparse.rs
tests/lu_markowitz.rs
tests/lu_real_bases.rs
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

(truncated from 363 lines to 350 line budget)
