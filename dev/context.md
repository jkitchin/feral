# FERAL Context (auto-generated)

Generated: 2026-08-30T03:50:39Z

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
7b42414 fix(solve): an interrupt during the MC64 retry must not be swallowed
942d130 Merge origin/main into claude/issue-194-p6gri8
f1dc5ee Merge pull request #191 from jkitchin/fix/componentwise-refine-default
9710b0d Merge origin/main into fix/componentwise-refine-default
205cbbb Merge pull request #193 from jkitchin/claude/issue-192-bepseb
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

test result: ok. 464 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 3.53s

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

**Scope of the change.** The escalation ladder is untouched — its rungs,
the `0.75` exponent, `pivtol_max` — and unit tests U1–U5 pass unchanged,
so a caller that never calls `reset_quality` sees byte-identical
behaviour. The reset touches only the two escalated parameters and the
level, mirroring what `increase_quality` leaves alone: the cached
symbolic factorization survives (scaling-invariant since the β refactor
moved scaling to the numeric phase), so re-baselining costs no
re-analysis exactly as escalating costs none. Pinned by integration test
`i9_reset_quality_rebaselines_without_invalidating_symbolic`.

## 2026-08-29 — the escalation baseline is snapshotted lazily, not at construction (issue #192)

**Decision.** `reset_quality` restores a `QualityBaseline { scaling,
pivot_threshold }` captured on the transition *out of*
`QualityLevel::Baseline` — i.e. at the instant the ladder starts — held
in `Option<QualityBaseline>` and cleared by every reset. Not captured in
`with_params`.

**Why.** The `with_*` builders are consuming and run *after*
`with_params`, so `Solver::with_params(np, sn).with_scaling(Identity)`
would have a construction-time snapshot recording `np`'s strategy, and a
reset would silently discard the caller's builder configuration. The
lazy snapshot also makes the round trip exact by construction: it
records a state the solver demonstrably occupied, so `reset` →
`increase` retraces the same rungs a freshly constructed `Solver` would
— the property downstream needs when re-baselining at a loop boundary.
Pinned by `r6_reset_quality_preserves_builder_configured_scaling` (would
fail under a construction-time snapshot) and
`r4_reset_quality_from_exhausted_restarts_identical_ladder`.

## Recent Tried-and-Rejected
semantically identical, and giving the compiler `&[[f64; 4]]` instead of
`&[f64]` may well be neutral or better for codegen. That is a
*hypothesis*. This is the hottest loop in the factorization; its unroll
depth and `into_remainder()` cleanup are a measured design
(`dev/research/dense-kernel-*.md`), and the container this was found in
has no corpus, so the change could not be benchmarked — the exit
partition reports N/A here. Landing an unmeasured edit to that loop to
satisfy a style lint inverts the project's order of operations.

**What was done instead.** A file-scoped
`#![allow(clippy::chunks_exact_to_as_chunks)]` in `schur_kernel.rs` with
the reasoning in a comment beside it. The three other sites the lint
flagged — `diag_schur_parity.rs` (x2) and `diag_acopr14.rs` — are
byte-decoding loops in diagnostic binaries, not hot paths, so those took
the real rewrite (and lost a `copy_from_slice` each).

**Still open.** Whether `as_chunks` in the kernel is neutral, a win, or
a loss is unmeasured and unclaimed. A session with corpus access should
sweep it and either land the rewrite with numbers or record the
regression here. Until then the `allow` is a deferral, not a verdict.

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
tests/issue190_componentwise_default.rs
tests/issue190_refine_target.rs
tests/issue194_factor_interrupt.rs
tests/issue194_interrupt_during_mc64_retry.rs
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

(truncated from 366 lines to 350 line budget)
