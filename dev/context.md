# FERAL Context (auto-generated)

Generated: 2026-05-15T18:15:35Z

## Latest Session
File: dev/sessions/2026-05-15-04.md
```
# Session 2026-05-15-04

## Bench vs. prior session

Synthetic-only bench, ~10% variance against session 2026-05-15-03
(corpus not present on this machine). Hot path untouched —
pool reuse only affects the parallel-fan-out path which the
synthetic bench doesn't exercise.

```
spd_10             10           83           30     (10, 0, 0)
spd_50             50           44            5     (50, 0, 0)
spd_100           100          157            9    (100, 0, 0)
spd_200           200          780           31    (200, 0, 0)
kkt_10_3           13            6            1     (10, 3, 0)
kkt_30_10          40           42            2    (30, 10, 0)
kkt_50_15          65          108            5    (50, 15, 0)
kkt_100_30        130          431           13   (100, 30, 0)
```

## Goal

Issue #19 follow-up: replace the implicit rayon scope context that
the parallel multifrontal driver currently builds per `factor()`
call with a `Solver`-owned `rayon::ThreadPool` that persists across
calls. Reduces the cv-wait wakeup cost that session 2026-05-15-03's
work-aware gate could only sidestep, not eliminate.

## Accomplished

### 1. `Solver::parallel_pool` lazy-init field

`src/numeric/solver.rs`:

- New field `parallel_pool: Option<Arc<rayon::ThreadPool>>`.
- New helper `ensure_parallel_pool() -> Option<Arc<ThreadPool>>`
  that lazy-builds on first call and clones the Arc on subsequent
  ones.
- Thread count matches `rayon::current_num_threads()` at build
  time (honors `RAYON_NUM_THREADS`, falls through to num_cpus as
  the global pool would).
- `Solver::factor` now reads the pool *before* taking the
  immutable `symbolic` borrow, so the mutable borrow on
  `parallel_pool` doesn't collide. The dispatcher runs inside
  `pool.install(...)` when both `use_parallel` is on and the pool
  built successfully.

Inside `pool.install(...)`, the inner driver's `rayon::scope` /
`current_thread_index` / `current_num_threads` calls bind to this
pool's workers instead of the global pool. No changes to the inner
```

## Git Status
```
2dc8fb3 chore(session): 2026-05-15-03 -- issue #19 work-aware parallel gate
19d7b03 fix(parallel): work-aware gate in should_parallelize_assembly (#19)
6de5790 chore(session): 2026-05-15-02 -- issue #18 refinement wire-up
597a90a fix(capi): default feral_solve to solve_many_refined for issues #17, #18
6a7d1d5 chore(session): 2026-05-15-01 -- issue #17 diagnosis
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-6872324a4f072be5)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_solve_parity_tiny_real_matrix ... ok
test test_determinism_tiny ... ok
test test_gate_boundary_n_16 ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 1 test
test src/symbolic/profiler.rs - symbolic::profiler::SymbolicProfiler (line 27) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; no session checkpoint with bench)
```

## Recent Decisions
Implementation: `Solver::factor` calls `ensure_parallel_pool()`
before borrowing `last_symbolic`, then runs the parallel driver
inside `pool.install(|| ...)`. Inside `install`, all
`rayon::scope` / `current_thread_index` / `current_num_threads`
in the inner driver bind to this pool's workers.

**Why.** Issue #19 (sessions 2026-05-15-03/04) flagged rayon
spawn / cv-wait wakeup as 53% of sys time on `robot_1600`. The
work-aware gate added in session 2026-05-15-03 sidesteps this
cost by *not firing parallel*; the pool reuse decision instead
*amortises* the cost when parallel does fire. Complementary, not
substitutive.

**No user-facing toggle.** Pool reuse is strictly dominant over
per-call construction (lower sys, same wall worst case). The
existing `with_parallel(false)` toggle already disables the
parallel path *including* pool construction — pinned by test
`solver_with_parallel_false_does_not_build_pool`.

**Evidence.** robot_1600 force-parallel (200 iters, M4 Pro): sys
time 24.7 s → 17.9 s (**-28%**). Wall on M4 Pro unchanged because
cv-wait wasn't yet wall-dominant locally; on the issue reporter's
hardware where it reportedly was, this should translate to a wall
win too. `cargo test --lib --release` → 256 passed (254 prior + 2
new pool-reuse tests).

**References.**
- feral GitHub issue #19.
- `dev/sessions/2026-05-15-04.md`.
- `dev/journal/2026-05-15-04.org`.

## Recent Tried-and-Rejected
robot_1600 in 40 iters / 6.1 s vs cb=default's MaxIter at 200
iters / 53 s.

**Why rejected.** Cascade-break is the cascade-arm gate shipped
by #15 and is calibrated across the bench corpus to help on a
specific class of matrices. Disabling it by default would
regress those without addressing the underlying mechanism in
robot_1600. The 2026-05-15 decision (`dev/decisions.md`)
established the failure is a *solve-accuracy* regression (~5-OOM
on identical inertia), not an *inertia-counting* one. Fixing it
upstream by removing cascade-break trades one regression for
another.

**Status.** Issue #17 is being addressed downstream: wire
`Solver::solve_refined` into `pounce-feral/src/lib.rs:107` so
F2.3 iterative refinement absorbs the perturbation. Pursued in
next session.

References: `dev/sessions/2026-05-15-01.md`,
`dev/decisions.md` 2026-05-15 entry, issue #17 thread.

## Source Files
```
src/bin/alloc_probe.rs
src/bin/bench_fma_phase3.rs
src/bin/bench_issue8.rs
src/bin/bench_one_matrix.rs
src/bin/bench_orderings.rs
src/bin/bench_solver_corpus.rs
src/bin/bench_solver_reuse.rs
src/bin/bench.rs
src/bin/blas3_prototype.rs
src/bin/d3_probe.rs
src/bin/d4_probe.rs
src/bin/diag_acopr.rs
src/bin/diag_acopr14.rs
src/bin/diag_amalgamation.rs
src/bin/diag_amd_substages.rs
src/bin/diag_amf_vs_amd.rs
src/bin/diag_cascade_default_evidence.rs
src/bin/diag_cascade_ratio_distribution.rs
src/bin/diag_chainwoo_profile.rs
src/bin/diag_chainwoo.rs
src/bin/diag_compress_costbenefit.rs
src/bin/diag_compress_profile.rs
src/bin/diag_compression_bench.rs
src/bin/diag_cond_parity.rs
src/bin/diag_dense_tail.rs
src/bin/diag_etree_shape.rs
src/bin/diag_factor_nnz_accounting.rs
src/bin/diag_fill_parity.rs
src/bin/diag_fill_tail.rs
src/bin/diag_inertia_mismatch.rs
src/bin/diag_leaf_profile.rs
src/bin/diag_max_ncol.rs
src/bin/diag_mc64_cycles.rs
src/bin/diag_mittelmann.rs
src/bin/diag_orbit2_quotient.rs
src/bin/diag_ordering_race.rs
src/bin/diag_par_firstdiff.rs
src/bin/diag_par_frontal_hash.rs
src/bin/diag_par_repeat.rs
src/bin/diag_parent_unique.rs
src/bin/diag_phase_b_nemin_sweep.rs
src/bin/diag_pinene_0009_profile.rs
src/bin/diag_pinene_amd.rs
src/bin/diag_pinene_pivot_cliff.rs
src/bin/diag_pinene_static_pivot.rs
src/bin/diag_poisson_kkt.rs
src/bin/diag_qcqp_knobs.rs
src/bin/diag_qcqp_profile.rs
src/bin/diag_robot1600_eigs.rs
src/bin/diag_schur_parity.rs
src/bin/diag_small_leaf_gate.rs
src/bin/diag_small_leaf.rs
src/bin/diag_sparse_memory.rs
src/bin/diag_split_tail.rs
src/bin/diag_strategy_compare.rs
src/bin/diag_supernode_cost.rs
src/bin/diag_swopf_w22x2.rs
src/bin/diag_symbolic_stages.rs
src/bin/dump_diff.rs
src/bin/hs85_diag.rs
src/bin/parallel_corpus_parity.rs
src/bin/polak6_diag.rs
src/bin/policy4_diag.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_panel_attribution.rs
src/bin/produce_dense_schur.rs
src/bin/profile_hot.rs
src/bin/profile_sparse.rs
src/bin/profile_supernode_distribution.rs
src/bin/solve_microbench.rs
src/bin/vesuvio_diag.rs
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
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
tests/fma_opt_in_roundtrip.rs
tests/growth_flag.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_2_kkt_ls_init.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/mc64_end_to_end.rs
tests/mc64_scaling.rs
tests/multi_rhs.rs
tests/parallel_parity.rs
tests/parity.rs
tests/pivot_rejection.rs
tests/pounce_interface.rs
tests/profiler_smoke.rs
tests/property_tests.rs
tests/rook_rescue_kkt.rs
tests/rook_rescue.rs
tests/small_leaf_parity.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
