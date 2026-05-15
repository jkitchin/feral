# FERAL Context (auto-generated)

Generated: 2026-05-15T15:29:27Z

## Latest Session
File: dev/sessions/2026-05-15-02.md
```
# Session 2026-05-15-02

## Bench vs. prior session

Synthetic bench only — corpus matrices are gitignored and not
present on this machine (the prior session's corpus output was
from a machine with them). No regression suspected: refinement
is solve-side; factor hot path untouched.

```
spd_10             10           81           11     (10, 0, 0)
spd_50             50           22            3     (50, 0, 0)
spd_100           100           79            5    (100, 0, 0)
spd_200           200          401           16    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           30            1    (30, 10, 0)
kkt_50_15          65           53            2    (50, 15, 0)
kkt_100_30        130          205            7   (100, 30, 0)
```

## Goal

Address feral issue #18 (NARX_CFy stall) and the still-open #17
(robot_1600) by acting on task 1 from session 2026-05-15-01's
"next session should": *wire `Solver::solve_refined` into the IPM
consumers*. Verify on both reproducers.

## Accomplished

### 1. Triaged #18 against #17

#18's `NARX_CFy.nl` stall trajectory matches #17's `robot_1600`
exactly: early progress, then α stuck in [0.05, 0.30], inf_du
~1e-3 indefinitely. Issue's hypothesis #1 (backsolve residual
floor) aligns with the session 2026-05-15-01 forensic conclusion:
cascade-break's L-factor perturbation produces a per-pivot
residual ~1e-5 that exceeds the duality gap in late iters. Same
root cause; same fix.

### 2. Wired `solve_many_refined` into the C ABI

`src/capi.rs:feral_solve` now routes through
`Solver::solve_many_refined` against the cached `s.matrix`. Opt-
out via `FERAL_REFINE=0|false|off|no`. Added two inline unit tests
exercising both paths on the 2x2 indefinite `[[1,2],[2,1]]` (#17
forensics canonical example).

Justification for using `s.matrix` as the refinement reference:
Ipopt's `MultiSolve` protocol guarantees no values writes between
`feral_factor` and `feral_solve` — Ipopt only writes via the
```

## Git Status
```
6a7d1d5 chore(session): 2026-05-15-01 -- issue #17 diagnosis
921cb23 diag(issue-17): bin to compare cb=off vs cb=default per-matrix
6e95b82 docs(CLAUDE.md): note core.hooksPath workaround for pre-commit
e8dab31 style: cargo fmt on src/numeric/solve.rs
1f25a54 feat(bench): synthetic-matrix scaling bench vs MUMPS + MA57
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-6872324a4f072be5)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_solve_parity_tiny_real_matrix ... ok
test test_gate_boundary_n_16 ... ok
test test_determinism_tiny ... ok

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

**Why.** Per the 2026-05-15-01 decision and forensics: cascade-
break perturbs the L factor (not just D), producing a per-pivot
backsolve residual ~1e-5 that exceeds the IPM duality gap in late
iters. The unrefined backsolve was the binding constraint for
feral#17 (`robot_1600`) and feral#18 (`NARX_CFy`). One round of
refinement against the cached original matrix closes the gap.

**Cost.** Per backsolve: one sparse SymV (mat-vec) + one extra
forward/back substitution. For NARX_CFy that maps to ~3.2× the
wallclock of ipopt-MUMPS at the same iter band — orthogonal to
the stall failure mode and addressable separately.

**Evidence.**
- `ipopt-feral NARX_CFy.nl ... max_iter=500` → Optimal, 485
  iters, 498 s (was: TIMEOUT @ 250 s, iter 279).
- `ipopt-feral robot_1600.nl ... max_iter=500` → Optimal, 301
  iters, 19.3 s (was: MaxIter @ 3000 iters, 395 s on pounce; or
  MaxIter @ 200 with the issue's stale opt-file cap).
- `cargo test --lib --release` → 248 passed including two new
  `capi::tests::capi_factor_and_refined_solve` /
  `capi_solve_unrefined_opt_out`.
- `cargo test --release -p pounce-feral` → 6 passed.

**References.**
- feral GitHub issues #17, #18.
- `dev/sessions/2026-05-15-02.md`.
- `dev/journal/2026-05-15-02.org`.
- Prior decision block (2026-05-15-01) for the forensic
  groundwork this builds on.

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
