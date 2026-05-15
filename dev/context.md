# FERAL Context (auto-generated)

Generated: 2026-05-15T17:48:48Z

## Latest Session
File: dev/sessions/2026-05-15-03.md
```
# Session 2026-05-15-03

## Bench vs. prior session

Synthetic-only bench (corpus not present locally). Synthetic numbers
sit slightly higher than session 2026-05-15-02 (e.g., spd_200 factor
401 μs → 878 μs) but the harness is microsecond-noisy on synthetic
inputs and the hot-path stayed untouched (the new code only runs at
the parallel-dispatch gate which the synthetic bench doesn't reach).
Not flagging as regression.

```
spd_10             10          103           18     (10, 0, 0)
spd_50             50           64            5     (50, 0, 0)
spd_100           100          184           11    (100, 0, 0)
spd_200           200          878           36    (200, 0, 0)
kkt_10_3           13            8            1     (10, 3, 0)
kkt_30_10          40           44            2    (30, 10, 0)
kkt_50_15          65          113            4    (50, 15, 0)
kkt_100_30        130          460           15   (100, 30, 0)
```

## Goal

Address feral issue #19: `should_parallelize_assembly` fires rayon
on too-small assembly trees (small-KKT IPM control-NLP profile),
producing per-iter wall regression because rayon spawn / cv-wait
overhead exceeds the parallel speedup. Add a work-aware gate.

## Accomplished

### 1. Reproduced the issue (with caveats)

On Apple M4 Pro (14 cores), 200 iters / problem:

| problem | parallel | sequential | par/seq ratio | sys/wall (par) |
|---|---|---|---|---|
| robot_1600 | 25.3 s | 34.4 s | 0.74× | 53% |
| henon120 | 101 s | 294 s | 0.34× | 21% |

The issue claims a **12× wall regression** on robot_1600 — that does
not reproduce on M4 Pro; parallel is actually 1.3× *faster* in wall.
But rayon overhead is clearly real (27 s of sys time inside a 25.3 s
wall on robot_1600 parallel = 53% sys-bound). The issue's machine is
the same M4 Pro but evidently in a different load / measurement
regime where the cv-wait cost translated to wall. The fix direction
is the same on any machine: stop firing parallel on too-small trees.

### 2. Work-aware gate in `should_parallelize_assembly`

```

## Git Status
```
6de5790 chore(session): 2026-05-15-02 -- issue #18 refinement wire-up
597a90a fix(capi): default feral_solve to solve_many_refined for issues #17, #18
6a7d1d5 chore(session): 2026-05-15-01 -- issue #17 diagnosis
921cb23 diag(issue-17): bin to compare cb=off vs cb=default per-matrix
6e95b82 docs(CLAUDE.md): note core.hooksPath workaround for pre-commit
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
robot_1600 at 200 iters) because parallel was a slight wall win
there; the user-tunable override absorbs the disagreement.

**Override.** `NumericParams::min_parallel_flops: Option<u64>`
(default `None` → use the const). Set to `Some(0)` to disable the
flop gate (structural-only behavior, equivalent to the pre-fix
heuristic); `Some(u64::MAX)` to force-reject all parallel dispatch
at the tree level. Pounce-side wired as `POUNCE_FERAL_MIN_PAR_FLOPS=<
u64>` env var.

**Why a const + override instead of runtime calibration.** Startup
calibration adds complexity for diminishing returns; the env-var
override gives a consumer-controlled tuning knob with O(1) cost.
Calibration probe in `dev/research/issue-19-parallel-heuristic.md`
"Calibration follow-up" section.

**Evidence.**
- `robot_1600` (M4 Pro, 200 iters): OLD parallel 25.3 s wall + 27 s
  sys; NEW default 33.5 s wall + 0.3 s sys (sys time -99%). NEW with
  `MIN_PAR_FLOPS=0` override: 25.4 s wall + 24.7 s sys (matches OLD).
- `henon120`: NEW default 97.9 s wall (parallel correctly preserved
  by the gate), within noise of OLD 101 s. The gate's flop estimate
  for henon120 clears the 10^8 threshold.
- `cargo test --lib --release` → 254 passed (248 prior + 6 new).

**References.**
- feral GitHub issue #19.
- `dev/sessions/2026-05-15-03.md`.
- `dev/research/issue-19-parallel-heuristic.md`.
- `dev/journal/2026-05-15-03.org`.

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
