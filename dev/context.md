# FERAL Context (auto-generated)

Generated: 2026-05-16T17:39:54Z

## Latest Session
File: dev/sessions/2026-05-16-04.md
```
# Session 2026-05-16-04

## Goal

Close issue #27 (M4 synthetic generators for saddle-rankdef,
wide-frontal, MC64-resistant, and Stokes pathologies in the stress
suite) and the issue #31 follow-up asking for an explicit exact-zero
rankdef matrix to expose the dispersed-null-space failure mode.

## Accomplished

Added five new generator families to
`external_benchmarks/stress/synth.py`, all seeded for bit-
reproducibility and verified against NumPy oracles:

| generator                  | matrix size | nnz   | expected inertia |
|----------------------------|-------------|-------|------------------|
| `rankdef_exact_50_5`       | 50          | 1275  | (21, 24, 5)      |
| `rankdef_exact_100_10`     | 100         | 5050  | (50, 40, 10)     |
| `saddle_rankdef_50_10_3`   | 90          | 3315  | (50, 37, 3)      |
| `saddle_rankdef_100_20_5`  | 180         | 13130 | (100, 75, 5)     |
| `wide_frontal_616`         | 616         | 178982| data-dependent   |
| `mc64_resistant_200`       | 200         | 20100 | (107, 93, 0)     |
| `stokes_q1p0_8`            | 162         | 866   | (98, 62, 2)      |

`report.py` extended with regex-based oracle dispatch for each new
naming convention (`rankdef_exact_<n>_<k>`,
`saddle_rankdef_<n>_<k>_<r>`, `stokes_q1p0_<h>` map to expected zero
counts of k, r, and 2 respectively). The rank-deficient-refusal
short-circuit broadened from `category == "rankdef"` to also cover
`saddle_rankdef` and `stokes`. Manifest gains seven new rows.

Math, oracle derivations, and the abandoned mc64_resistant first
attempt are documented in
`dev/research/synthetic-generators-m4.md`.

End-to-end verification:
- `python3 external_benchmarks/stress/synth.py --only <name>` produces
  each matrix with deterministic header (`50 50 1275`, etc.).
- `cargo build --release --bin bench_one_matrix` succeeds.
- `python3 external_benchmarks/stress/run.py --category
  rankdef,saddle_rankdef,stokes,wide_frontal,mc64_resistant` factors
  all 7 new matrices in <14ms total factor time; the largest
  individual factor is `wide_frontal_616` at 12.9ms.
- `python3 external_benchmarks/stress/report.py` flags none. All seven
  matrices have `rel_res < 1e-13`. `stokes_q1p0_8` returns the exact
  expected inertia `(98, 62, 2)` — feral's BK pivoting correctly
  recovers the 2-dimensional LBB defect.

## Benchmark Results
```

## Git Status
```
74f5f26 merge(#30): IR convergence policy research — keep current loop
d431362 chore(context): refresh dev/context.md after issue #30 research
2630770 research(#30): IR convergence policy -- keep residual-based exit
594db5a merge(#31): near-singular eps boundary certification (worktree agent)
7657b0d docs(inertia): certify near-singular detection boundary p=6 (#31)
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-e78ae26ec1799036)

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
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-16-04.md)


`cargo run --bin bench --release` (no synthetic matrices in
`data/benchmark-config.toml`, so the standard suite was unaffected by
this session's changes):

name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10           43            0     (10, 0, 0)
spd_50             50           21            3     (50, 0, 0)
spd_100           100           86            5    (100, 0, 0)
spd_200           200          422           20    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           21            1    (30, 10, 0)
kkt_50_15          65           49            2    (50, 15, 0)
kkt_100_30        130          206            7   (100, 30, 0)

8 matrices benchmarked

Stress-suite factor times for the new matrices (from `run.py`):

rankdef_exact_50_5      58 μs
rankdef_exact_100_10   119 μs
saddle_rankdef_50_10_3  91 μs
stokes_q1p0_8          133 μs
saddle_rankdef_100_20_5 603 μs
mc64_resistant_200     853 μs
wide_frontal_616     12866 μs

All well under the 1-second target.

```

## Recent Decisions
matrices) costs zero extra IR solves under the current code.

**Evidence.**
- `dev/research/ir-convergence-policy.md` — methodology, raw
  per-matrix table, bucket A/B/C analysis,
  `external_benchmarks/stress/out/ir_probe/*.out` sidecars.
- κ̂(A) distributions overlap between the "IR helps" bucket
  (κ̂ ∈ [1.16e3, 8.00e22]) and the "IR no-op" bucket
  (κ̂ ∈ [9.94e1, 2.29e29]); no κ̂ threshold separates them.
  Routing `bratu3d` (κ̂=1.16e3) into a skip path would lose
  10.24 decades of residual.
- 4 stagnant matrices cost ≤3 IR solves each (the existing
  `max_stagnant_steps=2` rule). Total "wasted" IR work across
  the corpus is ≤12 extra solve-calls — bounded and small.
- `cargo test` and `cargo clippy --all-targets -- -D warnings`
  clean (no implementation change in `src/`; only the probe
  binary and analysis script were added).

**Escape hatches for callers who want to bypass IR.** They
already exist: `Solver::solve`, `solve_sparse`, and
`solve_sparse_many` call back-substitution directly without IR.
The skip-IR knob is a method-selection decision at the call
site, not a parameter inside `solve_sparse_refined`.

**References.**
- `dev/research/ir-convergence-policy.md`
- `src/bin/probe_ir_trajectory.rs`
- `external_benchmarks/stress/analyze_ir.py`
- `external_benchmarks/stress/out/ir_probe/`
- `src/numeric/solve.rs` lines 640–897 (the unchanged loop)

## Recent Tried-and-Rejected
scaling to fail at — the matrix is already perfectly equilibrated.

Direct verification on n=200, seed=601:
- `np.linalg.cond(A) = 1.48` before any scaling
- after a symmetric row-max scaling (proxy for MC64-style scaling):
  `cond = 1.48` (unchanged, as expected)

The "rank-1 perturbation of a diagonally dominant skeleton" framing
suggested in the issue was misleading: a low-rank update of an O(1)
diagonal redistributes O(1) mass; it does not produce the dispersed
ill-conditioning that defeats diagonal scaling.

Replaced with `A = Q D Q^T` construction where Q is a random dense
orthonormal basis and D has one eigenvalue at `1e-8` with the rest
O(1). Now `cond(A) = 2e8` before *and* after symmetric scaling — the
small eigenvalue is in a basis direction that diagonal scaling
cannot reach.

Documented in `dev/research/synthetic-generators-m4.md` §4. The
current generator uses the Q D Q^T construction.

## Source Files
```
src/bin/alloc_probe.rs
src/bin/bench_axpy_small.rs
src/bin/bench_fma_phase3.rs
src/bin/bench_issue8.rs
src/bin/bench_one_matrix.rs
src/bin/bench_orderings.rs
src/bin/bench_solver_corpus.rs
src/bin/bench_solver_reuse.rs
src/bin/bench.rs
src/bin/blas3_prototype.rs
src/bin/calibrate_par_min_flops.rs
src/bin/d3_probe.rs
src/bin/d4_probe.rs
src/bin/diag_acopp30_residual.rs
src/bin/diag_acopr.rs
src/bin/diag_acopr14.rs
src/bin/diag_amalgamation.rs
src/bin/diag_amd_substages.rs
src/bin/diag_amf_vs_amd.rs
src/bin/diag_cascade_default_evidence.rs
src/bin/diag_cascade_ratio_distribution.rs
src/bin/diag_chainwoo_profile.rs
src/bin/diag_chainwoo.rs
src/bin/diag_clnlbeam_maxfromm.rs
src/bin/diag_clnlbeam_slb.rs
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
src/bin/diag_near_singular_sweep.rs
src/bin/diag_orbit2_quotient.rs
src/bin/diag_ordering_panel.rs
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
src/bin/probe_acopp30_64.rs
src/bin/probe_cascade_perturb.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_ir_trajectory.rs
src/bin/probe_issue_19.rs
src/bin/probe_panel_attribution.rs
src/bin/probe_scaling_policy4.rs
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
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/maxfromm_parity.rs
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
tests/solver_with_ordering.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
