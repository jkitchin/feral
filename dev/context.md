# FERAL Context (auto-generated)

Generated: 2026-05-31T15:10:34Z

## Latest Session
File: dev/sessions/2026-05-31-02.md
```
# Session 2026-05-31-02

## Goal

Resolve the feral side of pounce#79: determine whether the parallel
multifrontal driver `run_parallel_task` keeps worker-stack depth O(1) in
elimination-tree height (or whether a deep/path-like tree can overflow a
rayon worker's ~2 MiB stack), and either bound it or document + guard.
Earlier in the session: review pounce#79, post the resolution comment,
and merge feral PR #59.

## Accomplished

- **pounce#79 reviewed and answered.** The per-instance parallelism
  toggle the issue asks for already exists (`Solver::with_parallel`,
  per-instance, Rust + Python); the only `FERAL_PARALLEL` env read is the
  C ABI (`capi.rs:134`). The env-var dance is a pounce-side fix
  (per-worker `with_parallel(false)`); no feral API change needed.
  Resolution comment posted to pounce#79.
- **feral PR #59 merged** (squash, normal merge; "perf-review: analysis
  and verification of intra-front parallelism" — docs + two probe bins,
  no production code).
- **Parallel worker-stack depth: investigated → O(1) in tree height →
  documented + guarded, no behavioral change.** The leaf→root climb in
  `run_parallel_task` (`factorize.rs:3232`) is trampolined through
  rayon's task queue (`scope.spawn`), not native recursion, so native
  stack depth is O(1) in tree height. Verified structurally and by
  measurement:
  - c-big (n=345 241, supernode-tree height 1521 — deepest in corpus):
    parallel factor succeeds on worker stacks all the way down to
    **32 KiB** (~64× under the ~2 MiB default).
  - bratu3d (height 154): factors at a requested 1 KiB stack.
  - Every optimization/KKT corpus matrix has supernode-tree height ≤ 9.
  Changes landed: doc section on `run_parallel_task` (`factorize.rs`);
  regression test `deep_chain_tree_no_stack_overflow`
  (`tests/parallel_parity.rs`, tridiagonal SPD n=8000, default ordering →
  deep supernode chain height ~500); research note
  `dev/research/parallel-stack-depth-pounce79.md`.

## Benchmark Results

No benchmark-affecting change this session — the only source change is a
doc comment plus a new test. `bench` numbers are unchanged from
2026-05-31-01 (PR #59); not re-run (the test gate for doc/test-only
changes is satisfied by the green gates below).

## Decisions Made

- **No behavioral change for pounce#79's feral side.** Depth is already
  O(1) in tree height; an enlarged `ensure_parallel_pool` `stack_size`
```

## Git Status
```
8902f0f docs+test(parallel): document O(1) worker-stack depth, add deep-tree guard (pounce#79)
14cff66 perf-review: analysis and verification of intra-front parallelism (#59)
cd12735 release: v0.9.0
c51ddfd docs: notebook + book now show the batched refined win (#58)
2e096e9 perf(solve): drop the 0-step allocations in batched refinement (#58)
```

## Test Status
```
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test numeric::solver::tests::mc64_fallback_surfaces_via_solver_api ... ok
test scaling::tests::auto_falls_back_to_infnorm_on_mss1_0009 ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_resolves_to_amd_when_bisection_degenerates ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok

test result: ok. 317 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 0.43s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-31-02.md)


No benchmark-affecting change this session — the only source change is a
doc comment plus a new test. `bench` numbers are unchanged from
2026-05-31-01 (PR #59); not re-run (the test gate for doc/test-only
changes is satisfied by the green gates below).

```

## Recent Decisions
than its unrefined solve.

**Decision 3 — active-column compaction each step.** Each refinement step
gathers only the un-converged columns into the batched solve. This bounds
the batched path at ≤ the per-column work even for heterogeneous
convergence (most columns done in 1 step, a few needing 10), where
solving the full batch every step would otherwise regress.

**Decision 4 — threshold dispatch at `BLAS3_REFINE_THRESHOLD = 16`.**
`nrhs < 16` keeps the literal per-column loop (the IPM predictor-
corrector, `nrhs = 2`, and other narrow refined solves stay on the
proven, bit-identical path). 16 (below the 32 panel crossover) because
the batched *solve* amortizes from ~16, and the batched refiner is
provably bit-identical to the per-column loop for `16 ≤ nrhs < 32` (the
rank-1 solve is bit-identical per column there), so there is no accuracy
risk in that band.

**Rejected.** Global-norm refinement loop (drops per-column best-iterate
— accuracy regression risk on near-singular columns). No compaction
(heterogeneous-convergence perf regression). Single-pass batched SpMV for
the residual (deferred — helps dense inputs only; per-column symv is
cache-friendly and reuses tested code).

**Measured.** Bit-identical band verified (`max|batched − per-column| ==
0` at nrhs=24 SPD and nrhs=20 indefinite). Panel band (nrhs=64) matches
the oracle to ≤1e-15 with per-column relative residual ≤1e-15. Lib 317
pass; bench_multirhs refined ratio ~0.34–0.40.

**References.** `dev/research/issue-58-batched-refinement.md`,
`dev/journal/2026-05-30-01.org`, issue #58.

## Recent Tried-and-Rejected
**c-block outer / m-tile inner** would keep a small `B`-block
L1-resident and cut the dominant re-streaming by the factor `NR/MR = 2`.

Tried the swap. **Measured: no improvement.** n=1024 stayed at ratio
~1.0–1.2 (still a regression), and n=484/2025 were within noise of the
m-outer order. The loop order was not the bottleneck at these sizes.

Reverted to the simpler m-outer kernel (the comment claiming the swap
fixed n=1024 would have been false). The actual n=1024 cause was the
**stride-`n` gather/scatter** reading the column-major `y` — power-of-
two `n` aliased RHS columns into the same cache sets. Flipping the
internal `y` buffer to row-major (contiguous memcpy gather/scatter)
fixed it (ratio 1.2 → 0.33) and ~halved wide-solve time everywhere.
See `dev/research/issue-57-blas3-panel.md` Results and the
`dev/decisions.md` 2026-05-30 entry.

**Lesson.** Diagnose the bottleneck before micro-optimizing the kernel:
the transpose in the gather/scatter dominated, not the GEMM's operand
re-streaming. A loop-order change to the GEMM was wasted motion until
the layout (row-major `y`) was fixed.

## Source Files
```
src/bin/alloc_probe.rs
src/bin/bench_axpy_small.rs
src/bin/bench_dense_multirhs.rs
src/bin/bench_fma_phase3.rs
src/bin/bench_issue8.rs
src/bin/bench_multirhs.rs
src/bin/bench_one_matrix.rs
src/bin/bench_orderings.rs
src/bin/bench_solver_corpus.rs
src/bin/bench_solver_reuse.rs
src/bin/bench_sqd.rs
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
src/bin/diag_fbrain3ls_pivtol_sweep.rs
src/bin/diag_fill_parity.rs
src/bin/diag_fill_tail.rs
src/bin/diag_inertia_mismatch.rs
src/bin/diag_issue50_auto_validate.rs
src/bin/diag_issue50_inventory.rs
src/bin/diag_issue50_large_sparse_scan.rs
src/bin/diag_issue50_numeric_inventory.rs
src/bin/diag_issue50_symbolic.rs
src/bin/diag_leaf_profile.rs
src/bin/diag_max_ncol.rs
src/bin/diag_mc64_cycles.rs
src/bin/diag_mittelmann.rs
src/bin/diag_narx_kernel_gflops.rs
src/bin/diag_near_singular_sweep.rs
src/bin/diag_nemin_amalgamation_panel.rs
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
src/bin/diag_small_sparse_inventory.rs
src/bin/diag_sparse_memory.rs
src/bin/diag_split_tail.rs
src/bin/diag_strategy_compare.rs
src/bin/diag_supernode_cost.rs
src/bin/diag_swopf_w22x2.rs
src/bin/diag_symbolic_stages.rs
src/bin/dump_diff.rs
src/bin/feral_replay.rs
src/bin/feral_time.rs
src/bin/hs85_diag.rs
src/bin/parallel_corpus_parity.rs
src/bin/phase0_cb_on_revalidation.rs
src/bin/polak6_diag.rs
src/bin/policy4_diag.rs
src/bin/probe_acopp30_64.rs
src/bin/probe_cache_sequence.rs
src/bin/probe_cascade_perturb.rs
src/bin/probe_clnlbeam_refine.rs
src/bin/probe_clnlbeam_shape.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_dtoc2_mc64.rs
src/bin/probe_explicit_zeros.rs
src/bin/probe_f01.rs
src/bin/probe_fbrain.rs
src/bin/probe_fma_kernel.rs
src/bin/probe_front_concentration.rs
src/bin/probe_hang_loop.rs
src/bin/probe_intrafront_schur.rs
src/bin/probe_ir_trajectory.rs
src/bin/probe_issue_19.rs
src/bin/probe_issue45_ordering.rs
src/bin/probe_issue45.rs
src/bin/probe_issue46_preprocess.rs
src/bin/probe_issue46_supernode.rs
src/bin/probe_issue46.rs
src/bin/probe_issue49.rs
src/bin/probe_issue54_alpha_shift.rs
src/bin/probe_issue54_cascade.rs
src/bin/probe_issue54_ma57_alpha.rs
src/bin/probe_issue54.rs
src/bin/probe_kkt_replay.rs
src/bin/probe_marine_shape.rs
src/bin/probe_marine_time.rs
src/bin/probe_mc64_spread.rs
src/bin/probe_mc64_synth.rs
src/bin/probe_narx_factor.rs
src/bin/probe_narx_phases.rs
src/bin/probe_panel_attribution.rs
src/bin/probe_pinene_issue38_fix.rs
src/bin/probe_rkt_shape.rs
src/bin/probe_robot_profile.rs
src/bin/probe_rocket_profile.rs
src/bin/probe_rocket_residuals.rs
src/bin/probe_rocket_slow.rs
src/bin/probe_scaling_policy4.rs
src/bin/probe_static_pivot_inertia.rs
src/bin/probe_supernode_widths.rs
src/bin/probe_thomson_hessian.rs
src/bin/probe_value_determinism.rs
src/bin/probe_warm_cascade.rs
src/bin/probe_wide_supernode.rs
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
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/factor_scratch_parity.rs

(truncated from      389 lines to 350 line budget)
