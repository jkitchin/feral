# FERAL Context (auto-generated)

Generated: 2026-05-30T17:53:15Z

## Latest Session
File: dev/sessions/2026-05-30-01.md
```
# Session 2026-05-30-01

## Goal

Commit the in-progress issue #57 fix #1 (row-major `w` for the multi-RHS
solve), then **finish the BLAS-3 implementation** (issue #57 fix #2) to
reach the 5–10× per-RHS regime. Then, on request: surface the work in
the Python interface and a motivating performance notebook, and do a
completeness pass over the mdBook (including new Scaling and Ordering
chapters with verified citations).

## Benchmark Results

**Unfavorable note (per the hard rule):** the **factor** benchmark's
dense p90 ratios drifted up vs the previous session — small-frontal
1.29 → 1.34, medium 1.67 → 1.74. This session changed **only the solve
path** (`solve_sparse_core_many_into`), not factorization, so this is
machine/measurement noise, not a regression. All gates still PASS.

`cargo run --bin bench --release` (factor ratio vs MUMPS):

```
--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.34     <= 2.0     PASS
medium (<500)            152145     1.74     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.50     <= 2.0     PASS
medium (<500)            153560     1.50     <= 3.0     PASS

Top worst factor-ratio vs MUMPS: KIRBY2_0007 n=458 ratio 8.76,
CRESC132_0000 n=5314 ratio 6.63, MUONSINE_0000 n=1537 ratio 5.63.
```

**The session's actual performance story** is the multi-RHS solve
(`cargo run --release --bin bench_multirhs`, 2-D Laplacians, idle
machine, per-RHS batched/looped ratio at nrhs ∈ {64, 256}):

| n    | ratio        | speedup | before fix #2 (row-major w only) |
|------|--------------|---------|----------------------------------|
| 484  | 0.18–0.24    | ~4–5×   | ~0.34                            |
| 1024 | 0.32–0.34    | ~3×     | ~1.0–1.2 (REGRESSION)            |
| 2025 | 0.17–0.23    | ~5–6×   | ~0.35                            |

Parity vs single-RHS oracle: `max|many − single| ≤ 1.6e-15` (machine
precision; 1e-12 gate, tolerance untouched).

## Accomplished
```

## Git Status
```
92b8150 bench: add dense-Hessian multi-RHS probe (#58 repro)
2369fce fix(solve): batched iterative refinement for wide multi-RHS (#58)
f8f324b session: 2026-05-30-01 checkpoint (#57 multi-RHS BLAS-3 + docs)
401486e docs(book): add Scaling and Fill-reducing ordering chapters
0fcb010 docs(book): complete the mdBook — fix stale examples, add sparse/multi-RHS/Python (#57)
```

## Test Status
```
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test scaling::tests::auto_solves_below_guard_matrix_correctly ... ok
test scaling::tests::auto_falls_back_to_infnorm_on_mss1_0009 ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_resolves_to_amd_when_bisection_degenerates ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok

test result: ok. 317 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 0.41s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-30-01.md)


**Unfavorable note (per the hard rule):** the **factor** benchmark's
dense p90 ratios drifted up vs the previous session — small-frontal
1.29 → 1.34, medium 1.67 → 1.74. This session changed **only the solve
path** (`solve_sparse_core_many_into`), not factorization, so this is
machine/measurement noise, not a regression. All gates still PASS.

`cargo run --bin bench --release` (factor ratio vs MUMPS):

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.34     <= 2.0     PASS
medium (<500)            152145     1.74     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.50     <= 2.0     PASS
medium (<500)            153560     1.50     <= 3.0     PASS

Top worst factor-ratio vs MUMPS: KIRBY2_0007 n=458 ratio 8.76,
CRESC132_0000 n=5314 ratio 6.63, MUONSINE_0000 n=1537 ratio 5.63.

**The session's actual performance story** is the multi-RHS solve
(`cargo run --release --bin bench_multirhs`, 2-D Laplacians, idle
machine, per-RHS batched/looped ratio at nrhs ∈ {64, 256}):

| n    | ratio        | speedup | before fix #2 (row-major w only) |
|------|--------------|---------|----------------------------------|
| 484  | 0.18–0.24    | ~4–5×   | ~0.34                            |
| 1024 | 0.32–0.34    | ~3×     | ~1.0–1.2 (REGRESSION)            |
| 2025 | 0.17–0.23    | ~5–6×   | ~0.35                            |

Parity vs single-RHS oracle: `max|many − single| ≤ 1.6e-15` (machine
precision; 1e-12 gate, tolerance untouched).

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
src/bin/probe_hang_loop.rs
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

(truncated from      416 lines to 350 line budget)
