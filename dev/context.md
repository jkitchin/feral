# FERAL Context (auto-generated)

Generated: 2026-05-30T17:03:49Z

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
401486e docs(book): add Scaling and Fill-reducing ordering chapters
0fcb010 docs(book): complete the mdBook — fix stale examples, add sparse/multi-RHS/Python (#57)
8908d32 docs(python): add example notebooks; motivate multi-RHS perf demo (#57)
9c2c716 perf(solve): BLAS-3 panel kernels + row-major y for wide multi-RHS (#57 fix #2)
80348f9 perf(solve): row-major working buffer for multi-RHS sparse solve (#57)
```

## Test Status
```
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
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

test result: ok. 317 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 0.40s

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
cascade; back differs by float reassociation (~κ·eps), inside the
1e-12 parity gate (observed ≤ 1.6e-15). The dual path isolates the new
kernels from the single-RHS and small-`nrhs` paths.

**Rejected alternatives.**
- *Keep `y` column-major and only tune the GEMM* — leaves the
  stride-`n` gather/scatter regression on power-of-two `n` and caps
  every size; the GEMM was not the bottleneck.
- *GEMM loop reorder (c-block outer) alone* — no measurable effect;
  the bottleneck was the transpose, not B re-streaming at these sizes.
- *Global BLAS-3 for all `nrhs`* — would route the IPM hot path (small
  `nrhs`, bit-identical today) onto the non-bit-identical back-sub for
  no benefit; threshold keeps it off.

**Measured (idle, `bench_multirhs`, 2-D Laplacians, nrhs ∈ {64,256}).**
Per-RHS batched/looped ratio: n=484 ~0.18–0.24 (~4–5×), n=1024
~0.32–0.34 (~3×), n=2025 ~0.17–0.23 (~5–6×). Lib tests 317 pass;
multi-RHS parity 10/10 at ≤ 1.6e-15.

**Deferred.** Packing the column-major `L` panel into a contiguous
buffer (BLIS-style) to remove the strided `L` access inside the GEMM —
the next lever, most relevant to power-of-two front dimensions
(n=1024). Not pursued until a workload demands it.

**References.**
- `dev/research/issue-57-blas3-panel.md` — design, bit-exactness
  analysis, and the Results section with the regression diagnosis.
- `dev/research/issue-57-multirhs-row-major.md` — fix #1 (row-major `w`).
- `dev/journal/2026-05-30-01.org` — real-time work log.
- Issue #57 — original report (column-major layout, 5–10× target).

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
src/scaling/hungarian.rs

(truncated from      415 lines to 350 line budget)
