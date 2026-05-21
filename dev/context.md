# FERAL Context (auto-generated)

Generated: 2026-05-21T15:13:30Z

## Latest Session
File: dev/sessions/2026-05-20-03.md
```
# Session 2026-05-20-03

## Goal

Fix issue #46 — FERAL's LDLᵀ ~160× slower than MA57 on the POUNCE CHO
`parmest` IPM KKT (n=43332): a delayed-pivot cascade on a saddle-point
KKT with a structurally-zero (2,2) block (28M factor-nnz, ~17 s
factor). Post a status comment on #46, then fix it correctly.

## Benchmark Results

No regression. Bench numbers are flat vs the prior session
(2026-05-20-02). The single trivially-worse number — dense medium p90
1.70 → 1.71 (+0.01) — is benchmark noise: an interim run this session
read 1.74, and the final run below reverted to baseline. The #46 fix
is a numeric-kernel change that does not touch ordering, and none of
the bench corpus is a zero-(2,2)-block saddle KKT, so no real effect is
expected.

```
--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.32     <= 2.0     PASS
medium (<500)            152145     1.71     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.57     <= 2.0     PASS
medium (<500)            153560     1.57     <= 3.0     PASS

Top 10 worst factor-ratio vs MUMPS:
MUONSINE_0000  18.96   ACOPR30_0001  12.31   ACOPR14_0211  9.93
ACOPR30_0039    8.14   KIRBY2_0007    7.92   ACOPR14_0128  7.03
ACOPR14_0365    6.93   KIRBY2_0006    6.91   ACOPR14_0187  6.75
ACOPR14_0472    6.66
```

(Prior session 2026-05-20-02: Dense 1.32 / 1.70, Sparse 1.57 / 1.57.)

## Accomplished

**Diagnosed #46 — and overturned the three-agent research diagnosis.**
The session opened with a three-agent research phase (feral source,
MUMPS 5.8.2, SPRAL SSIDS) that converged on "analysis-phase ordering
failure, fix = broaden `pick_ordering_preprocess`'s activation
predicate". Two ground-truth probes on the real CHO KKT refuted every
load-bearing claim:

- `probe_issue46_preprocess` — feral stores only the lower triangle, so
  KKT constraint columns are stored-degree 0/1, *not* high-degree. The
```

## Git Status
```
c990def feat(scaling): value-bounded MC64 scaling cache (B2) + fix External 10× bug
9512c0a perf(profiler): instrument numeric prologue sub-phases (Track B1)
8a481b0 docs(plan): reference filed issue #48 for Track C
60febc5 docs(research): profile the per-factor cost cluster — two mechanisms
c898f71 fix(ci): commit arch-unstable rankdef synth fixtures to fix stress-smoke (#46)
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-8af795e9f581145d)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_gate_boundary_n_16 ... ok
test test_solve_parity_tiny_real_matrix ... ok
test test_determinism_tiny ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 1 test
test src/symbolic/profiler.rs - symbolic::profiler::SymbolicProfiler (line 27) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-20-03.md)


No regression. Bench numbers are flat vs the prior session
(2026-05-20-02). The single trivially-worse number — dense medium p90
1.70 → 1.71 (+0.01) — is benchmark noise: an interim run this session
read 1.74, and the final run below reverted to baseline. The #46 fix
is a numeric-kernel change that does not touch ordering, and none of
the bench corpus is a zero-(2,2)-block saddle KKT, so no real effect is
expected.

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.32     <= 2.0     PASS
medium (<500)            152145     1.71     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.57     <= 2.0     PASS
medium (<500)            153560     1.57     <= 3.0     PASS

Top 10 worst factor-ratio vs MUMPS:
MUONSINE_0000  18.96   ACOPR30_0001  12.31   ACOPR14_0211  9.93
ACOPR30_0039    8.14   KIRBY2_0007    7.92   ACOPR14_0128  7.03
ACOPR14_0365    6.93   KIRBY2_0006    6.91   ACOPR14_0187  6.75
ACOPR14_0472    6.66

(Prior session 2026-05-20-02: Dense 1.32 / 1.70, Sparse 1.57 / 1.57.)

```

## Recent Decisions
   matches from iter 2 on, but the value-bound gate rejects *every*
   warm iter on condition 1 (ratio growth: 1.9e8, 7.8e8, 2.5e10,
   5.6e10 vs budgets 1.2e8 … 3.7e10). The baseline `r0 ≈ 5.8e7`: the
   MC64-scaled KKT is not remotely diagonally dominant. Root cause —
   the KKT (2,2)-block rows have a tiny δ-regularized diagonal (≈1e-8)
   against ≈1 off-diagonals; the off/diag ratio is ≈1/δ, and as the
   IPM drives δ→0 the ratio explodes 1e8→1e10. The value-bound metric
   (diagonal dominance of `D·A·D`) tracks the IPM's regularization
   trajectory, not scaling staleness — it is the wrong instrument, and
   no `GROWTH_FACTOR` recalibration fixes a confounded metric.

3. **Cost share.** pinene_3200's 10 iters total 493.9 s; iters 6-9
   alone are 64.8/77.8/135.7/208.2 s — the per-factor cost-cluster
   blowup, 98 % of wall time. The MC64 Hungarian B2 eliminates is
   ≤6 s across all 10 iters. B2 optimizes a <2 % slice.

**Consequence.** The cache (`Solver::with_mc64_cache`, default on) and
the `External` correctness fix stay. They are harmless: the value-bound
check is O(nnz), well under 1 % of factor cost, and on a genuine hit it
is provably bit-identical to the no-cache path. The B2 *approach* —
caching MC64 across IPM iterations gated by a cheap value proxy — is
recorded as not-yet-viable in `tried-and-rejected.md`. Effort moves to
the iter 6-9 factor-time explosion, where feral's per-factor cost
actually lives.

**References.**
- `dev/journal/2026-05-21-01.org` §18:40, §19:30.
- `dev/plans/mc64-value-bounded-cache.md` — B2 plan.
- `src/scaling/value_bound.rs` — the (confounded) gate.
- `dev/plans/per-factor-cost-cluster.md` — the cluster the pivot targets.

## Recent Tried-and-Rejected

- Even with a perfect gate, B2 targets <2 % of the cost. pinene_3200's
  10 iters total 493.9 s; iters 6-9 are 64.8/77.8/135.7/208.2 s (the
  cost-cluster blowup, 98 %). The MC64 Hungarian is ≤6 s total.

- The named target rocket_12800 cannot even exhibit a hit: its 2-iter
  dump changes pattern between iters (332793→435190 nnz).

**What was kept.** The cache wiring (`Solver::with_mc64_cache`),
`src/scaling/value_bound.rs`, and — separately — the `External`
scaling correctness fix B2 surfaced (see `decisions.md` 2026-05-21).
All correct and tested; the *approach* of a cheap value-proxy gate
for cross-iteration MC64 reuse is what is rejected.

**Lesson.** Validate the cost model before building the optimization.
B2 assumed "MC64 Hungarian reruns every IPM iter and dominates" — true
for rocket_12800's iter-0 profile, false for pinene's actual 10-iter
trajectory where the delayed-pivot blowup dwarfs everything. A
per-factor profile of the *named target's full iteration sequence*,
not a single iteration, should precede the plan.

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
src/bin/diag_leaf_profile.rs
src/bin/diag_max_ncol.rs
src/bin/diag_mc64_cycles.rs
src/bin/diag_mittelmann.rs
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
src/bin/polak6_diag.rs
src/bin/policy4_diag.rs
src/bin/probe_acopp30_64.rs
src/bin/probe_cascade_perturb.rs
src/bin/probe_clnlbeam_refine.rs
src/bin/probe_clnlbeam_shape.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_dtoc2_mc64.rs
src/bin/probe_f01.rs
src/bin/probe_fbrain.rs
src/bin/probe_fma_kernel.rs
src/bin/probe_ir_trajectory.rs
src/bin/probe_issue_19.rs
src/bin/probe_issue45_ordering.rs
src/bin/probe_issue45.rs
src/bin/probe_issue46_preprocess.rs
src/bin/probe_issue46_supernode.rs
src/bin/probe_issue46.rs
src/bin/probe_kkt_replay.rs
src/bin/probe_marine_shape.rs
src/bin/probe_marine_time.rs
src/bin/probe_mc64_spread.rs
src/bin/probe_mc64_synth.rs
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
tests/factor_workspace_parity.rs
tests/fma_opt_in_roundtrip.rs
tests/growth_flag.rs

(truncated from      382 lines to 350 line budget)
