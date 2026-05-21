# FERAL Context (auto-generated)

Generated: 2026-05-21T12:20:56Z

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
c898f71 fix(ci): commit arch-unstable rankdef synth fixtures to fix stress-smoke (#46)
70649e7 ci: generate synth stress fixtures before cargo test
b3e4d3e docs(journal): record dropping the unused from_triplets_strict
672e0c5 docs(context): complete dev/context.md regeneration
9016ef9 docs(session): checkpoint 2026-05-20-03 — fix #46 zero-(2,2)-block cascade
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-8af795e9f581145d)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_gate_boundary_n_16 ... ok
test test_determinism_tiny ... ok
test test_solve_parity_tiny_real_matrix ... ok

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
`parmest` KKT). The analysis-phase `OrderingPreprocess::LdltCompress`
already co-locates every MC64-matched saddle partner at the adjacent
column, so `k+1` is the numerically correct partner the argmax search
was missing.

**Consequence — a soft analysis→numeric coupling.** The numeric kernel
now opportunistically benefits from the analysis phase having placed the
matched partner at `k+1`. This is *not* a hard dependency: tier 2 is
guarded by `a[k,k+1] != 0`, and a `{k,k+1}` candidate that is
numerically unsound still fails the Duff–Reid growth bound and the
SSIDS determinant floor and falls through to the last-resort 1×1
exactly as before. The change widens the 2×2 *search*; it does not
relax the stability gate. With no co-located partner the behavior is
bit-identical to the pre-#46 kernel. Future work that changes how
`LdltCompress` lays out matched pairs should be aware the kernel reads
`k+1` as the preferred saddle partner.

**Evidence.** CHO KKT: factor 11.7 s → 0.20 s (57×), factor-nnz
28.05M → 3.35M, inertia `(21672, 21660, 0)` unchanged. Regression test
`tests/issue_46_saddle_kkt_cascade.rs` verified against a
temporarily-reverted kernel: pre-#46 → 61× fill blowup (test fails),
fixed → 0.83× (test passes).

**References.**
- `src/dense/factor.rs` — `scalar_pivot_step`, the 2×2 partner block.
- `dev/research/kkt-zero-2x2-block-cascade-2026-05-20.md` — corrected
  diagnosis (the original three-agent "ordering failure" diagnosis was
  overturned by ground-truth probes).
- `tests/issue_46_saddle_kkt_cascade.rs` — committed regression test.
- Issue #46 (resolved).

## Recent Tried-and-Rejected
  blowup), max supernode `ncol = 133` (no giant root supernode),
  **20 918 / 21 660 pairs co-located in the same supernode, 20 794 at
  adjacent columns (96.6 %)**. The ordering is fine and the pairs are
  co-located; the cascade is purely a numeric delayed-pivot blowup.
- The MUMPS/SSIDS conclusion "matching-based ordering *is* the fix" was
  incomplete: MUMPS/MA57 also rely on MC64 *scaling* (makes matched
  entries magnitude ≈ 1 so BK's argmax hits the partner). feral's MC64
  scaling is degenerate on saddles (#45) and rejected — feral cannot use
  that mechanism. Worse, on CHO `preprocess=None` actually produced
  *less* fill (21.9M) than `LdltCompress` (28M).

**The actual bug** was the numeric kernel: `scalar_pivot_step`'s 2×2
partner search only considered the magnitude-argmax row `r` and, when
`r` was out-of-front, delayed instead of using the co-located partner
at `k+1`. Fixed there (see `decisions.md` 2026-05-20 #46 entry).

**Lesson.** A convergent multi-agent diagnosis is not evidence — three
agents reading reference solvers agreed on a story that a single probe
on the real matrix overturned in minutes. Probe the actual failing
matrix *before* writing a research note's "recommended fix", not after.

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

(truncated from      381 lines to 350 line budget)
