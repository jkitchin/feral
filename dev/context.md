# FERAL Context (auto-generated)

Generated: 2026-05-20T22:51:17Z

## Latest Session
File: dev/sessions/2026-05-20-02.md
```
# Session 2026-05-20-02

## Goal
Fix the defect underlying GitHub issue #45: the CHO `parmest` KKT factors
successfully with correct inertia but the back-solve returns garbage
(residual ~7e11) while `factor()` silently reports `Success`. Issue #46
(LDLᵀ ~160× slower than MA57) was explicitly scoped OUT as a separate,
larger performance effort — "correctness before performance, always".

## Accomplished

### Root cause isolated (issue #45)
A long diagnostic chain (full trace in `dev/journal/2026-05-20-02.org`)
overturned two earlier wrong hypotheses (duplicate-coordinate doubling;
ordering needs a complete diagonal) and converged on the real defect:

- `ScalingStrategy::Auto` applies an MC64 symmetric scaling vector whose
  own spread `max|s| / min|s|` exceeds `1/EPS ≈ 4.5e15`. On the CHO KKT
  MC64 produced spread ≈ **3e82** (`min 2.89e-42 .. max 8.88e40`).
- Such a scaling is degenerate to working precision: `D = diag(s)` is
  singular, `D·A·D` underflows, Bunch-Kaufman force-accepts exact-`0.0`
  1×1 pivots (`min pivot mag = 0.00e0`), the solve is garbage, and
  `factor()` still returns `Success` with correct inertia.
- The existing issue-#24 guard missed it: `compute_scaling_auto_with_cache`
  had a fast-path `if raw_diag_range(matrix) >= RAW_GUARD(1e6) { return mc64 }`
  that committed to MC64 *without* inspecting the produced vector. The CHO
  KKT is ill-conditioned (raw range ≥ 1e6) so it took that fast-path and
  the `mc_off` catastrophe diagnostic was never reached.
- Diagnosis note: `dev/research/kkt-mc64-scaling-blowup-2026-05-20.md`.

### Fix — MC64 catastrophic-spread guard
`src/scaling/mod.rs`, `compute_scaling_auto_with_cache`:
- New `const MC64_SPREAD_GUARD: f64 = 1.0 / f64::EPSILON` (≈ 4.5036e15).
  Corpus max MC64 spread is 3.27e15 (ssine) — a 67-order gap below the
  CHO catastrophe; the guard catches CHO and clears the whole corpus.
- New `Mc64FallbackReason::Mc64ScalingDegenerate` variant.
- The MC64 branch now computes `(mc_vec, mc_info)` **once**, then
  `if scaling_spread(&mc_vec) > MC64_SPREAD_GUARD` returns the
  already-computed InfNorm vector tagged `Mc64ScalingDegenerate`.
  Placed **before** the `raw_diag_range` fast-path so it fires regardless
  of raw conditioning — that fast-path was the #45 bypass.
- `src/bin/bench_one_matrix.rs`: exhaustive `Mc64FallbackReason` match
  extended with `"mc64_scaling_degenerate"`.

### Verification
- Real CHO KKT via `probe_issue45_ordering` (added an `Auto` row to its
  scaling loop): `completed Auto` went from **relres 7.149e11 → 2.455e-8**,
  inertia (21672,21660,0) unchanged. `Auto` now == `InfNorm` on the
  diagonal-completed CHO KKT (the POUNCE live-KKT form that triggers #45).
  #45 closed.
```

## Git Status
```
eb77966 test(issue46): ground-truth probes for the zero-(2,2)-block cascade
070840b fix(ldlt): break the zero-(2,2)-block KKT delayed-pivot cascade (#46)
d432086 docs(session): checkpoint 2026-05-20-02 — issue #45 MC64 spread guard
6bda61d test(probe): add issue #45/#46 diagnostic and oracle probes
b017beb fix(scaling): guard against catastrophic MC64 scaling spread (#45)
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
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-20-02.md)

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
MUONSINE_0000                 1537        10486          376      27.89
KIRBY2_0007                    458          901          119       7.57
KIRBY2_0006                    458          886          127       6.98
KIRBY2_0008                    458          727          122       5.96
KIRBY2_0009                    458          673          128       5.26
KIRBY2_0011                    458          598          120       4.98
KIRBY2_0010                    458          648          133       4.87
CRESC132_0000                 5314        59673        12266       4.86
KIRBY2_0012                    458          453          118       3.84
GROUPING_0299                  225          396          117       3.38

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.32     <= 2.0     PASS
medium (<500)            152145     1.70     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.57     <= 2.0     PASS
medium (<500)            153560     1.57     <= 3.0     PASS
No regression vs the previous session — both phase exit partitions still
PASS. The fix touches only the `ScalingStrategy::Auto` router on matrices
whose MC64 spread exceeds `1/EPS`; empirically none in the parity corpus,
so corpus benchmark numbers are unchanged.

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
