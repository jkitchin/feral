# FERAL Context (auto-generated)

Generated: 2026-05-17T16:17:25Z

## Latest Session
File: dev/sessions/2026-05-17-01.md
```
# Session 2026-05-17-01

## Goal

Localize and remediate the per-iter factor cost gap vs MA57 on the
ipopt-feral Mittelmann sweep. Going in, the suspect was MC64 cost
(98% of warm wall on `probe_rocket_slow.rs` against the pounce
corpus dumps).

## Accomplished

### Investigation pivot: MC64 is rocket-specific, cascade is general

1. **MC64 live trace.** Added `MC64_RECOMPUTE_COUNT` +
   `FERAL_MC64_TRACE=1` to `src/scaling/mc64.rs`. Live ipopt-feral
   run on rocket_12800: 30 MC64 recomputes (one per warm factor —
   confirms #38 `db20166` invalidation), MC64 = **55%** of total
   Ipopt wall (not 98% as the dumped corpus suggested). Per-call
   wall ranges 14–2482 ms.

2. **Per-supernode profile on robot_1600** (`probe_robot_profile.rs`).
   Found MC64 is only **2.6%** of robot_1600 warm wall (350 ms of
   13.7 s). The 32× MA57 gap there lives elsewhere.

3. **Factor trace** (`FERAL_FACTOR_TRACE=1` in `src/capi.rs`).
   Per-factor wall + `sum_delayed` + `max_delayed` exposed
   smoking-gun: robot_1600 late-IPM factors have
   `sum_delayed = 30k–60k` on n=24000 — classic delayed-pivot
   cascade (the same mechanism issue #38 fixed for pinene_3200).

4. **Auto-CB was dead code.** `Solver::with_auto_cascade_break(β)`
   (the warm cascade-break auto-arm from #38) was never wired into
   the capi, so ipopt-feral never benefited. Wired it as the
   default with `FERAL_AUTO_CB_BETA` env (default 0.05).

Spot-check on 10 Mittelmann problems (`benchmarks/mittelmann_ipopt`):

| problem        | CB=off    | auto-CB   | Δ        |
|----------------|-----------|-----------|----------|
| robot_1600     | 13.81 s   |  3.58 s   |  -74 %   |
| marine_1600    | 470.87 s  | 58.13 s   |  -88 %   |
| clnlbeam       | 361.26 s  | 47.73 s   |  -87 %   |
| corkscrw       | 53.89 s   | 15.43 s   |  -71 %   |
| camshape_6400  |  6.67 s   |  2.09 s   |  -69 %   |
| dtoc2          | timeout   | 78.0 s    | rescued  |
| bearing_400    |  6.21 s   |  4.86 s   |  -22 %   |
| rocket_12800   |  8.73 s   | 10.55 s   |  +21 %   |
| arki0003       |  3.10 s   |  3.17 s   |   ~0 %   |
| pinene_3200    | timeout   | timeout   | needs CB=on |

```

## Git Status
```
ad48ab2 chore(dev-tools): add Makefile wrapper + pounce KKT replay/time probes
f1da854 feat(capi): auto-arm cascade-break by default to rescue delayed-pivot cascades
a28cfec fix(scaling): require dense arrow head, not just diag-only mass, for MC64 routing (#68)
0d874f5 feat(numeric): MA57-style static-pivot perturbation knob (#38)
1c36f2d feat(dense): closed-form 2x2 eigenvalue inertia classifier (#38)
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-f15e862c5b10279a)

running 5 tests
test test_gate_tiny_sparse_in ... ok
test test_gate_just_outside_n_tiny ... ok
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
(skipped: pass --with-bench to re-run; no session checkpoint with bench)
```

## Recent Decisions
Numbers come from `probe_fma_kernel` on M-series (commit ee46d72)
and ubuntu-latest x86_64 (CI run 25971444759 via commit f1f9894).
The aarch64 regression is intrinsic to the kernel body — `mul + sub`
exposes more ILP than `mul_add` on NEON pipes — while x86 V3
(AVX2+FMA) gets the textbook 1.5x speedup.

Two paths were considered and rejected:

1. **Gate `fma = true` to `cfg(target_arch = "x86_64")` in
   `BunchKaufmanParams`.** Rejected because it would silently
   override an explicit caller opt-in; downstream tooling that uses
   the flag for parity/regression bisection (e.g. probe binaries)
   would lose the ability to time the FMA path on aarch64 even when
   that's the explicit measurement goal.
2. **Remove the FMA path.** Rejected because x86 callers do get the
   1.5x and the path's correctness is well-tested
   (`schur_kernel.rs` has bit-exact rank-1 reference tests on both
   variants).

Production default `fma = false` already gives every arch its best
kernel, so no runtime change is needed. Callers building on x86 can
opt in via `Solver::new().with_fma(true)`.

References:
- `dev/research/fma-kernel-aarch64-regression-2026-05-16.md` (probe
  methodology + aarch64 numbers).
- `dev/research/fma-kernel-opt-in.md` (original opt-in design).
- Probe: `src/bin/probe_fma_kernel.rs`.
- Commits: ee46d72 (probe + note), f1f9894 (CI wiring), this entry.
- GH: #35.

## Recent Tried-and-Rejected

**Disposition.** Replaced with an in-module unit test
(`numeric::solver::tests::mc64_cache_invalidated_after_factor_issue_38`)
that inspects `last_symbolic.cached_mc64` directly and asserts it is
`None` after one `factor()` call. The pub(crate) field is only accessible
from `super::*` so the test had to move from `tests/` to the in-module
`#[cfg(test)]` block. Verified the unit test fails when the fix is
removed (panics on the assertion) and passes when restored.

**Lesson.** Behavioural tests for scaling-related bugs need either (a) a
matrix large enough to expose BK pivot-threshold sensitivity, which means
shipping corpus data, or (b) a direct-field assertion on the cache state.
For one-shot caches that should be cleared per call, (b) is cheaper and
more targeted than (a).

**Evidence.** See `dev/research/mc64-cache-staleness-2026-05-16.md` and
the diagnostic tables in `dev/journal/2026-05-16-30.org`. Verification
procedure (toggle fix, run `cargo test --release --lib
numeric::solver::tests::mc64_cache_invalidated_after_factor_issue_38`,
observe pass/fail) reproducible from HEAD.

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
src/bin/probe_fma_kernel.rs
src/bin/probe_ir_trajectory.rs
src/bin/probe_issue_19.rs
src/bin/probe_kkt_replay.rs
src/bin/probe_marine_shape.rs
src/bin/probe_marine_time.rs
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
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
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
tests/sqd_fast_path.rs
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
