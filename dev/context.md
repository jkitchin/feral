# FERAL Context (auto-generated)

Generated: 2026-05-16T21:46:35Z

## Latest Session
File: dev/sessions/2026-05-16-31.md
```
# Session 2026-05-16-31 — Land the #38 MC64-cache-staleness fix

## Goal

Land the fix identified in session-30 (stale MC64 cache reused across
warm `Solver::factor` calls produces silently wrong inertia and
explodes cost on real arrow-KKTs). Per session-30's "next session
should":

- Write research note `dev/research/mc64-cache-staleness-2026-05-16.md`
- Land the fix + regression test
- Re-test #37 (pinene_3200) under the fix
- Post finding-comment on #38

## Accomplished

- **Fix landed: db20166** `fix(#38): invalidate stale MC64 cache
  between factor() calls`. One-shot cache invalidation at the end of
  `Solver::factor`: clears `last_symbolic.cached_mc64` after every
  numeric call. The cache stays valid for the first numeric call
  after symbolic (values match by construction); subsequent calls
  fall through to a fresh `mc64::compute_symmetric(matrix)` against
  current values. Cost: one extra MC64 (~100–200 ms on n ≈ 1e5) per
  warm refactor when scaling resolves to `Mc64Symmetric`.

- **Regression test**
  `numeric::solver::tests::mc64_cache_invalidated_after_factor_issue_38`.
  Inspects `last_symbolic.cached_mc64` directly after one `factor()`
  call and asserts it is `None`. Field-inspection rather than
  behavioural: Sylvester's law keeps inertia invariant under any
  symmetric scaling on well-conditioned small matrices, so the
  downstream wrong-inertia symptom only manifests on large arrow-KKTs
  — a 4×4 reproducer is insensitive. Verified the test fails when the
  fix is removed (panics on the assertion) and passes when restored.

- **Test suite green.** `cargo test --release` exit 0;
  `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings`
  pass via pre-commit hooks.

- **#38 finding-comment posted**:
  <https://github.com/jkitchin/feral/issues/38#issuecomment-4468206937>.
  Reframes the issue as silent inertia corruption (not just "warm
  slowdown") with the four diagnostic tables (warm replay,
  warm-vs-fresh, PAR=0 localisation, InfNorm control) from
  session-30. Closes Failure A; leaves issue open for Failure B
  (CB=on iter-4 disagreement vs MA57) which is a separate
  investigation.

- **Research note written**:
  `dev/research/mc64-cache-staleness-2026-05-16.md`. Covers the
```

## Git Status
```
db20166 fix(#38): invalidate stale MC64 cache between factor() calls
c0cceea chore(session): 2026-05-16-30 -- #37 -> #38 stale MC64 cache finding
87e6be3 chore(session): 2026-05-16-29 -- #13 merge + #18 residual gates + #11 close
40f687c docs(#11): reject SmallLeafBatch::On default flip on post-SIMD+APP re-eval
0b60d9a perf(dense): in-place scratch-pooled scalar fallback (#13)
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-86a6a76d01c16bc9)

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
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-16-31.md)


`cargo run --bin bench --release` — both Phase 2.8.1 partition gates
pass at p90, no regressions:

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.37     <= 2.0     PASS
medium (<500)            152145     1.87     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.74     <= 2.0     PASS
medium (<500)            153560     1.75     <= 3.0     PASS

Top-10 worst per-matrix ratios unchanged from session-30 baseline
(MUONSINE_0000 30.6×, KIRBY2_*  ~6-8×, SWOPF_* ~6×). The fix only
touches a single boolean assignment in `Solver::factor`; bench
matrices factor cold so the cache path isn't exercised here.

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
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_fma_kernel.rs
src/bin/probe_ir_trajectory.rs
src/bin/probe_issue_19.rs
src/bin/probe_panel_attribution.rs
src/bin/probe_pinene_issue38_fix.rs
src/bin/probe_scaling_policy4.rs
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
