# FERAL Context (auto-generated)

Generated: 2026-05-16T21:01:59Z

## Latest Session
File: dev/sessions/2026-05-16-30.md
```
# Session 2026-05-16-30 — Issue #37 reframe via #38: stale MC64 cache root-cause

## Goal

User assertion: #37 (pinene_3200 CB=off regression) "relies on a workaround
that should not be needed". Investigate whether the workaround
(`Solver::with_cascade_break(0.5)` builder, or pounce-side
`feral_cascade_break=yes` ipopt option) can be eliminated by a feral-side
default fix.

Mid-session redirect: user pointed at issue #38 (sister case to #37 on
`rocket_12800` where BOTH CB modes fail), which reframes the question from
"flip the right default" to "fix the underlying mechanism". Pivoted to
chasing #38's Failure A (warm-Solver-state slowdown).

## Accomplished

- **AutoRace ordering hypothesis killed by diag data.** Ran
  `diag_pinene_amd` on `pinene_3200_{0008,0009}` + `robot_1600_0003` to
  test whether switching `Solver::new()`'s default from
  `OrderingMethod::Auto` to `OrderingMethod::AutoRace` would close #37 by
  preferring AMD on pinene. Result: under CB=off, AMD on pinene_3200 is
  **~10× WORSE** than MetisND (917s and 1055s vs MetisND's 88s, with
  ~13.5M delayed pivots), not 4.5× better as the c92cafe commit message
  claimed. c92cafe's benchmark predates 585d739 (cascade-break became
  opt-in), so it was measuring AMD-with-CB-armed. Robot_1600_0003 was
  clean (both AMD and MetisND produce neg=9601 matching the MUMPS oracle,
  AMD 1.4× faster), so the regression guard would not have fired.
  Conclusion: ordering choice cannot fix #37; the mechanism is the issue.

- **Reproduced #38 Failure A** (warm-Solver-state slowdown) on the 18
  `/tmp/rkt_*.bin` rocket dumps. Default `Solver::new()` (CB=off,
  parallel=on, scaling=Auto):

  | call | factor   | neg   | comment            |
  |-----:|---------:|------:|--------------------|
  | #000 |  0.320s  | 38400 | cold (incl. symbolic) |
  | #009 |  0.023s  | 38400 | warm, stable        |
  | #010 |  0.022s  | 38395 | inertia drift starts |
  | #014 |  0.024s  | 38145 | drift continuing    |
  | #015 |  0.056s  | 37513 | cost begins to climb |
  | #016 |  2.093s  | 35900 | cost explodes       |
  | #017 | 43.216s  | 31720 | runaway             |

  Matches issue body's qualitative description.

- **NEW FINDING (bigger than #38's framing): warm Solver under default
  `ScalingStrategy::Auto` produces SILENTLY WRONG INERTIA on the same
  matrix that a fresh Solver factors correctly.** Side-by-side fresh
  vs warm on calls #014..#017:
```

## Git Status
```
87e6be3 chore(session): 2026-05-16-29 -- #13 merge + #18 residual gates + #11 close
40f687c docs(#11): reject SmallLeafBatch::On default flip on post-SIMD+APP re-eval
0b60d9a perf(dense): in-place scratch-pooled scalar fallback (#13)
ea98f98 test(#18): add refined-solve residual gate (1e-10) to NARX_CFy corpus
a520e73 docs: close #35 — FMA per-arch asymmetry confirmed, defaults stay
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-6bd386c5815a5f71)

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
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-16-30.md)


No `cargo run --bin bench --release` this session — pure investigation,
no code changes to the solver. The numbers above (rocket replay sweeps)
are the relevant measurements. No regression run because no change
landed.

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
So the regression guard would not have caught this — the failure
is specific to pinene's elimination tree shape, not to AMD itself.

**Disposition.**
- `Solver::new()` retains `OrderingMethod::Auto` as the default.
- The c92cafe claim about AMD speedup on pinene is now historical
  (CB=on regime only) and should not be cited as motivation for
  default-ordering changes.
- Closing #37 requires fixing the underlying CB-mechanism gap, not
  the ordering choice. The follow-up investigation (see
  `dev/sessions/2026-05-16-30.md` and `dev/journal/2026-05-16-30.org`)
  redirects to issue #38 and surfaces a separate silent-correctness
  bug — stale MC64 cache producing wrong inertia on warm IPM
  re-factors — that is the more likely root cause for both #37 and #38.

**Evidence.** Diag outputs at
`/private/tmp/claude-501/-Users-jkitchin-projects-feral/<session>/tasks/{br9xq3zub,bh8zlfol7,b0ozjbi4h}.output`
(retained for the session; reproducible via
`cargo run --release --bin diag_pinene_amd -- pinene_3200_0009`,
`pinene_3200_0008`, and `robot_1600_0003` from a checkout at HEAD).

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
