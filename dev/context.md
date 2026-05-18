# FERAL Context (auto-generated)

Generated: 2026-05-17T22:01:57Z

## Latest Session
File: dev/sessions/2026-05-17-02.md
```
# Session 2026-05-17-02

Continuation of 2026-05-17-01. After auto-CB + scaling fix landed
feral at MA57 parity on the Mittelmann panel, this session shipped
v0.4.0 — cleanly — to PyPI and Crates.io, fixed the wheel pipeline,
and tightened the parity-test gate to match the project's actual
correctness contract.

## Goal

1. Re-audit the README "Known limitations" section against current
   parity-test status.
2. Run the 13 `#[ignore]`'d parity tests on current `main` and
   un-ignore any that close cold.
3. Switch the parity gate from MUMPS-only to oracle-consensus
   (CLAUDE.md correctness contract).
4. Document the Python interface in the README.
5. Publish `feral-solver==0.4.0` to PyPI.
6. Fix whatever the v0.4.0 publish run breaks.

## Accomplished

### Parity — un-ignored 7 panel matrices, filed one real regression

Reran the 13 `#[ignore]`'d parity tests cold against current
`main`. Two passed under the MUMPS-only gate (CERI651DLS_0618 and
ROSZMAN1_0241) and were un-ignored directly.

Switched `tests/parity.rs` from MUMPS-only to oracle-consensus:
feral inertia must match **either** MUMPS 5.8.2 **or** SPRAL SSIDS.
This is verbatim the CLAUDE.md contract:

> Inertia must be exactly correct on non-singular matrices. On
> matrices where the canonical Fortran direct solvers (MUMPS 5.8.2
> and SPRAL SSIDS) disagree on inertia, feral must agree with at
> least one of them.

Updated the generator (`examples/select_parity_panel.rs`) to emit
the new gate and a `try_read_oracle()` helper. Hand-edited
`tests/parity.rs` to match (the example was non-runnable —
`autoexamples = false` in `Cargo.toml`).

Result: **20 passed / 0 failed / 6 ignored** (was 13/0/13 cold).
Newly-passing under oracle-consensus: ACOPP14_{0001,0003},
ACOPP30_{0000,0001}, CERI651CLS_0486.

Genuine outliers still ignored:

| matrix             | reason                                                                |
|--------------------|-----------------------------------------------------------------------|
```

## Git Status
```
2442d1f ci(python-wheels): switch wheel matrix to maturin-action
07d385e ci(python-wheels): bootstrap rustup in manylinux + drop uv frontend
b0c7521 chore(python): bump feral-solver to 0.4.0 to match Rust crate
48c5d13 docs(readme): add Python bindings section
c966c61 test(parity): oracle-consensus gate matches CLAUDE.md correctness contract
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-cdac38fde24c2943)

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
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-17-02.md)


`cargo run --bin bench --release` on `main` at `2442d1f` (output
truncated to the perf summary — the dense small-matrix table that
prints before this is unchanged from the prior session and verbose):

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.43       0.30       1.74       3.09      36.35
solve/MUMPS        153560       0.08       0.08       0.16       0.87       3.57
factor/SSIDS       154500       0.04       0.03       0.36       0.92       7.79
solve/SSIDS        154500       0.97       1.00       3.00     10.40     130.00
nnzL/MUMPS         153560       0.62       0.58       0.77       4.50      23.11
nnzL/SSIDS         154500       0.90       1.00       1.00       4.50       5.00

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.37     <= 2.0     PASS
medium (<500)            152145     1.78     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.74     <= 2.0     PASS
medium (<500)            153560     1.74     <= 3.0     PASS

All four 2.8.1 exit-partition gates still PASS. No work this session
touched numeric kernels, so no perf delta is expected and none
appears.

Worst sparse factor outlier vs MUMPS is now MUONSINE_0000 at
36.35× (n=1537). The CRESC100 / GAUSS2 cluster at ~42× shown
earlier in the run is the small-frontal warm-bench table, not the
panel-bench partition that gates 2.8.1.

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
   (`schur_kernel.rs` has bit-exact rasrc/bin/alloc_probe.rs
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
src/bin/diag_cascade_defaudirect equivalent. The `test` job (linux × py3.10/3.12/3.13) and the
`smoke-test` job (linux wheel + `uv pip install` + quickstart.py)
still gate the release, so coverage for the platforms that matter
most for the release gate is intact. Per-platform wheel-pytest could
be added back as a separate job that downloads the wheel artifact
and runs pytest against it, but for v0.4.0 it was not worth the
churn.

**Lesson.** cibuildwheel is the wrong tool when the Python crate has
a sibling-path Rust dependency. The package-dir-only copy semantics
fundamentally cannot see the parent. Reach for `maturin-action`
first when the layout is a Rust workspace with a Python crate
inside it.

**Evidence.** Run 26002981755 (initial failure, 4/4 wheels red).
Run 26003051776 (after fix 1, 3/4 still red — log of job
76429732912 contains the cargo manifest error verbatim). Run
26003260115 (after switching to maturin-action, 4/4 wheels green).
Run 26003542088 (re-cut v0.4.0 release event, full pipeline green
through PyPI publish).

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
src/bin/prtests/amf_corpus_oracle.rs
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
numeric/solver.rs
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


(truncated from      381 lines to 350 line budget)
(truncated from      381 lines to 350 line budget)
