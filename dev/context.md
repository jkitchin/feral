# FERAL Context (auto-generated)

Generated: 2026-05-14T20:42:04Z

## Latest Session
File: dev/sessions/2026-05-14-02.md
```
# Session 2026-05-14-02

## Bench vs. prior session

Phase 2.8.1 corpus bench not re-run this session — work was on
external benchmark infrastructure, not on the multifrontal hot
path. The most recent numbers are from session `2026-05-14-01`
(both dense and sparse Phase 2.8.1 gates PASS, p90 ratios in
[1.32, 1.70] dense / [1.56, 1.56] sparse). No code in `src/`
that the corpus bench exercises changed this session.

## Goal

User asked first about the `assemble-context.sh` hang from the
prior session, then requested a synthetic-matrix scaling
benchmark comparing feral, MUMPS, and MA57, with plots and an
org-mode write-up. Scope:

1. Diagnose and fix the `assemble-context.sh` slowness (the
   "hang" was actually the full corpus bench running for ~3.5
   minutes).
2. Scaffold a scaling benchmark at `external_benchmarks/scaling/`
   covering four matrix families (dense SI, banded SPD, 2D
   Laplacian, saddle-point KKT) across multiple sizes.
3. Generate plots (log-log factor time, ratio vs MUMPS, residual
   quality, four-panel overview).
4. Produce an org-mode report distilling the actionable
   improvement targets vs. fundamental constraints visible in
   the data.

## Accomplished

### 1. `dev/assemble-context.sh` — bench is now opt-in

- Added `--with-bench` flag (default: skip). Default mode
  sources the `## Benchmark Results` section from the latest
  dated session checkpoint instead of re-running
  `cargo run --bin bench --release`.
- Fixed a pre-existing glob bug: `ls dev/sessions/*.md | sort |
  tail -1` returned `phase-2-baseline.md` because it sorts
  after `2026-...` lexicographically. The glob is now
  `dev/sessions/[0-9]*.md` which restricts to dated sessions.
- Result: refresh time drops from **3m32s → 3.25s** for the
  default case. `--with-bench` recovers the original behaviour.
- Verified end-to-end: `./dev/assemble-context.sh` writes a
  352-line `dev/context.md` with the bench section correctly
  sourced from `dev/sessions/2026-05-14-01.md`.

### 2. Scaling benchmark — `external_benchmarks/scaling/`

```

## Git Status
```
7b2a061 chore(session): 2026-05-14-01 -- F2.3 diagnostics + cond-parity audit
c6eee1f feat(F2.3): RefinementDiagnostics for iterative refinement
aca9f3d fix(issue-15): skip cascade-arm gate test when corpus missing
b19b94c journal(2026-05-13-04): cascade-break ratio distribution + symbolic-arm gate
c7d2048 diag(issue-15): per-family probes for cascade-break gate calibration
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-e512e6534a50691d)

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
**Why the collapse:**

1. The C ABI is small (7 functions, ~250 lines) and tied
   1:1 to types already public in the core crate
   (`CscMatrix`, `Solver`, `FactorStatus`). A separate
   workspace member would have re-exported these or
   wrapped them with no added isolation.
2. Single `cargo build` produces both the rlib for Rust
   consumers and the staticlib for the C++ shim — no
   second crate to coordinate. Pure-Rust consumers
   ignore the staticlib artifact.
3. The FFI safety surface is still localized to one file
   (`src/capi.rs`) with a clear module boundary. The
   "audit FFI in one place" property the prior decision
   wanted is preserved.

**What's *not* changed:**

- The CLAUDE.md "pure Rust core, zero non-Rust deps"
  constraint scope clarification from the prior entry
  still stands: feral exposing a C ABI is not the same
  as feral consuming a non-Rust dependency.
- The `feral-ipopt-shim/` in-tree-during-bring-up
  decision still stands.

**References.**
- `src/capi.rs` (7 `extern "C"` functions, status codes).
- `Cargo.toml:39-45` (lib crate-type).
- `src/lib.rs` (`pub mod capi;`).
- `feral-ipopt-shim/` (consumer, in-tree).

## Recent Tried-and-Rejected
  on the no-swap branch are already eliminated.
- The 32×32 SIMD body (`block_ldlt32`, landed `d3f1132`
  2026-05-13) puts trailing-update FLOPs for the dominant CHAINWOO
  front shape through a quad pulp dispatch. The dispatch at
  `factor.rs:1189-1193` routes `nrow == ncol == 32` fronts to
  `factor_block32` before the panel path is reached.

**Decision.** Do not implement APP. Recorded in `dev/decisions.md`
2026-05-13. Full analysis in `dev/research/dense-app-path.md`.

**Lesson.** Same as the 2026-05-12 (c) BLAS-3 quad parking: the
session-checkpoint "Next session should" list is not a substitute
for re-measuring the gate. The previous session
(`dev/sessions/2026-05-13-02.md`) advanced #10 as the next target
on the strength of #9 having landed, without re-running
`diag_supernode_cost`. One binary run was the difference between
implementing dead code and recording a clean closure.

References: `dev/research/dense-app-path.md`,
`dev/decisions.md` 2026-05-13 entry, issue #10 thread.

## Source Files
```
src/bin/alloc_probe.rs
src/bin/bench_fma_phase3.rs
src/bin/bench_issue8.rs
src/bin/bench_one_matrix.rs
src/bin/bench_orderings.rs
src/bin/bench_solver_corpus.rs
src/bin/bench_solver_reuse.rs
src/bin/bench.rs
src/bin/blas3_prototype.rs
src/bin/d3_probe.rs
src/bin/d4_probe.rs
src/bin/diag_acopr.rs
src/bin/diag_acopr14.rs
src/bin/diag_amalgamation.rs
src/bin/diag_amd_substages.rs
src/bin/diag_amf_vs_amd.rs
src/bin/diag_cascade_default_evidence.rs
src/bin/diag_cascade_ratio_distribution.rs
src/bin/diag_chainwoo_profile.rs
src/bin/diag_chainwoo.rs
src/bin/diag_compress_costbenefit.rs
src/bin/diag_compress_profile.rs
src/bin/diag_compression_bench.rs
src/bin/diag_cond_parity.rs
src/bin/diag_dense_tail.rs
src/bin/diag_etree_shape.rs
src/bin/diag_factor_nnz_accounting.rs
src/bin/diag_fill_parity.rs
src/bin/diag_fill_tail.rs
src/bin/diag_inertia_mismatch.rs
src/bin/diag_leaf_profile.rs
src/bin/diag_max_ncol.rs
src/bin/diag_mc64_cycles.rs
src/bin/diag_mittelmann.rs
src/bin/diag_orbit2_quotient.rs
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
src/bin/hs85_diag.rs
src/bin/parallel_corpus_parity.rs
src/bin/polak6_diag.rs
src/bin/policy4_diag.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_panel_attribution.rs
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
tests/issue_2_kkt_ls_init.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
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
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
