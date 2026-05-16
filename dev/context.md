# FERAL Context (auto-generated)

Generated: 2026-05-16T15:11:28Z

## Latest Session
File: dev/sessions/2026-05-16-01.md
```
# Session 2026-05-16-01

## Goal
Investigate and close the 6 ACOPP30 plateau-2 matrices revealed by the
full 105-matrix corpus sweep (issue #23, expanded scope). After the
SSIDS scale-invariant det floor closed the original 8 suspects, iters
59, 63, 64, 65, 66, 67 stayed at rel_ref 1.88e-6..1.74e-1 under default
`Solver`, all reporting `min|D|=0`.

## Accomplished

### Investigation (commit d02b39a)
- Built `src/bin/diag_acopp30_residual.rs`: reproducer for #23. Sweeps
  all 105 ACOPP30 matrices with `--all`; `DIAG_SCALING` env var
  overrides scaling strategy. Reports inertia, min|D|, rel_raw, rel_ref.
- Built `src/bin/probe_acopp30_64.rs`: NumericParams knob sweep. Proved
  pivot_threshold / on_zero_pivot / PerturbToEps cannot rescue any of
  the 6; only changing the scaling strategy changes the pivot pattern
  (115×1+47×2 broken vs 101×1+54×2 working).
- Built `src/bin/probe_scaling_policy4.rs`: 9-matrix Policy 4 validation
  panel diagnostic. Discovered `in_spread = max|s|/min|s|` of the
  InfNorm scaling vector is the clean discriminator between
  matrices-where-MC64-helps and matrices-where-MC64-hurts.
- Drafted `dev/research/acopp30-plateau-2.md` with hypothesis tree,
  evidence tables, root-cause analysis, and proposed fix.

### Fix (commit 8986679)
Landed `IN_SPREAD_GUARD = 1e3` pre-MC64 InfNorm trial in
`src/scaling/mod.rs::compute_scaling_auto_with_cache`. Before the
existing `raw_drng >= 1e6 → MC64 unconditionally` fast-path, run
InfNorm Knight-Ruiz; if the resulting scaling vector has
`max|s|/min|s| < 1e3`, accept InfNorm and skip MC64. The `in_vec` is
hoisted and reused by the existing `mc_off/in_off` ratio test, so the
net cost is one InfNorm pass per Auto invocation that reaches the
MC64 leg.

Test changes:
- Renamed `auto_keeps_mc64_on_hs75_0000` → `auto_picks_infnorm_on_hs75_0000`.
  The old test asserted MC64 as "the win" with a 4-order residual
  improvement; current per-matrix probe shows InfNorm = 4.20e-17
  strictly beats MC64 = 1.31e-16. Updated to match new (correct)
  behavior.
- Added `auto_picks_infnorm_on_acopp30_0064` regression test asserting
  `pick_scaling_strategy` still picks MC64 (arrow-KKT shape) but
  `compute_scaling(Auto)` resolves to InfNorm via IN_SPREAD_GUARD.

### Validation
- `DIAG_SCALING=auto cargo run --release --bin diag_acopp30_residual -- --all`:
  **105/105 ACOPP30 matrices pass rel_ref < 1e-10** (previously 99/105).
- Policy 4 panel: zero regressions. MEYER3NE / VESUVIA / VESUVIO /
```

## Git Status
```
8986679 fix(scaling): pre-MC64 InfNorm trial in Policy 4 (clostest result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-72ad14a7f1d9bf8c)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_gate_boundary_n_16 ... ok
test test_determinism_tiny ... ok
test test_solve_paritytest result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-72ad14a7f1d9bf8c)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_determinism_tiny ... ok
test test_gate_boundary_n_16 ... ok
test test_solve_parity_tiny_real_matrix ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 1 test
test src/symbolic/profiler.rs - symbolic::profiler::SymbolicProfiler (line 27) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-16-01.md)


=== Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.38     <= 2.0     PASS
medium (<500)            152145     1.80     <= 3.0     PASS

=== Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.74     <= 2.0     PASS
medium (<500)            153560     1.74     <= 3.0     PASS

Bench partitions stable vs prior session (small-frontal sparse 1.69 →
1.74 is within run-to-run noise; both gates pass).

```

## Recent Decisions
this session via `probe_cascade_perturb`) is fully recoverable
with a single builder call. The `Solver::with_cascade_break_eps`
and `Solver::with_cascade_break` builders are unchanged. Tests
that exercise the gate continue to construct `NumericParams`
with explicit `Some(...)` values.

**Evidence.**
- `probe_cascade_perturb` on `robot_1600_0004` (n=24000):
  cb=off residual 6.24e-7; cb=default residual 1.06e-5;
  cb=fa residual 2.10e+2.
- `probe_cascade_perturb` on `pinene_3200_0009` (n=127995):
  cb=off factor 94 s, residual 2.27e-2; cb=default factor 33 ms,
  residual 7.99e-2 (with inertia preserved); cb=fa factor 36 ms
  but wrong inertia and residual 5.34e+3.
- `cargo test --lib --release` → 256 passed; integration tests
  pass; `cargo clippy --all-targets --release -- -D warnings` clean;
  `cargo fmt --check` clean.
- `cargo run --release --bin bench` Phase 2.8.1 dense+sparse
  small-frontal and medium buckets all PASS; bench numbers within
  noise of session 2026-05-15-06.

**References.**
- `dev/research/cascade-break-l-perturbation-2026-05-15.md` —
  the corrected forensics (the note's original "zero L" proposal
  was rejected; the note now records both the wrong premise and
  the right outcome).
- `dev/tried-and-rejected.md` — 2026-05-15 "Zero L on
  `PerturbToEps`" entry.
- `src/bin/probe_cascade_perturb.rs` — the probe that produced
  the residual numbers.

## Recent Tried-and-Rejected
**Resolution.**

1. Code revert: no change to the `PerturbToEps` branches.
2. Docstring corrected (`src/dense/factor.rs` `PerturbToEps`,
   `src/numeric/solver.rs` `with_cascade_break_eps`) to honestly
   describe the perturbation structure.
3. Cascade-break flipped to **opt-in** by default
   (`NumericParams::default()` now has
   `cascade_break_ratio = None, cascade_break_eps = None`).
   MUMPS and MA57 don't ship an equivalent of cascade-break-eps;
   auto-arming a non-standard mechanism was creating surprises and
   the prior tried-and-rejected entry above ("Default
   `cascade_break_ratio = None` to fix issue #17") was based on the
   wrong assumption that the win-case had no opt-in path. The win
   case (`pinene_3200_0009`, 88.6 s → 34 ms) is preserved via
   explicit `Solver::with_cascade_break(0.5).with_cascade_break_eps(1e-10)`.

References: `dev/research/cascade-break-l-perturbation-2026-05-15.md`,
session 2026-05-15-02 (original 1.4e-5 measurement), session
2026-05-15-07 (this entry).

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
src/bin/hs85_diag.rs
src/bin/parallel_corpus_parity.rs
src/bin/polak6_diag.rs
src/bin/policy4_diag.rs
src/bin/probe_acopp30_64.rs
src/bin/probe_cascade_perturb.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_issue_19.rs
src/bin/probe_panel_attribution.rs
src/bin/probe_scaling_policy4.rs
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
