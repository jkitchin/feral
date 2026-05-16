# FERAL Context (auto-generated)

Generated: 2026-05-16T01:44:30Z

## Latest Session
File: dev/sessions/2026-05-15-07.md
```
# Session 2026-05-15-07

## Goal

Build a thorough, pip-installable, uv-compatible Python interface
to feral, targeting the IPM in the user's `discopt` project as the
primary consumer. GH issue #20.

## Accomplished

Shipped the entire `python/` subtree:

- **Rust binding** (`python/src/lib.rs`, ~600 lines): PyO3 0.22 +
  rust-numpy 0.22. Surfaces `CscMatrix`, `Solver`,
  `Inertia`, `FactorStatus`, `QualityLevel`, plus an exception
  hierarchy (`FeralError` → `FactorError` → `SingularError` /
  `WrongInertiaError` / `NumericFailure`; `SolveError`,
  `PatternMismatch`, `FeralIOError`). GIL is released around
  factor/solve/refactor via `py.allow_threads`.

- **Pure-Python layer** (`python/feral/`):
  - `__init__.py` — re-exports + `FactorStatus`/`QualityLevel`
    IntEnums + `from_scipy`/`to_scipy` adapters.
  - `ipm.py` — `KktSolver` dataclass implementing the
    Wächter–Biegler 2006 §3.1 perturbation escalation, plus
    `FactorReport`. `solve_pair` for Mehrotra predictor-corrector.

- **Tests** (23/23 pass):
  - `test_basic.py` (15): CscMatrix construction, factor/solve,
    inertia, refactor symbolic reuse, PatternMismatch detection,
    multi-RHS (1D and 2D), solve_refined, exceptions, repr.
  - `test_scipy_interop.py` (4): from_scipy in full/lower-triangle
    forms, solve-matches-dense, to_scipy round-trip.
  - `test_ipm.py` (4): basic KKT factor+solve, 20-iteration
    symbolic-reuse loop (`symbolic_call_count == 1`), inertia
    perturbation, `solve_pair`.

- **Examples** (`python/examples/`):
  - `quickstart.py` — minimal SPD factor/solve.
  - `discopt_ipm_kkt.py` — HS071 KKT Newton loop demo. Factor
    time drops 0.37 ms → 0.02 ms cold → warm; symbolic count
    stays at 1 across the loop.

- **Distribution**:
  - `python/pyproject.toml` — PEP 517 maturin backend, abi3-py310,
    `feral-solver` name, numpy>=1.23, `[scipy]` extra.
  - `python/Cargo.toml` — empty `[workspace]` opts out of root.
  - `python/README.md` — quickstart, IPM usage, scipy interop,
    build-from-source.
  - Built `feral_solver-0.3.0-cp310-abi3-macosx_11_0_arm64.whl`
```

## Git Status
```
585d739 fix(numeric): make cascade-break opt-in; correct PerturbToEps docs
2b1c3d2 chore(session): 2026-05-15-06 -- close issue #19, PAR_MIN_FLOPS=1e7
b12e03c perf(factor): lower PAR_MIN_FLOPS from 1e8 to 1e7 (#19 closeout)
25926cc feat(bench): probe_issue_19 binary
30a30fc chore(session): 2026-05-15-05 -- PAR_MIN_FLOPS calibration
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-e39990277cb5fde0)

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
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-15-07.md)


No Rust benchmark changes this session (no changes to `src/`).
Existing session-06 bench numbers stand; the python crate is
out-of-workspace and does not touch the hot path.

spd_10             10           56            0     (10, 0, 0)
spd_50             50           27            3     (50, 0, 0)
spd_100           100           89            5    (100, 0, 0)
spd_200           200          423           20    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           25            1    (30, 10, 0)
kkt_50_15          65           55            2    (50, 15, 0)
kkt_100_30        130          223            7   (100, 30, 0)

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
src/bin/probe_cascade_perturb.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_issue_19.rs
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
