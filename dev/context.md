# FERAL Context (auto-generated)

Generated: 2026-05-20T01:38:05Z

## Latest Session
File: dev/sessions/2026-05-19-01.md
```
# Session 2026-05-19-01

## Goal

Close the gap that prevents an IPM backed by FERAL (`pounce`) from bumping
its Hessian perturbation `δ_w` on KKT systems that are ill-conditioned but
land on the correct inertia. Ipopt+MA57 solves three Mittelmann-class
problems in 100–291 iters; pounce+FERAL stalls because the perturbation
handler never fires. The agreed fix: have FERAL report a near-singularity
signal — `min|λ(D)|`, the smallest accepted pivot magnitude — that the
perturbation handler can threshold.

## Accomplished

- **Research note + plan** — `dev/research/near-singularity-signal.md`,
  `dev/plans/near-singularity-signal.md`. Root cause traced: FERAL's default
  `ZeroPivotAction::ForceAccept` force-accepts a near-singular pivot and
  returns `FactorStatus::Success`; the only near-singularity-adjacent fact
  reaching the IPM is `needs_refinement`, which is internal, a coarse
  boolean, and already true on healthy cascade-break factorizations. MA57
  reports the analogous case via `CNTL(2)` → `INFO(1)==4` → Ipopt
  `SYMSOLVER_SINGULAR` → `PerturbForSingularity`.

- **Rust API** — `SparseFactors::{min,max}_pivot_magnitude` over a shared
  `pivot_magnitude_extent()` pass, mirroring the existing `min_diagonal()`.
  `Solver::{min,max}_pivot_magnitude` delegate. 2×2 smaller magnitude
  computed `|det|/larger` to stay cancellation-free on near-singular blocks.
  Kept deliberately distinct from `min_diagonal()` (signed-min vs.
  magnitude-min).

- **C ABI** — `feral_min_pivot` / `feral_max_pivot` (`-1.0` sentinel on
  no-factor / null handle). Declared in `feral-ipopt-shim/include/feral_capi.h`.

- **Evidence** — 5 new tests, all with hand-computed oracles external to the
  implementation:
  - `diag(5,-2,3,-7)`: `min|λ|=2`, `max|λ|=7`, `min_diagonal=-7`
  - `[[0,1],[1,0]]`: 2×2 block, `min=max=1` (`|smaller eig|`, not
    `d_diag[0]=0`)
  - `diag(1,1e-14,-3)`: inertia `(2,1,0)` still correct, `min|λ|≈1e-14`,
    ratio `min/max≈3e-15` — thresholdable where inertia alone is silent
  - `None` / `-1.0` sentinel before any factor
  - C ABI `capi_min_max_pivot`: `[[1,2],[2,1]]` under identity scaling →
    `min|λ(D)|=1`, `max|λ(D)|=3`
  Full `cargo test` exit 0; `cargo clippy --all-targets -- -D warnings`
  clean.

- Three atomic commits (research+plan; Rust API; C ABI). CHANGELOG
  Unreleased and `dev/decisions.md` updated.

## Benchmark Results
```

## Git Status
```
40b5612 fix(stress): allowlist 3 #28 cross-arch BK-pivot divergences
8298d7b chore(session): 2026-05-19-01 -- near-singularity signal min|λ(D)|
f6640eb feat(capi): feral_min_pivot / feral_max_pivot near-singularity ABI
cb03009 feat(numeric): min/max pivot magnitude near-singularity signal
5b81db0 docs(research): plan near-singularity signal (min|λ(D)|)
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-1f9bf40140b78005)

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
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-19-01.md)


Purely additive change (new query accessors; no factorization/solve path
touched). Phase 2.8.1 exit partition all PASS, in line with the prior
session:

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.33     <= 2.0     PASS
medium (<500)            152145     1.70     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.58     <= 2.0     PASS
medium (<500)            153560     1.58     <= 3.0     PASS

(Microbenchmark noise: a second run read 1.36 / 1.74 / 1.56 / 1.56 — same
verdicts. Worst factor-ratio outliers unchanged: `MUONSINE_0000` ≈28–30×,
`KIRBY2_*` 4–8×, `CRESC132_0000` ≈4.8×.)

```

## Recent Decisions
and returns `FactorStatus::Success`. MA57 reports the analogous case
via its `CNTL(2)` small-pivot threshold → `INFO(1)==4` →
Ipopt `SYMSOLVER_SINGULAR` → `PerturbForSingularity`.

Two alternatives were considered and rejected:

1. **Add a `FactorStatus::NearSingular` variant** (FERAL decides the
   threshold and reports a distinct status). Rejected: it bakes a
   policy threshold into the solver, is an ABI break, and forces every
   caller to handle a status that only matters to perturbation-driven
   IPMs. The threshold is caller-specific (it is pounce's analog of
   `CNTL(2)`), so the solver should not own it.
2. **Paper over it inside FERAL** — MA57-style internal static-pivot
   bending (issue #38, `dev/research/static-pivot-perturbation-2026-05-17.md`).
   Already a separate opt-in lever; it perturbs the factor instead of
   informing the caller, which is the wrong fix when the *IPM* is the
   component that should react.

Decision: FERAL stays policy-free. It reports the magnitude; the
caller thresholds it (recommended: the scale-free ratio
`min|λ(D)| / max|λ(D)| ≈ 1/κ(D)`) and decides whether to treat the
factor as singular. `min|λ(D)|` is computed for free in a pass that
mirrors the existing `min_diagonal()` — no factorization/solve cost.

References:
- `dev/research/near-singularity-signal.md`, `dev/plans/near-singularity-signal.md`.
- `factorize.rs` `min_diagonal()` — the signed-min precedent this
  magnitude-min signal is deliberately kept distinct from.
- Issue #38 / `static-pivot-perturbation-2026-05-17.md` — the rejected
  "paper over it" lever.

## Recent Tried-and-Rejected
direct equivalent. The `test` job (linux × py3.10/3.12/3.13) and the
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
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_dtoc2_mc64.rs
src/bin/probe_f01.rs
src/bin/probe_fbrain.rs
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

(truncated from      365 lines to 350 line budget)
