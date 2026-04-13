# Changelog

All notable changes to FERAL will be documented in this file.

## [Unreleased]

### Known issues (Phase 2 in progress)

- **The sparse path produces catastrophically wrong residuals on
  matrices with n > 500.** Phase 1 validation only measured
  matrices with n ≤ 500 (the bench harness enforced this via a
  Phase 1a hold-over filter that was not removed until Phase 2).
  When the filter was lifted in Phase 2.1.2, the sparse path
  produced residuals 10⁴ to 10¹⁴ on larger matrices already
  present in the corpus (CHWIRUT1 through CRESC132 at n=5314),
  while canonical MUMPS and SPRAL/SSIDS produced residuals at
  machine precision. Root cause: missing global MC64
  matching-based scaling. Fix in progress as Phase 2.2.1. Until
  it lands, do not use feral on matrices the dense path cannot
  handle.
- **Phase 1 residual pass rate is not a numerical quality
  measurement**, it is a measurement against the bench tolerance
  `n · ε · 10⁶`. On small matrices this tolerance is loose enough
  (≈ 10⁻⁷ at n=500) to accept feral residuals that are already
  6–8 orders of magnitude worse than canonical solvers. Phase 1's
  99.7% sparse residual pass rate survives this re-reading; what
  does not survive is any implicit claim that feral is numerically
  comparable to canonical solvers at those residual levels.

### Phase 1b Exit (2026-04-12)

Phase 1b closed under the multi-source consensus exit criterion on
the n ≤ 500 subset of the KKT corpus. Feral matches canonical
Fortran MUMPS 5.8.2 on **99.97%** of that subset's inertia — higher
than the agreement between canonical MUMPS and canonical SPRAL/SSIDS
(98.25%). See `dev/sessions/2026-04-12-01.md` and the Known issues
above for the limits of this claim.

### Added
- Sparse multifrontal LDLᵀ solver (`factorize_multifrontal`,
  `solve_sparse`, `solve_sparse_refined`)
- CSC sparse matrix infrastructure (`CscMatrix`, `CscPattern`)
- AMD ordering, elimination tree, postorder, column counts, supernode
  detection with nemin amalgamation (CHOLMOD-style pipeline)
- Symbolic factorization (`symbolic_factorize`) with postorder
  composition of AMD permutation
- Bench failure analysis: family-grouped failure tables, top-worst
  residual lists, dense ∩ sparse cross-comparison
- Bench `FERAL_EMIT_SIDECARS` environment variable: emits canonical
  `.feral.json` sidecars alongside each matrix for consensus analysis
- External benchmark infrastructure (`external_benchmarks/`):
  - Native Fortran MUMPS 5.8.2 oracle (build from `ref/mumps`,
    manifest-based driver, Python JSON wrapper)
  - Native Fortran SPRAL/SSIDS oracle (meson + METIS build, same
    driver pattern)
  - Multi-source consensus computation (Python), applies
    Definitive / Borderline / NumericallyIntractable / Excluded
    verdicts per matrix across four oracles
- Dense LDLᵀ factorization with Bunch-Kaufman pivoting (scalar, unblocked)
- Full 7-step solve sequence with equilibration
- Iterative refinement (`solve_refined`) with best-iterate strategy
- Iterative infinity-norm equilibration (Knight-Ruiz)
- Benchmark harness with built-in dense matrix timing
- CI workflow (test, clippy, fmt, no-unwrap)
- Property-based tests and stress tests (121 total tests)
- Fused update+argmax optimization (halves memory traffic per pivot step)

### Fixed
- **Phase 2.3 — delayed pivoting + sign-preservation fix**: the
  sparse multifrontal path now delays rejected pivots (both 1×1
  column-relative and 2×2 Duff-Reid growth-bound) from non-root
  supernodes to their parent, giving them a landing zone where
  child contributions have been assembled and the block is more
  likely to pivot cleanly. At root supernodes where no further
  delay is possible, `try_reject_1x1_frontal` preserves the
  pivot's sign in the `ForceAccept` fallback: small-but-nonzero
  pivots are accepted with `inertia.positive`/`negative` (not
  counted as zero) and flagged for iterative refinement. Only
  `|d| <= zero_tol ≈ eps` counts as a zero pivot. Evidence:
  sparse KKT sweep worst residual `2.31e+11 → 3.22e-4` (15 orders
  of magnitude across Phase 2.3), sparse-only failure count
  `3328 → 64`, parity panel `11/28 → 22/28`. Dense KKT numbers
  unchanged (99.0% inertia, 99.7% residual pass, 3.99e-2 worst
  on ACOPP30_0002) because the sparse-only `pivot_threshold =
  0.01` config is scoped to `params_kkt_sparse` and
  `BunchKaufmanParams::default()` stays at `0.0`. See
  `dev/sessions/2026-04-13-02.md`, `03.md`, and `04.md`.
- **Phase 2.2.2 — ACOPP30 MC64 regression**: Phase 2.2.1 MC64
  scaling improved 6 of 7 sanity-panel matrices but pushed
  ACOPP30_0000 from a pre-MC64 residual of `2.84e+16` to
  `2.27e+46` — a 30-order-of-magnitude regression caused by 5
  forced-zero pivots in the `ForceAccept` branch interacting with
  the unscaled residual recompose. Phase 2.2.2 adds
  `BunchKaufmanParams::pivot_threshold` (a column-relative 1×1
  rejection clause matching MUMPS CNTL(1) / SSIDS `options%u`,
  default `0.01`) plus the Duff-Reid 2×2 growth bound. MC64
  callers (`tests/mc64_regression.rs::ldlt_params`,
  `src/bin/bench.rs::params_kkt`,
  `examples/triage_large_cresc132.rs`) opt in at `u = 0.01`.
  ACOPP30_0000 residual drops `2.27e+46 → 1.076e-1` (47 orders),
  now ~17 orders better than the pre-MC64 Identity baseline. The
  remaining 3 regression targets (CHWIRUT1, CRESC100, CRESC132)
  are unchanged — their inertia is already exact or ±2, so the
  column-relative rejection has nothing to fire on. Full closure
  of the MC64 residual gap requires delayed pivoting (Phase 2.3).
  Validation: `dev/validation/phase-2.2.2-pivot-rejection.md`.
- **Postorder pipeline bug**: `symbolic_factorize` did not apply
  postorder to the elimination tree before supernode amalgamation,
  causing merged supernodes to have non-contiguous columns while
  downstream code assumed contiguous ranges. Closed MGH10S_0000
  (inertia (50,1,0) → (35,16,0), residual 2.61e21 → 1.10e-16).
- **Pivot threshold mismatch**: factor flagged pivots as zero at
  `100*eps` while solve divided by them at `eps*1e-10`. The band in
  between produced catastrophic cancellation. `Factors` and
  `FrontalFactors` now carry `zero_tol`/`zero_tol_2x2`; both solve
  paths skip any pivot the factor counted as zero. Closed POLAK6_0021
  (residual 8.97e-1 → 4.6e-17).
- **Best-iterate refinement**: `solve_refined` and
  `solve_sparse_refined` now track the smallest `||r||` across
  refinement steps and return the corresponding `x`, guaranteeing the
  refined answer is no worse than the unrefined one on rank-deficient
  matrices where ForceAccept produced a wrong `A⁻¹`.
- **`zero_tol` default lowered** from `100 * EPSILON` to `EPSILON`.
  The 100× safety margin was flagging tiny-but-legitimately-positive
  pivots as zero on small SPD matrices. Verified against canonical
  Fortran MUMPS, SPRAL/SSIDS, and rmumps on CERI651DLS_0534 and
  FBRAIN3LS_0788. Closed the final 32 Definitive feral failures.

### Changed
- Phase 1b exit criterion redefined from "100% correct inertia +
  solution vs rmumps" to multi-source consensus across feral, rmumps,
  canonical MUMPS 5.8.2, and SPRAL/SSIDS. Recorded in
  `dev/decisions.md` (entry 2026-04-12) with a reconsideration clause.
- Bench no longer prints per-row PASS lines for the 153k KKT corpus
  (~153k lines removed from stdout, runtime reduced). The bench now
  emits summary tables with family-grouped failure analysis and a
  dense ∩ sparse cross-comparison.
