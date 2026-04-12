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
