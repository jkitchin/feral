# Changelog

All notable changes to FERAL will be documented in this file.

## [Unreleased]

### Phase 1b Exit (2026-04-12)

Phase 1b closed under the multi-source consensus exit criterion. Feral
matches canonical Fortran MUMPS 5.8.2 on **99.97%** of the 153k KKT
corpus — higher than the agreement between canonical MUMPS and
canonical SPRAL/SSIDS (98.25%). See `dev/sessions/2026-04-12-01.md`.

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
