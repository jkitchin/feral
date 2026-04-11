# Changelog

All notable changes to FERAL will be documented in this file.

## [Unreleased]

### Added
- Dense LDLᵀ factorization with Bunch-Kaufman pivoting (scalar, unblocked)
- Full 7-step solve sequence with equilibration
- Iterative refinement (solve_refined)
- Iterative infinity-norm equilibration (Knight-Ruiz)
- Benchmark harness skeleton
- CI workflow (test, clippy, fmt, no-unwrap)
- Property-based tests (6) and stress tests (8) — 31 total tests
- Fused update+argmax optimization (halves memory traffic per pivot step)
- Benchmark harness with built-in dense matrix timing
