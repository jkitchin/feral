# FERAL Context (auto-generated)

Generated: 2026-04-11T14:31:45Z

## Latest Session
File: dev/sessions/2026-04-11-01.md
```
# Session 2026-04-11-01

## Goal
Bootstrap Session 1: initialize project structure and implement the dense LDLᵀ
factorization with Bunch-Kaufman pivoting per Section 13.2 of the spec.

## Accomplished
- Initialized Cargo project with full directory structure (Section 13.1)
- Created all dev files, templates, LICENSE, CHANGELOG.md
- Implemented core data structures: SymmetricMatrix, Inertia, Factors, FeralError,
  BunchKaufmanParams, ZeroPivotAction — all matching Section 2.5
- Implemented scalar (unblocked) dense LDLᵀ with full Bunch-Kaufman pivoting:
  - LAPACK-style 3-way pivot selection (Tests 3, 5, 6, and 2×2 fallback)
  - Full symmetric row search for γᵣ
  - Normalized 2×2 computation (divide by |a₁₀|, faer approach)
  - Correct inertia counting via determinant for 2×2 blocks
  - ZeroPivotAction::ForceAccept and Fail
- Implemented solve (7-step sequence) and solve_refined (iterative refinement)
- Implemented iterative infinity-norm equilibration (Knight-Ruiz, max 10 iterations)
- Wired lib.rs reexports per Section 13.2
- Wrote 17 exact tests, all passing
- Wrote benchmark harness skeleton (exact expected output verified)
- Set up CI workflow (test, clippy, fmt, no-unwrap, commit-body advisory)
- Wrote dev/assemble-context.sh and generated dev/context.md

## Bugs Found and Fixed
1. **2×2 rank-2 update ordering**: L column stores happened before Schur complement
   inner loop, corrupting diagonal update when i==j. Fix: store after inner loop.
2. **L extraction for 2×2 blocks**: included D block off-diagonal in L. Fix: skip
   intra-block entries using subdiag discriminant.
3. **Verify function**: computed L·D·L instead of L·D·Lᵀ (wrong transpose index).

## Benchmark Results
```
FERAL benchmark harness
Loading matrices from data/benchmark-config.toml ... not found
0 matrices benchmarked
```
(No matrices yet — harness skeleton confirmed working.)

## Decisions Made
None requiring recording in decisions.md.

## Abandoned Approaches
None.

## Next Session Should
1. Implement blocked BK LDLᵀ with W-panel technique (Section 3.3 step 2)
2. Add property-based tests (randomized inertia invariant checks)
3. Begin benchmark matrix collection (Section 13.4)
```

## Git Status
```
4023006 Add dense LDLt research note (mandatory pre-implementation)
b31fbe2 Add ref/ to .gitignore and remove from tracking
de46d32 Initial commit: project spec, CLAUDE.md agent protocol, and references
```

## Test Status
```
test test_force_accept_with_refinement ... ok
test test_identity ... ok
test test_kkt_5x2 ... ok
test test_kkt_structure ... ok
test test_mixed_pivots_4x4 ... ok
test test_nan_rejected ... ok
test test_negative_definite ... ok
test test_solve_dimension_mismatch ... ok
test test_zero_dimension_rejected ... ok
test test_zero_pivot_fail ... ok
test test_bench_harness_output ... ok

test result: ok. 17 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.19s

   Doc-tests feral

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
FERAL benchmark harness
Loading matrices from data/benchmark-config.toml ... not found
0 matrices benchmarked
```

## Recent Decisions
# Architectural Decisions Log

Append-only. Do not modify existing entries.

## Recent Tried-and-Rejected
# Tried and Rejected Log

Append-only. Do not modify existing entries.

## Source Files
```
src/bin/bench.rs
src/dense/equilibrate.rs
src/dense/factor.rs
src/dense/matrix.rs
src/dense/mod.rs
src/dense/solve.rs
src/error.rs
src/inertia.rs
src/lib.rs
```

## Test Files
```
tests/dense_ldlt.rs
```
