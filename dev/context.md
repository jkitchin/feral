# FERAL Context (auto-generated)

Generated: 2026-04-11T15:07:47Z

## Latest Session
File: dev/sessions/2026-04-11-02.md
```
# Session 2026-04-11-02

## Goal
Harden Phase 1a: add property-based and stress tests, implement fused
update+argmax optimization, wire benchmark harness with timing.

## Accomplished
- 6 property-based tests: random SPD, indefinite, KKT, badly-scaled matrices
  at sizes n=2..50, verifying inertia, reconstruction, solve, permutation,
  L structure, D block consistency
- 8 stress tests: LAPACK extension (Test 6), arrow matrix, all-2×2-pivots,
  large SPD (n=50), large indefinite (n=100), extreme scaling (1e-8 to 1e8),
  tridiagonal (n=30), KKT trajectory (δ from 1e-2 to 1e-10)
- Fused update+argmax: rank-1 and rank-2 updates now compute the next column's
  off-diagonal argmax during the update pass, eliminating the separate O(n) scan
- Benchmark harness: 8 built-in matrices with timing (factor + solve in μs)
- Test count: 17 exact + 6 property + 8 stress = 31 total, all passing

## Benchmark Results
```
name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10            8           17     (10, 0, 0)
spd_50             50           80            5     (50, 0, 0)
spd_100           100          308            8    (100, 0, 0)
spd_200           200         1273           27    (200, 0, 0)
kkt_10_3           13           10            1     (10, 3, 0)
kkt_30_10          40           40            2    (30, 10, 0)
kkt_50_15          65          101            4    (50, 15, 0)
kkt_100_30        130          543           12   (100, 30, 0)

8 matrices benchmarked
```

## Decisions Made
None requiring recording in decisions.md.

## Abandoned Approaches
None.

## Next Session Should
1. Investigate KKT reconstruction error (0.22 relative for n=20 random KKT) —
   may indicate growth factor issue in scalar BK for ill-conditioned matrices
2. Begin Phase 1b planning (AMD ordering, elimination tree, sparse infrastructure)
3. Consider blocked BK LDLᵀ with W-panel technique for larger dense matrices
4. Profile fused vs non-fused update on n=500+ matrices to quantify improvement
```

## Git Status
```
7537160 Add KKT-specific hardening tests (8 tests, 39 total)
c30a454 Session 2026-04-11-02 checkpoint
e8b4eba Wire benchmark harness with dense matrix timing
ad05ff4 Implement fused update+argmax optimization
56631e8 Add property-based and stress tests (31 total)
```

## Test Status
```
     Running tests/stress_tests.rs (target/debug/deps/stress_tests-a5f908df781f8ee1)

running 8 tests
test test_all_2x2_pivots ... ok
test test_arrow_matrix ... ok
test test_extreme_scaling ... ok
test test_lapack_extension_branch ... ok
test test_kkt_trajectory ... ok
test test_tridiagonal ... ok
test test_large_spd_50 ... ok
test test_large_indefinite_100 ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s

   Doc-tests feral

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
FERAL benchmark harness
Loading matrices from data/benchmark-config.toml ... not found

name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10           10            0     (10, 0, 0)
spd_50             50           72            4     (50, 0, 0)
spd_100           100          284            8    (100, 0, 0)
spd_200           200         1304           25    (200, 0, 0)
kkt_10_3           13            9            1     (10, 3, 0)
kkt_30_10          40           39            2    (30, 10, 0)
kkt_50_15          65          130            6    (50, 15, 0)
kkt_100_30        130          586           13   (100, 30, 0)

8 matrices benchmarked
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
tests/kkt_hardening.rs
tests/property_tests.rs
tests/stress_tests.rs
```
