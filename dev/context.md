# FERAL Context (auto-generated)

Generated: 2026-04-12T23:40:46Z

## Latest Session
File: dev/sessions/2026-04-12-01.md
```
# Session 2026-04-12-01 — Phase 1b Exit

## Goal

Close Phase 1b. Investigate remaining failures, build a principled
multi-source ground truth (rmumps + canonical Fortran MUMPS 5.8.2 +
canonical Fortran SSIDS + feral), compute consensus verdicts, and
declare Phase 1b complete under a criterion that is honest about
where solvers legitimately disagree.

## Accomplished

### Structural bug fixes (rewriting the solver's correctness baseline)

1. **Postorder pipeline fix** (`7a303d0`) — `symbolic_factorize` never
   applied postorder to the elimination tree before supernode
   detection, so merged supernodes had non-contiguous columns while
   downstream code assumed contiguous ranges. This was the root cause
   of the sparse path's wrong inertia and catastrophic residuals.
   Closed MGH10S_0000 (inertia (50,1,0) → (35,16,0), residual
   2.61e21 → 1.10e-16). Per-corpus delta: sparse inertia match
   98.6% → 99.3%, sparse worst residual 2.61e21 → 3.14e-4.

2. **Best-iterate refinement** (`d954c73`, follows `eab0042`) — naive
   refinement was amplifying error on rank-deficient matrices where
   ForceAccept produced a wrong `A⁻¹`. Best-iterate strategy tracks
   the smallest `||r||` seen across refinement steps and returns the
   corresponding `x`. Closed POLAK6_0021 refinement regression
   (8.97e-1 → 4.6e-17 after threshold-mismatch fix).

3. **Threshold-mismatch fix** (`95e6760`) — factor flagged pivots as
   zero at `100*eps = 2.22e-14` while solve divided by them at
   `eps*1e-10 = 2.22e-26`. The band in between produced catastrophic
   cancellation. Added `zero_tol` + `zero_tol_2x2` fields to `Factors`
   and `FrontalFactors`; both solve paths now consult the stored
   threshold.

4. **`zero_tol = eps` default** (`ef9e103`) — the `100*eps` default
   was too aggressive. Tiny-but-legitimately-positive pivots (e.g.
   `1.83e-14` on CERI651DLS_0534) were being flagged as zero on small
   SPD matrices. Changed the default to `f64::EPSILON` after verifying
   that canonical MUMPS, SSIDS, and rmumps all classify such pivots
   as positive. Closed the final 32 Definitive feral failures.

### New infrastructure

5. **Native Fortran MUMPS 5.8.2 oracle** (`1b5a44e`) — built
   `ref/mumps` with gfortran 15.2 + OpenBLAS; wrote
   `external_benchmarks/mumps_oracle/{Makefile, Makefile.inc.mumps,
   mumps_bench.F, run_mumps.py}`. The Fortran driver reads a manifest
```

## Git Status
```
3d0716b Phase 2.2.1 Step 8 triage: ACOPP30 regression investigation
8a95825 Phase 2.2.1 Step 8: MC64 validation sweep (partial; 4 regressions open)
0a13515 Phase 2.2.1 Steps 6+7: apply MC64 scaling in factorize assembly and solve
67954d9 Phase 2.2.1 Step 5: integrate MC64 scaling into symbolic_factorize
321568e Phase 2.2.1 Step 4: MC64 wrapper (compute_symmetric)
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

     Running tests/threshold_consistency.rs (target/debug/deps/threshold_consistency-c660296127e8afca)

running 6 tests
test polak6_0021_residual_after_threshold_fix ... ignored
test factor_inertia_force_accept_implies_solve_skip_invariant ... ok
test factors_carry_zero_tol_from_params ... ok
test dense_solve_skips_zero_pivots_rank_deficient ... ok
test refinement_does_not_amplify_error_on_rank_deficient_matrix ... ok
test sparse_solve_skips_zero_pivots_rank_deficient ... ok

test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

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
spd_10             10           35            0     (10, 0, 0)
spd_50             50           21            3     (50, 0, 0)
spd_100           100           80            5    (100, 0, 0)
spd_200           200          395           17    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           20            1    (30, 10, 0)
kkt_50_15          65           47            2    (50, 15, 0)
kkt_100_30        130          202            7   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 154588 matrices loaded

KKT summary: 154588 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000)
  Inertia match: 152979/154481 (99.0%)
  Residual pass: 154051/154481 (99.7%)
  Worst residual: 3.99e-2 (ACOPP30_0002)

--- Sparse solver validation ---
Sparse solver: 154588/154588 total
  Inertia match vs MUMPS: 153007/154588 (99.0%)
  Residual pass: 154210/154588 (99.8%)
  Worst residual: 1.20e9 (POLAK6_0021)

--- Dense failure analysis (1928 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       1.76e-12
QPNBLEND                    362        362          0       7.56e-16
MSS1                        240        240          0       1.40e-15
CORE1                       141        141          0       3.56e-16
CRESC50                      97         97          0       2.08e-16
ACOPP30                      68          0         68        3.99e-2
FBRAIN3LS                    59          6         57        4.11e-7
CERI651DLS                   51          3         48        1.69e-7
HS46                         42          0         42        7.51e-8
PFIT4                        38         38          0       3.27e-14
CERI651A                     37         37          0       9.15e-14
DEVGLA2                      37          0         37        2.77e-6
PFIT2                        24          0         24        8.14e-6
CERI651ALS                   24          2         22        1.79e-7
CERI651CLS                   23          1         22        2.78e-7
PALMER1ENE                   23          0         23        1.79e-8
CRESC100                     19         19          0       9.89e-16
KIRBY2                       12         12          0       1.76e-13
HATFLDFL                     12          0         12        2.49e-9
MISTAKE                      12          0         12        1.83e-6
ALLINITA                      9          2          7        5.43e-7
SNAKE                         9          0          9        6.99e-9
BENNETT5                      8          8          0       1.70e-13
DISCS                         8          8          0       6.31e-16
DJTL                          7          0          7        1.01e-6
  ... and 44 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
ACOPP30_0002                   209      3.99e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0037                   209      2.92e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0026                   209      2.80e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0018                   209      2.76e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0045                   209      2.76e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0000                   209      2.74e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0012                   209      2.69e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0065                   209      2.64e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0046                   209      2.64e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0036                   209      2.63e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0051                   209      2.58e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0024                   209      2.54e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0057                   209      2.53e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0055                   209      2.49e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0013                   209      2.46e-2   (72, 137, 0)   (72, 137, 0)

--- Sparse failure analysis (1937 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.25e-13
QPNBLEND                    362        362          0       7.33e-16
MSS1                        241        240          1        3.05e-7
CORE1                       141        141          0       3.27e-16
CRESC50                      97         97          0       1.66e-16
ACOPP30                      68         67         20        4.01e-2
FBRAIN3LS                    61          4         59        3.38e-7
CERI651DLS                   53          4         49        8.65e-8
HS46                         42          0         42        1.42e-7
PFIT4                        38         38          0       1.99e-14
CERI651A                     37         37          0       8.38e-14
DEVGLA2                      26          0         26        2.20e-6
PFIT2                        24          0         24        1.22e-5
PALMER1ENE                   23          0         23        1.89e-8
CERI651ALS                   23          2         21        1.48e-7
CERI651CLS                   22          1         21        2.02e-7
CRESC100                     19         19          0       8.68e-16
HATFLDFL                     14          0         14        2.04e-9
KIRBY2                       12         12          0       1.87e-13
MISTAKE                      11          0         11        9.24e-7
SNAKE                        11          0         11        3.75e-9
ALLINITA                     10          2          8        9.17e-7
VESUVIO                      10         10          0       5.70e-13
BENNETT5                      8          8          0       9.59e-14
DISCS                         8          8          0       5.48e-16
  ... and 49 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
POLAK6_0021                      9       1.20e9      (5, 4, 0)      (3, 4, 2)
ACOPP30_0059                   209      4.01e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0063                   209      6.24e-4   (72, 137, 0)   (71, 137, 1)
ACOPP30_0064                   209      5.08e-4   (72, 137, 0)   (71, 137, 1)
ACOPP30_0066                   209      4.42e-4   (72, 137, 0)   (71, 137, 1)
ACOPP30_0067                   209      4.38e-4   (72, 137, 0)   (71, 137, 1)
ACOPP30_0065                   209      3.85e-4   (72, 137, 0)   (71, 137, 1)
ERRINBAR_0824                   27      3.50e-4     (18, 9, 0)     (18, 9, 0)
ACOPP30_0058                   209      3.90e-5   (72, 137, 0)   (71, 137, 1)
PFIT2_0329                       6      1.22e-5      (3, 3, 0)      (3, 3, 0)
PFIT2_0327                       6      1.22e-5      (3, 3, 0)      (3, 3, 0)
PFIT2_0328                       6      1.22e-5      (3, 3, 0)      (3, 3, 0)
ACOPP30_0049                   209      8.40e-6   (72, 137, 0)   (71, 137, 1)
ACOPP30_0051                   209      7.79e-6   (72, 137, 0)   (71, 137, 1)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  1870
Failed in dense only:             58
Failed in sparse only:            67
```

## Recent Decisions
within the loose absolute tolerance. The Phase 1 "99.7% sparse
residual pass rate" was therefore a measurement of *whether feral
met an absolute tolerance*, not a measurement of *whether feral
was producing answers comparable to canonical solvers*. The
former claim is accurate as stated. The latter is what a casual
reader of the exit summary would assume, and that assumption does
not hold.

**What this changes.** Nothing about the Phase 1b exit commit or
session file is undone. The retrospective
(`dev/phase1-retrospective.org`) already documents the scope caveat
in its "honest assessment of success" section; that caveat is now
a concrete failure mode with measurements attached, and the README
and CHANGELOG have been updated to reflect the revised
interpretation. The Phase 2 plan ordering (`dev/plans/phase-2-planning.md`)
remains correct: Phase 2 opens with measurement infrastructure
(which surfaced the bug in its first hour), followed by the
deferred correctness fixes (MC64 scaling as Phase 2.2.1 and the
trace fix as Phase 2.2.2), followed by pivoting and performance
work. The sanity check the plan called for in §2.1.2 did exactly
what a gate is supposed to do, which was to stop us from
proceeding with corpus expansion on top of a broken sparse path.

**Commitment.** Feral's README will not advertise scale-related
correctness (n > 500 matrices, production KKT workloads, or
performance parity with canonical solvers) until Phase 2.2.1 is
complete and the sanity check panel is re-run with residuals
within 2–3 orders of magnitude of canonical solvers. This is not
a target to aspire to after Phase 2; it is a precondition for
advertising feral as a working sparse solver at all.

## Recent Tried-and-Rejected
   trace fix is more correct in absolute terms but moves feral
   away from the current oracle.

**Decision.** Revert and re-attempt after canonical Fortran MUMPS becomes
available as a second oracle (per `dev/plans/phase-1b-consensus-exit.md`).
At that point we can verify whether canonical MUMPS uses trace-based or
a00-based inertia counting on the 16 regressed matrices and reapply the
fix in the direction that the canonical solver agrees with.

**Code state.** A `KNOWN BUG` comment is left in
`src/dense/factor.rs::count_2x2_inertia` documenting the issue and
linking back here. The function signature remains unchanged so we don't
need `#[allow(clippy::too_many_arguments)]` for code that we know will
need to change again.

**Symptoms.** Inertia error pattern `(p+1, n+1, 0) → (p, n, +1)` on
matrices with zero-diagonal Hessian rows. The "lost positive" appears
as a "gained zero" in feral's output. Most visible on the ACOPP30
family (68 matrices, all with the same `(72,137,0) → (71,137,1)`
mismatch).

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
src/io/mod.rs
src/io/mtx.rs
src/io/sidecar.rs
src/lib.rs
src/numeric/factorize.rs
src/numeric/mod.rs
src/numeric/solve.rs
src/ordering/amd.rs
src/ordering/elimination_tree.rs
src/ordering/mod.rs
src/ordering/postorder.rs
src/scaling/hungarian.rs
src/scaling/mc64.rs
src/scaling/mod.rs
src/sparse/csc.rs
src/sparse/mod.rs
src/symbolic/column_counts.rs
src/symbolic/mod.rs
src/symbolic/supernode.rs
```

## Test Files
```
tests/dense_ldlt.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/mc64_end_to_end.rs
tests/mc64_regression.rs
tests/mc64_scaling.rs
tests/property_tests.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/stress_tests.rs
tests/threshold_consistency.rs
```
