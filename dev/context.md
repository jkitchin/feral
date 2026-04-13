# FERAL Context (auto-generated)

Generated: 2026-04-13T01:38:51Z

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
09955c2 Phase 2.2.2 Steps 5-6: MC64 callers opt in; green test sweep
1f7d878 Phase 2.2.2 Step 4: implement Duff-Reid 2x2 growth bound
2b086bd Phase 2.2.2 Step 3: implement column-relative 1x1 pivot threshold
286e506 Phase 2.2.2 Steps 1-2: pivot_threshold field and failing tests
404b45e Phase 2.2.2: implementation plan for scaling-aware pivot rejection
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s

     Running tests/threshold_consistency.rs (target/debug/deps/threshold_consistency-c660296127e8afca)

running 6 tests
test polak6_0021_residual_after_threshold_fix ... ignored
test factors_carry_zero_tol_from_params ... ok
test factor_inertia_force_accept_implies_solve_skip_invariant ... ok
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
spd_10             10           29            0     (10, 0, 0)
spd_50             50           43            5     (50, 0, 0)
spd_100           100          116            6    (100, 0, 0)
spd_200           200          471           19    (200, 0, 0)
kkt_10_3           13            4            0     (10, 3, 0)
kkt_30_10          40           23            1    (30, 10, 0)
kkt_50_15          65           61            2    (50, 15, 0)
kkt_100_30        130          355           13   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 154588 matrices loaded

KKT summary: 154588 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000)
  Inertia match: 150094/154481 (97.2%)
  Residual pass: 151212/154481 (97.9%)
  Worst residual: 1.57e7 (DEGENLPA_0061)

--- Sparse solver validation ---
Sparse solver: 154588/154588 total
  Inertia match vs MUMPS: 152968/154588 (99.0%)
  Residual pass: 154202/154588 (99.8%)
  Worst residual: 1.20e9 (POLAK6_0021)

--- Dense failure analysis (4742 failures) ---

family                    total    inertia   residual      worst_res
BATCH                      1589       1589       1588         2.34e1
SWOPF                      1185       1185       1185         1.19e1
HAHN1                       498        498         24         6.26e0
QPNBLEND                    362        362          0       7.56e-16
MSS1                        240        240          0       1.40e-15
CORE1                       141        141          0       3.56e-16
CRESC50                      97         97          0       2.08e-16
ACOPP30                      68         68         68         1.14e5
FBRAIN3LS                    59          6         57        4.11e-7
CERI651DLS                   51          3         48        1.69e-7
CERI651A                     42         42          7         8.72e0
HS46                         42          0         42        7.51e-8
PFIT4                        38         38          0       3.27e-14
DEVGLA2                      37          0         37        2.77e-6
CERI651ALS                   24          2         22        1.79e-7
PFIT2                        24          0         24        8.14e-6
PALMER1ENE                   23          0         23        1.79e-8
CERI651CLS                   23          1         22        2.78e-7
CRESC100                     19         19          0       9.89e-16
MISTAKE                      16          6         16         5.58e5
ALLINITA                     15          8         13        8.10e-4
HATFLDFL                     12          0         12        2.49e-9
KIRBY2                       12         12          0       1.76e-13
HS114                         9          7          9        7.21e-1
SNAKE                         9          0          9        6.99e-9
  ... and 52 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
DEGENLPA_0061                   35       1.57e7    (20, 15, 0)    (18, 15, 2)
MISTAKE_0101                    22       5.58e5     (9, 13, 0)     (8, 13, 1)
MISTAKE_0102                    22       4.81e5     (9, 13, 0)     (8, 13, 1)
DEGENLPB_0045                   35       2.15e5    (20, 15, 0)    (19, 15, 1)
ACOPP30_0001                   209       1.14e5   (72, 137, 0)   (71, 137, 1)
ACOPP30_0000                   209       1.08e5   (72, 137, 0)   (71, 137, 1)
ACOPP30_0002                   209       3.36e4   (72, 137, 0)   (71, 137, 1)
ACOPP30_0067                   209       6.84e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0066                   209       6.74e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0065                   209       6.64e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0064                   209       6.53e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0063                   209       6.43e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0062                   209       6.33e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0061                   209       6.22e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0060                   209       6.12e2   (72, 137, 0)   (71, 137, 1)

--- Sparse failure analysis (1975 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.25e-13
QPNBLEND                    362        362          0       7.33e-16
MSS1                        241        240          1        3.05e-7
CORE1                       141        141          0       3.27e-16
CRESC50                      97         97          0       1.66e-16
ACOPP30                      68         68          0       1.68e-14
FBRAIN3LS                    61          4         59        3.38e-7
CERI651DLS                   53          4         49        8.65e-8
HS46                         42          0         42        1.42e-7
CERI651A                     38         38          4         1.82e0
PFIT4                        38         38          0       1.99e-14
DEVGLA2                      37         11         37        2.20e-6
HIMMELBI                     25         25         12        6.76e-6
PFIT2                        24          0         24        1.22e-5
PALMER1ENE                   23          0         23        1.89e-8
CERI651ALS                   23          2         21        1.48e-7
CERI651CLS                   22          1         21        2.02e-7
CRESC100                     19         19          0       8.68e-16
HATFLDFL                     14          0         14        2.04e-9
KIRBY2                       12         12          0       1.87e-13
SNAKE                        11          0         11        3.75e-9
MISTAKE                      11          0         11        9.24e-7
ALLINITA                     10          2          8        9.17e-7
VESUVIO                      10         10          0       5.70e-13
BENNETT5                      8          8          0       9.59e-14
  ... and 50 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
POLAK6_0021                      9       1.20e9      (5, 4, 0)      (3, 4, 2)
CERI651A_0076                  190       1.82e0   (129, 61, 0)   (128, 61, 1)
CERI651A_0136                  190      1.09e-2   (129, 61, 0)   (127, 62, 1)
CERI651A_0135                  190      2.90e-3   (129, 61, 0)   (127, 62, 1)
ERRINBAR_0824                   27      3.50e-4     (18, 9, 0)     (18, 9, 0)
HS114_0270                      21      2.80e-4    (10, 11, 0)     (9, 11, 1)
CERI651A_0166                  190      1.09e-4   (129, 61, 0)   (128, 61, 1)
PFIT2_0329                       6      1.22e-5      (3, 3, 0)      (3, 3, 0)
PFIT2_0327                       6      1.22e-5      (3, 3, 0)      (3, 3, 0)
PFIT2_0328                       6      1.22e-5      (3, 3, 0)      (3, 3, 0)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0340                       6      6.78e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0338                       6      6.78e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0339                       6      6.78e-6      (3, 3, 0)      (3, 3, 0)
HIMMELBI_0013                  112      6.76e-6   (100, 12, 0)    (98, 12, 2)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  1870
Failed in dense only:             2872
Failed in sparse only:            105
```

## Recent Decisions
col_max`) rejects these pivots before they reach `ForceAccept`,
and the solve then sees a proper rank-deficient factor rather
than 5 forced zeros interacting with the exp-scaled rescale.
ACOPP30_0000 residual drops `2.27e+46 → 1.076e-1` (47 orders).

The 6 other sanity-panel matrices show no change, because their
pivot streams are already well-conditioned at the absolute
`zero_tol` — the column-relative rejection has nothing to fire
on. This is evidence that Phase 2.2.2 is a *correctness fix*
rather than a general-purpose improvement.

**Explicit deferral: delayed pivoting → Phase 2.3.** Phase 2.2.2
implements MUMPS-style column-relative rejection only. It does
*not* implement SPRAL SSIDS's delayed-pivot mechanism
(`ldlt_tpp.cxx`, where a rejected pivot is carried forward to the
parent front rather than forced-accepted). Three of the four
`tests/mc64_regression.rs` targets (CRESC132, CHWIRUT1, CRESC100)
did not improve under `u = 0.01` and plateau at `1e+02 – 1e+05`;
full closure of their residual gap is expected to require delayed
pivoting in Phase 2.3 plus a separate investigation of
solve-side rounding / refinement convergence on large KKT
systems. The 4 regression tests remain `#[ignore]`'d with updated
Post-2.2.2 status comments. No test tolerances were loosened.

**Commitment.** The README sparse-status section is *not* updated
by Phase 2.2.2. The broader MC64 residual gap remains open. Phase
2.2.2 closes the ACOPP30 correctness regression but does not
promote feral to "competitive on KKT matrices"; that claim still
waits on Phase 2.3. Validation evidence:
`dev/validation/phase-2.2.2-pivot-rejection.md`.

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
tests/pivot_rejection.rs
tests/property_tests.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/stress_tests.rs
tests/threshold_consistency.rs
```
