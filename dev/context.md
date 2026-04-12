# FERAL Context (auto-generated)

Generated: 2026-04-12T16:33:11Z

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
ef9e103 Lower zero_tol from 100*eps to eps — closes all 32 Definitive failures
8b4ec3b Add consensus computation; Phase 1b is essentially closed
3365d35 Add native SPRAL/SSIDS oracle and reveal 4-way oracle disagreement
1b5a44e Add native Fortran MUMPS oracle for consensus benchmark
383c5ca ERRINBAR_0824 triage: fundamental ill-conditioning, not a sparse bug
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

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
spd_10             10           20            0     (10, 0, 0)
spd_50             50           20            3     (50, 0, 0)
spd_100           100           76            5    (100, 0, 0)
spd_200           200          391           16    (200, 0, 0)
kkt_10_3           13            2            0     (10, 3, 0)
kkt_30_10          40           20            1    (30, 10, 0)
kkt_50_15          65           48            2    (50, 15, 0)
kkt_100_30        130          200            7   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 153151 matrices loaded

KKT summary: 153151/153151 total
  Inertia match: 152147/153151 (99.3%)
  Residual pass: 152721/153151 (99.7%)
  Worst residual: 3.99e-2 (ACOPP30_0002)

--- Sparse solver validation ---
Sparse solver: 153151/153151 total
  Inertia match vs MUMPS: 152043/153151 (99.3%)
  Residual pass: 152788/153151 (99.8%)
  Worst residual: 3.14e-4 (ERRINBAR_0824)

--- Dense failure analysis (1430 failures) ---

family                    total    inertia   residual      worst_res
QPNBLEND                    362        362          0       7.56e-16
MSS1                        240        240          0       1.40e-15
CORE1                       141        141          0       3.56e-16
CRESC50                      97         97          0       2.08e-16
ACOPP30                      68          0         68        3.99e-2
FBRAIN3LS                    59          6         57        4.11e-7
CERI651DLS                   51          3         48        1.69e-7
HS46                         42          0         42        7.51e-8
PFIT4                        38         38          0       3.27e-14
DEVGLA2                      37          0         37        2.77e-6
CERI651A                     37         37          0       9.15e-14
CERI651ALS                   24          2         22        1.79e-7
PFIT2                        24          0         24        8.14e-6
PALMER1ENE                   23          0         23        1.79e-8
CERI651CLS                   23          1         22        2.78e-7
CRESC100                     19         19          0       9.89e-16
KIRBY2                       12         12          0       1.76e-13
MISTAKE                      12          0         12        1.83e-6
HATFLDFL                     12          0         12        2.49e-9
ALLINITA                      9          2          7        5.43e-7
SNAKE                         9          0          9        6.99e-9
BENNETT5                      8          8          0       1.70e-13
DISCS                         8          8          0       6.31e-16
DJTL                          7          0          7        1.01e-6
LSC2LS                        6          0          6        2.88e-8
  ... and 43 more families

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

--- Sparse failure analysis (1459 failures) ---

family                    total    inertia   residual      worst_res
QPNBLEND                    362        362          0       6.95e-16
MSS1                        240        240          0       1.37e-15
CORE1                       141        141          0       3.53e-16
CRESC50                      97         97          0       2.19e-16
ACOPP30                      67         67         12        6.71e-7
PALMER1ENE                   64         41         23        1.67e-8
FBRAIN3LS                    59          2         57        8.63e-7
CERI651DLS                   50          3         47        1.07e-7
HS46                         46          0         46        7.86e-8
PFIT4                        38         38          0       4.19e-14
CERI651A                     37         37          0       1.20e-13
DEVGLA2                      26          0         26        5.72e-7
PFIT2                        24          0         24        1.09e-5
CERI651CLS                   24          1         23        1.88e-7
CERI651ALS                   24          2         22        1.39e-7
CRESC100                     19         19          0       1.01e-15
KIRBY2                       12         12          0       1.45e-13
MISTAKE                      11          0         11        1.47e-6
HATFLDFL                     10          0         10        2.62e-9
SNAKE                        10          0         10        3.73e-9
ALLINITA                      9          2          7        5.67e-7
BENNETT5                      8          8          0       1.20e-13
DJTL                          8          0          8        5.33e-7
DISCS                         8          8          0       7.01e-16
LSC2LS                        5          0          5        2.88e-8
  ... and 43 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
ERRINBAR_0824                   27      3.14e-4     (18, 9, 0)     (18, 9, 0)
PFIT2_0393                       6      1.09e-5      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      9.38e-6     (12, 4, 0)     (12, 4, 0)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0340                       6      4.10e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0588                       6      4.08e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0589                       6      4.08e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0590                       6      4.08e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0547                       6      3.79e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0545                       6      3.79e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0546                       6      3.79e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0330                       6      3.56e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0390                       6      2.73e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0338                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0339                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  1368
Failed in dense only:             62
Failed in sparse only:            91
```

## Recent Decisions
1. ~880 matrices where feral solves correctly (residual at machine
   precision) but disagrees with rmumps on the inertia label of
   boundary pivots — feral is not wrong, the oracle disagrees with it
   on a definitional choice.
2. ~400 matrices in problem families (ACOPP30, FBRAIN3LS, CERI*, HS46,
   PFIT2, ...) where ForceAccept on rank-deficient KKTs produces wrong
   `A⁻¹`. The principled fix is delayed pivoting, a Phase 2 feature.
3. 88 sparse-only failures, possibly a sparse-pipeline bug like the
   postorder issue.

The deeper concern: rmumps is a Rust port of MUMPS authored by the same
person developing feral. Treating it as ground truth means a bug in
rmumps and a matching bug in feral would both look like "100% pass"
forever. A multi-oracle consensus catches this class of failure and is
also more honest about matrices where the right answer is genuinely
ambiguous in double precision.

**Reconsideration clause.** This decision is **revisitable**. If running
the consensus across all four solvers reveals that the canonical Fortran
oracles agree with rmumps to within float64 precision on essentially the
entire corpus, then the multi-source machinery has not improved the
ground truth and the original strict criterion can be reinstated. If
the oracles disagree substantially, the consensus criterion stays. The
data from Phases 3-5 of `phase-1b-consensus-exit.md` will tell us which
world we live in.

**Constraints unchanged.** Feral itself remains pure Rust with zero
non-Rust dependencies in the core solver. The Fortran oracles live in a
new top-level `external_benchmarks/` directory, are not built by cargo,
and are not in CI. They are run manually as one-time test infrastructure.

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
tests/property_tests.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/stress_tests.rs
tests/threshold_consistency.rs
```
