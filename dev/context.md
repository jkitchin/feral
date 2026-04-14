# FERAL Context (auto-generated)

Generated: 2026-04-14T15:37:19Z

## Latest Session
File: dev/sessions/phase-2-baseline.md
```
# Phase 2 Performance Baseline Report

**Date:** 2026-04-14
**Head commit under test:** `e08c7a1` (Triage: ERRINBAR_0824 and ACOPP30_0004)
**Corpus:** 154588 KKT matrices in `data/matrices/kkt/` (dense-eligible: 154481
at `n <= 1000`; 107 skipped at `n > 1000`)
**Oracles:** canonical Fortran MUMPS 5.8.2 and SPRAL SSIDS (via
`external_benchmarks/{mumps,ssids}_oracle`), per-matrix `factor_us` /
`solve_us` in `*.mumps.json` / `*.ssids.json` sidecars.

This is the Phase 2.1.8 baseline required by
`dev/plans/phase-2-planning.md` §2.1.8. Every later optimization in Phase
2.4 (dense perf) and Phase 2.5 (sparse perf) is measured against these
numbers.

## Harness additions (Phase 2.1.7)

`src/bin/bench.rs` gained:

- `OracleTiming` + `read_oracle_timing` — parses the `factor_us` /
  `solve_us` fields out of oracle JSON sidecars.
- `KktEntry::{mumps_timing, ssids_timing}` — populated in `load_kkt_dir`
  by `with_extension("mumps.json")` / `with_extension("ssids.json")`;
  missing files leave the fields as `None`.
- `MatrixTiming` — per-matrix feral factor+solve μs, collected in both
  the dense and sparse loops.
- Sparse-loop `Instant::now()` calls — the old sparse loop reported
  inertia and residual but not timings; now records `sp_factor_us`
  (symbolic + numeric combined, matching the semantics of MUMPS's and
  SSIDS's single `factor_us` field) and `sp_solve_us`.
- `print_perf_comparison` — joins feral timings against
  `{mumps,ssids}_timing`, emits overall ratio distribution
  (geomean, p50, p90, p99, max), per-family geomean, and top-10 worst
  factor-ratio matrices vs MUMPS.

Ratio clamp: both sides use `.max(1) μs` so that sub-microsecond
matrices at the clock-resolution floor produce ratio = 1.0 rather than
collapsing the log-space geomean.

## Overall results — ratio = feral_μs / oracle_μs

Lower ratio = feral is faster. Ratio < 1.0 means feral beats the oracle.

### Dense path (`factor_single_front` + `solve_refined`), 154481 matrices

| metric        | count  | geomean |   p50 |   p90 |   p99 |      max |
|---------------|-------:|--------:|------:|------:|------:|---------:|
| factor/MUMPS  | 153472 |    0.23 |  0.11 |  2.27 | 28.99 |   296.45 |
| solve/MUMPS   | 153472 |    0.37 |  0.25 |  2.00 | 23.40 |   523.76 |
| factor/SSIDS  | 154393 |    0.01 |  0.00 |  0.34 |  8.04 |    48.23 |
```

## Git Status
```
4696e13 Phase 2.4.2 Step 3: pulp-dispatched SIMD kernels for axpy_minus and axpy2_minus
a174ec1 Phase 2.4.2 Step 2: schur_kernel module with scalar kernels + test harness
bf97ae6 Phase 2.4.2 Step 1: add pulp 0.22.2 dep for SIMD kernel
ff72c53 Phase 2.4.2 planning: SIMD Schur kernel via pulp
6ca8832 Phase 2.4.1a null result: contribution-block deferral reverted
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

     Running tests/threshold_consistency.rs (target/debug/deps/threshold_consistency-804225430b8dfbb6)

running 6 tests
test polak6_0021_residual_after_threshold_fix ... ignored
test factors_carry_zero_tol_from_params ... ok
test dense_solve_skips_zero_pivots_rank_deficient ... ok
test factor_inertia_force_accept_implies_solve_skip_invariant ... ok
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
spd_100           100           82            5    (100, 0, 0)
spd_200           200          400           21    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           19            1    (30, 10, 0)
kkt_50_15          65           46            2    (50, 15, 0)
kkt_100_30        130          214            8   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 154588 matrices loaded

KKT summary: 154588 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000)
  Inertia match: 152911/154481 (99.0%)
  Residual pass: 154207/154481 (99.8%)
  Worst residual: 1.87e-4 (ERRINBAR_0824)

--- Sparse solver validation ---
Sparse solver: 154588/154588 total
  Inertia match vs MUMPS: 153009/154588 (99.0%)
  Residual pass: 154329/154588 (99.8%)
  Worst residual: 2.50e-4 (ERRINBAR_0824)

--- Dense failure analysis (1840 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.78e-13
QPNBLEND                    362        362          0       2.77e-15
MSS1                        240        240          0       2.80e-15
CORE1                       141        141          0       1.07e-15
CRESC50                      97         97          0       2.71e-15
ACOPP30                      68         68          0       4.24e-14
FBRAIN3LS                    50          6         48        2.82e-7
CERI651DLS                   42          3         39        7.06e-8
PFIT4                        38         38          0       2.53e-14
CERI651A                     37         37          0       8.64e-14
HS46                         27          0         27        7.51e-8
PFIT2                        23          0         23        5.39e-6
CERI651CLS                   21          1         20        2.06e-7
CRESC100                     19         19          0       4.76e-15
CERI651ALS                   17          2         15        4.31e-8
PALMER1ENE                   17          0         17        1.22e-8
DEVGLA2                      15          0         15        1.50e-7
KIRBY2                       12         12          0       1.30e-13
MISTAKE                      11          0         11        1.33e-6
ALLINITA                      9          2          7        5.43e-7
DISCS                         8          8          0       1.98e-15
BENNETT5                      8          8          0       4.75e-14
DJTL                          7          0          7        5.33e-7
SNAKE                         6          0          6        1.83e-9
LSC2LS                        5          0          5        1.95e-8
  ... and 45 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
ERRINBAR_0824                   27      1.87e-4     (18, 9, 0)     (18, 9, 0)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0248                       6      5.39e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0548                       6      3.66e-6      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0340                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0338                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0339                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0299                       6      1.37e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0297                       6      1.37e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0298                       6      1.37e-6      (3, 3, 0)      (3, 3, 0)
MISTAKE_0100                    22      1.33e-6     (9, 13, 0)     (9, 13, 0)
TRO3X3_0637                     43      9.07e-7    (30, 13, 0)    (30, 13, 0)
PFIT2_0390                       6      6.86e-7      (3, 3, 0)      (3, 3, 0)
PFIT2_0545                       6      6.76e-7      (3, 3, 0)      (3, 3, 0)

--- Sparse failure analysis (1837 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.22e-13
QPNBLEND                    362        362          0       2.78e-15
MSS1                        240        240          0       1.68e-15
CORE1                       141        141          0       8.83e-16
CRESC50                      97         97          0       5.12e-16
ACOPP30                      67         67          0       1.63e-14
FBRAIN3LS                    52          3         50        2.79e-7
CERI651DLS                   39          3         36        1.93e-7
PFIT4                        38         38          0       1.69e-14
CERI651A                     37         37          0       7.97e-14
HS46                         29          0         29        3.56e-8
PFIT2                        23          0         23        2.42e-6
CERI651CLS                   21          1         20        2.53e-7
CRESC100                     19         19          0       2.40e-15
PALMER1ENE                   16          0         16        1.22e-8
CERI651ALS                   15          2         13        1.28e-7
DEVGLA2                      15          0         15        7.78e-7
KIRBY2                       12         12          0       1.52e-13
MISTAKE                      10          0         10        1.17e-6
VESUVIO                      10         10          0       1.40e-13
ALLINITA                      9          2          7        4.84e-7
BENNETT5                      8          8          0       8.69e-14
DISCS                         8          8          0       2.09e-15
DJTL                          7          0          7        5.33e-7
SNAKE                         5          0          5        2.42e-9
  ... and 44 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
ERRINBAR_0824                   27      2.50e-4     (18, 9, 0)     (18, 9, 0)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)
FLETCHER_0002                   16      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0390                       6      2.42e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0591                       6      1.70e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0329                       6      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0327                       6      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0328                       6      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0547                       6      1.35e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0545                       6      1.35e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0546                       6      1.35e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0248                       6      1.22e-6      (3, 3, 0)      (3, 3, 0)
MISTAKE_0100                    22      1.17e-6     (9, 13, 0)     (9, 13, 0)
TRO3X3_0637                     43      9.18e-7    (30, 13, 0)    (30, 13, 0)
DEVGLA2_0417                     5      7.78e-7      (5, 0, 0)      (5, 0, 0)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  1808
Failed in dense only:             32
Failed in sparse only:            29

Shared failure mode breakdown:
  Inertia mismatch on BOTH paths:          1566
  Residual-only fail on BOTH paths:         239
  Mixed (one inertia, other residual):        3

Shared failure size class breakdown:
  n <=  100:     314
  n <= 1000:    1494
  n >  1000:       0

Top 25 families in shared failures:
family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.78e-13
QPNBLEND                    362        362          0       2.78e-15
MSS1                        240        240          0       2.80e-15
CORE1                       141        141          0       1.07e-15
CRESC50                      97         97          0       2.71e-15
ACOPP30                      67         67          0       4.24e-14
FBRAIN3LS                    48          3         42        2.82e-7
PFIT4                        38         38          0       2.53e-14
CERI651DLS                   38          3         35        1.93e-7
CERI651A                     37         37          0       8.64e-14
HS46                         23          0         23        7.51e-8
PFIT2                        22          0         22        5.39e-6
CERI651CLS                   21          1         20        2.53e-7
CRESC100                     19         19          0       4.76e-15
PALMER1ENE                   16          0         16        1.22e-8
DEVGLA2                      15          0         15        7.78e-7
CERI651ALS                   14          2         12        1.28e-7
KIRBY2                       12         12          0       1.52e-13
MISTAKE                      10          0         10        1.33e-6
ALLINITA                      9          2          7        5.43e-7
BENNETT5                      8          8          0       8.69e-14
DISCS                         8          8          0       2.09e-15
DJTL                          7          0          7        5.33e-7
LSC2LS                        4          0          4        1.95e-8
CONGIGMZ                      3          2          1        9.85e-9
  ... and 40 more families

Top 15 worst shared residuals:
name                             n    dense_res   sparse_res       expected     actual(sp)
ERRINBAR_0824                   27      1.87e-4      2.50e-4     (18, 9, 0)     (18, 9, 0)
PRICE4_0002                      2      7.74e-6      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0248                       6      5.39e-6      1.22e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0548                       6      3.66e-6      1.43e-8      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      3.63e-6      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0340                       6      2.71e-6      5.92e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0338                       6      2.71e-6      5.92e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0339                       6      2.71e-6      5.92e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0390                       6      6.86e-7      2.42e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0591                       6      3.07e-7      1.70e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0299                       6      1.37e-6      6.12e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0297                       6      1.37e-6      6.12e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0298                       6      1.37e-6      6.12e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0329                       6      1.64e-7      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0327                       6      4.15e-8      1.36e-6      (3, 3, 0)      (3, 3, 0)

=== Dense perf vs canonical oracles (154481 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153472       0.22       0.11       2.00      24.96     101.66
solve/MUMPS        153472       0.37       0.25       1.89      22.31     141.03
factor/SSIDS       154393       0.01       0.00       0.31       7.38      22.71
solve/SSIDS        154393       1.47       1.00       8.00      76.33     272.43

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
SSINE                    3000       0.07       0.08       0.41
SSI                      3000       0.09       0.10       0.67
HS90                     3000       0.10       0.10       7.20
HATFLDH                  3000       0.09       0.09       0.23
HS89                     3000       0.10       0.10       0.22
BIGGSC4                  3000       0.08       0.08       0.89
HS118                    3000       0.33       0.31       7.40
CONCON                   3000       0.45       0.44       7.38
MCONCON                  3000       0.39       0.40       8.53
HATFLDBNE                3000       0.10       0.10       0.45
ALLINITC                 3000       0.09       0.08       0.20
ALLINITA                 3000       0.09       0.08       0.58
MGH10LS                  3000       0.11       0.11       0.25
HS91                     3000       0.10       0.10       0.91
HS92                     3000       0.09       0.10       1.08
PALMER5A                 3000       0.09       0.09       7.69
HS13                     3000       0.11       0.11       1.11
DJTL                     3000       0.11       0.11       1.14
PALMER7A                 3000       0.09       0.10      14.11
AVION2                   2682       1.68       1.76       2.40
CERI651ALS               2331       0.08       0.08       0.23
PFIT4                    2286       0.08       0.08       0.62
CERI651C                 2233       0.08       0.08       0.33
CERI651CLS               2227       0.08       0.08       0.42
BATCH                    2054       3.13       3.19       6.88

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
CRESC100_0000                  806        20332          200     101.66
HAHN1_0461                     715        14774          178      83.00
HAHN1_0371                     715        15006          186      80.68
HAHN1_0471                     715        14788          184      80.37
HAHN1_0497                     715        14938          186      80.31
HAHN1_0017                     715        14929          187      79.83
HAHN1_0152                     715        15195          191      79.55
HAHN1_0484                     715        14943          188      79.48
HAHN1_0258                     715        14701          186      79.04
HAHN1_0485                     715        14854          188      79.01

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.64       0.50       2.90      10.29      92.59
solve/MUMPS        153560       0.46       0.36       2.60      13.82      59.54
factor/SSIDS       154500       0.03       0.02       0.44       2.59      21.45
solve/SSIDS        154500       1.83       1.40      12.00      38.67     127.71

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
CONCON                   3000       1.09       1.00       5.07
PALMER5A                 3000       0.48       0.50       5.10
PALMER7A                 3000       0.36       0.40       1.44
ALLINITA                 3000       0.51       0.50       7.24
HS89                     3000       0.21       0.20       2.50
MGH10LS                  3000       0.21       0.22       0.89
HS91                     3000       0.40       0.40       1.30
SSINE                    3000       0.22       0.25       1.00
HS118                    3000       1.29       1.31       4.00
HS13                     3000       0.23       0.22      12.50

(truncated from      483 lines to 350 line budget)
