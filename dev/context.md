# FERAL Context (auto-generated)

Generated: 2026-04-14T16:34:23Z

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
7158eed Phase 2.5 profile: AMD is the small-frontal bottleneck, not column counts
37bf148 Phase 2.8.1: bench partition reveals sparse small-frontal fails exit bar
d40a988 Session 2026-04-14-02 checkpoint: Phase 2.4.3 landing
d9887a9 Phase 2.4.3: non-FMA bit-exact unroll4 Schur kernel wired in
4696e13 Phase 2.4.2 Step 3: pulp-dispatched SIMD kernels for axpy_minus and axpy2_minus
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s

     Running tests/threshold_consistency.rs (target/debug/deps/threshold_consistency-804225430b8dfbb6)

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
spd_10             10           43            1     (10, 0, 0)
spd_50             50           45            5     (50, 0, 0)
spd_100           100          164            9    (100, 0, 0)
spd_200           200          730           31    (200, 0, 0)
kkt_10_3           13            6            1     (10, 3, 0)
kkt_30_10          40           36            2    (30, 10, 0)
kkt_50_15          65           91            4    (50, 15, 0)
kkt_100_30        130          385           14   (100, 30, 0)

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
DISCS                         8          8          0       2.09e-15
BENNETT5                      8          8          0       8.69e-14
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
DISCS                         8          8          0       2.09e-15
BENNETT5                      8          8          0       8.69e-14
DJTL                          7          0          7        5.33e-7
LSC2LS                        4          0          4        1.95e-8
HS118                         3          0          3        9.68e-8
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
PFIT2_0298                       6      1.37e-6      6.12e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0297                       6      1.37e-6      6.12e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0329                       6      1.64e-7      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0327                       6      4.15e-8      1.36e-6      (3, 3, 0)      (3, 3, 0)

=== Dense perf vs canonical oracles (154481 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153472       0.22       0.11       1.91      23.73      96.29
solve/MUMPS        153472       0.36       0.25       1.83      21.27     120.15
factor/SSIDS       154393       0.01       0.00       0.30       7.08      22.12
solve/SSIDS        154393       1.43       1.00       8.00      73.00     275.25

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
DJTL                     3000       0.11       0.11       0.14
BIGGSC4                  3000       0.08       0.08       0.71
MGH10LS                  3000       0.11       0.11       0.33
SSINE                    3000       0.07       0.08       0.36
MCONCON                  3000       0.37       0.39       7.61
SSI                      3000       0.09       0.10       0.18
HS90                     3000       0.10       0.10       0.12
ALLINITC                 3000       0.09       0.08       0.50
CONCON                   3000       0.43       0.44       1.06
HATFLDBNE                3000       0.10       0.10       0.64
HS89                     3000       0.10       0.10       0.14
HS118                    3000       0.30       0.28       1.40
HS13                     3000       0.11       0.11       0.14
PALMER5A                 3000       0.08       0.09       0.55
HS92                     3000       0.09       0.10       0.27
PALMER7A                 3000       0.09       0.10       0.12
ALLINITA                 3000       0.09       0.08       0.92
HATFLDH                  3000       0.09       0.09       1.22
HS91                     3000       0.10       0.10       2.36
AVION2                   2682       1.59       1.67       3.28
CERI651ALS               2331       0.08       0.08       1.89
PFIT4                    2286       0.08       0.08       0.70
CERI651C                 2233       0.09       0.08       0.58
CERI651CLS               2227       0.08       0.08       0.50
BATCH                    2054       2.98       3.04       4.02

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
CRESC100_0000                  806        19258          200      96.29
HAHN1_0461                     715        14070          178      79.04
HAHN1_0490                     715        14675          190      77.24
HAHN1_0471                     715        14163          184      76.97
HAHN1_0445                     715        14426          188      76.73
HAHN1_0292                     715        13948          183      76.22
HAHN1_0499                     715        14405          189      76.22
HAHN1_0421                     715        14129          186      75.96
HAHN1_0266                     715        14043          185      75.91
HAHN1_0497                     715        14103          186      75.82

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.63       0.50       2.86       9.78      91.86
solve/MUMPS        153560       0.45       0.33       2.33      12.90      59.79
factor/SSIDS       154500       0.03       0.02       0.43       2.51      20.82
solve/SSIDS        154500       1.76       1.00      11.50      35.00      86.50

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
SSI                      3000       0.18       0.20       1.20
MGH10LS                  3000       0.21       0.22       1.33
HATFLDH                  3000       0.64       0.64       1.80
PALMER5A                 3000       0.45       0.50       1.67
HATFLDBNE                3000       0.44       0.40       5.07
PALMER7A                 3000       0.36       0.40       0.64
DJTL                     3000       0.11       0.11       1.78
HS90                     3000       0.30       0.30       0.78
HS92                     3000       0.40       0.40       1.50
HS118                    3000       1.22       1.27       3.87

(truncated from      493 lines to 350 line budget)
