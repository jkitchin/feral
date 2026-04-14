# FERAL Context (auto-generated)

Generated: 2026-04-14T18:14:26Z

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
9265f1f Phase 2.5.1': AMD + symbolic + numeric fixes close Phase 2.8.1 gate
881d785 Session 2026-04-14-03 checkpoint: Phase 2.8.1 verdict + Phase 2.5 profile
7158eed Phase 2.5 profile: AMD is the small-frontal bottleneck, not column counts
37bf148 Phase 2.8.1: bench partition reveals sparse small-frontal fails exit bar
d40a988 Session 2026-04-14-02 checkpoint: Phase 2.4.3 landing
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

     Running tests/threshold_consistency.rs (target/debug/deps/threshold_consistency-804225430b8dfbb6)

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
spd_10             10           66            1     (10, 0, 0)
spd_50             50           75            8     (50, 0, 0)
spd_100           100          271           13    (100, 0, 0)
spd_200           200         1499           42    (200, 0, 0)
kkt_10_3           13            8            1     (10, 3, 0)
kkt_30_10          40           62            2    (30, 10, 0)
kkt_50_15          65          139            5    (50, 15, 0)
kkt_100_30        130          634           16   (100, 30, 0)

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
BENNETT5                      8          8          0       4.75e-14
DISCS                         8          8          0       1.98e-15
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
HAHN1                       498        498          0       2.67e-13
QPNBLEND                    362        362          0       2.77e-15
MSS1                        240        240          0       1.94e-15
CORE1                       141        141          0       8.83e-16
CRESC50                      97         97          0       4.94e-16
ACOPP30                      67         67          0       1.37e-14
FBRAIN3LS                    52          3         50        2.79e-7
CERI651DLS                   39          3         36        1.93e-7
PFIT4                        38         38          0       1.69e-14
CERI651A                     37         37          0       8.65e-14
HS46                         29          0         29        3.56e-8
PFIT2                        23          0         23        2.42e-6
CERI651CLS                   21          1         20        2.53e-7
CRESC100                     19         19          0       3.69e-15
PALMER1ENE                   16          0         16        1.22e-8
CERI651ALS                   15          2         13        1.28e-7
DEVGLA2                      15          0         15        7.78e-7
KIRBY2                       12         12          0       1.32e-13
MISTAKE                      10          0         10        1.17e-6
VESUVIO                      10         10          0       2.53e-13
ALLINITA                      9          2          7        4.84e-7
DISCS                         8          8          0       1.41e-15
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
QPNBLEND                    362        362          0       2.77e-15
MSS1                        240        240          0       2.80e-15
CORE1                       141        141          0       1.07e-15
CRESC50                      97         97          0       2.71e-15
ACOPP30                      67         67          0       4.24e-14
FBRAIN3LS                    48          3         42        2.82e-7
CERI651DLS                   38          3         35        1.93e-7
PFIT4                        38         38          0       2.53e-14
CERI651A                     37         37          0       8.65e-14
HS46                         23          0         23        7.51e-8
PFIT2                        22          0         22        5.39e-6
CERI651CLS                   21          1         20        2.53e-7
CRESC100                     19         19          0       4.76e-15
PALMER1ENE                   16          0         16        1.22e-8
DEVGLA2                      15          0         15        7.78e-7
CERI651ALS                   14          2         12        1.28e-7
KIRBY2                       12         12          0       1.32e-13
MISTAKE                      10          0         10        1.33e-6
ALLINITA                      9          2          7        5.43e-7
BENNETT5                      8          8          0       8.69e-14
DISCS                         8          8          0       1.98e-15
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
PFIT2_0298                       6      1.37e-6      6.12e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0297                       6      1.37e-6      6.12e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0329                       6      1.64e-7      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0328                       6      4.15e-8      1.36e-6      (3, 3, 0)      (3, 3, 0)

=== Dense perf vs canonical oracles (154481 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153472       0.23       0.11       2.16      26.19     104.47
solve/MUMPS        153472       0.37       0.25       2.00      22.45     139.83
factor/SSIDS       154393       0.01       0.00       0.34       7.78      24.87
solve/SSIDS        154393       1.47       1.00       8.00      77.33     276.57

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
HS91                     3000       0.10       0.10       0.50
HS92                     3000       0.09       0.10       1.30
SSI                      3000       0.09       0.10       0.54
ALLINITA                 3000       0.09       0.08       0.27
MCONCON                  3000       0.41       0.44       3.66
PALMER5A                 3000       0.09       0.09       1.10
CONCON                   3000       0.45       0.47       1.16
HS90                     3000       0.10       0.10       0.80
PALMER7A                 3000       0.09       0.10       0.67
HS13                     3000       0.11       0.11       0.22
MGH10LS                  3000       0.11       0.11       0.56
BIGGSC4                  3000       0.08       0.08       0.29
HATFLDH                  3000       0.09       0.09       0.45
HS118                    3000       0.37       0.35       1.73
DJTL                     3000       0.11       0.11       0.14
HATFLDBNE                3000       0.10       0.10       0.70
HS89                     3000       0.10       0.10       0.55
SSINE                    3000       0.07       0.08       0.64
ALLINITC                 3000       0.09       0.08       0.20
AVION2                   2682       1.78       1.88       2.50
CERI651ALS               2331       0.08       0.08       0.12
PFIT4                    2286       0.08       0.08       0.67
CERI651C                 2233       0.08       0.08       0.20
CERI651CLS               2227       0.08       0.08       0.22
BATCH                    2054       3.29       3.38       4.48

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
CRESC100_0000                  806        20895          200     104.47
HAHN1_0070                     715        18926          197      96.07
HAHN1_0461                     715        15484          178      86.99
HAHN1_0471                     715        15698          184      85.32
HAHN1_0292                     715        15568          183      85.07
HAHN1_0484                     715        15991          188      85.06
HAHN1_0378                     715        16245          191      85.05
HAHN1_0441                     715        15855          187      84.79
HAHN1_0440                     715        15998          189      84.65
HAHN1_0437                     715        15885          188      84.49

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.44       0.33       2.12       4.72      98.28
solve/MUMPS        153560       0.47       0.38       2.50      13.73      62.16
factor/SSIDS       154500       0.02       0.01       0.32       1.20      21.81
solve/SSIDS        154500       1.85       1.43      12.00      38.33     105.00

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
HATFLDH                  3000       0.48       0.50       1.44
HATFLDBNE                3000       0.33       0.30       1.54
SSINE                    3000       0.16       0.17       1.87
HS91                     3000       0.25       0.25       1.12
HS13                     3000       0.12       0.11       1.00
MGH10LS                  3000       0.11       0.11       1.00
BIGGSC4                  3000       0.42       0.42       2.23
SSI                      3000       0.10       0.10       0.90
HS92                     3000       0.28       0.30       1.11
ALLINITA                 3000       0.36       0.33       2.33

(truncated from      493 lines to 350 line budget)
