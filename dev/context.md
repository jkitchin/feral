# FERAL Context (auto-generated)

Generated: 2026-04-19T01:53:30Z

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
b5c67cb symbolic: pin KaHIP-not-default decision with research note + test
86cf1e8 vesuvio_diag: bin to localize VESUVIO factor outlier
34f02d9 solve: amortize workspace across sparse refinement steps
05eb8ab Session 2026-04-18-07: refinement 2-strike + bordered-KKT routing
824d3e6 ordering: bordered-KKT fallback to MetisND in symbolic_factorize default
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s

     Running tests/threshold_consistency.rs (target/debug/deps/threshold_consistency-c006c53777591831)

running 6 tests
test polak6_0021_residual_after_threshold_fix ... ignored
test factor_inertia_force_accept_implies_solve_skip_invariant ... ok
test refinement_does_not_amplify_error_on_rank_deficient_matrix ... ok
test factors_carry_zero_tol_from_params ... ok
test dense_solve_skips_zero_pivots_rank_deficient ... ok
test sparse_solve_skips_zero_pivots_rank_deficient ... ok

test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
FERAL benchmark harness
  ordering: default (symbolic_factorize heuristic)
Loading matrices from data/benchmark-config.toml ... not found

name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10           63           16     (10, 0, 0)
spd_50             50           63            6     (50, 0, 0)
spd_100           100          243           11    (100, 0, 0)
spd_200           200         1307           33    (200, 0, 0)
kkt_10_3           13           13            0     (10, 3, 0)
kkt_30_10          40           55            2    (30, 10, 0)
kkt_50_15          65          147            4    (50, 15, 0)
kkt_100_30        130          577           14   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 154588 matrices loaded

KKT summary: 154588 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000)
  Inertia match: 152911/154481 (99.0%)
  Residual pass: 154207/154481 (99.8%)
  Worst residual: 1.87e-4 (ERRINBAR_0824)

--- Sparse solver validation ---
Sparse solver: 154588/154588 total
  Inertia match vs MUMPS: 153008/154588 (99.0%)
  Residual pass: 154241/154588 (99.8%)
  Worst residual: 2.69e-4 (ERRINBAR_0824)

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

--- Sparse failure analysis (1926 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.97e-13
QPNBLEND                    362        362          0       1.79e-15
MSS1                        240        240          0       1.65e-15
CORE1                       141        141          0       7.42e-16
CRESC50                      97         97          0       3.98e-16
ACOPP30                      68         68          0       2.37e-14
FBRAIN3LS                    60          3         58        2.79e-7
CERI651DLS                   48          3         45        1.94e-7
HS46                         43          0         43        1.11e-7
PFIT4                        38         38          0       2.54e-14
CERI651A                     37         37          0       1.05e-13
DEVGLA2                      26          0         26        1.58e-6
CERI651CLS                   26          1         25        3.20e-7
PFIT2                        24          0         24        5.92e-6
PALMER1ENE                   22          0         22        1.87e-8
CERI651ALS                   21          2         19        1.45e-7
CRESC100                     19         19          0       3.65e-15
KIRBY2                       12         12          0       1.68e-13
HATFLDFL                     11          0         11        2.97e-9
MISTAKE                      11          0         11        7.35e-7
VESUVIO                      10         10          0       5.04e-13
SNAKE                         9          0          9        3.25e-9
ALLINITA                      9          2          7        5.48e-7
DISCS                         8          8          0       1.77e-15
BENNETT5                      8          8          0       1.00e-13
  ... and 50 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
ERRINBAR_0824                   27      2.69e-4     (18, 9, 0)     (18, 9, 0)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0594                       6      5.92e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0248                       6      5.39e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0548                       6      3.66e-6      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0330                       6      3.56e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0545                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0546                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0547                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0591                       6      2.16e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0297                       6      2.04e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0298                       6      2.04e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0299                       6      2.04e-6      (3, 3, 0)      (3, 3, 0)
DEVGLA2_0417                     5      1.58e-6      (5, 0, 0)      (5, 0, 0)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  1822
Failed in dense only:             18
Failed in sparse only:            104

Shared failure mode breakdown:
  Inertia mismatch on BOTH paths:          1567
  Residual-only fail on BOTH paths:         252
  Mixed (one inertia, other residual):        3

Shared failure size class breakdown:
  n <=  100:     327
  n <= 1000:    1495
  n >  1000:       0

Top 25 families in shared failures:
family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.97e-13
QPNBLEND                    362        362          0       2.77e-15
MSS1                        240        240          0       2.80e-15
CORE1                       141        141          0       1.07e-15
CRESC50                      97         97          0       2.71e-15
ACOPP30                      68         68          0       4.24e-14
FBRAIN3LS                    49          3         43        2.82e-7
CERI651DLS                   40          3         37        1.94e-7
PFIT4                        38         38          0       2.54e-14
CERI651A                     37         37          0       1.05e-13
HS46                         25          0         25        1.11e-7
PFIT2                        23          0         23        5.92e-6
CERI651CLS                   21          1         20        3.20e-7
CRESC100                     19         19          0       4.76e-15
PALMER1ENE                   17          0         17        1.87e-8
DEVGLA2                      15          0         15        1.58e-6
CERI651ALS                   14          2         12        1.45e-7
KIRBY2                       12         12          0       1.68e-13
MISTAKE                      10          0         10        1.33e-6
ALLINITA                      9          2          7        5.48e-7
DISCS                         8          8          0       1.98e-15
BENNETT5                      8          8          0       1.00e-13
DJTL                          7          0          7        9.27e-7
LSC2LS                        5          0          5        2.88e-8
SNAKE                         5          0          5        3.25e-9
  ... and 41 more families

Top 15 worst shared residuals:
name                             n    dense_res   sparse_res       expected     actual(sp)
ERRINBAR_0824                   27      1.87e-4      2.69e-4     (18, 9, 0)     (18, 9, 0)
PRICE4_0002                      2      7.74e-6      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0594                       6      6.93e-8      5.92e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0248                       6      5.39e-6      5.39e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0548                       6      3.66e-6      3.66e-6      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      3.63e-6      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0330                       6      3.11e-8      3.56e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0340                       6      2.71e-6      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0338                       6      2.71e-6      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0339                       6      2.71e-6      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0546                       6      6.76e-7      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0545                       6      6.76e-7      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0547                       6      6.76e-7      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0591                       6      3.07e-7      2.16e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0298                       6      1.37e-6      2.04e-6      (3, 3, 0)      (3, 3, 0)

=== Dense perf vs canonical oracles (154481 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153472       0.22       0.11       2.03      24.82     100.94
solve/MUMPS        153472       0.37       0.25       1.86      21.91     128.30
factor/SSIDS       154393       0.01       0.00       0.31       7.40      22.57
solve/SSIDS        154393       1.46       1.00       8.00      76.33     283.75

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
HS92                     3000       0.09       0.10       0.40
MGH10LS                  3000       0.11       0.11       0.33
HS91                     3000       0.10       0.10       0.44
HS118                    3000       0.35       0.33       5.00
BIGGSC4                  3000       0.08       0.08       0.41
HATFLDH                  3000       0.09       0.09       0.33
CONCON                   3000       0.44       0.44       2.15
SSI                      3000       0.09       0.10       0.29
DJTL                     3000       0.11       0.11       0.14
PALMER5A                 3000       0.09       0.09       1.46
HS13                     3000       0.11       0.11       0.14
ALLINITA                 3000       0.09       0.08       0.31
ALLINITC                 3000       0.09       0.08       0.25
PALMER7A                 3000       0.09       0.10       8.56
HS89                     3000       0.10       0.10       0.25
HS90                     3000       0.10       0.10       0.33
MCONCON                  3000       0.39       0.41       9.13
SSINE                    3000       0.07       0.08       0.70
HATFLDBNE                3000       0.10       0.10       0.55
AVION2                   2682       1.64       1.71       2.67
CERI651ALS               2331       0.08       0.08       0.12
PFIT4                    2286       0.08       0.08       0.20
CERI651C                 2233       0.08       0.08       0.38
CERI651CLS               2227       0.08       0.08       0.46
BATCH                    2054       3.04       3.12       4.92

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
CRESC100_0000                  806        20187          200     100.94
HAHN1_0461                     715        14892          178      83.66
HAHN1_0437                     715        15352          188      81.66
HAHN1_0391                     715        15175          188      80.72
HAHN1_0421                     715        14868          186      79.94
HAHN1_0292                     715        14592          183      79.74
HAHN1_0445                     715        14975          188      79.65
HAHN1_0380                     715        14870          187      79.52
HAHN1_0258                     715        14785          186      79.49
HAHN1_0266                     715        14677          185      79.34

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.44       0.33       1.86       3.72      87.46
solve/MUMPS        153560       0.29       0.20       1.00       5.45      22.38
factor/SSIDS       154500       0.02       0.01       0.28       0.94      21.22
solve/SSIDS        154500       1.16       1.00       4.00      14.75      66.50

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
DJTL                     3000       0.11       0.11       0.33
HATFLDH                  3000       0.49       0.50       1.90
SSI                      3000       0.15       0.18       0.44
HS118                    3000       0.97       1.00       2.87
MGH10LS                  3000       0.11       0.11       1.78
PALMER7A                 3000       0.27       0.30       1.80
HS90                     3000       0.20       0.20       0.80
BIGGSC4                  3000       0.44       0.45       2.09
HS13                     3000       0.18       0.22       2.50

(truncated from      498 lines to 350 line budget)
