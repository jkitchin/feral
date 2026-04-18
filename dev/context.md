# FERAL Context (auto-generated)

Generated: 2026-04-18T19:19:15Z

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
38b925a Mongoose research note: GPL-3.0, cherry-pick ideas not dedicated crate
36e400d feral-kahip K4: flow-based node separator
68d55bc Session 2026-04-18-06: feral-kahip K3 complete
7651ba7 feral-kahip K3: flow-based edge refinement
cb14eea Session 2026-04-18-05: feral-kahip K2 complete
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s

     Running tests/threshold_consistency.rs (target/debug/deps/threshold_consistency-5816576ec7052023)

running 6 tests
test polak6_0021_residual_after_threshold_fix ... ignored
test dense_solve_skips_zero_pivots_rank_deficient ... ok
test factors_carry_zero_tol_from_params ... ok
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
spd_10             10           84           21     (10, 0, 0)
spd_50             50           64            7     (50, 0, 0)
spd_100           100          266           13    (100, 0, 0)
spd_200           200         1553           45    (200, 0, 0)
kkt_10_3           13           10            1     (10, 3, 0)
kkt_30_10          40           71            3    (30, 10, 0)
kkt_50_15          65          163            6    (50, 15, 0)
kkt_100_30        130          728           20   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 154588 matrices loaded

KKT summary: 154588 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000)
  Inertia match: 152911/154481 (99.0%)
  Residual pass: 154207/154481 (99.8%)
  Worst residual: 1.87e-4 (ERRINBAR_0824)

--- Sparse solver validation ---
Sparse solver: 154588/154588 total
  Inertia match vs MUMPS: 153008/154588 (99.0%)
  Residual pass: 154327/154588 (99.8%)
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
PALMER1ENE                   17          0         17        1.22e-8
CERI651ALS                   17          2         15        4.31e-8
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

--- Sparse failure analysis (1840 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       1.11e-13
QPNBLEND                    362        362          0       1.79e-15
MSS1                        240        240          0       1.65e-15
CORE1                       141        141          0       7.42e-16
CRESC50                      97         97          0       3.98e-16
ACOPP30                      68         68          0       1.80e-14
FBRAIN3LS                    52          3         50        2.79e-7
CERI651DLS                   39          3         36        1.93e-7
PFIT4                        38         38          0       9.30e-15
CERI651A                     37         37          0       8.84e-14
HS46                         30          0         30        5.30e-8
PFIT2                        24          0         24        3.66e-6
CERI651CLS                   21          1         20        2.53e-7
CRESC100                     19         19          0       3.65e-15
PALMER1ENE                   16          0         16        1.22e-8
CERI651ALS                   15          2         13        1.28e-7
DEVGLA2                      15          0         15        7.78e-7
KIRBY2                       12         12          0       1.26e-13
VESUVIO                      10         10          0       2.43e-13
ALLINITA                      9          2          7        5.48e-7
MISTAKE                       9          0          9        7.35e-7
DISCS                         8          8          0       1.77e-15
BENNETT5                      8          8          0       7.79e-14
DJTL                          7          0          7        5.33e-7
SNAKE                         5          0          5        3.25e-9
  ... and 44 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
ERRINBAR_0824                   27      2.69e-4     (18, 9, 0)     (18, 9, 0)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0548                       6      3.66e-6      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0545                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0546                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0547                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0297                       6      2.04e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0298                       6      2.04e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0299                       6      2.04e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0588                       6      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0589                       6      1.36e-6      (3, 3, 0)      (3, 3, 0)
TRO3X3_0637                     43      9.68e-7    (30, 13, 0)    (30, 13, 0)
DEVGLA2_0417                     5      7.78e-7      (5, 0, 0)      (5, 0, 0)
MISTAKE_0100                    22      7.35e-7     (9, 13, 0)     (9, 13, 0)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  1810
Failed in dense only:             30
Failed in sparse only:            30

Shared failure mode breakdown:
  Inertia mismatch on BOTH paths:          1567
  Residual-only fail on BOTH paths:         240
  Mixed (one inertia, other residual):        3

Shared failure size class breakdown:
  n <=  100:     315
  n <= 1000:    1495
  n >  1000:       0

Top 25 families in shared failures:
family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.78e-13
QPNBLEND                    362        362          0       2.77e-15
MSS1                        240        240          0       2.80e-15
CORE1                       141        141          0       1.07e-15
CRESC50                      97         97          0       2.71e-15
ACOPP30                      68         68          0       4.24e-14
FBRAIN3LS                    48          3         42        2.82e-7
CERI651DLS                   38          3         35        1.93e-7
PFIT4                        38         38          0       2.53e-14
CERI651A                     37         37          0       8.84e-14
PFIT2                        23          0         23        5.39e-6
HS46                         22          0         22        7.51e-8
CERI651CLS                   21          1         20        2.53e-7
CRESC100                     19         19          0       4.76e-15
PALMER1ENE                   16          0         16        1.22e-8
DEVGLA2                      15          0         15        7.78e-7
CERI651ALS                   14          2         12        1.28e-7
KIRBY2                       12         12          0       1.30e-13
MISTAKE                       9          0          9        1.33e-6
ALLINITA                      9          2          7        5.48e-7
BENNETT5                      8          8          0       7.79e-14
DISCS                         8          8          0       1.98e-15
DJTL                          7          0          7        5.33e-7
SNAKE                         4          0          4        3.25e-9
LSC2LS                        4          0          4        1.95e-8
  ... and 40 more families

Top 15 worst shared residuals:
name                             n    dense_res   sparse_res       expected     actual(sp)
ERRINBAR_0824                   27      1.87e-4      2.69e-4     (18, 9, 0)     (18, 9, 0)
PRICE4_0002                      2      7.74e-6      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0248                       6      5.39e-6      5.74e-7      (3, 3, 0)      (3, 3, 0)
PFIT2_0548                       6      3.66e-6      3.66e-6      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      3.63e-6      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0340                       6      2.71e-6      4.36e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0339                       6      2.71e-6      4.36e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0338                       6      2.71e-6      4.36e-8      (3, 3, 0)      (3, 3, 0)
PFIT2_0545                       6      6.76e-7      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0546                       6      6.76e-7      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0547                       6      6.76e-7      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0297                       6      1.37e-6      2.04e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0298                       6      1.37e-6      2.04e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0299                       6      1.37e-6      2.04e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0589                       6      2.13e-8      1.36e-6      (3, 3, 0)      (3, 3, 0)

=== Dense perf vs canonical oracles (154481 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153472       0.22       0.11       1.97      23.86     973.79
solve/MUMPS        153472       0.38       0.25       2.00      30.73     213.62
factor/SSIDS       154393       0.01       0.00       0.30       7.07     142.15
solve/SSIDS        154393       1.51       1.00       9.00      78.00     641.57

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
SSI                      3000       0.09       0.10       0.22
HS13                     3000       0.11       0.11       1.75
HS89                     3000       0.10       0.10       0.14
ALLINITA                 3000       0.09       0.08       0.62
MCONCON                  3000       0.39       0.41       0.76
PALMER7A                 3000       0.09       0.10       0.27
SSINE                    3000       0.07       0.08       0.11
HS92                     3000       0.09       0.10       0.22
HATFLDH                  3000       0.09       0.09       0.22
HS118                    3000       0.31       0.29       1.00
HS91                     3000       0.10       0.10       0.18
BIGGSC4                  3000       0.08       0.08       0.69
ALLINITC                 3000       0.09       0.08       0.90
CONCON                   3000       0.44       0.44       1.54
HS90                     3000       0.10       0.10       0.12
DJTL                     3000       0.11       0.11       0.14
HATFLDBNE                3000       0.10       0.10       0.64
MGH10LS                  3000       0.11       0.11       0.82
PALMER5A                 3000       0.09       0.09       0.30
AVION2                   2682       1.71       1.79       3.00
CERI651ALS               2331       0.08       0.08       0.58
PFIT4                    2286       0.08       0.08       0.20
CERI651C                 2233       0.08       0.08       0.18
CERI651CLS               2227       0.08       0.08       0.17
BATCH                    2054       2.97       3.04       3.93

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
HAHN1_0010                     715       215208          221     973.79
HAHN1_0011                     715        24773          204     121.44
CRESC100_0000                  806        19531          200      97.66
HAHN1_0012                     715        16605          203      81.80
HAHN1_0461                     715        14245          178      80.03
HAHN1_0258                     715        14464          186      77.76
HAHN1_0036                     715        15075          194      77.71
HAHN1_0292                     715        14206          183      77.63
HAHN1_0476                     715        14391          187      76.96
HAHN1_0471                     715        14142          184      76.86

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.42       0.33       1.78       3.58     484.79
solve/MUMPS        153560       0.45       0.33       2.33      12.61     115.39
factor/SSIDS       154500       0.02       0.01       0.27       0.90      21.89
solve/SSIDS        154500       1.76       1.00      11.50      35.25     109.89

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
HS89                     3000       0.20       0.20       0.42
SSI                      3000       0.15       0.18       0.61
CONCON                   3000       0.75       0.77       2.44
BIGGSC4                  3000       0.42       0.42       1.36
MGH10LS                  3000       0.11       0.11       0.33
PALMER5A                 3000       0.30       0.30       1.08
ALLINITC                 3000       0.17       0.17       1.18
MCONCON                  3000       0.64       0.67       4.74
DJTL                     3000       0.11       0.11       0.62
HS118                    3000       0.94       0.94       2.64

(truncated from      494 lines to 350 line budget)
