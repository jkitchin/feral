# FERAL Context (auto-generated)

Generated: 2026-04-21T00:23:49Z

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
aaf37f4 plan: Phase 2.4.2 addendum -- updated kernel surface post-2.4.1b
d279810 session: 2026-04-20-07 -- Phase 2.4.1 closes (blocked kernel wired)
23cae7b session: 2026-04-20-06 -- Phase 2.4.1b Step 5 GREEN (may_delay wiring)
6606446 session: 2026-04-20-05 -- Phase 2.4.1b Step 4b GREEN (peek-ahead panel)
0160dad session: 2026-04-20-04 — Phase 2.4.1b Step 4a GREEN (thin delegation)
```

## Test Status
```

test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-47e687b76ffbe0b0)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_determinism_tiny ... ok
test test_gate_boundary_n_16 ... ok
test test_solve_parity_tiny_real_matrix ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
FERAL benchmark harness
  ordering: default (symbolic_factorize heuristic)
  scaling: default (SupernodeParams::default)
Loading matrices from data/benchmark-config.toml ... not found

name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10           37           11     (10, 0, 0)
spd_50             50           25            3     (50, 0, 0)
spd_100           100           83            5    (100, 0, 0)
spd_200           200          409           16    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           34            1    (30, 10, 0)
kkt_50_15          65           59            2    (50, 15, 0)
kkt_100_30        130          203            7   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 154588 matrices loaded

KKT summary: 154588 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000)
  Inertia match: 152911/154481 (99.0%)
  Residual pass: 154207/154481 (99.8%)
  Worst residual: 1.87e-4 (ERRINBAR_0824)

--- Sparse solver validation ---
Sparse solver: 154588/154588 total
  Inertia match vs MUMPS: 153008/154588 (99.0%)
  Residual pass: 154262/154588 (99.8%)
  Worst residual: 2.99e8 (POLAK6_0021)

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

--- Sparse failure analysis (1903 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.10e-12
QPNBLEND                    362        362          0       1.19e-15
MSS1                        240        240          0       1.76e-15
CORE1                       141        141          0       1.19e-15
CRESC50                      97         97          0       4.52e-16
ACOPP30                      66         66          0       1.45e-14
FBRAIN3LS                    59          4         57        6.09e-7
CERI651DLS                   49          3         46        8.88e-8
HS46                         43          0         43        9.26e-8
PFIT4                        38         38          0       2.51e-14
CERI651A                     37         37          0       8.74e-14
CERI651CLS                   26          1         25        2.41e-7
PFIT2                        23          0         23        3.55e-6
CERI651ALS                   22          2         20        1.13e-7
CRESC100                     19         19          0       2.57e-15
PALMER1ENE                   18          0         18        1.56e-8
DEVGLA2                      15          0         15        7.45e-7
KIRBY2                       12         12          0       1.68e-13
ALLINITA                     12          2         10        5.58e-7
MISTAKE                      11          0         11        2.34e-6
VESUVIO                      10         10          0       2.52e-13
DISCS                         8          8          0       2.20e-15
HATFLDFL                      8          0          8        2.69e-9
BENNETT5                      8          8          0       1.48e-13
SNAKE                         7          0          7        2.92e-9
  ... and 49 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
POLAK6_0021                      9       2.99e8      (5, 4, 0)      (3, 4, 2)
ERRINBAR_0824                   27      2.96e-4     (18, 9, 0)     (18, 9, 0)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)
FLETCHER_0002                   16      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0300                       6      3.55e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0390                       6      2.73e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0340                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0338                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0339                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
MISTAKE_0100                    22      2.34e-6     (9, 13, 0)     (9, 13, 0)
DJTL_0016                        2      1.80e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0297                       6      1.42e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0298                       6      1.42e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0299                       6      1.42e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0588                       6      1.36e-6      (3, 3, 0)      (3, 3, 0)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  1814
Failed in dense only:             26
Failed in sparse only:            89

Shared failure mode breakdown:
  Inertia mismatch on BOTH paths:          1566
  Residual-only fail on BOTH paths:         246
  Mixed (one inertia, other residual):        2

Shared failure size class breakdown:
  n <=  100:     321
  n <= 1000:    1493
  n >  1000:       0

Top 25 families in shared failures:
family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.10e-12
QPNBLEND                    362        362          0       2.77e-15
MSS1                        240        240          0       2.80e-15
CORE1                       141        141          0       1.19e-15
CRESC50                      97         97          0       2.71e-15
ACOPP30                      66         66          0       4.24e-14
FBRAIN3LS                    47          4         41        6.09e-7
CERI651DLS                   40          3         37        8.88e-8
PFIT4                        38         38          0       2.53e-14
CERI651A                     37         37          0       8.74e-14
HS46                         27          0         27        9.26e-8
PFIT2                        22          0         22        5.39e-6
CERI651CLS                   20          1         19        2.41e-7
CRESC100                     19         19          0       4.76e-15
CERI651ALS                   15          2         13        1.13e-7
DEVGLA2                      15          0         15        7.45e-7
PALMER1ENE                   13          0         13        1.56e-8
KIRBY2                       12         12          0       1.68e-13
MISTAKE                      11          0         11        2.34e-6
ALLINITA                      9          2          7        5.58e-7
BENNETT5                      8          8          0       1.48e-13
DISCS                         8          8          0       2.20e-15
DJTL                          7          0          7        1.80e-6
LSC2LS                        5          0          5        2.88e-8
SNAKE                         4          0          4        2.84e-9
  ... and 42 more families

Top 15 worst shared residuals:
name                             n    dense_res   sparse_res       expected     actual(sp)
POLAK6_0021                      9     9.21e-17       2.99e8      (5, 4, 0)      (3, 4, 2)
ERRINBAR_0824                   27      1.87e-4      2.96e-4     (18, 9, 0)     (18, 9, 0)
PRICE4_0002                      2      7.74e-6      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0248                       6      5.39e-6      1.26e-7      (3, 3, 0)      (3, 3, 0)
PFIT2_0548                       6      3.66e-6      2.86e-8      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      3.63e-6      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0390                       6      6.86e-7      2.73e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0340                       6      2.71e-6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0338                       6      2.71e-6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0339                       6      2.71e-6      2.71e-6      (3, 3, 0)      (3, 3, 0)
MISTAKE_0100                    22      1.33e-6      2.34e-6     (9, 13, 0)     (9, 13, 0)
DJTL_0016                        2      3.55e-7      1.80e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0298                       6      1.37e-6      1.42e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0297                       6      1.37e-6      1.42e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0299                       6      1.37e-6      1.42e-6      (3, 3, 0)      (3, 3, 0)

=== Dense perf vs canonical oracles (154481 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153472       0.22       0.11       1.97      24.10     387.64
solve/MUMPS        153472       0.38       0.25       2.09      31.33    2193.00
factor/SSIDS       154393       0.01       0.00       0.31       5.96     102.70
solve/SSIDS        154393       1.52       1.00       9.00      81.50    3198.12

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
PALMER5A                 3000       0.08       0.09       0.12
MGH10LS                  3000       0.11       0.11       0.14
HATFLDH                  3000       0.09       0.09       0.22
SSI                      3000       0.09       0.10       0.12
HS118                    3000       0.36       0.33       2.00
ALLINITA                 3000       0.09       0.08       0.27
HATFLDBNE                3000       0.11       0.10       0.54
BIGGSC4                  3000       0.08       0.08       0.18
CONCON                   3000       0.45       0.44       1.31
DJTL                     3000       0.11       0.11       0.14
PALMER7A                 3000       0.09       0.10       0.12
HS92                     3000       0.10       0.10       0.25
SSINE                    3000       0.07       0.08       0.11
HS13                     3000       0.11       0.11       0.14
ALLINITC                 3000       0.09       0.08       0.12
MCONCON                  3000       0.37       0.39       0.76
HS90                     3000       0.10       0.10       0.12
HS89                     3000       0.10       0.10       0.25
HS91                     3000       0.10       0.10       0.12
AVION2                   2682       1.68       1.75       4.17
CERI651ALS               2331       0.08       0.08       0.12
PFIT4                    2286       0.08       0.08       0.11
CERI651C                 2233       0.08       0.08       0.12
CERI651CLS               2227       0.08       0.08       0.12
BATCH                    2054       3.02       3.10       7.31

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
HAHN1_0252                     715        87607          226     387.64
CRESC100_0000                  806        77487          200     387.44
HAHN1_0210                     715        43429          217     200.13
HAHN1_0261                     715        33448          204     163.96
HAHN1_0222                     715        33752          212     159.21
GAUSS2_0033                    758        45268          285     158.84
HAHN1_0014                     715        32498          211     154.02
HAHN1_0062                     715        33718          220     153.26
HAHN1_0153                     715        32192          224     143.71
HAHN1_0127                     715        29397          206     142.70

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.35       0.27       1.67       3.52      11.09
solve/MUMPS        153560       0.29       0.20       1.09       4.82      16.12
factor/SSIDS       154500       0.02       0.01       0.25       0.83       4.16
solve/SSIDS        154500       1.15       1.00       3.67      13.50      46.67

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
HS13                     3000       0.11       0.11       0.25
ALLINITC                 3000       0.17       0.17       0.27
CONCON                   3000       0.68       0.71       1.69
PALMER7A                 3000       0.18       0.20       0.33
SSI                      3000       0.09       0.10       0.20
DJTL                     3000       0.11       0.11       0.14
HS89                     3000       0.16       0.18       0.29
SSINE                    3000       0.15       0.17       0.82

(truncated from      514 lines to 350 line budget)
