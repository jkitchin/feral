# FERAL Context (auto-generated)

Generated: 2026-04-19T21:47:16Z

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
8e68482 research: sparse-tail perf survey — D.2 ruled out, D.1 next
af9315d scaling: Policy 4 - Auto fallback to InfNorm when MC64 misfires
0c39e27 scaling: flip default ScalingStrategy from InfNorm to Auto
1819480 schur_kernel: dispatch _nofma kernels on aarch64 + x86_64
5ed26b6 journal: bench p90 1% "regression" is noise, not real
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s

     Running tests/threshold_consistency.rs (target/debug/deps/threshold_consistency-1a3fc6fe10f6c962)

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
  ordering: default (symbolic_factorize heuristic)
  scaling: default (SupernodeParams::default)
Loading matrices from data/benchmark-config.toml ... not found

name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10           29            9     (10, 0, 0)
spd_50             50           19            2     (50, 0, 0)
spd_100           100           75            5    (100, 0, 0)
spd_200           200          409           17    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           20            1    (30, 10, 0)
kkt_50_15          65           49            2    (50, 15, 0)
kkt_100_30        130          204            7   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 154588 matrices loaded

KKT summary: 154588 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000)
  Inertia match: 152911/154481 (99.0%)
  Residual pass: 154207/154481 (99.8%)
  Worst residual: 1.87e-4 (ERRINBAR_0824)

--- Sparse solver validation ---
Sparse solver: 154588/154588 total
  Inertia match vs MUMPS: 153009/154588 (99.0%)
  Residual pass: 154233/154588 (99.8%)
  Worst residual: 1.31e13 (POLAK6_0021)

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

--- Sparse failure analysis (1932 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.10e-12
QPNBLEND                    362        362          0       1.19e-15
MSS1                        240        240          0       1.76e-15
CORE1                       141        141          0       8.00e-16
CRESC50                      97         97          0       4.52e-16
ACOPP30                      66         66          0       1.45e-14
FBRAIN3LS                    60          3         58        2.79e-7
CERI651DLS                   48          3         45        1.94e-7
HS46                         43          0         43        1.11e-7
PFIT4                        38         38          0       2.51e-14
CERI651A                     37         37          0       8.74e-14
CERI651CLS                   26          1         25        3.20e-7
DEVGLA2                      26          0         26        1.58e-6
PFIT2                        24          0         24        7.84e-6
PALMER1ENE                   22          0         22        1.87e-8
CERI651ALS                   21          2         19        1.45e-7
CRESC100                     19         19          0       2.57e-15
SNAKE                        15          0         15        6.57e-9
HATFLDFL                     14          0         14        2.77e-9
KIRBY2                       12         12          0       1.68e-13
MISTAKE                      11          0         11        7.13e-7
ALLINITA                     11          2          9        5.50e-7
VESUVIO                      10         10          0       2.52e-13
BENNETT5                      8          8          0       1.48e-13
DISCS                         8          8          0       2.20e-15
  ... and 48 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
POLAK6_0021                      9      1.31e13      (5, 4, 0)      (3, 4, 2)
ERRINBAR_0824                   27      2.74e-4     (18, 9, 0)     (18, 9, 0)
PFIT2_0390                       6      7.84e-6      (3, 3, 0)      (3, 3, 0)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0297                       6      5.43e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0298                       6      5.43e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0299                       6      5.43e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0590                       6      4.08e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0588                       6      4.08e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0589                       6      4.08e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0548                       6      3.66e-6      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0341                       6      3.59e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0327                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0328                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  1822
Failed in dense only:             18
Failed in sparse only:            110

Shared failure mode breakdown:
  Inertia mismatch on BOTH paths:          1565
  Residual-only fail on BOTH paths:         254
  Mixed (one inertia, other residual):        3

Shared failure size class breakdown:
  n <=  100:     329
  n <= 1000:    1493
  n >  1000:       0

Top 25 families in shared failures:
family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.10e-12
QPNBLEND                    362        362          0       2.77e-15
MSS1                        240        240          0       2.80e-15
CORE1                       141        141          0       1.07e-15
CRESC50                      97         97          0       2.71e-15
ACOPP30                      66         66          0       4.24e-14
FBRAIN3LS                    49          3         43        2.82e-7
CERI651DLS                   40          3         37        1.94e-7
PFIT4                        38         38          0       2.53e-14
CERI651A                     37         37          0       8.74e-14
HS46                         25          0         25        1.11e-7
PFIT2                        23          0         23        7.84e-6
CERI651CLS                   21          1         20        3.20e-7
CRESC100                     19         19          0       4.76e-15
PALMER1ENE                   17          0         17        1.87e-8
DEVGLA2                      15          0         15        1.58e-6
CERI651ALS                   14          2         12        1.45e-7
KIRBY2                       12         12          0       1.68e-13
MISTAKE                      10          0         10        1.33e-6
ALLINITA                      9          2          7        5.50e-7
BENNETT5                      8          8          0       1.48e-13
DISCS                         8          8          0       2.20e-15
DJTL                          7          0          7        1.80e-6
SNAKE                         6          0          6        6.57e-9
LSC2LS                        5          0          5        1.95e-8
  ... and 42 more families

Top 15 worst shared residuals:
name                             n    dense_res   sparse_res       expected     actual(sp)
POLAK6_0021                      9     9.21e-17      1.31e13      (5, 4, 0)      (3, 4, 2)
ERRINBAR_0824                   27      1.87e-4      2.74e-4     (18, 9, 0)     (18, 9, 0)
PFIT2_0390                       6      6.86e-7      7.84e-6      (3, 3, 0)      (3, 3, 0)
PRICE4_0002                      2      7.74e-6      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0297                       6      1.37e-6      5.43e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0298                       6      1.37e-6      5.43e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0299                       6      1.37e-6      5.43e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0248                       6      5.39e-6      7.57e-7      (3, 3, 0)      (3, 3, 0)
PFIT2_0590                       6      1.06e-8      4.08e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0589                       6      2.13e-8      4.08e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0588                       6      2.13e-8      4.08e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0548                       6      3.66e-6      3.66e-6      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      3.63e-6      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0341                       6      1.40e-8      3.59e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0328                       6      4.15e-8      2.71e-6      (3, 3, 0)      (3, 3, 0)

=== Dense perf vs canonical oracles (154481 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153472       0.22       0.11       2.06      25.25     176.63
solve/MUMPS        153472       0.37       0.25       1.86      22.20     208.97
factor/SSIDS       154393       0.01       0.00       0.31       7.39      41.93
solve/SSIDS        154393       1.45       1.00       8.00      75.67     260.12

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
BIGGSC4                  3000       0.08       0.08       0.53
MCONCON                  3000       0.39       0.41       0.95
HS90                     3000       0.10       0.10       0.12
HS13                     3000       0.11       0.11       0.14
HS92                     3000       0.09       0.10       2.88
HATFLDH                  3000       0.09       0.09       1.83
HS91                     3000       0.10       0.10       2.20
PALMER5A                 3000       0.09       0.09       0.38
HS118                    3000       0.34       0.33       2.41
ALLINITA                 3000       0.09       0.08       0.33
ALLINITC                 3000       0.09       0.08       0.25
HS89                     3000       0.10       0.10       0.25
HATFLDBNE                3000       0.10       0.10       4.95
MGH10LS                  3000       0.11       0.11       0.14
PALMER7A                 3000       0.09       0.10       0.40
SSINE                    3000       0.07       0.08       0.36
DJTL                     3000       0.11       0.11       0.14
CONCON                   3000       0.44       0.47       2.25
SSI                      3000       0.09       0.10       1.00
AVION2                   2682       1.65       1.71       4.74
CERI651ALS               2331       0.08       0.08       1.71
PFIT4                    2286       0.08       0.08       0.36
CERI651C                 2233       0.09       0.08       1.69
CERI651CLS               2227       0.08       0.08       0.80
BATCH                    2054       3.16       3.20       6.13

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
CHWIRUT1_0081                  645        42392          240     176.63
HAHN1_0476                     715        29774          187     159.22
CRESC100_0000                  806        20808          200     104.04
HAHN1_0477                     715        18153          189      96.05
HAHN1_0479                     715        16378          190      86.20
HAHN1_0485                     715        16162          188      85.97
HAHN1_0016                     715        16115          192      83.93
HAHN1_0421                     715        15474          186      83.19
HAHN1_0420                     715        15615          188      83.06
HAHN1_0475                     715        15840          192      82.50

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.48       0.36       1.98       4.03      11.52
solve/MUMPS        153560       0.30       0.20       1.25       5.71      19.78
factor/SSIDS       154500       0.02       0.01       0.29       0.93       4.31
solve/SSIDS        154500       1.19       1.00       4.29      16.00      77.00

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
CONCON                   3000       0.85       0.85       5.15
HATFLDH                  3000       0.55       0.55       2.30
PALMER7A                 3000       0.27       0.30       0.56
HS90                     3000       0.20       0.20       0.56
HS91                     3000       0.28       0.30       3.80
HS118                    3000       1.05       1.07       8.29
DJTL                     3000       0.12       0.11       0.40
HS13                     3000       0.22       0.22       0.47

(truncated from      506 lines to 350 line budget)
