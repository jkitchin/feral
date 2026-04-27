# FERAL Context (auto-generated)

Generated: 2026-04-27T01:47:34Z

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
53c07bb feat(bench): streaming load + FERAL_SPARSE_MAX cap
35247f0 docs: correct session 03 expanded-corpus finding
cfbc548 session: 2026-04-26-03 sparsity gap resolved + corpus expansion blocked
ae81b81 fix(factor_nnz): match SSIDS lower-tri-with-diag accounting
976dcad docs(polak6): bench-path follow-up to the 2026-04-19 triage
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-4143909cb5d74be9)

running 5 tests
test test_gate_tiny_sparse_in ... ok
test test_gate_just_outside_n_tiny ... ok
test test_determinism_tiny ... ok
test test_solve_parity_tiny_real_matrix ... ok
test test_gate_boundary_n_16 ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 1 test
test src/symbolic/profiler.rs - symbolic::profiler::SymbolicProfiler (line 27) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
FERAL benchmark harness
  ordering: default (symbolic_factorize heuristic)
  scaling: default (SupernodeParams::default)
Loading matrices from data/benchmark-config.toml ... not found

name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10           45            0     (10, 0, 0)
spd_50             50           26            3     (50, 0, 0)
spd_100           100           80            4    (100, 0, 0)
spd_200           200          389           16    (200, 0, 0)
kkt_10_3           13            2            0     (10, 3, 0)
kkt_30_10          40           20            1    (30, 10, 0)
kkt_50_15          65           60            2    (50, 15, 0)
kkt_100_30        130          200            7   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 156927 matrices loaded

156927 KKT matrices total

KKT summary: 156927 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000, 2339 parse-skipped)
  Inertia match: 154408/154481 (100.0%)
  Residual pass: 154207/154481 (99.8%)
  Parse-skipped: 2339
  Worst residual: 1.87e-4 (ERRINBAR_0824)

--- Sparse solver validation ---
Sparse solver: 154588/156927 total
  Inertia match vs MUMPS: 154528/154588 (100.0%)
  Residual pass: 154254/154588 (99.8%)
  Parse-skipped: 2339
  Worst residual: 2.99e8 (POLAK6_0021)

--- Dense failure analysis (343 failures) ---

family                    total    inertia   residual      worst_res
ACOPP30                      68         68          0       3.02e-14
FBRAIN3LS                    48          4         48        2.82e-7
CERI651DLS                   39          0         39        7.06e-8
HS46                         27          0         27        7.51e-8
PFIT2                        23          0         23        5.39e-6
CERI651CLS                   20          0         20        2.06e-7
PALMER1ENE                   17          0         17        1.22e-8
CERI651ALS                   15          0         15        4.31e-8
DEVGLA2                      15          0         15        1.50e-7
MISTAKE                      11          0         11        1.33e-6
DJTL                          7          0          7        5.33e-7
ALLINITA                      7          0          7        5.43e-7
SNAKE                         6          0          6        1.83e-9
LSC2LS                        5          0          5        1.95e-8
EQC                           3          0          3        8.12e-8
HS118                         3          0          3        9.68e-8
PALMER2E                      3          0          3        6.94e-9
PALMER3E                      2          0          2        3.36e-9
HATFLDFL                      2          0          2        1.56e-9
TRUSPYR2                      2          0          2        1.70e-8
PALMER4E                      2          0          2        4.84e-9
ERRINBAR                      2          0          2        1.87e-4
LEVYMONE5                     1          0          1        7.46e-9
PRICE4                        1          0          1        7.74e-6
PALMER1D                      1          0          1        2.19e-9
  ... and 13 more families

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

--- Sparse failure analysis (389 failures) ---

family                    total    inertia   residual      worst_res
ACOPP30                      63         57          8        1.94e-5
FBRAIN3LS                    57          2         57        6.09e-7
CERI651DLS                   46          0         46        8.88e-8
HS46                         43          0         43        9.26e-8
CERI651CLS                   25          0         25        2.41e-7
PFIT2                        23          0         23        3.55e-6
CERI651ALS                   20          0         20        1.13e-7
PALMER1ENE                   18          0         18        1.56e-8
DEVGLA2                      15          0         15        7.45e-7
MISTAKE                      11          0         11        2.34e-6
ALLINITA                     10          0         10        5.58e-7
HATFLDFL                      8          0          8        2.69e-9
DJTL                          7          0          7        1.80e-6
SNAKE                         7          0          7        2.92e-9
LSC2LS                        6          0          6        2.88e-8
EQC                           3          0          3        1.47e-7
HS118                         3          0          3        1.31e-7
HS114                         2          0          2        3.65e-8
ERRINBAR                      2          0          2        2.96e-4
CERI651BLS                    2          0          2        2.28e-9
TRO3X3                        1          0          1        9.11e-7
FLETCHER                      1          0          1        3.63e-6
HS13                          1          0          1        3.52e-9
PALMER4E                      1          0          1        2.85e-9
PALMER5A                      1          0          1        2.20e-9
  ... and 13 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
POLAK6_0021                      9       2.99e8      (5, 4, 0)      (4, 4, 1)
ERRINBAR_0824                   27      2.96e-4     (18, 9, 0)     (18, 9, 0)
ACOPP30_0061                   209      1.94e-5   (72, 137, 0)   (71, 138, 0)
ACOPP30_0060                   209      1.80e-5   (72, 137, 0)   (71, 138, 0)
ACOPP30_0059                   209      1.08e-5   (72, 137, 0)   (72, 137, 0)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)
FLETCHER_0002                   16      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0300                       6      3.55e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0390                       6      2.73e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0340                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0338                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0339                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
ACOPP30_0056                   209      2.63e-6   (72, 137, 0)   (72, 137, 0)
ACOPP30_0057                   209      2.39e-6   (72, 137, 0)   (72, 137, 0)
MISTAKE_0100                    22      2.34e-6     (9, 13, 0)     (9, 13, 0)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  314
Failed in dense only:             29
Failed in sparse only:            75

Shared failure mode breakdown:
  Inertia mismatch on BOTH paths:            60
  Residual-only fail on BOTH paths:         246
  Mixed (one inertia, other residual):        8

Shared failure size class breakdown:
  n <=  100:     251
  n <= 1000:      63
  n >  1000:       0

Top 25 families in shared failures:
family                    total    inertia   residual      worst_res
ACOPP30                      63         57          0        1.94e-5
FBRAIN3LS                    45          2         41        6.09e-7
CERI651DLS                   37          0         37        8.88e-8
HS46                         27          0         27        9.26e-8
PFIT2                        22          0         22        5.39e-6
CERI651CLS                   19          0         19        2.41e-7
DEVGLA2                      15          0         15        7.45e-7
CERI651ALS                   13          0         13        1.13e-7
PALMER1ENE                   13          0         13        1.56e-8
MISTAKE                      11          0         11        2.34e-6
DJTL                          7          0          7        1.80e-6
ALLINITA                      7          0          7        5.58e-7
LSC2LS                        5          0          5        2.88e-8
SNAKE                         4          0          4        2.84e-9
HS118                         3          0          3        1.31e-7
EQC                           3          0          3        1.47e-7
ERRINBAR                      2          0          2        2.96e-4
CONGIGMZ                      1          0          1        9.85e-9
PALMER4E                      1          0          1        3.30e-9
HS13                          1          0          1        3.52e-9
POLAK6                        1          1          0         2.99e8
BROWNBS                       1          0          1        2.11e-8
LEVYMONE5                     1          0          1        7.46e-9
TRO3X3                        1          0          1        9.11e-7
TRUSPYR2                      1          0          1        1.70e-8
  ... and 10 more families

Top 15 worst shared residuals:
name                             n    dense_res   sparse_res       expected     actual(sp)
POLAK6_0021                      9     9.21e-17       2.99e8      (5, 4, 0)      (4, 4, 1)
ERRINBAR_0824                   27      1.87e-4      2.96e-4     (18, 9, 0)     (18, 9, 0)
ACOPP30_0061                   209     1.39e-14      1.94e-5   (72, 137, 0)   (71, 138, 0)
ACOPP30_0060                   209     1.29e-14      1.80e-5   (72, 137, 0)   (71, 138, 0)
ACOPP30_0059                   209     1.69e-14      1.08e-5   (72, 137, 0)   (72, 137, 0)
PRICE4_0002                      2      7.74e-6      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0248                       6      5.39e-6      1.26e-7      (3, 3, 0)      (3, 3, 0)
PFIT2_0548                       6      3.66e-6      2.86e-8      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      3.63e-6      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0390                       6      6.86e-7      2.73e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0340                       6      2.71e-6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0339                       6      2.71e-6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0338                       6      2.71e-6      2.71e-6      (3, 3, 0)      (3, 3, 0)
ACOPP30_0056                   209     1.19e-14      2.63e-6   (72, 137, 0)   (72, 137, 0)
ACOPP30_0057                   209     2.30e-14      2.39e-6   (72, 137, 0)   (72, 137, 0)

--- Sparse-only failures (75 matrices: sparse fail, dense pass) ---

family                    total    inertia   residual      worst_res
HS46                         16          0         16        4.56e-9
FBRAIN3LS                    12          0         12        1.36e-8
CERI651DLS                    9          0          9        5.25e-9
CERI651ALS                    7          0          7        2.99e-9
HATFLDFL                      7          0          7        2.29e-9
CERI651CLS                    6          0          6        3.41e-9
PALMER1ENE                    5          0          5        1.24e-8
SNAKE                         3          0          3        2.92e-9
ALLINITA                      3          0          3        8.64e-9
CERI651BLS                    1          0          1        2.28e-9
PFIT2                         1          0          1        3.55e-6
HS114                         1          0          1        6.16e-9
PALMER2E                      1          0          1        1.80e-9
PALMER5A                      1          0          1        2.20e-9
PALMER1NE                     1          0          1        3.45e-9
LSC2LS                        1          0          1       8.45e-10

Top 25 worst sparse-only residuals (triage candidates):
name                             n       sp_res       expected     actual(sp)  i_ok  r_ok
PFIT2_0300                       6      3.55e-6      (3, 3, 0)      (3, 3, 0)  true false
FBRAIN3LS_0736                   6      1.36e-8      (6, 0, 0)      (6, 0, 0)  true false
PALMER1ENE_0107                  8      1.24e-8      (8, 0, 0)      (8, 0, 0)  true false
ALLINITA_0750                    8      8.64e-9      (4, 4, 0)      (4, 4, 0)  true false
PALMER1ENE_0110                  8      7.08e-9      (8, 0, 0)      (8, 0, 0)  true false
FBRAIN3LS_0732                   6      7.03e-9      (6, 0, 0)      (6, 0, 0)  true false
HS114_0758                      21      6.16e-9    (10, 11, 0)    (10, 11, 0)  true false
FBRAIN3LS_0844                   6      5.80e-9      (6, 0, 0)      (6, 0, 0)  true false
CERI651DLS_0642                  7      5.25e-9      (7, 0, 0)      (7, 0, 0)  true false
CERI651DLS_0643                  7      5.25e-9      (7, 0, 0)      (7, 0, 0)  true false
HS46_0376                        7      4.56e-9      (5, 2, 0)      (5, 2, 0)  true false
ALLINITA_0756                    8      4.42e-9      (4, 4, 0)      (4, 4, 0)  true false
ALLINITA_0758                    8      3.88e-9      (4, 4, 0)      (4, 4, 0)  true false
FBRAIN3LS_0681                   6      3.76e-9      (6, 0, 0)      (6, 0, 0)  true false
PALMER1NE_0007                   4      3.45e-9      (4, 0, 0)      (4, 0, 0)  true false
CERI651CLS_0292                  7      3.41e-9      (7, 0, 0)      (7, 0, 0)  true false
PALMER1ENE_0106                  8      3.30e-9      (8, 0, 0)      (8, 0, 0)  true false
HS46_0296                        7      3.24e-9      (5, 2, 0)      (5, 2, 0)  true false
HS46_0331                        7      3.22e-9      (5, 2, 0)      (5, 2, 0)  true false
CERI651ALS_0364                  7      2.99e-9      (7, 0, 0)      (7, 0, 0)  true false
HS46_0363                        7      2.98e-9      (5, 2, 0)      (5, 2, 0)  true false
HS46_0295                        7      2.95e-9      (5, 2, 0)      (5, 2, 0)  true false
SNAKE_0614                       4      2.92e-9      (2, 2, 0)      (2, 2, 0)  true false
CERI651DLS_0477                  7      2.88e-9      (7, 0, 0)      (7, 0, 0)  true false
CERI651ALS_0330                  7      2.83e-9      (7, 0, 0)      (7, 0, 0)  true false

--- Dense-only failures (29 matrices: dense fail, sparse pass) ---
name                             n        d_res       expected      actual(d)
PFIT2_0341                       6      1.40e-8      (3, 3, 0)      (3, 3, 0)
PALMER2E_0144                    8      6.94e-9      (8, 0, 0)      (8, 0, 0)
PALMER2E_0143                    8      6.48e-9      (8, 0, 0)      (8, 0, 0)
TRUSPYR2_0199                   22      6.27e-9    (11, 11, 0)    (11, 11, 0)
PALMER4E_0041                    8      4.84e-9      (8, 0, 0)      (8, 0, 0)
CERI651DLS_0613                  7      4.18e-9      (7, 0, 0)      (7, 0, 0)
PALMER1ENE_0102                  8      4.16e-9      (8, 0, 0)      (8, 0, 0)
FBRAIN3LS_0827                   6      3.86e-9      (6, 0, 0)      (6, 0, 0)
PALMER2E_0142                    8      3.00e-9      (8, 0, 0)      (8, 0, 0)
PALMER1ENE_0108                  8      2.69e-9      (8, 0, 0)      (8, 0, 0)

=== Dense perf vs canonical oracles (154481 matrices with oracle timings) ===


(truncated from      625 lines to 350 line budget)
