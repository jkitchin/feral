# FERAL Context (auto-generated)

Generated: 2026-05-13T16:00:46Z

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
be5a024 docs(small-front-perf): retrospective on #9/#11/#12/#13 model
d3aa627 docs(issue-10): research-phase gate measurement; recommend closure
d7267fe docs(session): 2026-05-13-02 — issue #9 Step 2 dispatch land
d3f1132 feat(issue-9): Step 2 — wire 32×32 dispatch via do_*_update fast-paths
1b94119 docs(session): session 2026-05-13-01 — Phase C land + issue #9 Step 3
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-4431a482ae44d88b)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_gate_boundary_n_16 ... ok
test test_determinism_tiny ... ok
test test_solve_parity_tiny_real_matrix ... ok

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
spd_10             10          173           27     (10, 0, 0)
spd_50             50           88           12     (50, 0, 0)
spd_100           100          361           16    (100, 0, 0)
spd_200           200         1625           40    (200, 0, 0)
kkt_10_3           13           10            1     (10, 3, 0)
kkt_30_10          40           65            2    (30, 10, 0)
kkt_50_15          65          146            5    (50, 15, 0)
kkt_100_30        130          644           16   (100, 30, 0)

8 matrices benchmarked

  SKIP heart6_iter_a (no .json sidecar)
  SKIP heart6_iter_a_rhs (no .json sidecar)
  SKIP heart6_iter_b (no .json sidecar)
  SKIP heart6_iter_b_rhs (no .json sidecar)
  SKIP heart6_iter_c (no .json sidecar)
  SKIP heart6_iter_c_rhs (no .json sidecar)
Loading KKT matrices from data/matrices/kkt ... 156927 matrices loaded

156927 KKT matrices total

KKT summary: 156927 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000, 2339 parse-skipped)
  Inertia match: 154428/154481 (100.0%)
  Residual pass: 154207/154481 (99.8%)
  Parse-skipped: 2339
  Worst residual: 1.87e-4 (ERRINBAR_0824)

--- Sparse solver validation ---
Sparse solver: 154588/156927 total
  Inertia match vs MUMPS: 154545/154588 (100.0%)
  Residual pass: 154250/154588 (99.8%)
  Parse-skipped: 2339
  Worst residual: 2.94e-4 (ERRINBAR_0824)

--- Dense failure analysis (320 failures) ---

family                    total    inertia   residual      worst_res
FBRAIN3LS                    48          4         48        2.82e-7
ACOPP30                      43         43          0       3.02e-14
CERI651DLS                   39          1         39        7.06e-8
HS46                         27          0         27        7.51e-8
PFIT2                        23          0         23        5.39e-6
CERI651CLS                   20          2         20        2.06e-7
PALMER1ENE                   17          0         17        1.22e-8
DEVGLA2                      15          0         15        1.50e-7
CERI651ALS                   15          0         15        4.31e-8
MISTAKE                      11          0         11        1.33e-6
DJTL                          7          0          7        5.33e-7
ALLINITA                      7          0          7        5.43e-7
SNAKE                         6          0          6        1.83e-9
LSC2LS                        5          0          5        1.95e-8
PALMER2E                      3          0          3        6.94e-9
EQC                           3          0          3        8.12e-8
HS118                         3          0          3        9.68e-8
PALMER4E                      2          0          2        4.84e-9
PALMER3E                      2          0          2        3.36e-9
ERRINBAR                      2          0          2        1.87e-4
TRUSPYR2                      2          0          2        1.70e-8
HATFLDFL                      2          0          2        1.56e-9
ACOPP14                       2          2          0       6.35e-16
HS13                          1          0          1        1.56e-9
CERI651BLS                    1          0          1        2.12e-9
  ... and 14 more families

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

--- Sparse failure analysis (377 failures) ---

family                    total    inertia   residual      worst_res
FBRAIN3LS                    57          2         57        6.09e-7
CERI651DLS                   46          1         46        8.88e-8
HS46                         43          0         43        9.26e-8
ACOPP30                      42         36          6        6.62e-6
CERI651CLS                   26          2         25        2.41e-7
PFIT2                        24          0         24        7.46e-6
CERI651ALS                   20          0         20        1.13e-7
PALMER1ENE                   18          0         18        1.56e-8
DEVGLA2                      15          0         15        7.45e-7
HATFLDFL                     12          0         12        3.36e-9
MISTAKE                      11          0         11        7.29e-7
SNAKE                         8          0          8        6.64e-9
ALLINITA                      7          0          7        9.70e-7
DJTL                          7          0          7        1.00e-6
LSC2LS                        6          0          6        2.88e-8
HS118                         3          0          3        6.52e-8
EQC                           3          0          3        2.15e-7
CERI651BLS                    2          0          2        2.28e-9
ERRINBAR                      2          0          2        2.94e-4
CONGIGMZ                      2          0          2        1.76e-8
HS114                         2          0          2        3.30e-8
ACOPP14                       2          2          0       3.25e-16
TRUSPYR2                      2          0          2        1.24e-8
HS13                          1          0          1        3.52e-9
BROWNBS                       1          0          1        2.11e-8
  ... and 15 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
ERRINBAR_0824                   27      2.94e-4     (18, 9, 0)     (18, 9, 0)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0297                       6      7.46e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0298                       6      7.46e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0299                       6      7.46e-6      (3, 3, 0)      (3, 3, 0)
ACOPP30_0067                   209      6.62e-6   (72, 137, 0)   (72, 137, 0)
ACOPP30_0064                   209      5.75e-6   (72, 137, 0)   (72, 137, 0)
ACOPP30_0066                   209      5.45e-6   (72, 137, 0)   (72, 137, 0)
ACOPP30_0065                   209      5.42e-6   (72, 137, 0)   (72, 137, 0)
ACOPP30_0063                   209      4.93e-6   (72, 137, 0)   (72, 137, 0)
FLETCHER_0002                   16      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0300                       6      3.55e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0327                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0328                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0329                       6      2.71e-6      (3, 3, 0)      (3, 3, 0)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  298
Failed in dense only:             22
Failed in sparse only:            79

Shared failure mode breakdown:
  Inertia mismatch on BOTH paths:            42
  Residual-only fail on BOTH paths:         248
  Mixed (one inertia, other residual):        8

Shared failure size class breakdown:
  n <=  100:     255
  n <= 1000:      43
  n >  1000:       0

Top 25 families in shared failures:
family                    total    inertia   residual      worst_res
FBRAIN3LS                    45          2         41        6.09e-7
ACOPP30                      41         35          0        6.62e-6
CERI651DLS                   37          1         36        8.88e-8
HS46                         27          0         27        9.26e-8
PFIT2                        23          0         23        7.46e-6
CERI651CLS                   20          2         18        2.41e-7
DEVGLA2                      15          0         15        7.45e-7
CERI651ALS                   13          0         13        1.13e-7
PALMER1ENE                   13          0         13        1.56e-8
MISTAKE                      11          0         11        1.33e-6
ALLINITA                      7          0          7        9.70e-7
DJTL                          7          0          7        1.00e-6
SNAKE                         6          0          6        6.64e-9
LSC2LS                        5          0          5        2.88e-8
EQC                           3          0          3        2.15e-7
HS118                         3          0          3        9.68e-8
TRUSPYR2                      2          0          2        1.70e-8
ERRINBAR                      2          0          2        2.94e-4
ACOPP14                       2          2          0       6.35e-16
LEVYMONT5                     1          0          1        4.20e-8
LEVYMONE5                     1          0          1        3.19e-8
CONGIGMZ                      1          0          1        1.76e-8
CERI651BLS                    1          0          1        2.13e-9
PALMER3E                      1          0          1        3.36e-9
TRO3X3                        1          0          1        1.35e-6
  ... and 10 more families

Top 15 worst shared residuals:
name                             n    dense_res   sparse_res       expected     actual(sp)
ERRINBAR_0824                   27      1.87e-4      2.94e-4     (18, 9, 0)     (18, 9, 0)
PRICE4_0002                      2      7.74e-6      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0297                       6      1.37e-6      7.46e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0298                       6      1.37e-6      7.46e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0299                       6      1.37e-6      7.46e-6      (3, 3, 0)      (3, 3, 0)
ACOPP30_0067                   209     2.18e-14      6.62e-6   (72, 137, 0)   (72, 137, 0)
ACOPP30_0064                   209     2.14e-14      5.75e-6   (72, 137, 0)   (72, 137, 0)
ACOPP30_0066                   209     1.28e-14      5.45e-6   (72, 137, 0)   (72, 137, 0)
ACOPP30_0065                   209     1.65e-14      5.42e-6   (72, 137, 0)   (72, 137, 0)
PFIT2_0248                       6      5.39e-6      1.49e-7      (3, 3, 0)      (3, 3, 0)
ACOPP30_0063                   209     2.24e-14      4.93e-6   (72, 137, 0)   (72, 137, 0)
PFIT2_0548                       6      3.66e-6      2.86e-8      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      3.63e-6      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0327                       6      4.15e-8      2.71e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0328                       6      4.15e-8      2.71e-6      (3, 3, 0)      (3, 3, 0)

--- Sparse-only failures (79 matrices: sparse fail, dense pass) ---

family                    total    inertia   residual      worst_res
HS46                         16          0         16        4.56e-9
FBRAIN3LS                    12          0         12        1.36e-8
HATFLDFL                     11          0         11        3.36e-9
CERI651DLS                    9          0          9        5.25e-9
CERI651ALS                    7          0          7        2.99e-9
CERI651CLS                    6          0          6        3.41e-9
PALMER1ENE                    5          0          5        1.24e-8
SNAKE                         2          0          2        1.39e-9
CONGIGMZ                      1          0          1        4.65e-9
ACOPP30                       1          1          0        2.06e-9
EXPFITNE                      1          0          1        1.39e-7
SIMPLLPB                      1          0          1        1.11e-9
PALMER5A                      1          0          1        2.20e-9
PALMER2E                      1          0          1        1.80e-9
PFIT2                         1          0          1        3.55e-6
HS114                         1          0          1        6.43e-9
CERI651BLS                    1          0          1        2.28e-9
PALMER1NE                     1          0          1        3.45e-9
LSC2LS                        1          0          1        1.03e-9

Top 25 worst sparse-only residuals (triage candidates):
name                             n       sp_res       expected     actual(sp)  i_ok  r_ok
PFIT2_0300                       6      3.55e-6      (3, 3, 0)      (3, 3, 0)  true false
EXPFITNE_0000                    2      1.39e-7      (2, 0, 0)      (2, 0, 0)  true false
FBRAIN3LS_0736                   6      1.36e-8      (6, 0, 0)      (6, 0, 0)  true false
PALMER1ENE_0107                  8      1.24e-8      (8, 0, 0)      (8, 0, 0)  true false
PALMER1ENE_0110                  8      7.08e-9      (8, 0, 0)      (8, 0, 0)  true false
FBRAIN3LS_0732                   6      7.03e-9      (6, 0, 0)      (6, 0, 0)  true false
HS114_0758                      21      6.43e-9    (10, 11, 0)    (10, 11, 0)  true false
FBRAIN3LS_0844                   6      5.80e-9      (6, 0, 0)      (6, 0, 0)  true false
CERI651DLS_0643                  7      5.25e-9      (7, 0, 0)      (7, 0, 0)  true false
CERI651DLS_0642                  7      5.25e-9      (7, 0, 0)      (7, 0, 0)  true false
CONGIGMZ_0000                    8      4.65e-9      (3, 5, 0)      (3, 5, 0)  true false
HS46_0376                        7      4.56e-9      (5, 2, 0)      (5, 2, 0)  true false
FBRAIN3LS_0681                   6      3.76e-9      (6, 0, 0)      (6, 0, 0)  true false
PALMER1NE_0007                   4      3.45e-9      (4, 0, 0)      (4, 0, 0)  true false
CERI651CLS_0292                  7      3.41e-9      (7, 0, 0)      (7, 0, 0)  true false
HATFLDFL_0480                    3      3.36e-9      (3, 0, 0)      (3, 0, 0)  true false
PALMER1ENE_0106                  8      3.30e-9      (8, 0, 0)      (8, 0, 0)  true false
HS46_0296                        7      3.24e-9      (5, 2, 0)      (5, 2, 0)  true false
HS46_0331                        7      3.22e-9      (5, 2, 0)      (5, 2, 0)  true false
CERI651ALS_0364                  7      2.99e-9      (7, 0, 0)      (7, 0, 0)  true false
HS46_0363                        7      2.98e-9      (5, 2, 0)      (5, 2, 0)  true false
HS46_0295                        7      2.95e-9      (5, 2, 0)      (5, 2, 0)  true false
CERI651DLS_0477                  7      2.88e-9      (7, 0, 0)      (7, 0, 0)  true false
CERI651ALS_0330                  7      2.83e-9      (7, 0, 0)      (7, 0, 0)  true false
CERI651ALS_0614                  7      2.75e-9      (7, 0, 0)      (7, 0, 0)  true false

--- Dense-only failures (22 matrices: dense fail, sparse pass) ---
name                             n        d_res       expected      actual(d)
PALMER2E_0144                    8      6.94e-9      (8, 0, 0)      (8, 0, 0)
PALMER2E_0143                    8      6.48e-9      (8, 0, 0)      (8, 0, 0)
PALMER4E_0041                    8      4.84e-9      (8, 0, 0)      (8, 0, 0)
CERI651DLS_0613                  7      4.18e-9      (7, 0, 0)      (7, 0, 0)

(truncated from      663 lines to 350 line budget)
