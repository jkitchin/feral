# FERAL Context (auto-generated)

Generated: 2026-04-26T22:04:16Z

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
4757fdb feat(diag_fill_parity): three-ordering streaming fill-parity diagnostic
ff24741 feat(oracles): --max-n / --mem-gb / skip-list guards
202c5f5 fix(bench): make dense form lazy to fit kkt corpus in RAM
aa0d097 fix(feral-amd): exclusive pme2 to avoid -1i32 → usize::MAX wrap-around
7d22ee3 journal: 2026-04-26 mittelmann correctness + qcqp drill
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-b4cc4aabe0a8b5e9)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_determinism_tiny ... ok
test test_gate_boundary_n_16 ... ok
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
spd_10             10           46           10     (10, 0, 0)
spd_50             50           23            3     (50, 0, 0)
spd_100           100           84            5    (100, 0, 0)
spd_200           200          427           17    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           21            1    (30, 10, 0)
kkt_50_15          65           52            2    (50, 15, 0)
kkt_100_30        130          220            7   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 154588 matrices loaded

  SKIP DIXMAANJ_0001 (mtx parse error: I/O error: data/matrices/kkt-expansion/DIXMAANJ/DIXMAANJ_0001.mtx: line 3904978: invalid value '8.85215930281821155e-')
  SKIP DIXMAANL_0001 (mtx parse error: I/O error: data/matrices/kkt-expansion/DIXMAANL/DIXMAANL_0001.mtx: line 3928748: invalid value '4.81628433180690541e-')
  SKIP FLETCBV3_0001 (mtx parse error: I/O error: data/matrices/kkt-expansion/FLETCBV3/FLETCBV3_0001.mtx: line 3474057: expected 'i j value', got '1431')
  SKIP FMINSURF_0000 (no .json sidecar)
  SKIP ODNAMUR_0000 (no .json sidecar)
  SKIP ROCKET_0067 (no .json sidecar)
  SKIP ROSEPETAL_0000 (no .json sidecar)
  SKIP VAREIGVL_0000 (no .json sidecar)
Loading KKT matrices from data/matrices/kkt-expansion ... 12430 matrices loaded

  SKIP arki0003_0005 (mtx parse error: I/O error: data/matrices/kkt-mittelmann/arki0003/arki0003_0005.mtx: line 735245: expected 'i j value', got '931')
  SKIP cont5_1_l_0003 (mtx parse error: I/O error: data/matrices/kkt-mittelmann/cont5_1_l/cont5_1_l_0003.mtx: line 302696: expected 'i j value', got '133930')
Loading KKT matrices from data/matrices/kkt-mittelmann ... 596 matrices loaded

167614 KKT matrices total

KKT summary: 167614 matrices (157494 dense-eligible n <= 1000, 10120 skipped n > 1000)
  Inertia match: 157356/157494 (99.9%)
  Residual pass: 157220/157494 (99.8%)
  Worst residual: 1.87e-4 (ERRINBAR_0824)

--- Sparse solver validation ---
