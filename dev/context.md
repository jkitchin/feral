# FERAL Context (auto-generated)

Generated: 2026-04-20T14:25:16Z

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
26b5e01 session: 2026-04-20-02 — bench harness multi-sample denoise
df540ab session: 2026-04-20-01 — HS85 diagnosis + D.4 tiny-n fast-path
16fdd77 d4: stage-1 probe + stage-2 corpus bench — close D.4
ddefc2f d4: GREEN — add N_TINY=16 disjunct to dense-fast-path gate
d570960 d4: RED — tiny_fast_path tests against planned predicate
```

## Test Status
```
thread 'test_kkt_regression_spot_checks' (18037843) panicked at tests/blocked_ldlt.rs:266:71:
called `Result::unwrap()` on an `Err` value: InvalidInput("factor_frontal_blocked: Phase 2.4.1b not yet implemented")

---- test_2x2_at_block_boundary stdout ----

thread 'test_2x2_at_block_boundary' (18037840) panicked at tests/blocked_ldlt.rs:203:67:
called `Result::unwrap()` on an `Err` value: InvalidInput("factor_frontal_blocked: Phase 2.4.1b not yet implemented")


failures:
    test_2x2_at_block_boundary
    test_frontal_ncol_lt_nrow_parity
    test_indefinite_bk77_parity
    test_kkt_regression_spot_checks
    test_rejection_fallback
    test_spd_scalar_blocked_parity_size_sweep

test result: FAILED. 0 passed; 6 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

error: test failed, to rerun pass `--test blocked_ldlt`
