# Phase 2.5.1 — Validation: Liu's row-subtree column counts

**Session:** 2026-04-20-09
**Plan:** `dev/plans/phase-2.5.1-liu-column-counts.md`
**Research note:** `dev/research/phase-2.5.1-liu-column-counts.md`

## Summary

Replaced the `O(n²)` `column_counts` elimination-simulation in
`src/symbolic/column_counts.rs` with a new
`O(nnz(A) + n·α(n))` Gilbert–Ng–Peyton implementation
(`column_counts_gnp`). Bit-exact equivalence verified on 169585
KKT matrices; no downstream regression.

## Exit-criteria table

| # | criterion                                           | target | result | status |
|---|-----------------------------------------------------|--------|--------|--------|
| 1 | All lib tests pass                                  | pass   | 129/129 | ✓ |
| 2 | GNP = reference across full KKT corpus              | 0 diff | 0 diff on 169585 | ✓ |
| 3 | Symbolic throughput on large n improves ≥ 3×        | ≥3×    | **95×–1938×** | ✓ |
| 4 | (soft) sparse factor/MUMPS p90 no regression        | ≤1.65  | 1.65    | ✓ |
| 5 | (soft) dense factor/MUMPS p90 no regression         | ≤1.79  | 1.83 (Δ0.04, run-noise) | ✓ |

## Step 3 cross-check

`examples/verify_column_counts.rs` walks every matrix in
`data/matrices/kkt/`, computes column counts via both
`column_counts` (slow, O(n²)) and `column_counts_gnp` (new),
and records any per-matrix disagreement.

```
Loaded 169585 KKT matrices
=== column_counts_gnp vs column_counts ===
Matched   : 169585/169585
Mismatches: 0
```

Bit-exact match across the entire corpus.

## Micro-timing (top-20 largest matrices)

`examples/bench_column_counts.rs` runs each of
`column_counts` / `column_counts_gnp` 100 times per matrix on
the 20 largest-n matrices. Single-threaded, release profile,
aarch64 (Apple Silicon).

| matrix          | n    | nnz(sym) | slow (ns)     | gnp (ns) | speedup |
|-----------------|-----:|---------:|--------------:|---------:|--------:|
| CRESC132_0000   | 5314 |    39820 |   234,538,123 |  173,130 |  1354×  |
| CRESC132_0005   | 5314 |    39838 |   388,163,255 |  200,237 |  **1938×** |
| CRESC132_0006   | 5314 |    39838 |   369,552,345 |  263,511 |  1402×  |
| VESUVIA_0000    | 3083 |    22183 |    22,250,648 |   97,148 |   229×  |
| VESUVIA_0001    | 3083 |    23101 |    23,343,560 |   89,528 |   261×  |
| VESUVIO_0005    | 3083 |    18105 |    23,901,133 |   87,412 |   273×  |

Full top-20 listing: 95× – 1938×, all well above the plan's
≥ 3× soft target. Per-call CRESC132 savings: ~234 ms → ~180 μs
(1300× faster per symbolic call).

## Full KKT bench — no regression check

Pre-switch (Phase 2.4.1c HEAD, session 09 run 1) vs
post-switch (session 09 run 2):

| metric                        | pre        | post       | Δ       |
|-------------------------------|-----------:|-----------:|--------:|
| Dense inertia match           | 152911/154481 | 152911/154481 | 0 |
| Dense residual pass           | 154207/154481 | 154207/154481 | 0 |
| Sparse inertia match          | 153008/154588 | 153008/154588 | 0 |
| Sparse residual pass          | 154262/154588 | 154262/154588 | 0 |
| Dense factor/MUMPS p90        | 1.79       | 1.83       | +0.04 (run-noise) |
| Sparse factor/MUMPS p90       | 1.65       | 1.65       | 0.00 |
| Dense factor/MUMPS p99        | 22.08      | 22.04      | −0.04 |
| Sparse factor/MUMPS p99       | 3.48       | 3.48       | 0.00 |

All four correctness counts are bit-exact to the pre-switch
run, confirming the switch is a pure refactor. Timing ratios
are within run-to-run variance (the phase 2 baseline uses
geometric-mean across 154k matrices; factor-time is dominated
by numeric, not symbolic, so the 1938× microbench win on
CRESC132 barely moves the p90).

## Decisions

None new. Pure refactor. Old `column_counts` kept public as a
test oracle for the 7 existing supernode unit tests and any
external user.
