# Issue #237 step 3 — parallel Knight–Ruiz (∞-norm) sweeps

## Problem

`scaling::infnorm::compute_infnorm` runs on every factorization whose
scaling resolves to `InfNorm` (including `Auto` pinned to it). It is
serial, inside the driver prologue, outside the parallel tree. Issue
#237: 72 ms of a 256 ms factorization at 16 threads (28 %). Here (chain
KKT, n = 155 970, nnz = 5.97 M, 4 cores): 204–284 ms per call.

The sweep is cap-bound: from the second IPM iterate on it runs all
`max_iter = 10` passes without reaching `tol` (tried-and-rejected
2026-08-09 and 2026-08-15). Warm starts and lower caps are rejected; a
faster algorithm changes numerics. Parallelising a pass is the one lever
that keeps the output bit-identical.

## One pass

For lower-triangular CSC storage, column `j` contributes
`v = |d_i a_ij d_j|` to row `j` (register `col_max`, written once at
column end) and, for `i > j`, to row `i` (scatter into `row_max[i]`).
Then `d_i ← kr_guarded_update(d_i, row_max[i])` and `max_dev` is
reduced, O(n).

## Bit-identity argument

Every update is `if v > acc { acc = v }` starting from `0.0`. For non-NaN
`v ≥ 0` that is the exact maximum, independent of order and grouping;
a NaN `v` never wins a `>` comparison, so it is ignored in every order
too. Combining per-chunk partial maxima with the same `>` rule therefore
reproduces the serial `row_max` bit for bit. The `d` update and
`max_dev` reduction are unchanged (serial, O(n)).

## Design

- Partition columns into `T` contiguous ranges of roughly equal nnz
  (`T` = rayon's current thread count, capped so each range carries at
  least `MIN_NNZ_PER_CHUNK`). Lower-triangle columns `[j0, j1)` only
  write rows `≥ j0`, so chunk `t`'s private buffer covers `[j0, n)`.
- Pass: each chunk zero-fills its buffer and scans its columns exactly
  as the serial loop does (same `col_max` hoist). Then `row_max[i]` =
  max over the chunks whose range starts at or before `i`, computed in
  parallel over row blocks.
- Taken only on a rayon worker (inside a Solver's pool, or the
  parallel driver) and above an nnz threshold — the same gate as the
  permute gather (step 1). Otherwise the existing serial loop runs,
  untouched.
- Memory: `Σ_t (n − j0_t)` f64, at most `T·n`, allocated once per call
  and reused across the 10 passes. 16 threads × 156k × 8 B ≈ 20 MB worst
  case; in practice about half (ranges start progressively later).
- Cost: per pass, each chunk zero-fills `n − j0_t` entries and the merge
  reads them back: `O(T·n)` against `O(nnz)` of real work. For this KKT
  `T·n / nnz ≈ 0.1` at 4 threads, 0.4 at 16. If the merge shows up, the
  fallback is per-chunk touched-row ranges (lower triangle: the rows a
  column range touches are bounded below by `j0`, above by `n`, so no
  gain without the pattern's row extent; measure first).

## Tests / oracle

The serial `compute_infnorm` predates this session and is the oracle:
parallel output must equal it bitwise (`to_bits`) on matrices that hit
the parallel path (above the nnz threshold), on a rayon worker, at
several thread counts (1, 2, 3, 4, 7) — chunk boundaries move with `T`.
Include rows that are all-zero, a dense row (arrow), and explicit
zeros. The chain KKT harness digest must be unchanged.
