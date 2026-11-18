# Issue #237 step 1 — permute warm path without allocation

## Problem

On an IPM refactorization sequence (fixed pattern, drifting values) the
value permutation `P·A·Pᵀ` is serial work outside the parallel tree.
Issue #237 measured 23 ms of a 256 ms factorization at 16 threads on a
3.9 M-nnz collocation KKT. Reproduced here on a 6.0 M-nnz chain KKT
(`examples/issue237_permute_warm.rs`): 74.8 ms per warm call.

Breakdown of the warm branch of `permute_csc_values_with_cache`
(`src/numeric/factorize.rs`), throwaway instrumentation, per call:

| part | ms |
|---|---|
| compare `col_ptr`, `row_idx`, `perm_inv` against the cache | 6.4–7.4 |
| `vec![0.0; nnz]` | 1.9–2.6 |
| scatter `values[value_map[k]] += a[k]` | 35.5–40.1 |
| clone cached `permuted_col_ptr` + `permuted_row_idx` | 26.2–30.4 |

The scatter number includes first-touch page faults on the lazily
zeroed allocation. The clone is two full copies of the permuted
pattern, plus faults, on every call, though the pattern never changes.

## Prior work

- Issue #56 Lever A.2 introduced the cache (value map instead of a
  triplet sort). Issue #124 wired it into the parallel driver.
- REG-1: the cache is keyed on the full input pattern and `perm_inv`,
  compared exactly (not hashed) so a collision cannot produce a wrong
  factor. N7: one-shot callers (`pattern_reused_hint == false`) neither
  build nor keep a cache.
- `tried-and-rejected.md` has no entry on this warm path. The 2026-08-09
  KR warm-start entry concerns scaling (step 4 of the #237 review).

## Design

1. **Persistent permuted matrix.** `PermuteCache` holds the permuted
   `CscMatrix` itself (pattern + value buffer) instead of separate
   `permuted_col_ptr` / `permuted_row_idx`. A warm hit moves the matrix
   out (`Option::take`), overwrites its values, and hands it to the
   driver by value. The driver gives it back
   (`FactorWorkspace::restore_permuted`) once it no longer borrows it.
   If a driver returns early with an error the buffer is not returned;
   the next warm call then finds no buffer and takes the cold path,
   which rebuilds the cache. Correct, just slow, and only after an error.
   Net memory: the cached pattern was already resident; the value array
   (8·nnz bytes) is now resident between calls too, the same peak as
   during a factorization.
2. **No zero fill when the value map is a bijection.** When the input
   holds no `(i, j)` / `(j, i)` duplicate pair, every permuted slot
   receives exactly one input value. Recorded once at cache build
   (`value_map_bijective`). The write is `0.0 + a[k]`, not `a[k]`, so
   that a `-0.0` input still becomes `+0.0` exactly as the old
   zero-then-accumulate did: the factor stays bit-identical.
   (Found while writing the test: the *cold* path keeps `-0.0` as
   `-0.0`, so warm and cold already differed in the sign of a zero in
   0.23.0. The warm behaviour is kept; see journal 2026-11-18-01.)
   Non-bijective maps keep `fill(0.0)` + accumulate in the same `k`
   order (same rounding).
3. **Skip the pattern compare when the Solver already made it.**
   `Solver::factor_impl` sets `pattern_reused_hint` only after an exact
   `col_ptr` / `row_idx` comparison against its stored pattern. A new
   crate-private flag on `FactorWorkspace` (`pattern_verified`) records
   that; the driver consumes it (resets it to `false`) at the permute,
   so a later direct-API call through the same workspace cannot inherit
   it. With it set, the `O(nnz)` `col_ptr` / `row_idx` compare is
   skipped; the `O(n)` `perm_inv` compare and the size checks stay (the
   Solver's check covers the pattern, not the ordering). `NumericParams`
   is not touched: it is a public struct with all-public fields, and a
   private field would break struct-literal construction downstream.

   Why the cache cannot belong to a different pattern when the Solver
   says "verified": every Solver call with a changed pattern has
   `pattern_reused_hint == false`, and that path clears the cache (N7).
   Re-analysis trials also run with the hint off. So a cache present on
   a verified call was built from the pattern the Solver just matched.
   The `perm_inv` compare stays as defence against an ordering swap.

Parallel scatter (the review's "partition `value_map` by destination
column, or invert it to a gather") is deferred until 1–3 are measured:
if the remaining scatter is a few ms it is not worth the extra index
array.

## Oracle / tests

- Bit-identity: factors and solves must be byte-identical to the old
  path. Existing `permute_cache_*` tests compare warm and cold outputs.
  Add: warm output equals the cold `permute_csc_values` output bitwise,
  including a `-0.0` input entry and a duplicated `(i, j)`/`(j, i)`
  input (non-bijective map).
- Buffer loss: a warm call whose buffer is absent still returns the
  correct matrix (cold path) and rebuilds the cache.
- `pattern_verified` is consumed: after one driver call it is `false`.
- The analytic inertia of the chain KKT (`t·nx` positive, `(t−1)·nx`
  negative) is the external oracle in the example harness.
