# Phase 2.5.x — Pooling `CscMatrix::to_dense` via `FactorWorkspace`

## Motivation

Session 2026-04-20-01 (`dev/sessions/2026-04-20-01.md` §Next Session 3)
profiled the D.3 dense fast-path on TRO3X3 and reported that
`CscMatrix::to_dense` accounts for **~9 µs** per call, **22 %** of
the dense-total factor time. After the D.4 tiny-n extension (n ≤ 16
unconditionally routed through `dense_fast_factor`), the densify
fraction of the total dense-path work grew further: on TRO3X3-class
workloads the densify is now one of the top two cost centers in the
fast path, alongside `compute_scaling` and `factor_frontal_blocked`.

The comment at `src/numeric/factorize.rs:360`:

> On a gate hit `ws` is pass-through — the dense path allocates its
> own dense buffer (pooling it is a follow-up; see
> `dev/plans/sparse-tail-d3.md`).

explicitly flags the TODO. This research note closes that follow-up.

## What `to_dense` does today

`src/sparse/csc.rs:261`:

```rust
pub fn to_dense(&self) -> crate::dense::matrix::SymmetricMatrix {
    let entries: Vec<(usize, usize, f64)> = (0..self.n)
        .flat_map(|j| {
            (self.col_ptr[j]..self.col_ptr[j + 1])
                .map(move |k| (self.row_idx[k], j, self.values[k]))
        })
        .collect();
    crate::dense::matrix::SymmetricMatrix::from_lower_triangle(self.n, &entries)
}
```

Two allocations per call:

1. A `Vec<(usize, usize, f64)>` of length `nnz` (32 bytes each).
2. The `SymmetricMatrix::from_lower_triangle` call allocates the
   `n * n` f64 dense data array (through whatever constructor
   `from_lower_triangle` uses).

On a tiny-n matrix (n = 10, nnz ≈ 30) this is ~240 bytes for the
triplet vec and 800 bytes for the dense data — 1 kB per call. Not
huge in absolute terms, but allocating 1 kB from the system
allocator for every sub-100-µs factor call is measurable.

## Target: pool both the triplet vec and the dense data

The cleanest pool lives on `FactorWorkspace`, which already holds:
- `frontal_values: Vec<f64>` — the per-supernode frontal buffer,
  pooled via `std::mem::take` at the top of the supernode body and
  returned at the end.
- `row_map: Vec<usize>` — global→local map, pooled across calls.
- `build_{delayed,trailing,seen}` — row-index scratch.

The pattern for a densify pool is the same `std::mem::take` hand-off:
the caller of `dense_fast_factor` owns the workspace, `to_dense_into`
takes the empty/previously-sized buffer, reuses it (clear + resize to
`n * n` zeros), and returns a `SymmetricMatrix` borrowing-owned of
the reused storage.

## Plan

1. Add `FactorWorkspace::dense_values: Vec<f64>`. Default empty.
2. Add `CscMatrix::to_dense_into(&self, buf: Vec<f64>) -> SymmetricMatrix`
   that clears `buf`, resizes to `n*n` zeros, scatters the lower
   triangle, and wraps the buffer in a `SymmetricMatrix`.
   - Keep existing `to_dense` as a thin wrapper: `self.to_dense_into(Vec::new())`.
   - This preserves bit-identical semantics for existing callers.
3. Add `dense_fast_factor_with_workspace(matrix, params, ws)` that:
   - `std::mem::take`s `ws.dense_values`.
   - Calls `matrix.to_dense_into(buf)` to build the dense.
   - Proceeds with the existing scaling/factor.
   - After `factor_frontal_blocked` consumes the dense, returns
     `sym.data` to `ws.dense_values` (same pattern as `frontal_values`).
4. Update `factorize_multifrontal_with_workspace` at line 369 to
   call the workspace variant, passing `ws`.
5. Keep the old `dense_fast_factor(matrix, params)` as a thin wrapper
   that allocates a fresh workspace (so non-workspace callers and
   tests keep working).

## Risks / Non-risks

- **Risk: bit drift.** None expected — `to_dense_into` produces the
  same dense matrix as `to_dense` (byte-exact: the resize to zeros
  then scatter is the identical construction). Guardrail: the
  dense-fast-path parity tests (`tests/dense_fast_path.rs`) already
  compare dense-fast vs multifrontal bit-exact on in-gate matrices.
- **Risk: workspace lifetime.** `frontal_values` is already taken
  then returned in the supernodal path; `dense_values` follows the
  same discipline. Between calls both buffers sit on `ws` with
  their capacity intact.
- **Non-risk: concurrent use.** `FactorWorkspace` is `&mut` —
  no aliasing.

## Exit criteria

1. All lib tests pass (currently 129/129).
2. Dense-fast-path parity tests pass (`tests/dense_fast_path.rs`,
   `tests/tiny_fast_path.rs`).
3. Corpus bench: dense p90 not worse than baseline (1.79-1.83 range).
4. Micro-probe on TRO3X3-class matrix (or the D.4 probe bench):
   densify time reduced. Target ≥ 2× on the pooled buffer reuse
   (a fresh alloc vs a warm reuse should see the allocator cost
   disappear).
5. No new heap traffic beyond the first call for a given
   `FactorWorkspace` lifetime — verified by the `alloc_probe`
   bench not increasing on a sequence of same-size calls.

## References

- `dev/sessions/2026-04-20-01.md` §Next Session 3 (9 µs 22% TRO3X3)
- `src/numeric/factorize.rs:140` (`FactorWorkspace` struct)
- `src/numeric/factorize.rs:257,360` (densify site + TODO comment)
- `src/sparse/csc.rs:261` (`to_dense`)
- `dev/plans/sparse-tail-d3.md` (original D.3 plan where pooling
  was deferred)
