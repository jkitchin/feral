# Bump elimination Step 1 — sub-diagonal pivot index

**Status:** Plan
**Date:** 2026-06-18 (session 01)
**Origin:** discopt#229; `dev/research/bump-elimination-speedup-2026-06-18.md` Step 1.
**Reproducer/guard already landed (4b54b22):** `tests/lu_update_casctanks.rs`,
`benches/lu_update_trace.rs` (baseline 136 ms / 144 updates), fixture
`tests/data/lu_trace/casctanks.txt`.

## Goal

Remove the O(bump²) pivot-selection scan in
`SparseLu::eliminate_bump` (`src/lu/sparse_update.rs:305`) with **zero change to
the produced numerics** (identical `FtOp` sequence, identical U). The change must
keep strict partial pivoting (the bump always takes the max-magnitude pivot — the
reason the in-place scheme exists, per journal 2026-06-08-01.org).

## Non-goal

Not the asymptotic redesign (cyclic permutation / sparse BGR — that is the parked
spike `dev/research/asymptotic-bump-update-spike-2026-06-18.md`). Not the dense
workspace (Step 2, separate). No tolerance changes.

## Structural facts (rev 5ab8074)

- `u_rows: Vec<Vec<(usize, f64)>>` — one column-sorted row per pivot position,
  diagonal first. Normal invariant: a row's columns are all `>= row index`.
- The FT spike (`set_column_r`, `:291`) writes column `r` into rows `r..=h`,
  **temporarily violating** `column >= row` for rows `> r` — that sub-diagonal
  fill in the bump `[r, h]` is exactly what `eliminate_bump` removes.
- `eliminate_bump` does, for each pivot column `k ∈ [r, h]`:
  1. **scan** `for i in k..=h { get_col(&u_rows[i], k) }` to pick the
     max-magnitude pivot row — O(bump) probes per column ⇒ **O(bump²) total,
     independent of fill** (lines 321–327);
  2. swap `u_rows[k]`/`u_rows[pivot_row]` (records `FtOp::Swap`);
  3. **eliminate** `for i in k+1..=h { if get_col(.., k) { row_sub(..); FtOp::Axpy }}`
     — again iterates all tail rows (lines 343–353).

Both hot loops are dominated by iterating `k..=h` and probing column `k`.

## Design — bump-local sub-diagonal index `col_rows`

`col_rows[c]` = the rows `i ∈ [r, h]` that currently hold a nonzero in column `c`,
for `c ∈ [r, h]`. With it, step `k` iterates only the actual candidates
(`col_rows[k]`) instead of `k..=h`, in both the pivot scan and the elimination.

### Build (once per `eliminate_bump`)
For `i ∈ [r, h]`: extract the slice of `u_rows[i]` with column in `[r, h]` (rows
are column-sorted ⇒ two binary searches bound the slice), and push `i` into
`col_rows[c]` for each such `c`. Cost O(Σ bump-row entries in `[r,h]`) — one pass,
versus the current O(bump²) of pure probes.

### Maintenance across the **row swap** (the subtlety)
Swapping the contents of `u_rows[k]` and `u_rows[pivot_row]` relabels which row id
holds which column entries. Only columns `[k+1, h]` matter going forward (columns
`<= k` are finished). For each `c ∈ [k+1, h]` appearing in `support(old u_rows[k]) ∪
support(old u_rows[pivot_row])`, toggle membership of `k` / `pivot_row` in
`col_rows[c]` (if exactly one of the two was present, swap which one). Bounded by
the two rows' nnz — no asymptotic penalty. (Column `k` itself: we already hold its
candidate list; the pivot moves to position `k`.)

### Maintenance across `row_sub` (fill delta)
`u_rows[i] = row_sub(u_rows[i], pivot_data, mult, k)` drops column `k` and merges in
pivot-row entries (columns `> k`). Update `col_rows` by the delta over `[k+1, h]`:
before the call snapshot row `i`'s `[k+1,h]` columns; after, diff — remove `i` from
vanished columns, add `i` to new-fill columns. Remove `i` from `col_rows[k]`
(column `k` eliminated). Cost O(nnz(row i) in `[r,h]`) — same order as `row_sub`.

### Correctness argument (numerics identical)
`col_rows[k]` is, by construction, *exactly* the set `{ i ∈ [k,h] : get_col(u_rows[i],k) = Some }`
the current scan finds (the rest return `None`). Same candidate set ⇒ same
max-magnitude `pivot_row` ⇒ same `FtOp::Swap`/`Axpy` and same `row_sub` results ⇒
identical U and identical eta. The index only skips probes that are `None` today.

## Implementation notes
- Reuse a pooled workspace for `col_rows` (a `Vec<Vec<usize>>` sized to the bump,
  or a single arena) to avoid per-update allocation — mirror the existing scratch
  pooling (L3/L12). Bump indices are `[r, h]`; offset by `r`.
- Keep the existing `ztol`/`NeedsRefactor`, growth monitor, rollback, and the
  `u_above` reindex on commit **unchanged** — this change is confined to the pivot
  search + elimination iteration inside `eliminate_bump`.
- No `unwrap`/`expect` in `src/`; keep `Result` contracts.

## Validation
1. `cargo test` full suite, especially `tests/lu_sparse.rs` (the existing 25-update
   FT chain, dense↔sparse agreement, `ft_update_is_bump_local`).
2. `tests/lu_update_casctanks.rs` (committed fixture) — true residual must stay
   ~1e-13 and **unchanged**; dense cross-oracle (`FERAL_LU_TRACE_DENSE=1`) unchanged.
3. Full-trace replay (`FERAL_LU_TRACE=…/casctanks_trace.txt`) — true residual must
   match the pre-change 1.819e-12 to the bit (numerics-preserving), `refactor_signals=0`.
4. `cargo bench --bench lu_update_trace` — expect a large drop from 136 ms (the scan
   is ~half the cost in the discopt profile); record the ratio.
5. fmt + clippy clean (pre-commit gate).

## Risk / fallback
The swap-relabel maintenance is the only delicate part. If it proves error-prone,
fall back to doing Step 2 (dense bump workspace) first — a dense scatter makes both
pivot search and elimination contiguous without an index to maintain across swaps —
then revisit the index. Either way the guard in (1)–(3) gates correctness.
