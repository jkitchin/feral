# LU Forrest–Tomlin update: allocation pooling

*2026-06-19 — research note for Phase 1 of the measure-then-pool plan.*

## Motivation / Phase-0 evidence

The sparse FT update path (`src/lu/sparse_update.rs`) had no allocation work,
unlike its sibling `sparse_solve.rs` (pooled via `take_zeroed`, zero
steady-state allocs). Phase-0 probe (`tests/lu_update_alloc_probe.rs`,
counting `#[global_allocator]`, in-tree casctanks fixture m=2169, 144 updates):

    per-update: allocs=1804.2  reallocs=176.3  bytes=128610

Baseline update wall-time (`bench --bench lu_update_trace`): 12.351 ms / 144 =
**85.8 µs/update**. ~1980 alloc-ops against 85.8 µs is first-order. Gate passed.

Falsification risk (carried from `dev/tried-and-rejected.md`: multi-slot contrib
pool, arena refactor): pooling loses when push/pop/clear bookkeeping exceeds the
malloc/free removed. **The bench delta is the arbiter, not the count.**

## Allocation taxonomy (per update)

Classify each site as *churn* (allocated then freed within the update — poolable)
or *retained* (kept in `self` — pooling does not help):

| site | file:line | frequency | class |
|------|-----------|-----------|-------|
| `row_sub` `out` | :544 (called :381) | **per axpy** (inner loop) | churn |
| `pivot_data = u_rows[k].clone()` | :371 | per pivot (k∈[r,h]) | churn |
| `targets` collect | :376 | per pivot | churn |
| `col_rows` outer + inner vecs | :322 | per update (width inners) | churn |
| `saved` row clones | :124 | per update (×changed.len()) | churn |
| `touched` | :91 | per update | churn |
| `supp` | :95 | per update | churn |
| `changed` | :116 | per update | churn |
| `reach = touched.clone()` | :229 | per update | churn |
| `stack` (compute_spike) | :207 | per update | churn |
| `old_above` clone (set_column_r) | :292 | per update | churn |
| `ops` → `FtEta` | :311 | per update | **retained** (skip) |

`ops` is moved into `self.etas` (:168) and lives until refactor — legitimate
retained memory, not churn. Do **not** pool it.

The highest-frequency sites are in the bump loop: `row_sub` (per axpy),
`pivot_data` and `targets` (per pivot). Land those first.

## Pooling design

Add pooled scratch fields to `SparseLu` (mirroring `scratch_b/c/d`, `ft_work`),
all `clear()`+refill, taken via `std::mem::take` into locals at the top of
`eliminate_bump`/`update_sparse` to dodge borrow conflicts with `&mut self`, and
**restored on every return path** (success, `NeedsRefactor`, rollback):

- `pivot_scratch: Vec<(usize,f64)>` — replaces `u_rows[k].clone()`:
  `clear(); extend_from_slice(&u_rows[k])`. `row_sub` then reads `&pivot_scratch`
  (same borrow shape as the clone — a buffer separate from `u_rows`).
- `targets_scratch: Vec<usize>` — `clear(); extend(col_rows[kc].iter().copied()
  .filter(|&i| i > k))`.
- `row_pool: Vec<Vec<(usize,f64)>>` — free-list of row buffers. `row_sub` becomes
  `row_sub_into(dst, src, mult, drop, &mut out)` writing into a cleared pooled
  buffer; the displaced `old_row` (already `mem::take`-n out of `u_rows[i]`) is
  pushed back into `row_pool` for the next axpy. Steady state: zero alloc.
- `col_rows_pool: Vec<Vec<usize>>` — persistent outer vec; per update, ensure
  `len >= width` (recycling inner vecs) and `clear()` the first `width` inners.
- update-level: `supp`, `changed`, `touched`, `reach`, `stack`, `old_above` →
  pooled `Vec` fields, `clear()`+refill. `saved` row clones drawn from / returned
  to `row_pool`.

## Correctness contract (non-negotiable)

`tests/lu_update_casctanks.rs` already pins the FT eta `FtOp` sequence and the
solve residual bit-for-bit. Pooling changes *only where bytes come from*:
`extend_from_slice`/`clear`+refill reproduce identical contents in identical
order, and `row_sub_into` is `row_sub` with the `out` buffer supplied. So the
numerics — pivot choice, multipliers, fill pattern, eta ops — are unchanged.
That test is the guardrail and must stay green.

New: extend the alloc probe into a steady-state assertion — after warm-up the
per-update allocation count must drop sharply (target: near-zero growth across a
long chain), mirroring `SOLVE_SCRATCH_ALLOCS` in `sparse_solve.rs`.

## Plan order (each step: correctness test + alloc probe + bench)

1. Bump loop: `pivot_scratch`, `targets_scratch`, `row_pool`+`row_sub_into`,
   `col_rows_pool`. Measure bench delta. **If it washes → revert, record
   falsified, stop.**
2. If (1) pays: update-level pools (`supp`/`changed`/`touched`/`reach`/`stack`/
   `old_above`, `saved` via `row_pool`).
3. Lock in with the steady-state alloc assertion.
