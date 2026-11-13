# Implementation Plan: Contribution-block memory on wide elimination trees

**Date:** 2026-11-12
**Research note:** dev/research/wide-tree-contribution-memory.md
**Spec sections:** multifrontal assembly; symbolic `WorkEstimate` (issue #204)

Scope: the sequential and Schur-complement drivers. The parallel
task-graph driver keeps its square blocks and classic assembly. Two
commits, each passing the gates on its own.

## Change 1 — packed contribution blocks

### Files
- `src/dense/factor.rs`, `src/dense/block_ldlt32.rs`: crate-internal
  `ContribLayout`, crate-internal `_with_layout` kernel variants (the
  public kernels keep their signatures and square output); one
  `extract_contrib` for both layouts.
- `src/numeric/contrib.rs` (new): `ContribBlock<L: BlockLayout>` with
  `Square` and `Packed`, a length check, and one `extend_add`.
- `src/numeric/factorize.rs`: the sequential and Schur drivers hold
  `Packed` blocks, the parallel driver `Square`.

### Tests
- Unit: packed vs square kernel output bit for bit, every kernel path,
  with and without delays; square and packed extend-add alike.
- `tests/issue227_never_golden.rs` digests unchanged.

## Change 2 — planned accumulators (opt-in)

### Files
- `src/symbolic/mod.rs`: `accumulate_from` (the opening rule and its
  bound).
- `src/symbolic/mod.rs`: the plan as an iterator over the analysis
  (`wide_parent_plan`), derived by each factorization that asks.
- `src/numeric/contrib.rs`: `Accumulators` (plans sorted by parent, a
  min-heap of each plan's next expected child, one row map), `deposit`,
  `take`, `open_accumulator`; absorbed blocks are freed.
- `src/numeric/factorize.rs`: `EarlyAssembly {Off (default),
  WideParents, Always (internal feature)}` and `NumericParams::early_assembly`
  (needs `use_static_row_indices`); the sequential and Schur drivers
  deposit through `Accumulators` and adopt the accumulator as the
  frontal, then free it and grow the frontal pool by
  `dense::matrix::grow_pooled_front` (capacity only); only whole blocks
  are pooled.
- `src/dense/factor.rs`: an empty block takes no pooled buffer.

### Tests (written first)
- `tests/wide_tree_memory.rs`: a 1,500-leaf star, `WideParents` peak
  under a quarter of `Off`'s; generated trees (with and without
  zero-diagonal leaves, plus the review's pool-growth tree):
  `WideParents` peak at most `Off`'s plus a bookkeeping allowance,
  retained factor no larger, no factor owns contribution storage.
- `tests/early_assembly_parity.rs`: `Off` / `WideParents` / `Always`
  digests equal; parallel driver unchanged under each; the Schur driver
  with and without delays; `Off` and `WideParents` pinned to 0.22.0
  (computed on aarch64, equal on x86_64).
- Unit: plan and peak model on hand-built stars; an out-of-order deposit
  falls back to held blocks, bit-identical; a row outside the parent
  fails.

## Success Criteria
- Factors bit-identical to 0.22.0 under every scheme and driver.
- Parallel task-graph driver: same digests as 0.22.0, peaks no higher.
- Default never raises peak memory or measurably raises time on FERAL's
  corpus; `WideParents` peak live heap at most `Off`'s plus bookkeeping
  and the retained factor no larger, on any tree.
- Every unfavourable result reported.
