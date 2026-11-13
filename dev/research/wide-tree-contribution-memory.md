# Research Note: Contribution-block memory on wide elimination trees

**Status:** Complete
**Date:** 2026-11-12
**Related spec sections:** multifrontal assembly (`src/numeric/factorize.rs`
`factor_one_supernode`), symbolic peak model (`src/symbolic/mod.rs`
`compute_peak_contrib`, issue #204 `WorkEstimate`)
**Key references:** Duff & Reid (1983), "The multifrontal solution of
indefinite sparse symmetric linear equations"; Liu (1992), "The
multifrontal method for sparse matrix solution: theory and practice"
(§5, stack storage and its dependence on child order); MUMPS
`dfac_asm*.F` (parent allocated before its last child is assembled).

## Overview

The multifrontal method holds each child's contribution block until its
parent front is assembled, so a parent's assembly costs the *sum* of its
children's blocks at once. On ordinary trees that sum is a few fronts.
On a wide tree it is not: a star of many leaves coupled to a few shared
variables (the KKT system of a problem with many constraints over a few
variables, a bordered block-diagonal matrix) sends thousands of dense
blocks into one small front. 0.22.0 also held each block square
(`c²` f64s, of which only the lower triangle carries values).

Scope: the sequential driver and the Schur-complement driver. The
parallel task-graph driver keeps square blocks and classic assembly
(`early_assembly` ignored); it shares only the kernel's extract, so it
gains the empty-block rule below.

## Algorithm

1. **Packed blocks** (always on in the sequential drivers). A held block
   stores its lower triangle packed column-major (`c(c+1)/2` entries,
   column `j` at offset `j(2c−j+1)/2`). The kernel writes it so through
   crate-internal `_with_layout` variants; the public kernels keep their
   0.22.0 signatures and square output. `ContribBlock<L>` is generic over
   a zero-sized `BlockLayout` (`Square` for the parallel driver, `Packed`
   for the sequential ones), so one extend-add serves both with the
   layout fixed at compile time.

2. **Wide-parent accumulators** (opt-in, `EarlyAssembly::WideParents`;
   needs `use_static_row_indices`, the default). For a parent `p` with
   front area `A = order(p)²` and children's packed blocks `B(i)` in
   `children` order, `symbolic::accumulate_from` names the first child
   `j` with `Σ_{i≤j} B(i) > A` and `Σ_{i>j} B(i) ≥ A` (none if the
   children sum to less than `2A`). The plan is computed once per
   analysis and cached on it. When `j` is deposited, `p` gets an
   accumulator — its frontal over its static rows, seeded with its
   original entries — into which the held children `≤ j` and then every
   later child are extend-added in order. At `p`'s assembly the
   accumulator is the frontal; with delayed pivots its cells are copied
   into the larger frontal and the delayed remainders added. The
   sequential drivers deposit in ascending supernode order, which is
   `children` order, so a min-heap of each plan's next expected child
   recognises a member deposit in O(log plans). The hidden `Always`
   (open at every parent's first child) exists for the parity tests.

3. **Buffers follow the classic history.** The drivers recycle two
   buffers: the frontal pool and a one-slot contribution-block pool.
   Both only grow (`Vec::resize` / `reserve`), so the bound below holds
   for the whole heap only if `WideParents` never leaves a pooled buffer
   larger than `Off` would:
   - an adopted accumulator is freed once its front is factored, and the
     idle frontal pool grows by the one rule both paths use
     (`dense::matrix::fit_pooled_front`), as if it had held that front;
   - absorbed blocks and delayed-pivot remainders are freed, not pooled
     (only a whole block, assembled classically, returns to the pool);
   - a front whose block is empty (a root) takes no pooled buffer, in
     every driver, so no returned factor holds contribution storage.

## Why these defaults

- **The memory bound.** At the moment the accumulator opens, live are
  the accumulator (`A`), the held blocks `Σ_{i≤j} B(i)` and the driver's
  frontal buffer. The classic scheme holds all `Σ_i B(i) ≥ Σ_{i≤j} B(i)
  + A` at `p`'s assembly together with a frontal of at least `A`.
  Afterwards each later block is live only until absorbed, while the
  classic scheme keeps accumulating. With the buffer rules of step 3
  the peak live heap is `Off`'s plus the plan (a few words per planned
  parent), one block's row map, and `O(order(p))` words while `p`'s
  front is factored (the adopted accumulator beside the idle pool and the
  kernel's outputs). The retained factor is the same. `compute_peak_contrib`
  keeps the classic square model, an upper bound for every driver and
  scheme without delays.
- **The first version broke the bound.** It adopted the exact-size
  accumulator as the frontal pool and recycled absorbed blocks through
  the contribution pool; both left larger pooled buffers than `Off`
  (a hand-built tree 1.117× `Off`, retained factor larger on 1,126 of
  2,000 random trees). See `dev/tried-and-rejected.md`.
- **Time is not bounded by construction.** The extend-adds are the same
  ones, but an accumulator adds `O(A)` to lay it out, a binary search per
  block row and a heap operation per member deposit, and freed blocks are
  not recycled. On matrices with one or two planned parents whose fronts
  are small (order 10–16 under 1,000–2,600 tiny children) that is 2.6–4.1%
  of the factorization. Hence `WideParents` is opt-in and `Off` the
  default.
- **Why not `Always`.** Holding a whole front open across a parent's
  later child subtrees exceeds the classic stack: 1.6–2.0× the parent's
  peak on 3D grids, 5.5× on nuffield2.
- **Why a symbolic rule, not a runtime threshold.** The positions are
  fixed by the analysis, so the plan is reproducible, testable on
  hand-built trees and needs no tuning constant.
- **Parallel driver excluded.** Out-of-order deposits need per-parent
  synchronisation, and a bounded backlog behind the thread absorbing
  into a parent; that added locks and a tuning constant (a backlog
  limit) to the task graph. Deferred to a separate change
  (`dev/tried-and-rejected.md`). The default `Solver` runs the task
  graph when the tree has enough task parallelism, where `WideParents`
  does nothing (`Solver::with_parallel`).

## Bit-identity

Every frontal cell receives its original entry by assignment, then one
`+=` per child that reaches it, children in `children` order, zeros
skipped — in the classic scheme and through an accumulator alike. A
delayed column belongs to one child and has no original entry at the
parent, so its cells are `0.0 + v` either way. The parent's trailing
rows are its static trailing rows whether or not pivots were delayed,
so the accumulator can be laid out before the delays are known; a block
row outside them is a `FeralError`. Packing copies each lower-triangle
value to its packed slot; nothing is recomputed.

## Test Strategy

- `tests/early_assembly_parity.rs`: `Off` / `WideParents` / `Always`
  digests (factor, row indices, delayed-in count, cascade-break flag)
  equal on the parity corpus, the issue-#227 synthetic KKTs, stars with
  and without delays, a delay chain through an accumulator (issue-#221
  backstop), and the Schur driver with and without delays (frontal
  adoption with a Schur block); the parallel driver under each scheme
  must not change. `Off` and `WideParents` digests are pinned to 0.22.0
  for aarch64; the Schur kernel dispatches on run-time CPU features, so
  the pin is ignored on other targets rather than passing vacuously.
- `tests/wide_tree_memory.rs` (counting allocator): a 1,500-leaf star,
  peak live heap `Off` 65.73 MiB, `WideParents` 12.12 MiB; generated
  trees (the review's pool-growth tree, a delaying star, 150 random trees
  and 40 with zero-diagonal leaves): `WideParents` peak at most `Off`'s
  plus a bookkeeping allowance (16 words per planned parent, 9 words per
  row of the largest front), retained factor no larger, and no returned
  factor owns contribution storage. 192 matrices: below `Off` on 113, at
  most 7,208 B above. Reverting either buffer fix or the empty-block rule
  fails it.
- Unit (`numeric::contrib`, `symbolic`): square and packed extend-add
  alike; the plan on hand-built stars; `take` before every child is
  deposited, out-of-order deposits and a row outside the parent all fail.

## Measurements

Counting allocator (peak live heap of one factorization); 101 matrices
(parity corpus, stress fixtures, issue-#227 KKTs, nuffield2, 2D/3D
grids, synthetic stars and forests), 3 interleaved rounds × 7 runs
against 0.22.0, load average 16–26.

| | digests | peak vs 0.22.0 | time geomean, ≥ 0.5 ms (22) |
|---|---|---|---|
| `Off` (default) | 101/101 equal | 0.673–1.000 | 0.978 |
| `WideParents` | 101/101 equal | 0.320–1.000 (MUONSINE +64 B) | 0.997 (1.019 vs `Off`) |
| parallel, 1 thread | 101/101 equal | 0.673–1.000, none above | — |
| parallel, 4 threads | 101/101 equal | scheduling-dependent (below) | — |

Retained factor (every driver): 3D grid k = 32 79.3 → 69.1 MB, nuffield2
63.5 → 60.2 MB. Peaks: 3D grid k = 32 162.3 → 138.7 MB (`Off` and
`WideParents`); 1,024-leaf star 19.5 → 13.1 (`Off`) → 6.3 MB
(`WideParents`).

Review harness (random trees as in `tests/wide_tree_memory.rs`, fronts up
to 80 columns), `WideParents` against `Off`: 2,000 trees without delays,
above `Off` on 58, worst +1,488 B; 2,000 with zero-diagonal leaves,
above on 66, worst +2,168 B; none more than 2% above; retained factor
above `Off` on none; 0 digest mismatches. The pool-growth tree: +560 B.

Process footprint, FERAL alone (macOS peak `phys_footprint`, uniform
star of 10,982 leaves over `c` shared variables, default parameters,
sequential driver, `x` factorizations each built before the previous
is dropped), MiB, 0.22.0 / `Off` / `WideParents`:

| c | x1 | x12 |
|---|---|---|
| 40 | 255 / 180 / 101 | 258 / 199 / 116 |
| 64 | 513 / 401 / 166 | 513 / 399 / 198 |
| 80 | 887 / 505 / 197 | 1188 / 508 / 234 |
| 101 | 1124 / 780 / 257 | 1243 / 909 / 317 |
| 127 | 1690 / 1001 / 309 | 1690 / 1230 / 363 |
| 150 | 2265 / 1403 / 362 | 3935 / 1404 / 432 |

Unfavourable: focused timing (21 runs × 9 rounds, before the buffer
fixes), `WideParents` / `Off`: CRESC132 1.041, VESUVIO 1.031, sawpath
1.026, CRESC100 1.009, star:1024 0.980. On the corpus run after the
fixes, `WideParents` / `Off` on matrices ≥ 0.5 ms rose from 1.004 to
1.019 (multistar:8 1.093, saddle_rankdef_100 1.053, multistar:64 1.050),
consistent with freed rather than recycled blocks; not separated from
noise. `Off` / 0.22.0: MUONSINE 0.997 (1.02 in an earlier 6 × 401 run),
a 16-leaf star 1.038 (range 0.85–1.30). Four-thread peaks vary with
scheduling: grid3:32, 20 runs each, 0.22.0 175–193 MB and this change
175–190 MB, the same distribution. Raw data and scripts are outside
the repository.
