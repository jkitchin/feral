# Research: SCOTCH Direct Vertex Separator (S2)

**Date:** 2026-04-17
**Plan:** `dev/plans/ordering-scotch.md` §S2
**Status:** Pre-implementation

## Problem

Given a CSR graph and an initial edge bisection (`labels[v] ∈
{PART_A, PART_B}`), produce a *node* separator `S` such that every
edge has both endpoints in `PART_A ∪ S` or both in `PART_B ∪ S`.
Minimise the *weight* of `S` subject to a balance constraint on the
two sides after `S` is removed.

This is the SCOTCH-style alternative to feral-metis's
`construct_separator` + König min-cover step: rather than convert a
post-FM edge cut to a node separator only at the end, we initialise a
separator from the bisection's lighter side and *directly minimise its
weight* via two-sided Fiduccia-Mattheyses moves.

## Algorithm (Pellegrini 1996 §3, audit-corrected)

State per vertex:

- `WHERE_A = 0`, `WHERE_B = 1`, `WHERE_S = 2`.

Invariants maintained throughout a pass:

1. No `A↔B` edge exists (separator covers every cut edge).
2. `frontier_load[v]` for every `v ∈ S` is exact:
   `Σ vwgt[u]` over `u ∈ N(v)` currently in the *opposite* side from
   wherever this load entry tracks. Two arrays:
   - `load_a[v] = Σ vwgt[u]` for `u ∈ N(v) ∩ A`
   - `load_b[v] = Σ vwgt[u]` for `u ∈ N(v) ∩ B`

Gain when moving `v ∈ S → side`:

```
gain(v, side) = vwgt[v] - load_other(v, side)
```

where `load_other(v, A) = load_b[v]` and `load_other(v, B) = load_a[v]`
— exactly the audit-corrected formula in
`dev/plans/ordering-scotch.md` finding 1.

**Why this gain formula.** Moving `v` out of the separator into
`side` removes `vwgt[v]` from `S`; but every neighbour `u` currently
in the opposite side must enter `S` (otherwise we would have an
`A↔B` edge). The total weight added back is `Σ vwgt[u]` over
exactly those opposite-side neighbours. Hence the formula.

### Initial separator

Build from the boundary of the *smaller side* (audit finding 6 and
SCOTCH `vgraph_separate_fm.c`):

1. `total_a = Σ vwgt[v]` over `v ∈ A`, similarly `total_b`.
2. If `total_a <= total_b`, scan `A` and move every `v ∈ A` with
   `≥ 1` neighbour in `B` into `S`. Else mirror with `B`.
3. The resulting `S` is non-minimal (it covers all crossing edges by
   construction); FM will shrink it.

This is the smaller of the two trivial covers — choosing the lighter
side gives FM less to do on the first pass.

### FM pass

State that persists across moves within a pass:
- `lock[v] ∈ {false, true}` — locked once moved this pass.
- Two priority queues `pq_a`, `pq_b` keyed on gain (max-heap), each
  containing `(gain, v)` for separator vertices that *could* move to
  the corresponding side. A separator vertex appears in the PQ for a
  side iff it has at least one neighbour currently in that side
  (otherwise the move is uninteresting — it would just push `v` into
  one side and not pull anything out of the separator).

Loop until both PQs are exhausted or `move_cap` reached:

1. Pick the side with the larger current weight `<` opposite, or
   alternate strict to keep balance roughly maintained. (We use:
   pop from the side that currently weighs *less* — that's the side
   that needs vertices added.)
2. Pop highest-gain `v` from chosen side's PQ. Skip if `lock[v]` or
   stale (gain in PQ no longer matches actual `gain(v, side)`).
3. Compute pre-move balance check: if moving `v` would cause the
   destination side's weight to exceed `(1 + max_imbalance) *
   total / 2`, *skip the move* (per-move imbalance enforcement,
   audit finding 7).
4. Commit the move: `labels[v] = side`, lock it.
5. For each neighbour `u`:
   - If `u ∈ opposite_side`: pull `u` into `S`, locking it for this
     pass. Update load arrays and PQ entries as below.
   - If `u ∈ side` and was in `S`: nothing to do (already counted).
   - If `u ∈ S`: update `load_side[u] += vwgt[v]` (but `v` is no
     longer in opposite side — actually `u`'s view is that `v` is
     now in `side`, so the load it sees on `side` increased by
     `vwgt[v]` and the load it sees on the opposite side did not
     change). Re-stamp `u` in the PQs.
6. Track running `cur_sep_w` and `(prefix_a_w, prefix_b_w)`. After
   each accepted move, if `cur_sep_w < best_sep_w` AND the post-move
   `(a_w, b_w)` satisfies the *final-prefix* imbalance check (audit
   finding 7 part b), record this prefix as the new best.
7. At end of pass roll back to `best_prefix`.

Repeat passes until no improvement or `pass_cap` reached.

### Determinism

Tie-breaking inside the PQs: vertex index ascending. We use
`BinaryHeap<(i32, Reverse<i32>)>` exactly mirroring feral-metis's
edge-FM kernel — the `Reverse(v)` ensures stable lowest-index tie
break.

## Reuse from feral-metis

The following are pulled from `feral_metis::internals`:

- `graph::Graph` — same CSR layout.
- `initial_partition::{PART_A, PART_B, part_weight}` — same constants.
- `fm_refine::PART_SEP` — value `2`, our `WHERE_S`.

The two-sided node FM is *not* shared with feral-metis: feral-metis's
`refine_separator` is a single-sided greedy reducer
(`fm_refine.rs:31` notes "A full two-sided FM with negative-gain
acceptance is deferred until a concrete quality gap motivates it"),
which is exactly what S2 implements for SCOTCH.

## Test plan

Hand-verifiable cases:

1. **Path P_n (n=11)**: optimal separator is the single middle
   vertex, weight 1. Initial bisection of `[0..5] | [5..10]` puts 5
   on the boundary; FM should shrink to `{5}` (or its symmetry).
2. **Cycle C_8**: minimum separator weight is 2 (any two
   antipodal vertices). Verify result has weight ≤ 2.
3. **Complete bipartite K_{3,3}**: minimum separator is *one whole
   side*, weight 3. From an initial 3 vs 3 cut, FM cannot reduce
   below 3.
4. **Grid 5×5**: minimum row separator is one full row, weight 5.
   Verify final separator weight ≤ 5 + slack.
5. **Disconnected components**: graph = two disjoint K_4. Initial
   bisection puts one K_4 in A, one in B → no crossing edges → no
   separator. Verify `S = ∅`.
6. **Determinism**: run twice with same seed → identical labels.
7. **Validity invariant**: post-call, no edge has one end in
   PART_A and the other in PART_B.
8. **Balance invariant**: post-call, `max(a_w, b_w) ≤
   (1 + max_imbalance) * (a_w + b_w + s_w) / 2` where the
   denominator is the *original* total (separator vertices count
   for nothing in side weights but still count in the total).

## Out of scope for S2

- Halo extension (S3).
- Band extraction (S4).
- The recursive driver (S5).
- Tie-break by neighbour count (used in some SCOTCH variants); we
  stick with strict gain + index for now and revisit if quality
  benchmarks demand it.
