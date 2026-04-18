# Research: SCOTCH Band FM Refinement (S4)

**Date:** 2026-04-17
**Plan:** `dev/plans/ordering-scotch.md` §S4
**Status:** Pre-implementation

## Problem

Standard FM scans the whole graph. When the partition boundary is
small relative to total size, almost all of FM's work is wasted on
vertices that will never move. SCOTCH's `bgraph_bipart_bd.c` extracts
a "band" subgraph of width `w` around the current boundary, runs FM
inside the band, and projects the result back.

The audit (finding 2) flagged that a naive band extraction breaks
balance accounting: vertices outside the band are *invisible* to FM,
so FM happily empties one side of the band without realising the
global partition is now massively unbalanced. Fix: add two artificial
**anchor supervertices** (one per side) that absorb the weight of
all out-of-band vertices on each side. FM cannot move anchors but
their weight enters every balance check.

## Algorithm

**Inputs:** original graph `G`, edge bisection `labels[v] ∈
{PART_A, PART_B}`, band width `w`, balance tolerance.

**Build phase:**

1. BFS from every boundary vertex (vertex with at least one
   neighbour on the other side) to depth `w`. Mark visited
   vertices as in-band.
2. Allocate a sub-graph `B` with `n_band + 2` vertices: one
   slot per in-band vertex plus two anchors `anchor_a` (last-1)
   and `anchor_b` (last).
3. For every edge `(u,v)` in `G`:
   - If both ends in band: copy the edge to `B`.
   - If one end in band, other out of band on side `s`: add an
     edge in `B` from the band vertex to the corresponding anchor.
     Aggregate parallel edges into a single weighted edge.
   - If both out of band: skip (no effect on band's view).
4. Anchor weights: `vwgt[anchor_s] = sum of vwgt[v] over out-of-band
   v with labels[v] == s`. Anchor labels are set to `s`.
5. Run `refine_bisection` (or halo FM) on `B`. Anchors are *locked*
   for the entire pass — they cannot be moved.

**Project phase:**

After FM returns refined labels for `B`, copy each in-band vertex's
label back to `G`. Out-of-band vertices keep their original labels.

## Reuse from feral-metis / feral-scotch

- `Graph` from `feral_metis::internals::graph` for both `G` and `B`.
- `feral_metis::internals::initial_partition::{PART_A, PART_B,
  cut_size, part_weight}`.
- For the inner FM pass we use `crate::halo_fm::halo_fm_refine` to
  reuse the corrected sign convention; we extend it with a `locked
  set` parameter or wrap it. Cleanest: implement a tiny
  `band_fm_inner` that mirrors halo FM but takes a "vertices that
  cannot move" mask (anchors).

Actually the simpler design: anchors have `vwgt` equal to the
out-of-band weight on their side. Since they are connected to all
band-boundary vertices that abut them, an anchor's gain in a flip
would be enormous — but we explicitly skip anchor entries when
populating the FM heap. That's two lines of code at the candidate-set
boundary; no need to thread a mask through halo_fm.

To keep halo_fm.rs unchanged, S4 will implement its own minimal
band-FM pass inline (a stripped-down boundary FM with anchor
exclusion). Cost: ~70 lines of duplicated FM core. Trade: avoid
parametrising halo_fm with options it doesn't need.

## Determinism

BFS expansion: deterministic when starting frontier is processed in
ascending vertex order. We use a `VecDeque<u32>` and sort the
initial boundary vertices.

## Test plan

1. **Band of width 1 = boundary**: on a 5×5 grid with a row
   bisection, width=1 should produce the same band as
   `boundary ∪ {immediate neighbours}`. Verify in-band count.
2. **Anchor weights match**: build a band with width=1 on a 5×5
   grid, compute `vwgt[anchor_a]` directly, compare to expected.
3. **Cut never grows**: on a 6×6 grid, `band_fm_refine(width=2)`
   returns `cut_after ≤ cut_before`.
4. **Out-of-band vertices preserved**: after band FM, every
   out-of-band vertex's label equals its pre-call label.
5. **Anchor never moves**: post-call labels of `anchor_a, anchor_b`
   are unchanged (we don't expose them, but verifiable internally
   via the projection step — the build phase copies anchors with
   their assigned side label).
6. **Determinism**: same input → same output.
7. **Balance**: post-call satisfies global `max_imbalance` on `G`
   (NOT just on band).
8. **Edge case**: band covers whole graph → result identical to
   running FM on G directly (anchor weights = 0, anchors are
   isolated isolated-vertex no-ops).

## Out of scope

- Auto width selection (audit `RefineMethod::Auto`): defer to S5
  driver dispatch.
- Multilevel band FM (band-of-band): not in SCOTCH default
  pipeline either.
