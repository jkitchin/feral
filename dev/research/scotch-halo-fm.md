# Research: SCOTCH Halo FM Refinement (S3)

**Date:** 2026-04-17
**Plan:** `dev/plans/ordering-scotch.md` §S3
**Status:** Pre-implementation

## Problem

Standard boundary FM (feral-metis `refine_bisection`) considers only
vertices adjacent to the opposite side as move candidates. On graphs
with strong "shoulder" structure — vertices that are not on the
boundary but whose movement would unlock a chain of improving boundary
moves — boundary FM can leave gain on the table.

Halo FM widens the candidate set to include the *one-hop halo*: any
vertex with at least one neighbour on the boundary. The audit (finding
3) requires the halo be **DYNAMIC** — recomputed (or maintained
incrementally) during the pass as the boundary shifts.

## Algorithm

State per vertex:

- `labels[v] ∈ {PART_A, PART_B}` (edge bisection, no separator).
- `boundary_count[v]` = number of neighbours of `v` in the *opposite*
  side. `v ∈ boundary` iff `boundary_count[v] > 0`.
- `halo_count[v]` = number of neighbours of `v` that are *themselves*
  in the boundary set. `v ∈ halo` iff `halo_count[v] > 0` AND `v ∉
  boundary`.
- `gain[v] = e_other - e_same` where `e_X = Σ adjwgt(v,u) over u ∈ X`.
  This is the standard FM gain — moving `v` from its side to the
  other reduces cut by `gain[v]`.

Pass:

1. Compute `boundary_count`, `halo_count`, `gain` for every vertex.
2. Build a max-heap keyed on `gain`, populated with every vertex that
   is in `boundary ∪ halo`. (We allow non-boundary halo vertices to
   move; they will become boundary after the move with negative
   immediate gain but may unlock future moves.)
3. Pop highest-gain unlocked candidate `v`. Tie-break by ascending
   vertex index.
4. Tentatively move `v` to the other side. Apply per-move balance
   check (`max(a_w, b_w) ≤ (1+ε)*total/2`); if rejected, lock `v`
   and continue (do not remove from candidate set permanently — but
   for simplicity and correctness, we lock-and-continue per pass,
   matching feral-metis's `refine_bisection` semantics).
5. Update `cur_cut -= gain[v]`. Record move in undo log.
6. For each neighbour `u`:
   - `boundary_count[u]` flips: if `u` is now on the same side as
     the moved `v`, the edge no longer crosses, so `u`'s opposite-
     count decreases by 1 (if `u` was on `v`'s old side) or
     increases by 1 (if `u` was on `v`'s new side).
   - Concretely: if `labels[u] == old_side(v)` (where `v` left
     this side), `boundary_count[u]` increases (now an opposite
     neighbour, namely `v`, exists). If `labels[u] == new_side(v)`,
     `boundary_count[u]` decreases.
   - `gain[u]` updates: `gain[u] += 2 * adjwgt(u,v) * sign` (the
     standard FM update). `sign = +1` if `u` was on `v`'s old side
     (the cut edge with `v` was crossing before but the new
     same-side edge is not), `-1` otherwise.
7. **Halo update**: when `boundary_count[u]` transitions
   `0 → positive` or `positive → 0`, neighbours of `u` need their
   `halo_count` updated (`u` joined / left the boundary set). Push
   any newly-eligible halo vertices into the heap.
8. Track `best_cut` and `best_prefix` subject to balance.
9. End-of-pass: roll back to `best_prefix`.

Repeat passes until no improvement or `pass_cap`.

## Reuse vs duplication

feral-metis already has a `refine_bisection` that does steps 1–6 with
a *boundary-only* candidate set. The algorithmic core (gain
computation, lazy heap with stamped entries, best-prefix rollback,
balance check) is identical. The only differences:

1. Initial candidate set = `boundary ∪ halo` (not just boundary).
2. `boundary_count` and `halo_count` maintained incrementally to
   detect halo changes.
3. When boundary changes, halo may grow or shrink → push new
   candidates.

**Decision**: re-implement in feral-scotch rather than parametrise
feral-metis. Parametrisation would force feral-metis to carry halo
bookkeeping (`halo_count`, `pq` push on halo entry) it doesn't need
for METIS-style ordering. The implementations diverge enough that
cross-cutting parameters would obscure both.

The new module imports `Graph`, `cut_size`, `part_weight`, `PART_A`,
`PART_B` from `feral_metis::internals`.

## Test plan

1. **Cut never grows**: on a 6×6 grid with a hand-perturbed
   bisection, `halo_fm_refine` returns `cut_after ≤ cut_before`.
2. **Cut matches boundary FM on small graphs**: on a 4×4 grid with
   half-half bisection, halo FM and boundary FM produce the same
   cut (halo cannot find moves boundary couldn't, and the optimum
   is reached by both).
3. **Halo finds extra moves**: construct a graph where boundary FM
   leaves the cut unchanged but halo FM reduces it. Specifically:
   two cliques connected by a single edge, with a vertex slightly
   off-boundary — boundary FM cannot move it (not on boundary)
   but halo FM can. Quality assertion is necessarily probabilistic
   on random graphs; this targeted case lets us assert strict
   improvement.
4. **Determinism**: same input → same output.
5. **Balance**: post-call satisfies `max_imbalance`.
6. **Empty / trivial inputs**: n=0 returns 0; n=2 with cut 0
   stays 0.

## Out of scope

- AMD anchor handling — not part of halo FM (S4 band FM has anchors).
- Convergence theorem on unbounded passes — bounded by pass_cap=32.
