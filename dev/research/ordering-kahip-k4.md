# KaHIP Phase K4 — Flow-Based Node Separator

## Context

K4 transforms an edge bisection `π : V → {0, 1}` into a **node
separator** `S ⊂ V` such that `V \ S` decomposes into two parts
`A, B` with no edge crossing between `A` and `B`. The node
separator is what nested dissection actually uses: the elimination
order places `S` last, so the Schur complement on `S` is the only
coupling between `A` and `B`.

Reference: Sanders & Schulz 2011, "Engineering Multilevel Graph
Partitioning Algorithms", §4.4 ("From Edge to Node Separators").
Clean-room from the paper, not from KaHIP's C++ source.

Prerequisites delivered in earlier phases:
  - K2 `push_relabel(n, edges, s, t) -> (flow_value, is_source_side)`
    — the primitive we reuse with vertex splitting.
  - K3 `UndirectedGraph` CSR (n, xadj, adjncy, eweight) and the
    band/fixed-node pattern (not reused directly; K4 uses a
    different network construction).

## Problem statement

Given `G = (V, E, w_V, w_E)` with vertex weights `w_V(v) ≥ 1` and
edge weights `w_E(e) ≥ 1`, plus an edge bisection `π : V → {0, 1}`,
produce a separator `S ⊆ V` minimizing `Σ_{v ∈ S} w_V(v)` subject
to:

  1. **Separator property.** Removing `S` disconnects the two
     parts. Formally, for every edge `(u, v) ∈ E` with
     `π(u) = 0 ∧ u ∉ S` and `π(v) = 1 ∧ v ∉ S`, no such edge
     exists.
  2. **Balance.** `max(w_V(A), w_V(B)) ≤ (1 + ε) · ⌈W / 2⌉` where
     `W = Σ_{v ∉ S} w_V(v)` and `A = {v ∉ S : π(v) = 0}`,
     `B = {v ∉ S : π(v) = 1}`.

For v1 we assume unit vertex weights (`w_V ≡ 1`) — the same scope
as K3. Vertex weights enter when K5 passes coarsened-level graphs
through K4.

## Key idea: boundary + vertex splitting

### Boundary-only formulation

The separator need only be drawn from the **boundary** of the
bisection:

  `B_0 = { v ∈ V : π(v) = 0 ∧ ∃ (v, u) ∈ E with π(u) = 1 }`
  `B_1 = { v ∈ V : π(v) = 1 ∧ ∃ (v, u) ∈ E with π(u) = 0 }`

Any vertex strictly interior to part 0 or 1 has no cross-part
neighbor and thus need not be in the separator. So `S ⊆ B_0 ∪ B_1`.

The boundary-subgraph is bipartite in the sense that every edge
crossing the cut goes from `B_0` to `B_1` (edges internal to `B_0`
or `B_1` are not cut edges). The separator must hit every such
crossing edge: it is a **vertex cover of the bipartite cut-edge
graph `H = (B_0 ∪ B_1, E_cross)`**.

By König's theorem, the minimum vertex cover in a bipartite graph
equals the maximum matching, which in turn equals the minimum
`s-t` vertex cut in the standard flow reduction:

  source `s` → every `v ∈ B_0` with capacity `w_V(v)`;
  `u ∈ B_0`, `v ∈ B_1`, edge `(u, v) ∈ E_cross` → arc `u → v` with
    capacity `+∞`;
  every `v ∈ B_1` → sink `t` with capacity `w_V(v)`.

The min `s-t` cut in this network saturates only source/sink arcs
(inner arcs are ∞) and its saturated source/sink arcs correspond
to the vertices picked into `S`.

### Vertex splitting (general, not used for v1)

For a generic (non-bipartite) vertex-capacitated min-cut problem,
each vertex `v` with weight `w_V(v)` is split into `v_in → v_out`
with capacity `w_V(v)`, and every undirected edge `(u, v)` becomes
`u_out → v_in` and `v_out → u_in` with capacity `+∞`. The min cut
on this expanded graph saturates only split-arcs and corresponds
to a minimum vertex cut.

For K4 v1 we stick to the bipartite / boundary-vertex-cover
reduction since:
  - It is simpler (no edge explosion from splitting).
  - The König → max-matching equivalence makes the result exact
    and explainable.
  - It matches the Sanders-Schulz 2011 §4.4 construction
    literally.

If vertex weights are non-unit, the flow capacities (source-arc
and sink-arc) carry the vertex weight and the min-cut directly
gives the minimum-weight vertex cover. We handle this cleanly
because our K2 push-relabel takes `i64` capacities.

## Flow network construction (v1)

Inputs:
  - `graph: &UndirectedGraph` (from K3).
  - `where_: &[u8]` (bisection, `where_[v] ∈ {0, 1}`).
  - `vweight: &[i64]` (optional; default unit).

Build:

1. Compute boundary sets `B_0, B_1`.
2. Assign contiguous network ids:
   `b_0_id[v]` for `v ∈ B_0` → `0 .. |B_0|`;
   `b_1_id[v]` for `v ∈ B_1` → `|B_0| .. |B_0| + |B_1|`;
   super-source `s = |B_0| + |B_1|`;
   super-sink `t = |B_0| + |B_1| + 1`.
   Total: `n_net = |B_0| + |B_1| + 2`.
3. Edges:
   - For each `v ∈ B_0`: arc `(s, b_0_id[v], vweight[v])`.
   - For each `v ∈ B_1`: arc `(b_1_id[v], t, vweight[v])`.
   - For each `(u, v) ∈ E` with `π(u) = 0, π(v) = 1` (so `u ∈ B_0`
     and `v ∈ B_1`): arc `(b_0_id[u], b_1_id[v], INF)`. The reverse
     arc is not added — this is a DIRECTED acyclic network s → B_0
     → B_1 → t, matching the bipartite vertex-cover reduction.

`INF` = `(Σ vweight) + 1` (bounded by total vertex weight, which
bounds any vertex cover).

Run `push_relabel(n_net, edges, s, t)`. The returned
`is_source_side` gives a cut. The separator `S` is:

  `S = { v ∈ B_0 : the arc (s, b_0_id[v]) is saturated }`
    ∪ `{ v ∈ B_1 : the arc (b_1_id[v], t) is saturated }`

Equivalently, from `is_source_side`:

  `v ∈ B_0 is picked into S ⇔ ¬is_source_side[b_0_id[v]]`
  `v ∈ B_1 is picked into S ⇔ is_source_side[b_1_id[v]]`

(A source-arc is saturated iff its target is on the sink side; a
sink-arc is saturated iff its source is on the source side.)

## Producing the separator + parts

After extracting `S`:

  `A = { v ∈ V : π(v) = 0 ∧ v ∉ S }`
  `B = { v ∈ V : π(v) = 1 ∧ v ∉ S }`

**Separator property verification** (debug-assert): for every edge
`(u, v) ∈ E` with `π(u) ≠ π(v)`, at least one of `u, v` must be in
`S`. By König's theorem this holds whenever the min-cut is
computed correctly — but we assert it to catch bugs.

**Balance check.** Compute `w_A = Σ_{v ∈ A} w_V(v)` and `w_B`. If
`max(w_A, w_B) > (1 + ε) · ⌈(w_A + w_B) / 2⌉`, the separator is
unbalanced. The v1 policy: **return the separator anyway**, and
let the caller (K5/K6) decide whether to accept it. (KaHIP's own
driver retries with a different bisection on imbalance; that's K6
logic, not K4's.)

## API

```rust
pub(crate) struct NodeSeparator {
    /// `part[v] ∈ {0, 1, 2}`; 2 means separator.
    pub part: Vec<u8>,
    /// Total vertex weight of the separator.
    pub weight: i64,
}

/// Compute a min-weight node separator from an edge bisection
/// via the boundary-bipartite vertex-cover reduction.
///
/// `vweight = None` uses unit weights.
/// Returns `None` if the bisection has no cross-part edges
/// (empty separator is trivially optimal; caller handles the
/// empty-case path).
pub(crate) fn flow_node_separator(
    graph: &UndirectedGraph,
    where_: &[u8],
    vweight: Option<&[i64]>,
) -> Option<NodeSeparator>;
```

Error handling: malformed input (`where_` len mismatch, value
outside {0, 1}, `vweight` len mismatch) is a programming error;
debug-assert and return `None`.

## Test oracles

1. **Empty boundary.** All vertices in one part: returns `None`.
2. **Path graph.** `0-1-2-3-4-5-6-7-8`. Bisection
   `{0..3} | {4..8}`. Only crossing edge is `(3, 4)`; the
   minimum vertex cover = 1 vertex (either 3 or 4). Separator
   size = 1. Assert exactly one of `3, 4` is in `S`.
3. **7×7 grid horizontal bisection.** Part 0 = rows 0..3, part 1
   = rows 4..6. Crossing edges: `(r=3, c) - (r=4, c)` for
   `c = 0..6`, i.e., 7 edges arranged as a perfect matching in
   the bipartite cross graph. König: min vertex cover = max
   matching = 7. Assert separator size = 7.
4. **K_{3,3} bipartite.** With all left vertices in part 0, all
   right in part 1: every edge is crossing. Min vertex cover =
   `min(3, 3) = 3`. Assert separator size = 3.
5. **Determinism.** Same input produces same separator under
   repeated calls.
6. **Separator validity.** For arbitrary bisections on random
   graphs, every crossing edge must have at least one endpoint
   in `S`.
7. **Weight respect (vertex-weighted).** Assign large weights to
   some boundary vertices and small weights to others; assert
   the separator picks the small-weight covers whenever both
   options would cover the same edges.
8. **No interior vertex.** No vertex with all neighbors in its
   own part ever appears in the separator (sanity check on the
   boundary-only restriction).

## Design decisions

- **Boundary-bipartite reduction over generic vertex splitting.**
  Simpler, exact for bisections (König's theorem), fewer network
  vertices and edges, directly mirrors Sanders-Schulz §4.4.
- **Directed forward-only arcs.** Unlike K3 (undirected → anti-
  parallel), the bipartite reduction is intrinsically directional
  `s → B_0 → B_1 → t`. We do NOT add reverse cross-arcs.
- **Unit vertex weights in v1.** Matches K3 scope. K5 extends to
  coarsened vertex weights.
- **No balance enforcement inside K4.** The separator is always
  returned; balance checking is the caller's responsibility so K4
  is composable and deterministic.
- **INF capacity = `(Σ vweight) + 1`.** Bounded by total weight,
  safe against i64 overflow across any practical matrix size.
- **No multi-iteration loop.** One boundary extract + one max-flow
  call. Repeated improvement comes from K5's V/F-cycles (re-
  coarsen → re-refine → re-separate), not from iterating K4 in
  place.

## Relationship to K3 / K5

- **K3** refines a bisection's edge cut; it does NOT produce a
  separator. K4 consumes K3's output (a refined edge bisection)
  and converts it to a node separator.
- **K5** drives the pipeline: coarsen → initial partition →
  uncoarsen with K3 refinement at each level → K4 at the finest
  level. K4 is called once per nested-dissection recursion step;
  K5 is the within-step multilevel scheme.
- **K6** wraps K5 in the recursive ND driver: apply K5+K4 at the
  current level, recurse on `A` and `B`, stop when the subgraph
  is small enough (hand off to AMD).

## Out of scope (deferred)

- **Generic vertex-capacitated min-cut** (via node splitting).
  Not needed for the bipartite separator case; may be useful for
  K5's coarsened-level balance enforcement, revisit then.
- **Adaptive separator rebalancing.** If K4 returns an unbalanced
  separator, the caller's options are: (a) accept and move on
  (reduces parallelism but correct); (b) re-run K3 with a
  different seed; (c) reject and fall back to the prior level.
  K6 drives this choice; K4 stays stateless.
- **Weighted König-Egerváry.** The reduction handles weighted
  vertex covers directly via source/sink capacities = vertex
  weights; no separate algorithm needed.
- **Multi-way separators.** K4 is strictly bisection → separator.
  K-way is an entirely different KaHIP feature not in the plan.
