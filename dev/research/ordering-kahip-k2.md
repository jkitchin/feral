# KaHIP Phase K2 — Push-Relabel Max-Flow

## Context

Phase K2 delivers the max-flow / min-cut primitive that phases K3
(flow-based edge refinement) and K4 (flow-based node separator) consume.
No direct consumer until K3; this phase is purely the engine.

References (clean-room: papers and textbooks only, not the KaHIP C++):

- Goldberg & Tarjan 1988, "A New Approach to the Maximum Flow Problem"
  — original push-relabel.
- Cherkassky & Goldberg 1995, "On Implementing Push-Relabel Method for
  the Maximum Flow Problem" — gap relabeling + global relabeling
  heuristics, highest-label vs FIFO selection.
- Cormen, Leiserson, Rivest, Stein, *Introduction to Algorithms* 3e,
  Chapter 26 — textbook worked examples (Figure 26.1).
- Sanders & Schulz 2011, "Engineering Multilevel Graph Partitioning
  Algorithms" — usage as refinement primitive in nested dissection.

## Formal problem

Given a directed network `G = (V, E)` with nonnegative integer edge
capacities `c : E → ℤ≥0`, a source `s ∈ V`, and a sink `t ∈ V \ {s}`,
find a flow `f : E → ℤ≥0` satisfying:

1. **Capacity:** `0 ≤ f(u,v) ≤ c(u,v)` for every edge.
2. **Conservation:** for every `v ∈ V \ {s,t}`,
   `Σ f(u,v) = Σ f(v,w)`.

that maximizes `|f| = Σ f(s,v) = Σ f(u,t)`.

By max-flow / min-cut (Ford-Fulkerson): `|f*| = min {c(S, V\S) :
s ∈ S, t ∈ V\S}`. After solving, the set `S = {v : v reachable from s
in the residual graph `G_f*}` is one min-cut side.

## Push-relabel algorithm (Goldberg-Tarjan)

Maintain a *preflow* `f` (satisfies capacity; conservation is
relaxed to `excess(v) = Σ f(·,v) - Σ f(v,·) ≥ 0` for `v ≠ s`) and a
*height function* `h : V → ℤ≥0` satisfying:

- `h(s) = n`, `h(t) = 0`.
- For every residual edge `(u,v)` (with `c_f(u,v) > 0`):
  `h(u) ≤ h(v) + 1`.

An edge `(u,v)` is *admissible* iff `c_f(u,v) > 0` and `h(u) = h(v) + 1`.
A vertex `v ∉ {s,t}` is *active* iff `excess(v) > 0`.

**Init:** `h(s) = n`, all others 0; push `c(s,v)` along every
out-edge of `s`; `excess(s) = -Σ`, every neighbor of `s` becomes active.

**Operations:**

- **Push(u,v):** applicable when `u` active and `(u,v)` admissible.
  Send `δ = min(excess(u), c_f(u,v))` along `(u,v)`: decrease
  `excess(u)`, increase `excess(v)`, add `δ` to `f(u,v)`.
- **Relabel(u):** applicable when `u` active and no admissible
  out-edge. Set `h(u) = 1 + min {h(v) : (u,v) residual}`.

Terminate when no active vertex remains. At termination `f` is a valid
flow, and `|f| = -excess(s)`. A BFS on the residual graph from `s`
yields `S`.

Worst-case with highest-label selection: `O(V^2 √E)` (Cheriyan-Maheshwari
1989). For our use case — local bands of 100–5000 vertices inside K3 —
this is fast enough that the dominant cost per bisection level is the
coarsening, not the flow.

## Gap relabeling (required)

**Fact:** if at some point in the algorithm there exists a height
`0 < g < n` with no vertex at height `g`, then every vertex `u` with
`h(u) > g` is disconnected from `t` in the residual graph (there is
no residual path from `u` to `t` that avoids the gap, because any
such path would have to include a vertex at height `h(u) - 1, h(u) - 2,
…, 1` — but level `g` is empty, so no edge crosses it downward).

**Consequence:** such `u` will never again push flow to `t`. Raising
their heights to `n + 1` (or `n`) immediately removes them from the
active set, which avoids a cascade of wasted relabel operations.

**Implementation:** maintain `height_count[h]` (how many vertices sit
at height `h`). Every push/relabel that changes a height updates the
counts. After a relabel of `u`:

```
old_h = previous h(u)
height_count[old_h] -= 1
if old_h < n and height_count[old_h] == 0:
    for every v with old_h < h(v) < n:
        h(v) = n + 1
        height_count[h(v)] is no longer tracked
```

Cherkassky-Goldberg report 2–20× speedup from gap relabeling on their
DIMACS benchmark set. For small bands our absolute cost is small
either way, but the constant-factor win keeps K3 within the "Fast-mode
at METIS speed" budget.

## Highest-label selection + determinism (audit item 16)

Of the three standard selection rules (FIFO, highest-label,
lowest-label), highest-label has the best worst-case bound
(`O(V^2 √E)` vs `O(V^3)` for FIFO / lowest). We maintain `n + 1`
buckets of active vertices indexed by height; the main loop always
pulls from the highest non-empty bucket.

For deterministic tie-breaking:

- Within a bucket, use a **FIFO queue** (not a stack) so vertex
  promotion order is stable under seed changes.
- When scanning out-edges of `u` for an admissible one, iterate in
  **adjacency-list order** (i.e., by stored edge index) and take the
  first admissible edge. This yields the "lowest-index admissible
  neighbor" rule the audit calls for.

## Data structures

```
edges: Vec<Edge> where Edge { to: usize, rev: usize, cap: i64, flow: i64 }
adj:   Vec<Vec<usize>>   // adj[v] = indices into `edges`
h:     Vec<usize>        // height
excess: Vec<i64>
height_count: Vec<usize> // height_count[h] = #{v : h(v) == h}
active: VecDeque<usize> per height bucket  (vec of VecDeque)
```

Each undirected capacity-c edge `(u,v)` becomes two anti-parallel
entries: forward with `cap=c, flow=0` and reverse with `cap=c, flow=0`
(in max-flow textbooks the reverse is typically `cap=0`, but for the
**undirected flow model** K3 will need, we mirror capacities; callers
that want directed flow set the reverse capacity to 0 explicitly).
K2's signature exposes `(from, to, cap)` edges; whether to add a
reverse with capacity 0 or with capacity `c` is the caller's choice.
Our API: `push_relabel(n, edges: &[(usize, usize, i64)], source, sink)`
treats each `(from, to, cap)` as **directed**. K3 will double up
edges when it wants undirected semantics.

## Min-cut extraction

After termination, run a BFS from `source` over the residual graph
(edge `e` is residual iff `e.cap - e.flow > 0`). The visited set is
one valid min-cut partition (the source side). Return it as
`Vec<bool>` indexed by original vertex id.

## API

```rust
/// Solve max-flow / min-cut on a directed capacitated graph.
///
/// `edges` is a slice of `(from, to, cap)` with nonnegative integer
/// capacities. Multiple parallel edges are permitted and stack
/// (their capacities don't sum — each stays separate in the residual
/// graph). Self-loops `(v, v, _)` are ignored. `source == sink` or
/// any out-of-bounds vertex returns `MalformedInput`.
///
/// Returns `(flow_value, is_source_side)` where `is_source_side[v]`
/// is true iff `v` is on the source side of a min-cut (reachable
/// from `source` in the terminal residual graph).
pub(crate) fn push_relabel(
    n: usize,
    edges: &[(usize, usize, i64)],
    source: usize,
    sink: usize,
) -> Result<(i64, Vec<bool>), OrderingError>;
```

## Test oracle construction

Hand-computable oracles:

1. **Empty graph (n=2, source=0, sink=1, no edges):** max-flow = 0,
   source side = {0}.
2. **Single edge (0→1, cap=7):** max-flow = 7.
3. **Unit-capacity path (0→1→2→…→n-1, each cap=1):** max-flow = 1.
4. **Parallel edges (0→1 cap=3, 0→1 cap=5):** max-flow = 8.
5. **Diamond (0→1 cap=10, 0→2 cap=1, 1→3 cap=1, 2→3 cap=10):**
   bottleneck sum = 1 + 1 = 2.
6. **CLRS Figure 26.1** (classic network): max-flow = 23.
7. **k×k grid, horizontal cut, unit capacities:** max-flow = k.
8. **Bipartite matching as max-flow:** on `K_{3,3}` with
   super-source/sink, max-flow = 3.
9. **Max-flow = min-cut on random sparse:** for a random graph,
   verify `Σ (c - f) over forward cut edges + Σ f over backward cut
   edges = 0`, i.e., the cut is saturated forward and unused backward.

Not yet an oracle but a correctness check:

10. **Gap relabeling equivalence:** with and without the gap
    optimization, the returned `(flow, cut)` must be identical on a
    deterministic graph. (Cut may differ across tie-breaks; flow
    value must not.)

## Termination + correctness

- **Termination.** Goldberg-Tarjan 1988 §3: after at most `2n²` relabel
  operations the algorithm halts; every push is O(1). Gap relabeling
  can only decrease this count since it flips vertices out of the
  active set.
- **Correctness (end state is a max-flow).** At termination every
  `v ∉ {s,t}` has `excess(v) = 0` (otherwise there'd be an admissible
  edge to push along, because `h(v) < h(s) = n` + height function
  validity forces a residual path of descending heights to `t`, which
  contradicts termination). So the preflow is a flow, and it's
  maximum because any `s-t` path in the residual would have length
  at most `n - 1` but `h(s) - h(t) = n`, impossible.
- **Correctness of min-cut extraction.** The BFS-from-`s` set on the
  terminal residual graph is a standard consequence of max-flow /
  min-cut.

## Design decisions

- Capacity type: `i64`. K3 and K4 will use small weights (graph edge
  counts or small integer weights from coarsening), so 63 bits is
  overkill but costs nothing.
- Parallel edges kept separate (not summed). K3 may inject edges
  independently for different reasons.
- Self-loops ignored (they contribute nothing to max-flow).
- `source == sink` is a caller error, not "0 flow trivially": caught
  at the entry with `MalformedInput`.

## Out of scope (deferred to later phases)

- **Global relabeling** (periodic BFS from sink to reset heights).
  Beneficial on very large networks; our K3/K4 bands are small
  enough that we ship without it and revisit at K6 bench time.
- **Most-Balanced-Min-Cut** (Sanders-Schulz 2011 §4.3) — K3 concern,
  not K2.
- **Vertex-capacitated flow** (K4 concern). The standard reduction is
  node-splitting `v → (v_in, v_out)` with capacity `cap(v)` on the
  internal edge; that reduction happens in K4 on top of K2's
  edge-capacitated solver.
- **Parametric max-flow** (used for Gomory-Hu trees). Not needed.
