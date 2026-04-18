# Research Note: KaHIP Phase K1 — Data Reduction

**Session:** 2026-04-18-04
**Plan:** `dev/plans/ordering-kahip.md` (audit lines 17–131)
**Primary paper:** Ost, Schulz & Strash, "Engineering Data Reduction
for Nested Dissection" (2021), https://arxiv.org/abs/2004.11315.
**Scope:** Internal to `feral-kahip`. No wiring into `OrderingMethod`.

## What K1 does

Given a full-symmetric graph `G = (V, E)`, produce a reduced graph
`G' = (V', E')` with `V' ⊆ V` plus a stack of `ReductionOp` records
that lets us expand any permutation `π' : [|V'|] → V'` back to a
permutation `π : [|V|] → V` that preserves the fill-reducing
ordering property: eliminated vertices are placed *before* their
anchor in the final elimination order.

## The four rules

Let `N(v) = {u : (u,v) ∈ E}` (open neighborhood, no self-loops) and
`N[v] = N(v) ∪ {v}` (closed neighborhood). `deg(v) = |N(v)|`.

### Rule 1 — Degree-1 elimination

If `deg(v) = 1` with unique neighbor `u`, remove `v`.

- **Fill:** zero. `v` has no off-diagonal entries to generate fill.
- **Expansion:** `v` goes immediately before `u` in the elimination
  order. Record as `Degree1 { v, owner: u }`.
- **Cascading:** removing `v` may drop `u` to degree 1; `u` must be
  reconsidered in the same pass. Order of removal matters for
  expansion (the inner `v` was removed before `u` became
  degree-1-ready).

### Rule 2 — Degree-2 path compression (two sub-cases)

If `deg(v) = 2` with neighbors `p, q`:

- **Case A (simplicial, p ∼ q):** `v` is simplicial — eliminating
  `v` creates no new fill edge because the fill edge `(p,q)` already
  exists. Remove `v` with zero fill; do NOT add any edge.
- **Case B (non-simplicial, p ≁ q):** eliminate `v` and add the
  fill edge `(p, q)` to the reduced graph. One unit of fill cost.

Chains of degree-2 vertices between two "branch" endpoints can be
compressed together. Record the full interior path (in traversal
order) as `Degree2Path { u, w, path: [v_1, ..., v_k], simplicial }`.
During expansion, the interior is eliminated in path order before
both endpoints.

**Audit note (Bug 1):** the original plan only mentioned Case B.
Case A is strictly better than what Rule 4 catches generically.

### Rule 3 — Twin detection (open AND closed)

Two vertices `u, v` are *twins* if they have identical neighborhoods.

- **Open twins (`u ≁ v`):** `N(u) = N(v)`. They are not adjacent.
- **Closed twins (`u ∼ v`):** `N[u] = N[v]`. They are adjacent;
  each appears in the other's neighborhood.

Twins can be merged into one supervariable: eliminate `dup`, keep
`rep`, record `Twin { rep, dup, closed }`. `dup` is placed
immediately before `rep` in the expanded permutation (they share all
neighbors, so the second elimination creates no new fill).

**Canonical signatures** for fast grouping:
- `open_sig(v) = sorted(N(v))`
- `closed_sig(v) = sorted(N(v) ∪ {v})`

Group by signature, then within a group find pairs `(u, v)` with
`u < v` that satisfy the adjacency condition (non-adjacent for open,
adjacent for closed — closed pairs are guaranteed adjacent because
they share the signature that contains both).

**Audit note (Bug 5):** closed twins are common in KKT diagonal
blocks (e.g., two rows corresponding to equality constraints on the
same variable subset).

### Rule 4 — Neighborhood subset reduction

If `N(v) \ {u} ⊆ N(u)` for some `u ∈ N(v)`, vertex `v` is "dominated"
by `u`: all of `v`'s neighbors are also neighbors of `u` (modulo the
`u-v` edge itself). Eliminate `v` with no fill cost beyond what `u`
will already pay.

**Mark-array implementation** (Bug 7): for each `u`, mark `N(u)`.
For each `v ∈ N(u)`, walk `N(v)` and count how many are marked. If
the count equals `deg(v) - 1` (excluding the `u-v` edge), then
`N(v) \ {u} ⊆ N(u)`. Record as `SubsetElim { v, owner: u }`.

Cost is `O(Σ_u deg(u) · avg_deg_of_neighbors_of_u)`, not `O(|E|)`.

**Session scope decision:** Rule 4 is implemented minimally but with
conservative termination to avoid quadratic blow-up on dense graphs.
Full tuning deferred.

## Fixed-point loop

```
loop:
    progress = 0
    progress += apply_rule_1()  # with internal cascade
    progress += apply_rule_2()  # with internal cascade
    progress += apply_rule_3()
    progress += apply_rule_4()
    if progress == 0: break
```

Termination: `|V|` decreases monotonically. Each pass either
shrinks the graph or is the last pass. Worst case `O(|V|)` passes
but in practice 3–5 per Ost-Schulz-Strash 2021.

## Expansion: anchor-based union-find

Each eliminated vertex `v` has an *anchor*: a still-alive vertex that
`v` must be placed immediately before in the elimination order.

- `Degree1 { v, owner }`: `anchor(v) = owner`.
- `Degree2Path { u, w, path, _ }`: for the k-th path vertex, anchor
  at either endpoint — pick `u` by convention for determinism. The
  path interior is ordered `[path[0], ..., path[k-1]]`, all placed
  before `u` (which is placed before anything else whose anchor is
  `u`).
- `Twin { rep, dup, _ }`: `anchor(dup) = rep`.
- `SubsetElim { v, owner }`: `anchor(v) = owner`.

**Union-find with path compression:** if a vertex used as an owner
was itself later eliminated, we must chase `anchor(anchor(...))`
until we hit a surviving vertex. Implemented as a `parent[v]` array
where `parent[v] = v` for surviving vertices. At expansion, we
compress paths on read.

**Proof sketch (correctness of expansion):**
Each rule's elimination is "local" — it introduces at most the
minimal fill implied by its case. If the reduced graph is ordered
by any fill-reducing method, then inserting the eliminated vertices
immediately before their anchors produces an elimination order
whose fill is bounded above by (fill on reduced graph) + (the local
fill contributions from the eliminated-during-reduction vertices,
each of which is zero or one fill edge, already accounted for in
the reduced graph's structure per Rules 1–4).

Formal proof: Ost-Schulz-Strash 2021, Theorem 3.1.

## Test oracle construction

Tests rely ONLY on external oracles (graph structure, paper
definitions):

1. **Path(n=20):** Rule 1 removes both endpoints; Rule 2 compresses
   the interior. Final reduced graph has 0 or 1 vertex (depending
   on parity). Verified by hand.

2. **Star(n=10):** Rule 1 removes all 9 leaves; hub has degree 0
   after (no neighbors left, becomes an isolated vertex).

3. **Closed twins in K4 (complete graph):** Every pair of vertices
   in K4 is a closed twin. All reduced to a single vertex.

4. **Open twins in K_{2,3} (complete bipartite):** The three vertices
   on the size-3 side are pairwise open twins (share both vertices
   on the other side as neighbors, and are pairwise non-adjacent).

5. **Permutation validity:** For any small graph, `expand(reduce(G))`
   must produce a bijection of `[0, n)`.

6. **Fill monotonicity:** On a 4×4 grid, fill(reduced → expanded) ≤
   fill(naive identity permutation). This is a sanity check that
   the expansion orders eliminated vertices correctly.

## What ships this session

- `crates/feral-kahip/src/data_reduction.rs` with:
  - `ReductionOp` enum (4 variants).
  - `ReducedGraph` struct (new CSC + old_of_new + ops).
  - `reduce_graph(&CscPattern, max_ratio: f64) -> Option<ReducedGraph>`.
  - `expand_permutation(reduced: &ReducedGraph, reduced_perm: &[i32],
     n_original: usize) -> Vec<i32>`.
- Unit tests covering oracles 1–5 above.
- Fill-monotonicity test (oracle 6) deferred to K6 (needs a real
  ordering on the reduced graph to compare fills meaningfully).

## Out of scope (future phases)

- Push-relabel max-flow (K2).
- Any partitioning logic (K3–K5).
- Integration with `OrderingMethod::KahipND` (K6).
- Performance optimization of `BTreeSet`-backed adjacency: only if
  K2–K6 bench numbers show K1 as a bottleneck.
