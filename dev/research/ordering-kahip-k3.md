# KaHIP Phase K3 — Flow-Based Edge Refinement

## Context

K3 consumes the push-relabel primitive from K2 and an existing edge
bisection (produced later by K5's coarsening + initial partition)
to reduce the cut weight by solving a local max-flow / min-cut.

Reference: Sanders & Schulz 2011, "Engineering Multilevel Graph
Partitioning Algorithms", §4 (local-search refinement via
max-flow). Clean-room from the paper, not the C++ source.

## Problem statement

Given an undirected weighted graph `G = (V, E, w)` with a bisection
`π : V → {0, 1}` (where `w(e) > 0` is the edge weight and the
partition weight is in balance within some `ε`), produce a new
bisection `π'` with

  cut(π') ≤ cut(π)

while respecting the balance constraint
  max(|π'⁻¹(0)|, |π'⁻¹(1)|) ≤ (1 + ε) · ⌈|V|/2⌉

(weighted version uses vertex weights; for the K3 scope vertices are
unit-weight).

`cut(π)` = Σ{ w(u,v) : π(u) ≠ π(v) }.

## Boundary and band

- **Boundary** `B(π)` = { v ∈ V : ∃ u ~ v with π(u) ≠ π(v) }.
- **Band of radius `d`** = set of vertices whose graph distance to
  `B(π)` is at most `d`. We compute it with BFS seeded on every
  boundary vertex, recording the distance `dist[v] ∈ {0, 1, …, d}`.
- **Fixed nodes (source side)** = band vertices with `π(v) = 0` AND
  `dist[v] == d`.
- **Fixed nodes (sink side)** = band vertices with `π(v) = 1` AND
  `dist[v] == d`.
- **Internal band vertices** = band vertices with `dist[v] < d`.
  These are free to be reassigned by the min-cut.
- Vertices outside the band are not touched.

Per audit bug 2, "fixed nodes" means "every band vertex at distance
EXACTLY `d`", **not** "the single deepest vertex" — the super-source
connects to ALL such part-0 vertices with infinite capacity, and
similarly for the super-sink and part-1 vertices.

If `d == 0`, the band collapses to the boundary itself and no
internal vertices exist; refinement is a no-op. K3 requires `d ≥ 1`.

## Flow network construction

Build a directed capacitated network `N = (V_N, E_N, c)` as follows:

- `V_N = band ∪ { s, t }` where `s` = super-source, `t` = super-sink.
- For each undirected edge `(u, v) ∈ E` with both endpoints in
  the band and weight `w`, add **two anti-parallel directed edges**
  `(u, v, w)` and `(v, u, w)` (item 10 of the audit).
- For each part-0 fixed node `v` (`dist[v] == d, π(v) == 0`), add
  `(s, v, INF)`.
- For each part-1 fixed node `v` (`dist[v] == d, π(v) == 1`), add
  `(v, t, INF)`.

"`INF`" = large enough that the min-cut never saturates a super
edge. The sum of all in-band edge weights bounds the min-cut, so
`INF = (Σ w) + 1` suffices. (We use `i64::MAX / 4` for headroom;
the i64 capacity type makes over-provisioning free.)

Edges with only one endpoint in the band are **not** added to the
network — the cut in the band is what we are refining, so edges
crossing the band boundary don't participate. The band-exterior
cut weight is unchanged by K3; we only try to beat the in-band
cut weight.

## Running max-flow

Call `push_relabel(|V_N|, edges, s, t)` from K2. The returned
`is_source_side` gives a min-cut; partition the band according to:

  π'(v) = 0 iff `is_source_side[v]`   (for `v` in the band)
  π'(v) = π(v)                        (for `v` outside the band)

By the max-flow / min-cut theorem, the cut-weight within the band
equals the max-flow value. Fixed nodes cannot switch sides because
the super-source / super-sink pins them (infinite capacity).

## Most Balanced Min Cut — v1 scope

Multiple min-cuts may exist with equal value; which one we apply
affects balance. Full MBMC (Sanders-Schulz 2011 §4.3) manipulates
residual flow to walk across min-cuts. For this K3 v1 we implement a
**two-cut search**:

- **Source cut `S`**: `is_source_side` from residual BFS from `s`.
  This is the min-cut closest to the source (smallest source side).
- **Sink cut `T`**: `is_sink_side` from residual BFS from `t` on the
  reverse residual graph. This is the min-cut closest to the sink
  (largest source side).

`S ⊆ V_N \\ T ∪ {s, t}`; any min-cut lies between them (lattice of
min-cuts). We compute the band weights of both candidates and pick
the one that satisfies the balance constraint while minimizing cut
weight. If neither balances, we reject the refinement.

Full MBMC is deferred to a follow-up. The plan's audit item 3 asks
for it; we document v1 scope and a follow-up task.

## Monotonicity and acceptance

Compute `new_cut = cut(π')` via the standard edge-weight sum across
all graph edges `(u, v)` with `π'(u) ≠ π'(v)`. Apply iff

  new_cut < old_cut   AND   balance(π') within tolerance.

(Strict improvement: avoid thrashing on equal-weight cuts.)

## Iteration

A single K3 call does **one** band-extract + flow-solve. The
caller (K5 V/F-cycle controller or K6 driver) loops until no
improvement or an iteration cap. Per Sanders-Schulz 2011, 2-3 inner
iterations saturate on most instances; more is diminishing returns.

## API

```rust
pub(crate) struct UndirectedGraph {
    pub n: usize,
    pub xadj: Vec<usize>,     // length n+1
    pub adjncy: Vec<usize>,   // length xadj[n]
    pub eweight: Vec<i64>,    // parallel to adjncy; > 0
}

/// One iteration of flow-based edge refinement on a bisection.
///
/// Returns true if the bisection was improved (strictly lower cut
/// weight while respecting `max_imbalance`). The partition `where_`
/// is updated in place; on false return it is unchanged.
pub(crate) fn flow_refine_bisection(
    graph: &UndirectedGraph,
    where_: &mut [u8],        // ∈ {0, 1} per vertex
    bnd_distance: usize,      // ≥ 1
    max_imbalance: f64,       // e.g. 0.03 for ±3%
) -> bool;
```

Error handling: malformed input (graph/where size mismatch,
`where_[v]` outside {0,1}, `bnd_distance == 0`) is a programming
error; debug-assert and return `false` without mutating `where_`.

## Test oracles

1. **Empty / trivial.** `n == 0`, or `bnd_distance == 0`, or no
   boundary (all vertices in one part): returns `false`,
   `where_` unchanged.
2. **Pre-optimal bisection.** A path graph `0-1-...-n` with the
   cut at the midpoint (`w = 1`). Any flow refinement cannot
   improve; return `false`.
3. **Sub-optimal diagonal on a 7x7 grid.** Seed `where_` with a
   diagonal cut of weight ≥ 14. After K3 with `bnd_distance = 2`
   the cut drops to 7 (horizontal slice), which is the true
   minimum. Assert strict improvement and that the resulting cut
   is exactly 7.
4. **Determinism.** Run twice on the same input; both calls must
   produce the same `where_`.
5. **Balance enforcement.** On a graph where the min-cut has bad
   balance (e.g., `1` vs `n - 1`), `max_imbalance = 0.05` must
   reject it and return `false`.
6. **Fixed-node pinning.** After refinement, any vertex at depth
   `d` retains its original partition (verified by setting up a
   band with known fixed nodes).
7. **Non-worsening.** For a random graph with a random bisection,
   running K3 never increases `cut(where_)` (strict monotonicity
   guarantee).
8. **Band-boundary invariance.** Edges whose endpoints straddle
   the band boundary (one in band, one out) retain their cut
   contribution unchanged.

## Design decisions

- CSR graph representation. Sorted neighbor lists (easier for
  boundary detection and stable adjacency iteration). `eweight` is
  parallel to `adjncy`.
- Unit-weight vertices for K3. Vertex weights enter K4 (node
  separator min-cut) via the node-splitting reduction; we carry a
  `vweight` vector in a future patch.
- `bnd_distance` as a runtime parameter (not hard-coded). K6's
  Fast mode will call with d=3, Strong with d=5. Audit item 12.
- `max_imbalance` as a fraction; default 0.03 mirrors KaHIP.
- `INF = i64::MAX / 4`. Capacity arithmetic never overflows since
  residuals and pushes are bounded by source excess, which sums to
  `(#fixed source-side) * INF` — still `< i64::MAX`.
- Balance check: `max(|π'⁻¹(0)|, |π'⁻¹(1)|) ≤ (1 + ε) · ⌈n/2⌉`.
  Matches KaHIP's definition.

## Out of scope (deferred)

- **Full Most Balanced Min Cut** (Sanders-Schulz 2011 §4.3):
  K3 v1 implements a two-cut (source-BFS vs sink-BFS) search;
  full MBMC via residual-flow manipulation is K5/K6 follow-up.
- **Adaptive `bnd_distance`** (start at 2, grow until band >5% of
  |V|): deferred to K6 tuning.
- **Global relabeling in push-relabel**: still deferred as in K2.
- **Vertex-weighted balance**: unit weights only in K3; K4 will
  introduce vertex weights.
