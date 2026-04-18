# SCOTCH Graph Compression — Research Note

**Status:** Pre-implementation. Covers Phase S1 of `dev/plans/ordering-scotch.md`.
**Date:** 2026-04-17
**Plan:** `dev/plans/ordering-scotch.md` §"Graph Compression (`compress.rs`)"
**Wiki concept:** `.crucible/wiki/concepts/scotch-graph-ordering.org`
**Paper(s):** Pellegrini 1996 §3 (SCOTCH); Ashcraft 1995 (compression in
the AMD/MMD setting). MUMPS reference dispatch:
`mumps/PORD/lib/interface.c:84-99` (PORD-style top-of-recursion compression).
**Clean-room provenance:** algorithm is reconstructed from Pellegrini 1996 §3
and the conceptual description above. No SCOTCH source files
(`graph_coarsen.c`, `vgraph_separate_*.c`) are read or copied; constants and
data layout are independently chosen and documented per-decision.

---

## Problem

Two vertices `u, v` in a graph `G = (V, E)` are *indistinguishable* (also
called *supervariables* in the AMD literature) when their **closed**
neighborhoods are identical:

  N[u] = N[v]   where   N[u] = N(u) ∪ {u}.

Equivalently (since the graph is symmetric and we drop self-loops upfront):

  N(u) ∪ {u} == N(v) ∪ {v},
  i.e. N(u) \ {v} == N(v) \ {u}   AND   `(u,v) ∈ E ⇔ (v,u) ∈ E` (always true).

Two cases need merging:

1. `(u,v) ∉ E`: `N(u) == N(v)` exactly.
2. `(u,v) ∈ E`:  `N(u) \ {v} == N(v) \ {u}`.

In a fill-reducing context this matters because indistinguishable vertices
are eliminated in the same supernode anyway; collapsing them up-front
shrinks every coarsening level and the partitioning kernels by the same
factor.

## Algorithm

Hash-and-verify, in two passes:

1. **Hash pass.** For each vertex `v`, compute a hash `h(v)` of its
   *closed* sorted neighbor list. Closed-neighborhood is the simplest way
   to make case (1) and case (2) hash-equal; an open-neighborhood hash
   would distinguish `(u,v) ∈ E` from `(u,v) ∉ E` even when `u` and `v`
   are indistinguishable.

2. **Bucket-and-verify pass.** Group vertices by `h(v)`. Within each
   bucket, compare closed neighbor lists exactly to confirm equivalence
   classes (defeating hash collisions). Each equivalence class becomes
   one supervariable in the compressed graph.

3. **Build compressed graph.** Map each original vertex `v` to its
   supervariable index `c[v]`. For each edge `(u, w)` in `G`:
   - If `c[u] == c[w]`, it is an internal edge of a supervariable —
     drop it (no self-loops in the compressed graph).
   - Else, record the edge `(c[u], c[w])` once per direction with its
     weight contribution.
   - **Sum edge weights** across parallel edges that collapse onto the
     same compressed pair `(c[u], c[w])`.
   - **Sum vertex weights** across the original vertices in each class.

   Result: `xadj_c, adjncy_c, vwgt_c, adjwgt_c` for the compressed graph,
   plus `vertex_map: Vec<Vec<usize>>` recording the original vertices in
   each supervariable (for permutation expansion later).

## Invariants and correctness gates

- **Bijection on expansion.** The vertex_map's flattened image must equal
  `0..n` exactly once. Tested on every output.
- **Vertex-weight conservation.** `sum(vwgt_c) == sum(vwgt)` (assuming
  unit weights on input, this is `sum(vwgt_c) == n`).
- **Edge-weight conservation.** Total *external* edge weight (excluding
  edges that collapsed to self-loops) must equal `sum(adjwgt_c)`. Edges
  collapsed to self-loops are intentionally dropped — they carry no
  partitioning information.
- **Symmetry preserved.** For every directed edge `c[u] → c[w]` in the
  compressed graph there is a matching `c[w] → c[u]`. This is automatic
  because we visit each undirected edge from both endpoints when reading
  the original CSR.
- **No hash leakage.** Two vertices in the same compression class must
  have **bytewise identical** closed sorted neighbor lists. Hash equality
  is necessary but not sufficient.

## Edge-weight summing — why it matters

The audit (`dev/plans/ordering-scotch.md` finding 5) flags that vertex-
weight summing alone is insufficient. Concrete failure mode if edge
weights are not summed: when SCOTCH's two-sided FM later evaluates a move
of a supervariable `c` to the opposite part, the gain formula uses the
sum of `adjwgt[c → opposite]`. If two original edges
`(u₁ ∈ c) — (w₁ ∈ c')` and `(u₂ ∈ c) — (w₂ ∈ c')` collapse to a single
compressed edge with `adjwgt = 1`, FM thinks the cost is half what it
truly is, and may accept moves that grow the *true* cut. This is the
exact bookkeeping bug the SCOTCH source guards against in
`graph_coarsen.c`. We do not need to read that file to know the
invariant; the FM gain formula in §S2 of the plan determines it.

## Compression cadence — per-level vs. once

Two reasonable policies:

- **Once at top of recursion** (PORD-style): compress only the original
  graph; recurse on the compressed graph thereafter. Cheap, correct, but
  misses opportunities created by partitioning (interior subgraphs
  sometimes have new supervariables).

- **Per recursion level** (Plan choice): re-attempt compression at every
  recursion level. More aggressive, but wastes work on graphs without
  supervariables (random matrices, KKT off the structured-mesh path).

The plan's audit resolves this: do per-level, but **short-circuit** —
once a level returns "no usable compression", do not re-attempt deeper
in the same recursion subtree. This matches the audit's finding (item 5)
and is the policy we will implement at the driver level (S5), not in
`compress.rs` itself.

For S1 the contract is just: `compress_graph(g, min_ratio) -> Option<…>`.
The driver decides when to call it.

## Threshold

Plan default: `min_ratio = 0.7` (return `None` if the compression is less
aggressive than 30% reduction). PORD uses 0.75; the plan's choice of
0.70 is documented as "slightly more aggressive — defensible." We honor
the plan default and expose it via `ScotchOptions.compress_ratio` later.

A key subtlety: "compression ratio" in this plan means `n_compressed / n`
(smaller is better), so the threshold is *strict less than*:

  return Some(g_c) iff (n_compressed / n) < min_ratio.

This matches the plan wording "compress if ratio > threshold" — it reads
as *if savings > threshold*, with the plan's `compress_ratio = 0.7`
meaning "compress if at least 30% savings". To avoid confusion in code,
we will name the field `min_savings_ratio` internally with the
relationship `n_compressed / n <= 1.0 - min_savings_ratio`. The public
`ScotchOptions` field name stays `compress_ratio` per the plan, with a
docstring spelling out the convention.

(*Resolved at API time. For S1 itself the function takes a single
`max_compressed_ratio: f64` argument with explicit semantics, decoupled
from the public-facing field name.*)

## Hash design

We use Rust's `std::hash::DefaultHasher` (SipHash-1-3) seeded by hashing
the closed neighbor list in sorted order (`SortedNbrs`). This is:

- **Deterministic.** SipHash with a fixed empty key gives the same hash
  on every run on every platform — needed for reproducible orderings
  across reruns. (The `RandomState` used by `HashMap` is not
  deterministic; we do **not** use it. We use `BuildHasherDefault<…>`.)
- **Cheap.** O(deg(v)) per vertex; total O(|E|) across the graph.
- **Strong enough.** Cryptographic strength is not required; we do an
  exact verify pass anyway. We just need few enough collisions that the
  verify pass stays linear-amortized.

## Complexity

- Hash pass: `O(Σ deg(v)) = O(|E|)`.
- Verify pass: `O(|E|)` amortized assuming `O(1)` average bucket size.
  Worst case (pathological hashing) is `O(|E|² / |V|)` but does not
  arise with SipHash on real inputs.
- Build pass: `O(|E|)` to walk edges, plus `O(|E_compressed|)` to dedup
  parallel edges. We dedup by sorting each adjacency list and merging
  runs — `O(deg log deg)` per compressed vertex.
- Total: `O(|E| + |E_compressed| log d_max)`.
  Memory: `O(n + |E|)`.

## Test plan

Five tests, all standalone (no dependency on partitioning code yet):

1. **4×4 grid.** 16 vertices, but four corners (degree 2) and the eight
   edges (degree 3) of the boundary fall into degree-defined classes. In
   the grid, no two distinct vertices are *truly* indistinguishable
   because their adjacency sets differ even when their degree matches
   (e.g. corner (0,0) has neighbors {(0,1),(1,0)}; corner (0,3) has
   neighbors {(0,2),(1,3)} — these are not equal). So **the 4×4 grid
   actually compresses to itself** (`vertex_map[i] == [i]`, ratio = 1.0,
   `compress_graph` returns `None` at min_ratio = 0.7). This is the
   *negative* test — confirms the algorithm does not over-merge.

2. **Block diagonal of three identical 4×4 dense blocks.** 12 vertices.
   Within each block all four vertices have neighbors `{the other
   three}`, so all four are pairwise indistinguishable. Each block
   compresses to 1 supervariable; total 3 supervariables. Ratio 0.25,
   well under 0.7 → `Some(compressed)`. Verify `vwgt_c == [4, 4, 4]` and
   no edges in the compressed graph (the three blocks are disconnected
   and intra-block edges collapse to self-loops which we drop).

3. **Pure dense matrix (`K_n` for `n = 6`).** All vertices have neighbors
   = "everyone else" so all six are indistinguishable. Compresses to one
   supervariable of weight 6 with no edges. Ratio 1/6 ≈ 0.17 < 0.7 →
   `Some`. Confirms the algorithm correctly handles the all-merge case.

4. **Path graph (`P_n` for `n = 8`).** Each interior vertex has unique
   neighbors `{prev, next}`. Endpoints have unique singleton neighbors.
   No two are indistinguishable. Returns `None`.

5. **Bijection of expansion.** Run on test 2 (block-diagonal). Take the
   `vertex_map`, flatten into a single permutation by visiting
   compressed vertices in order, and assert it is a permutation of
   `0..12`. (This is the smoke test that a downstream
   "expand_permutation" can recover a full ordering.)

## Out of scope for S1

- The driver's compression-cadence policy (single-shot vs. per-level
  with short-circuit) lives in S5 (`node_nd.rs`).
- Vertex-weighted AMD base case (audit finding 9) is for the S5 leaf
  path, not for compression.
- PT-SCOTCH's distributed compression is Phase 4, not this plan.

## Open questions

None blocking. Logged here for context:

- Should the hash include the *vertex's own weight* in the bucket key?
  Pellegrini 1996 §3 does not specify because input weights are unit;
  we'll group by `(closed_neighbor_list, vwgt)` so two unit-weight
  vertices that share neighbors merge, but a vertex with non-unit input
  weight (e.g., from an outer compression already applied) does not
  merge with a unit-weight vertex unless their weights also match. This
  is conservative and correct. Resolved.

- Tie-breaking: when expanding the permutation later, in what order do
  we list the original vertices of a supervariable? S5 will decide
  (probably "input order" for determinism). S1 just preserves input
  order in `vertex_map[c]`.
