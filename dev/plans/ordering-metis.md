# Ordering Plan: METIS Nested Dissection

**Status:** Pre-implementation plan — audited 2026-04-16
**Date:** 2026-04-16
**Research note:** `.crucible/wiki/concepts/metis-graph-partitioning.org`
**Paper references:** `.crucible/wiki/summaries/karypis1998-metis.org`,
`.crucible/wiki/summaries/george1973-nested-dissection.org`
**Code reference:** METIS 5.2.0 `libmetis/` (Apache 2.0); MUMPS METIS
dispatch in `mumps/src/dana_aux.F:264-933` and
`mumps/src/ana_set_ordering.F:25,52-106`.
**Related:** `dev/plans/phase-2-planning.md` §2.6

---

## Audit Findings (mumps-expert, 2026-04-16)

The plan's module decomposition matches METIS 5 (`coarsen.c`, `fm.c`, `sfm.c`,
`separator.c`, `ometis.c`), but several decisions MUMPS depends on are missing
or misstated. Items below must be incorporated before coding.

1. **Connected-component handling is missing.** Real KKT matrices with
   bound-only constraints have isolated dual rows. METIS handles this with
   `MlevelNestedDissectionCC` (`METIS_OPTION_CCORDER`); FERAL must (a) detect
   components, (b) recurse on each, (c) concatenate permutations. Without
   this the recursion never bottoms out cleanly on disconnected graphs.

2. **Initial partition is wrong.** Plan says "GGP, METIS uses 1 trial." Actual
   METIS 5.2.0 default is `niparts = 7`; runs both GGP and "growing bisection
   from random seed", and **scores on post-FM cuts** (not raw GGP cuts). Each
   trial gets a `Balance2Way + FM_2WayCutRefine` pass before scoring.

3. **2-hop matching fallback is needed in v1**, not deferred. Power-law-degree
   KKT graphs (high-degree dual rows) cause SHEM's reduction ratio to fall
   below threshold; without a fallback, coarsening either stops too coarse or
   loops. Cite `Match_2Hop` in `libmetis/coarsen.c`.

4. **Promote v2 (Hopcroft-Karp min-vertex-cover) to v1.** "Lighter-side
   boundary" gives 1.5–2× larger top-level separators on KKT, multiplying
   root-frontal BLAS-3 work by ~4× and quadratically inflating peak frontal
   storage. Hopcroft-Karp on the boundary subgraph is `O(√n_boundary · |E|)`,
   cheap (boundary is `O(√n)` on 2D, `O(n^{2/3})` on 3D). Match METIS's
   `ConstructMinCoverSeparator` in `libmetis/separator.c`.

5. **Node-separator FM has two variants in METIS**: `FM_2WayNodeRefine2Sided`
   for upper levels, `FM_2WayNodeRefine1Sided` for the finest level
   (`libmetis/sfm.c`). Plan only describes two-sided. One-sided is
   significantly cheaper at fine levels; defer if needed but document.

6. **FM balance handling is too naive in plan.** METIS's FM tracks "best cut
   subject to balance" plus "current cut" separately. Treating balance as a
   hard wall at each move noticeably worsens cuts. Pass termination must use
   METIS's "no improvement in 50 moves" rule, not just "until no improvement."

7. **`MMD_SWITCH = 120` conflates two thresholds.** METIS uses 120 for the
   *coarsened* graph base case in initial bisection, **200** for the
   *uncoarsened* recursive ND→AMD switch. Expose both as separate parameters.

8. **Edge-weight summation on contraction must drop self-loops** (parallel
   edges between matched endpoints). Plan was silent on this.

9. **Heavy-edge matching tie-breaking and traversal order**: METIS's SHEM
   visits unmatched vertices in random order seeded by `METIS_OPTION_SEED`,
   picks max-`adjwgt` neighbor, ties broken by lower vertex id. Plan's
   "bucket sort by sqrt(degree)" is wrong — that's the *vertex-visit order*
   for SHEM, not bucket sort. Affects reproducibility against METIS.

10. **RNG / determinism**: don't try to match METIS byte-for-byte. Use a
    deterministic PRNG (e.g., `rand_chacha`) seeded from `SupernodeParams`,
    fixed-seed in inertia parity tests.

11. **MUMPS preprocessing the plan should be aware of (not necessarily
    replicate v1)**:
    - Schur compression (`KEEP(60)≠0`): listed Schur variables removed from
      the graph before METIS, reattached after (`dana_aux.F:264-292`).
    - LDLᵀ pair compression (MC64-driven, SYM=2): pairs of variables merged
      with weight 2; METIS run on compressed graph
      (`dana_aux.F:378-391,817-823`). For KKT matrices this is significant.
    - Vertex-weighted entry point: MUMPS uses `METIS_NodeWND` (with
      `FRERE(:)` weights), not `METIS_NodeND`, when compressed pairs exist.
      FERAL's `metis_node_nd` should accept optional vertex weights from
      the start (or plan a clean extension point).

12. **MUMPS's METIS configuration is otherwise minimal**: only sets
    `METIS_OPTION_NUMBERING = 1`. No override of `ufactor`, `niparts`,
    `nseps`, `pfactor`, `ccorder`, `compress`, `seed`, `dbglvl`. Matching
    MUMPS-quality output means matching METIS's *defaults*, listed in §3
    of the audit report.

13. **Export the implicit etree.** ND naturally produces parent pointers via
    the recursion structure (separator vertex's parent in elimination tree
    is the next-up separator vertex). MUMPS reconstructs this if not given;
    FERAL should expose it from `metis_node_nd` to skip a symbolic pass in
    `src/symbolic/mod.rs`.

---

## Goal

Implement a pure-Rust multilevel nested dissection ordering in
`src/ordering/metis/` that produces fill-reducing permutations via
recursive graph bisection. The implementation must:

- Produce fill quality competitive with C METIS on KKT matrices
  (target: within 10% of METIS 5.2.0 fill on the POUNCE corpus)
- Run in O(|E| log n) time
- Output `Vec<usize>` (new-to-old permutation), identical interface
  to `amd_order`, plugging into the existing symbolic pipeline
- Be a clean-room implementation from published papers (MIT-safe)

## Motivation

AMD is a local heuristic that struggles on KKT matrices — irregular
coupling between primal and dual blocks defeats minimum-degree
selection. METIS's global nested dissection finds separators that
respect the block structure, producing 2-5× less fill on these
matrices. For FERAL's target workload (Ipopt KKT systems), METIS
ordering is the highest-priority upgrade.

Theoretical gap on 3D problems: AMD produces O(N^{5/3}) fill vs
METIS's O(N^{4/3}). For N=100k this is ~4× more fill entries.

## Directory Structure

```
src/ordering/
├── mod.rs                    # add `pub mod metis;`
├── amd.rs                    # UNCHANGED (or amd_quotient.rs)
├── metis/
│   ├── mod.rs                # public API: metis_node_nd()
│   ├── graph.rs              # CSR graph representation (~150 lines)
│   ├── coarsen.rs            # heavy-edge matching + contraction (~300 lines)
│   ├── initial_partition.rs  # GGP + random bisection (~150 lines)
│   ├── fm_refine.rs          # Fiduccia-Mattheyses refinement (~400 lines)
│   ├── separator.rs          # edge bisection → node separator (~250 lines)
│   └── node_nd.rs            # recursive nested dissection driver (~200 lines)
├── elimination_tree.rs       # UNCHANGED
└── postorder.rs              # UNCHANGED
```

Estimated total: ~1800-2500 lines of Rust.

## Design

### Graph Representation (`graph.rs`)

```rust
/// CSR graph for partitioning algorithms.
/// Mirrors METIS's graph_t but uses Rust Vec types.
pub(crate) struct Graph {
    /// Number of vertices.
    pub nvtxs: usize,
    /// Adjacency offsets (length nvtxs + 1).
    pub xadj: Vec<usize>,
    /// Neighbor lists (length = 2 * |E|).
    pub adjncy: Vec<usize>,
    /// Vertex weights (length nvtxs). Default: all 1.
    pub vwgt: Vec<i32>,
    /// Edge weights (length = 2 * |E|). Default: all 1.
    pub adjwgt: Vec<i32>,
}

impl Graph {
    /// Build from a symmetric CscPattern. O(nnz).
    pub fn from_csc_pattern(pattern: &CscPattern) -> Self;

    /// Number of edges (undirected).
    pub fn nedges(&self) -> usize { self.adjncy.len() / 2 }

    /// Degree of vertex v.
    pub fn degree(&self, v: usize) -> usize {
        self.xadj[v + 1] - self.xadj[v]
    }
}
```

Conversion from `CscPattern` (which stores the full symmetric pattern)
is a direct copy: CSC col_ptr → CSR xadj, CSC row_idx → CSR adjncy.
The layouts are identical for symmetric matrices.

### Coarsening (`coarsen.rs`)

```rust
/// Result of one coarsening level.
pub(crate) struct CoarseGraph {
    pub graph: Graph,
    /// Maps fine vertex → coarse vertex.
    pub cmap: Vec<usize>,
    /// Number of coarse vertices.
    pub cnvtxs: usize,
}

/// Coarsen the graph by heavy-edge matching until small enough.
///
/// Returns a stack of coarsening levels (finest first, coarsest last).
/// Stops when nvtxs < coarsen_to or reduction ratio < 0.75.
pub(crate) fn coarsen_graph(
    graph: &Graph,
    coarsen_to: usize,
    match_type: MatchType,
) -> Vec<CoarseGraph>;

pub(crate) enum MatchType {
    /// Random matching (fast, lower quality).
    Random,
    /// Sorted heavy-edge matching (default, better quality).
    SortedHeavyEdge,
}
```

**Heavy-Edge Matching algorithm (matches METIS 5 SHEM):**
1. Visit unmatched vertices in **random permutation order** seeded by
   `params.seed` (use `rand_chacha` for determinism). Inside SHEM,
   re-sort the visit order by ascending degree.
2. For each unmatched vertex v, find its unmatched neighbor u with
   maximum `adjwgt(v, u)`; ties broken by **lower vertex id**.
3. Match (v, u) → merge into one coarse vertex
4. Coarse vertex weight = `vwgt(v) + vwgt(u)`
5. Coarse edges = union of v's and u's edges; parallel edges to the
   same coarse neighbor are **summed**; self-loops (edges between v and
   u themselves) are **dropped**.

**Implementation notes:**
- Matching uses a `Vec<Option<usize>>` match array
- **2-hop matching is required in v1**, not optional. When SHEM's
  reduction ratio falls below threshold (default 0.85), fall through to
  `Match_2Hop`-style "match isolated vertices to lightest 2-hop neighbor"
  (METIS `libmetis/coarsen.c`). Without this, KKT graphs with high-degree
  dual rows either stop coarsening too coarse or loop.
- Each level is O(|V| + |E|); O(log n) levels total.
- RNG: deterministic per `params.seed`, default fixed value for parity tests.

### Initial Partition (`initial_partition.rs`)

```rust
/// Bisect a small graph (< coarsen_to vertices).
///
/// Tries multiple random starts and keeps the best cut.
/// Returns partition labels: where[v] ∈ {0, 1}.
pub(crate) fn initial_bisect(
    graph: &Graph,
    n_trials: usize,
) -> Vec<u8>;
```

**Initial partition strategy (matches METIS 5 `Init2WayPartition`):**

Run `niparts` trials, alternating two methods (each followed by
`Balance2Way + FM_2WayCutRefine` before scoring):

1. **GGP (Greedy Graph Growing):**
   a. Pick a random seed vertex, assign to partition 0
   b. Greedily grow partition 0 by adding the boundary vertex that
      minimizes edge cut increase
   c. Stop when partition 0 has ~n/2 vertices
   d. Remaining vertices go to partition 1

2. **Random bisection from seed:**
   a. BFS from a random seed vertex, BFS-order-assign to partition 0
      until ~n/2 reached.

**Score on POST-FM cuts**, not raw GGP cuts. Keep the best.

Default `niparts = 7` (METIS 5.2.0 default, NOT 1 as the plan originally
claimed). FERAL exposes `params.niparts: usize` (default 7).

### FM Refinement (`fm_refine.rs`)

```rust
/// Fiduccia-Mattheyses refinement for edge bisection.
///
/// Refines a bisection of `graph` (labels in `where_`) to reduce
/// edge cut while maintaining balance within `max_imbalance`.
/// Modifies `where_` in place.
pub(crate) fn fm_refine_bisection(
    graph: &Graph,
    where_: &mut [u8],
    max_imbalance: f64,
    n_iter: usize,
);

/// FM refinement for node separator.
///
/// Refines a separator (where[v] ∈ {0, 1, 2=separator}).
/// Minimizes separator weight while maintaining partition balance.
pub(crate) fn fm_refine_separator(
    graph: &Graph,
    where_: &mut [u8],
    max_imbalance: f64,
    n_iter: usize,
);
```

**Edge-bisection FM algorithm (matches METIS `FM_2WayCutRefine`):**
1. Identify boundary vertices (neighbors in other partition)
2. For each boundary vertex v, compute gain:
   `gain(v) = edges_to_other[v] - edges_to_own[v]`
3. Insert into priority queue (bucket by gain; bucket width =
   `2 * max_adjwgt`)
4. Repeatedly move highest-gain vertex, updating neighbors' gains
5. Track **two state variables separately**: "current cut" and
   "best cut SUBJECT TO BALANCE constraint." Only the latter is the
   rollback target.
6. **Pass termination**: terminate when (a) all boundary vertices have
   been moved once, OR (b) `mincutorder - currentmove > 50`
   (no improvement in 50 moves) — METIS's exact rule.
7. Roll back to best balanced state.
8. Repeat for `n_iter` passes or until no improvement.

**Node-separator FM algorithm:**
- Two priority queues (one per side of separator)
- Moving separator vertex v to side `to`:
  - Removes v from separator (saves vwgt[v])
  - Pulls v's neighbors on the other side into separator (costs their weight)
- Gain: `vwgt[v] - sum(vwgt[u] for u in neighbors on other side not already in separator)`
- Two-sided variant: process both queues alternately

**Bucket priority queue:**
```rust
/// O(1) insert/delete/max priority queue for integer gains.
struct BucketPQ {
    buckets: Vec<Vec<usize>>,  // buckets[gain + offset] = list of vertices
    max_gain: isize,
    offset: usize,  // gain range is [-offset, max_possible]
}
```

### Node Separator (`separator.rs`)

```rust
/// Convert an edge bisection to a node separator.
///
/// Given where[v] ∈ {0, 1} (edge bisection), compute a vertex
/// separator S such that removing S disconnects parts 0 and 1.
/// Sets where[v] = 2 for separator vertices.
///
/// Uses minimum vertex cover of the bipartite boundary graph.
pub(crate) fn construct_separator(
    graph: &Graph,
    where_: &mut [u8],
);
```

**Algorithm (Hopcroft-Karp min vertex cover, promoted from v2 to v1):**
1. Identify boundary vertices: B = {v : ∃ neighbor u with where[u] ≠ where[v]}
2. Build bipartite graph of boundary vertices (edges crossing the cut)
3. Compute minimum vertex cover via maximum matching (König's theorem):
   - Run Hopcroft-Karp on bipartite boundary subgraph
   - Vertex cover = complement of maximum independent set
4. Mark cover vertices as separator (where[v] = 2)
5. Verify: removing separator disconnects parts 0 and 1

**Why not the v1 boundary heuristic:** "lighter-side boundary" produces
1.5–2× larger top-level separators on KKT, multiplying root-frontal
BLAS-3 work by ~4× and quadratically inflating peak frontal storage.
Hopcroft-Karp on the boundary subgraph (size `O(√n)` on 2D, `O(n^{2/3})`
on 3D) is cheap. Match METIS `ConstructMinCoverSeparator` in
`libmetis/separator.c`.

### Nested Dissection Driver (`node_nd.rs`)

```rust
/// Compute a fill-reducing nested dissection ordering.
///
/// Returns a permutation vector `perm` (new-to-old mapping),
/// identical interface to `amd_order`.
pub fn metis_node_nd(pattern: &CscPattern) -> Vec<usize>;
```

**Algorithm:**
```
function node_nd(graph, perm, offset, count):
    // Connected-components split (METIS MlevelNestedDissectionCC)
    components = find_connected_components(graph)
    if components.len() > 1:
        for comp in components:
            node_nd(comp.subgraph, perm, comp.offset, comp.count)
        return

    // Two distinct base-case thresholds:
    //   nd_to_amd_switch  (default 200): switch ND→AMD on uncoarsened recursion
    //   coarsen_floor     (default 120): stop coarsening at this size
    if graph.nvtxs < params.nd_to_amd_switch:
        order with amd_order, write into perm[offset..]
        return

    // Find node separator
    sep, left, right = multilevel_node_bisection(graph, params.niparts)

    // Number separator last (highest positions)
    for v in sep:
        perm[label[v]] = offset + count - 1 - sep_pos
        sep_pos += 1
        // Record etree parent: separator vertices' parent is the next
        // separator up the recursion (export this for symbolic pipeline)

    // Build subgraphs and recurse
    graph_left  = extract_subgraph(graph, left)
    graph_right = extract_subgraph(graph, right)
    node_nd(graph_left,  perm, offset, left.len())
    node_nd(graph_right, perm, offset + left.len(), right.len())
```

**Public API additions** to support MUMPS-equivalent preprocessing later:
```rust
pub struct MetisOutput {
    pub perm: Vec<usize>,
    /// Optional: parent[i] in the elimination tree induced by ND.
    /// Skipping a redundant symbolic pass downstream.
    pub etree_parent: Option<Vec<usize>>,
}
pub fn metis_node_nd(pattern: &CscPattern) -> MetisOutput;
pub fn metis_node_wnd(pattern: &CscPattern, vwgt: &[i32]) -> MetisOutput;
```
The vertex-weighted entry point lets a future MC64-driven pair compression
in `src/symbolic/` pass merged-weight inputs (analogous to MUMPS's
`METIS_NodeWND` call when `KEEP(95) ≥ 2`).

**Multilevel node bisection:**
1. Coarsen graph → hierarchy
2. Initial bisection at coarsest level
3. Uncoarsen with FM edge-bisection refinement at each level
4. Convert final edge bisection to node separator
5. Refine with node-separator FM

### Public API

```rust
// In src/ordering/metis/mod.rs:
pub fn metis_node_nd(pattern: &CscPattern) -> Vec<usize>
```

### Integration Point

In `src/symbolic/mod.rs`, the ordering call becomes selectable:

```rust
pub enum OrderingMethod {
    Amd,
    MetisND,
    // Custom(fn(&CscPattern) -> Vec<usize>),
}

// In symbolic_factorize:
let ordering_perm = match snode_params.ordering {
    OrderingMethod::Amd => amd_order(&full_pattern),
    OrderingMethod::MetisND => metis::metis_node_nd(&full_pattern),
};
```

This requires adding an `ordering: OrderingMethod` field to
`SupernodeParams` (default: `Amd`). The rest of the pipeline
(permute_pattern → etree → postorder → column_counts → supernodes)
is unchanged.

## Implementation Steps

### Phase M1: Graph infrastructure (~200 lines)

- `Graph` struct with CSR storage
- `Graph::from_csc_pattern()` conversion
- Unit tests: round-trip CSC → Graph → verify adjacency

### Phase M2: Coarsening (~350 lines)

- `MatchType::Random` matching
- `MatchType::SortedHeavyEdge` matching
- `coarsen_graph()` loop with stopping criteria
- Tests: verify coarse graph has ~half the vertices, edge weights
  are preserved, cmap is valid

### Phase M3: Initial partition (~150 lines)

- GGP (Greedy Graph Growing) bisection
- Multiple trials with best-cut selection
- Tests: partition is balanced (within 20%), cut is non-negative

### Phase M4: FM refinement — edge bisection (~300 lines)

- `BucketPQ` data structure
- `fm_refine_bisection()` with roll-back
- Boundary-restricted candidate selection
- Tests: refined cut ≤ initial cut, balance maintained

### Phase M5: FM refinement — node separator (~200 lines)

- `fm_refine_separator()` with two-sided moves
- Priority queue for separator vertices
- Tests: separator is valid (removal disconnects), weight ≤ initial

### Phase M6: Node separator construction (~200 lines)

- Simple boundary-based separator (v1: mark lighter-side boundary)
- Minimum vertex cover via augmenting paths (v2, optional)
- Tests: separator validity check

### Phase M7: Nested dissection driver (~200 lines)

- Recursive `node_nd()` with MMD base case
- Subgraph extraction (relabel vertices, map edges)
- Permutation assembly
- Tests: permutation is valid bijection

### Phase M8: Integration (~50 lines)

- Add `OrderingMethod` enum to `SupernodeParams`
- Wire into `symbolic_factorize`
- Default: `Amd` (no behavior change)

## Testing Plan

### Unit Tests (per module)

**T1. Graph construction** — Verify CSC → CSR conversion:
- Arrow(5): degree sequence matches
- Tridiagonal(10): each vertex has degree 1 or 2
- Dense(5): each vertex has degree 4
- Empty/diagonal: no edges

**T2. Coarsening correctness** — For each coarsening level:
- `cmap` maps every fine vertex to a valid coarse vertex
- Coarse graph has ≤ fine.nvtxs / 2 + 1 vertices
- Total coarse vertex weight = total fine vertex weight
- No self-loops in coarse graph

**T3. Initial partition balance** — GGP on known graphs:
- Path graph (n=100): partition within ±5 of balanced
- Complete graph (n=20): any bisection is balanced
- Verify `where_[v] ∈ {0, 1}` for all v

**T4. FM refinement quality** — On 2D grid (7×7):
- Refined cut < initial random cut (with high probability)
- Balance constraint maintained after refinement
- Verify edge cut computation is consistent:
  `cut = Σ adjwgt[e] for e crossing partition`

**T5. Separator validity** — After node separator construction:
- Removing separator vertices disconnects parts 0 and 1
  (BFS from any part-0 vertex should not reach part-1 vertices)
- All separator vertices have `where_[v] = 2`
- Separator weight is a meaningful fraction of total
  (not trivially all vertices)

**T6. Permutation validity** — `metis_node_nd` output:
- `perm.len() == n`
- `perm` is a bijection
- Separator vertices have higher positions than their subgraph
  vertices (nested dissection property)

**T7. Fill quality** — Compare against `amd_order`:
- Arrow(n): METIS fill ≈ 0 (separator is the hub)
- 2D grid (k×k): METIS fill ≤ AMD fill
- Random sparse: METIS fill ≤ 1.2 × AMD fill (allow some tolerance)

### Integration Tests

**T8. Full pipeline correctness** — For `tests/data/parity/` matrices:
- `symbolic_factorize` with `OrderingMethod::MetisND`
- Verify inertia matches MUMPS oracle
- Verify residuals within tolerance
- Compare factor NNZ vs AMD ordering

**T9. KKT-specific fill comparison** — For each parity matrix:
- `fill_metis = nnz(L) with METIS ordering`
- `fill_amd = nnz(L) with AMD ordering`
- Report `fill_metis / fill_amd` distribution
- **Expected:** METIS ≤ AMD on ≥ 60% of KKT matrices

### Benchmark Tests

**B1. Ordering time** — `metis_node_nd` vs `amd_order`:
- Synthetic: n ∈ {100, 500, 1000, 5000, 10000}
- Report: ordering time for both methods
- **Expected:** METIS 2-5× slower than AMD on small (n<500);
  competitive on large (n>5000) due to O(|E| log n) vs AMD's
  O(nnz) (but with better constant for complex structures)

**B2. Fill quality on KKT corpus** — Full parity panel:
- For each matrix: nnz(L) with AMD, nnz(L) with METIS
- Report: geometric mean fill ratio (METIS/AMD)
- **Gate:** geomean ≤ 0.95 (METIS should produce ≥ 5% less fill
  on average across KKT matrices)

**B3. End-to-end factorization** — `cargo run --bin bench --release`:
- Run with both orderings
- Compare: total solve time (ordering + symbolic + numeric + solve)
- **Gate:** METIS total time ≤ AMD total time on matrices where
  METIS fill < 0.8 × AMD fill (the fill savings should outweigh
  ordering overhead on fill-dominated problems)

**B4. Comparison vs C METIS** — For matrices where rmumps sidecar
has MUMPS-with-METIS data, and for direct comparison against the
`metis` Rust crate (FFI to C METIS) used as a ground-truth oracle:
- Compare FERAL METIS fill vs MUMPS METIS fill
- **Gate:** geometric-mean fill within **1.10×** of C METIS on the
  Schenk_IBMNA + GHS_indef set (per audit recommendation)
- **Per-matrix gate:** within 20% on ≥ 80% of matrices

**Concrete oracle matrices for KKT validation** (from audit):
- `Schenk_IBMNA/c-big`, `c-71`, `c-72`, `c-73` — IPM KKT, ~150k–500k
- `GHS_indef/ncvxqp1`, `ncvxqp3`, `ncvxqp5` — non-convex QP KKT
- `GHS_indef/cont-300`, `cont-201` — smooth QP KKT
- `Schenk_AFE/af_shell3`, `af_shell7`, `af_shell8` — saddle-point
- `GHS_indef/stokes128`, `qpband`, `copter2` — varied saddle-point
- DIMACS10 `delaunay_n14`...`delaunay_n20` — 2D ND validation
- DIMACS10 `333SP`, `AS365`, `M6` — 3D / road network reference

**B5. Scaling test** — Synthetic 2D grids at n = {100, 1k, 10k, 50k}:
- Verify O(|E| log n) scaling
- Plot ordering time vs |E|

## Open-Source Reference Implementations

**Primary reference — METIS 5.2.0 (C, Apache 2.0):**
- Repository: https://github.com/KarypisLab/METIS
- Key files for nested dissection:
  - `libmetis/ometis.c` — `METIS_NodeND` driver, `MlevelNestedDissection`
  - `libmetis/coarsen.c` — `CoarsenGraph`, `Match_RM`, `Match_SHEM`
  - `libmetis/separator.c` — `ConstructSeparator`, `ConstructMinCoverSeparator`
  - `libmetis/sfm.c` — `FM_2WayNodeRefine2Sided` (node-separator FM)
  - `libmetis/fm.c` — `FM_2WayRefine` (edge-bisection FM)
  - `libmetis/initpart.c` — `Init2WayPartition`, GGP
  - `libmetis/minconn.c` — minimum connectivity utilities
  - `include/metis.h` — public API and option constants
- License changed to Apache 2.0 in METIS v5.2.0 (2023)

**KaHIP (C++, MIT) — higher-quality partitioning:**
- Repository: https://github.com/KaHIP/KaHIP
- Relevant: `lib/partition/uncoarsening/refinement/kway_graph_refinement/`
  for flow-based refinement; `app/node_separator.cpp` for node separator
- Data reduction rules: https://arxiv.org/abs/2004.11315
  (6× faster than METIS with less fill on road networks)
- KaHIP produces measurably better separators than METIS; useful as a
  quality oracle for validating our implementation

**ParMETIS (C, Apache 2.0) — distributed memory reference:**
- Repository: https://github.com/KarypisLab/ParMETIS
- Relevant for Phase 4 only

**mt-METIS (C, BSD-2-Clause) — shared-memory parallel:**
- Repository: https://github.com/dlasalle/mt-metis
- Reference for threading the multilevel pipeline

**Rust crates (for API/idiom reference):**
- `metis` crate (FFI bindings): https://crates.io/crates/metis
  — shows Rust-idiomatic API wrapping C METIS; our clean-room
  implementation should provide a similar interface
- faer-rs: https://github.com/sarah-quinones/faer-rs
  — uses `metis` crate in `faer/src/sparse/linalg/cholesky.rs`
  via conditional compilation

## Verification Checklist

- [ ] All existing `cargo test` pass (no regressions from integration)
- [ ] `tests/parity.rs` — zero inertia regressions with METIS ordering
- [ ] Permutation is a valid bijection for all test matrices
- [ ] Separator validity: removal disconnects subgraphs (BFS check)
- [ ] Disconnected graph handling: per-component recursion correct
- [ ] 2-hop matching fallback engages when SHEM ratio < 0.85
- [ ] Edge-bisection FM: balance constraint via "best-balanced cut" rollback
- [ ] FM pass termination uses METIS's "no improvement in 50 moves" rule
- [ ] Hopcroft-Karp min vertex cover separator (not lighter-side boundary)
- [ ] `niparts = 7` default (matches METIS 5.2.0); GGP + random both run;
      scoring on post-FM cuts
- [ ] Both `coarsen_floor=120` and `nd_to_amd_switch=200` thresholds exposed
- [ ] Self-loops dropped on contraction; parallel edges summed
- [ ] Deterministic RNG (`rand_chacha`) with explicit seed parameter
- [ ] Fill quality: METIS ≤ AMD on ≥ 60% of KKT parity matrices
- [ ] Geometric-mean fill within 1.10× of C METIS on Schenk_IBMNA + GHS_indef
- [ ] `cargo run --bin bench --release` — works with both orderings
- [ ] No `unwrap()` or `expect()` in `src/ordering/metis/`
- [ ] `cargo clippy -- -D warnings` clean
- [ ] Default ordering remains AMD (backward compatible)
- [ ] `MetisOutput.etree_parent` exported and consumed by `src/symbolic/`
