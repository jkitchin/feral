# Ordering Plan: KaHIP Flow-Based Nested Dissection

**Status:** Pre-implementation plan — audited 2026-04-16
**Date:** 2026-04-16
**Research note:** `.crucible/wiki/concepts/kahip-graph-partitioning.org`
**Paper references:** `.crucible/wiki/summaries/sanders2011-kaffpa.org`,
`.crucible/wiki/summaries/ost2021-data-reduction-nd.org`
**Code reference:** KaHIP v3.16 (C++, MIT license — clean-room story is
straightforward but implement from papers, not source). SPRAL SSIDS is a
"consumer" of orderings — not a producer-side analogue, but its symbolic
amalgamation in `core_analyse.f90:536-642` is the closest in-tree mirror
to twin-style merging.
**Related:** `dev/plans/ordering-metis.md` (shared infrastructure)

---

## Audit Findings (spral-expert, 2026-04-16)

The plan is reasonable scaffolding. Three concrete bugs, one missing
design, and several over-claims must be addressed before K1 starts.

### Bugs to fix

1. **Degree-2 path compression is missing the simplicial sub-case (BUG).**
   Per Ost-Schulz-Strash 2021, Rule 2 has two cases: (a) endpoints u, w
   non-adjacent → compress with one fill edge (u, w); (b) endpoints
   **are** adjacent → v is simplicial and can be eliminated with **zero**
   fill. Plan only mentions (a). Case (b) is the more valuable one and
   is strictly better than what Rule 4 catches generically.

2. **Source/sink construction in flow refinement is ambiguous (BUG).**
   Plan says "Source nodes = band vertices at max distance in part 0."
   Correct wording: "vertices at distance EXACTLY `bnd_distance` from
   the boundary, in part 0, connected to a super-source with infinite
   capacity to fix them on the source side." The plan's wording can be
   misread as "the deepest part-0 vertex inside the band," which is wrong.

3. **Most Balanced Min Cut technique is missing (quality bug).** Plan
   rejects unbalanced improvements rather than searching for a balanced
   equivalent. KaHIP enumerates min-cuts by manipulating residual flow
   to find one satisfying balance (Sanders-Schulz 2011). Without this,
   significant quality is left on the table. Make it a v1 concern.

### Missing designs

4. **Expansion permutation across multiple reduction passes is
   undesigned.** After 3-5 fixed-point passes you have nested merges
   (a twin pair merged in pass 1 might be on a compressed path in pass
   2). The data structure must be a **stack of operations** popped in
   reverse during expansion. Plan's `Vec<Vec<usize>>` flattens this and
   loses the order needed for paths and cascading degree-1 removals.
   Design before coding.

5. **Closed twins.** Plan's "sorted neighbor list" twin detection only
   catches **open** twins (`N(v) = N(u)`, v∼u not adjacent). **Closed**
   twins (`N[v] = N[u]`, v∼u adjacent) are common in KKT diagonal blocks
   and need a separate comparison `N(v) ∪ {v}` vs `N(u) ∪ {u}`.

6. **Cascading degree-1 removals must record order.** Removing v can drop
   u to degree 1; the removal *order* matters for permutation
   reconstruction.

7. **Rule 4 (neighborhood subset) cost.** Naive check is
   `O(Σ deg(v)·deg(u))` over neighbor pairs, not `O(|E|)`. KaHIP uses a
   mark-array trick (`lib/node_ordering/`): mark all of N(u), then for
   each neighbor v of u, count marked neighbors; if count == deg(v) then
   N(v) ⊆ N(u). Plan should reference this.

### Push-relabel essentials

8. **Gap relabeling is essential (v1 gate), not optional.** On Strong-mode
   bands (d=5, possibly thousands of vertices) push-relabel without it
   is order-of-magnitude slower. KaHIP's `lib/algorithms/push_relabel.cpp`
   ships with it from day one. Make it part of K2, not a follow-up.

9. **Global relabeling (BFS from sink every O(V) ops to reset heights)**
   should at least be mentioned as a known optimization for Strong mode.

### Other corrections

10. **Edge capacities are undirected.** Modeled as two anti-parallel
    directed edges each with the edge weight as capacity. Plan implies
    directed without saying so.

11. **Fast mode "~METIS speed/quality" is overstated.** Almost certainly
    1.5–2× slower than METIS in pure Rust because (a) METIS has decades
    of low-level optimization, (b) data reduction is non-trivial overhead
    even when it shrinks the graph. Reword the claim to "comparable
    quality at 1.5–2× METIS time."

12. **Fixed `bnd_distance` defaults (3, 5) are reasonable starting points
    but suboptimal vs. KaHIP production.** KaHIP uses adaptive: start at
    d=2, grow until band exceeds ~5% of |V| or improvement saturates.
    Make `bnd_distance` configurable; document fixed defaults as v1.

13. **F-cycle cost estimate.** "3-5×" is plausible *only* with KaHIP's
    "branching factor 2 only at top few levels" tweak. Without it, cost
    is 2^depth. Document the tweak.

14. **Cycle vs data reduction interaction.** Data reduction is a one-shot
    preprocessing on the input graph; do NOT re-run inside the cycle
    loop, or monotonicity breaks. State this explicitly.

15. **Imbalance accumulation across recursion levels.** Per-level constant
    `max_imbalance` plus deep recursion gives wildly imbalanced leaves.
    KaHIP tightens per-level imbalance with depth — global balance
    constraint. Add `imbalance_decay: f64` parameter.

16. **Determinism in push-relabel.** Pick lowest-index admissible neighbor
    explicitly; otherwise tests like T6 ("mode hierarchy: Strong ≤ Eco
    ≤ Fast") become hard to debug.

17. **5–20% fill claim is plausible only at the upper end on KKT.** Hogg-
    Scott 2013 reports 5–15% on indefinite KKT (smaller than road-network
    gains where Sanders-Schulz showed 15–30%). **Eco gate of 0.95 (5%)
    is defensible; 0.90 (10%) for Strong is at the optimistic end.**

18. **SSIDS is METIS-only at the partitioning level.** No flow-based code,
    no iterated cycling. The closest in-tree analogue to twin merging is
    SSIDS's symbolic supernode amalgamation in
    `core_analyse.f90:536-642`. Don't over-claim SSIDS provenance.

19. **LOC estimate.** 1500-2000 LOC is correct for **bug-free Eco**.
    Strong mode with most-balanced-min-cut and adaptive `bnd_distance`
    is closer to **2500-3000 LOC**.

20. **License caution.** KaHIP is MIT-licensed since v3.00 (early 2022).
    Earlier versions were dual-licensed; some research-code branches are
    GPL. The clean-room story is fine **provided** implementers actually
    work from the papers and consult the C++ only as a reference for
    behavior, not for constants/structure copying. Add a one-line note.

---

## Goal

Implement a pure-Rust KaHIP-style nested dissection ordering in
`src/ordering/kahip/` that uses max-flow-based refinement and data
reduction preprocessing to produce higher-quality fill-reducing
permutations than METIS. The implementation must:

- Produce fill quality measurably better than METIS on KKT matrices
  (target: ≥ 5% less fill than feral METIS on the POUNCE corpus)
- Offer three quality modes: Fast (~METIS speed), Eco (2-3x, better
  fill), Strong (5-10x, best fill)
- Output `Vec<usize>` permutation, identical interface to `amd_order`
  and `metis_node_nd`
- Be a clean-room implementation from published papers (MIT-safe)

## Motivation

METIS uses FM refinement — a greedy hill-climbing algorithm that moves
one vertex at a time. It finds good partitions but can get stuck in
local optima. KaHIP's flow-based refinement solves a local max-flow
problem to find the exact minimum cut in a region around the partition
boundary. This consistently produces 10-30% smaller separators,
translating to 5-20% less fill during factorization.

For FERAL's KKT matrices (factored repeatedly with the same sparsity
pattern during IPM iterations), the extra ordering time of KaHIP-Eco
is amortized over many factorizations, making the fill improvement
essentially free.

Additionally, KaHIP's data reduction rules (degree-1/2 elimination,
twin detection, neighborhood subset reduction) can shrink the graph
by 30-80% before any partitioning work begins, benefiting all modes.

## Directory Structure

```
src/ordering/
├── mod.rs                       # add `pub mod kahip;`
├── amd.rs                       # UNCHANGED
├── metis/                       # METIS implementation (shared infra)
│   ├── graph.rs                 # SHARED: CSR graph type
│   ├── coarsen.rs               # SHARED: heavy-edge matching
│   ├── initial_partition.rs     # SHARED: GGP bisection
│   ├── fm_refine.rs             # SHARED: FM + BucketPQ
│   ├── separator.rs             # METIS-specific separator
│   └── node_nd.rs               # METIS-specific driver
├── kahip/
│   ├── mod.rs                   # public API: kahip_node_nd()
│   ├── data_reduction.rs        # graph simplification rules (~250 lines)
│   ├── push_relabel.rs          # max-flow solver (~300 lines)
│   ├── flow_refine.rs           # flow-based partition refinement (~250 lines)
│   ├── flow_separator.rs        # flow-based node separator (~200 lines)
│   ├── cycle.rs                 # V-cycle / F-cycle controller (~150 lines)
│   └── node_nd.rs               # recursive ND driver with modes (~200 lines)
├── elimination_tree.rs          # UNCHANGED
└── postorder.rs                 # UNCHANGED
```

**Prerequisite:** The METIS implementation (`ordering-metis.md`) must
be completed first, as KaHIP reuses its graph representation,
coarsening, initial partition, and FM refinement modules.

Estimated KaHIP-specific code: ~1500-2000 lines of Rust.

## Design

### Data Reduction (`data_reduction.rs`)

```rust
/// Result of graph reduction.
///
/// IMPORTANT: `Vec<Vec<usize>>` is NOT enough to reconstruct the
/// permutation across nested merges (see audit finding #4). Use a
/// stack of operations popped in reverse during expansion.
pub(crate) struct ReducedGraph {
    pub graph: Graph,
    /// Stack of reduction operations, in application order.
    /// Expansion replays them in REVERSE.
    pub ops: Vec<ReductionOp>,
    /// Reduction ratio (reduced / original vertex count).
    pub ratio: f64,
}

pub(crate) enum ReductionOp {
    /// Removed degree-1 vertex `v` whose only neighbor was `owner`.
    /// In expansion: place v's ordering immediately before owner's.
    Degree1 { v: usize, owner: usize },
    /// Compressed path with ordered interior vertices `path` between
    /// endpoints `(u, w)`. Two sub-cases:
    ///   - simplicial == true: u-w were adjacent (zero fill case)
    ///   - simplicial == false: u-w not adjacent (one fill edge added)
    Degree2Path {
        u: usize,
        w: usize,
        path: Vec<usize>,
        simplicial: bool,
    },
    /// Open or closed twin merge.
    Twin { rep: usize, dup: usize, closed: bool },
    /// Neighborhood-subset elimination: v ⊆ u in adjacency.
    SubsetElim { v: usize, owner: usize },
}

/// Apply exhaustive data reduction rules to shrink the graph.
///
/// Rules are applied in a fixed-point loop until no further reduction.
/// Returns None if reduction ratio exceeds `max_ratio` (not worth it).
pub(crate) fn reduce_graph(
    graph: &Graph,
    max_ratio: f64,
) -> Option<ReducedGraph>;

/// Expand a permutation from the reduced graph to the original graph.
pub(crate) fn expand_permutation(
    reduced_perm: &[usize],
    vertex_map: &[Vec<usize>],
    n_original: usize,
) -> Vec<usize>;
```

**Four reduction rules applied per pass:**

1. **Degree-1 elimination:** Scan for vertices with degree 1. Remove
   them (they'll be ordered first with zero fill). Mark their single
   neighbor as the "owner" for expansion. **Record removal order** —
   cascading removals (removing v drops u to degree 1) require ordered
   replay during expansion.

2. **Degree-2 path compression (TWO sub-cases — was missing):**
   For interior path vertices (degree 2, neighbors p and q):
   - **Sub-case A (simplicial, p ∼ q already adjacent):** v can be
     eliminated with **zero fill**. Strictly better than what Rule 4
     catches generically.
   - **Sub-case B (p, q not adjacent):** compress with one fill edge
     (p, q). Standard path compression.
   Record path interior order keyed to the surviving edge for expansion.

3. **Twin detection (open AND closed — was incomplete):**
   - **Open twins**: `N(v) = N(u)`, v∼u not adjacent. Hash sorted
     neighbor lists; compare within buckets.
   - **Closed twins**: `N[v] = N[u]`, v∼u adjacent (where N[x] =
     N(x) ∪ {x}). Compare `N(v) ∪ {v}` vs `N(u) ∪ {u}`. **Common in
     KKT diagonal blocks.**
   Merge confirmed twins into a single supervariable with summed weight
   (and summed edge weights to neighbors). Record `closed` flag.

4. **Neighborhood subset reduction (mark-array trick — cost detail):**
   For each vertex u, mark all of N(u). For each neighbor v of u, count
   marked neighbors of v; if `count == deg(v)` then `N(v) ⊆ N(u)` and
   v can be eliminated early. Cost is `O(Σ deg(u)·avg_deg)`, NOT
   `O(|E|)` — naive plan was overly optimistic.

**Fixed-point loop:** Apply rules 1-4 in order. After each pass, check
if vertex count decreased. Repeat until stable. Typically 3-5 passes.

**Cycle interaction (was missing):** Data reduction is a one-shot
preprocessing on the input graph. Do **NOT** re-run inside the V/F-cycle
loop, or the monotonicity invariant breaks.

### Push-Relabel Max-Flow (`push_relabel.rs`)

```rust
/// Solve max-flow/min-cut on a directed capacitated graph.
///
/// Returns the max flow value and a partition of vertices into
/// source-side and sink-side (the min-cut).
pub(crate) fn push_relabel(
    n: usize,
    edges: &[(usize, usize, i64)],  // (from, to, capacity)
    source: usize,
    sink: usize,
) -> (i64, Vec<bool>);  // (flow_value, is_source_side[v])
```

**Implementation:** Standard highest-label push-relabel:
1. Initialize: set height[source] = n, push excess from source on
   all outgoing edges
2. While active vertices exist (excess > 0, not source/sink):
   a. Select highest-label active vertex v
   b. If any admissible edge (v,u) with height[v] = height[u] + 1
      and residual capacity > 0: push flow
   c. Else: relabel v (increase height to min neighbor height + 1)
3. After termination: BFS from source on residual graph gives min-cut

**Complexity:** O(V^2 · sqrt(E)) for general graphs. On the small
local subgraphs used in refinement (|V| ≈ 100-1000), this is fast.

**Gap relabeling optimization:** Track height distribution. When a
height h has no vertices, all vertices with height > h are unreachable
from sink — reassign to height n (disconnect from sink). This provides
a significant speedup in practice.

### Flow-Based Refinement (`flow_refine.rs`)

```rust
/// Improve a bisection using max-flow/min-cut on a local area.
///
/// Grows an area of radius `bnd_distance` around the partition
/// boundary, constructs a flow network, solves it, and applies
/// the min-cut if it improves the partition.
pub(crate) fn flow_refine_bisection(
    graph: &Graph,
    where_: &mut [u8],
    bnd_distance: usize,
    max_imbalance: f64,
);
```

**Algorithm:**
1. **BFS band extraction:** Starting from boundary vertices (those
   with neighbors in both partitions), BFS outward to distance
   `bnd_distance`. Collect all vertices in the band.

2. **Flow network construction (CORRECTED):** Within the band:
   - **Source-side fixed nodes** = vertices at distance EXACTLY
     `bnd_distance` from the boundary, in part 0. These connect to
     a **super-source with infinite capacity** to PIN them on the
     source side. (NOT "deepest part-0 vertex" — that wording was
     ambiguous and wrong.)
   - **Sink-side fixed nodes** = vertices at distance EXACTLY
     `bnd_distance` from the boundary, in part 1. Connected to a
     **super-sink with infinite capacity** to pin them.
   - **Internal band vertices** are free to be re-assigned by the cut.
   - Edge capacities = edge weights from the original graph.
     Modeled as **two anti-parallel directed edges** (graph is
     undirected); each carries the full weight as capacity.

3. **Solve:** Call `push_relabel` on the flow network. **Gap relabeling
   is required**, not optional. Global relabeling recommended for
   Strong mode.

4. **Apply (with Most Balanced Min Cut):** Multiple min-cuts often
   exist with equal value but different balances. Search for a
   balanced equivalent by manipulating residual flow (Sanders-Schulz
   2011) instead of rejecting unbalanced improvements outright. Apply
   the best balanced min-cut.

5. **Iterate:** Re-identify boundary, re-grow band, re-solve.
   Stop after 2-3 iterations or when no improvement.

### Flow-Based Node Separator (`flow_separator.rs`)

```rust
/// Compute a vertex separator using flow-based methods.
///
/// Given an edge bisection (where[v] ∈ {0, 1}), compute a vertex
/// separator S (where[v] = 2) using max-flow on the boundary
/// bipartite graph. Produces tighter separators than METIS's
/// minimum vertex cover approach.
pub(crate) fn flow_node_separator(
    graph: &Graph,
    where_: &mut [u8],
    max_imbalance: f64,
);
```

**Algorithm:**
1. Identify boundary bipartite graph: vertices in part 0 adjacent to
   part 1, and vice versa
2. Construct flow network: boundary-0 vertices on source side,
   boundary-1 on sink side, capacities = vertex weights
3. Solve max-flow: the min vertex cut gives the minimum-weight set of
   boundary vertices whose removal disconnects parts 0 and 1
4. Mark min-cut vertices as separator (where[v] = 2)
5. Optionally refine with localized flow improvement on the separator

This is tighter than METIS's approach (minimum vertex cover of
boundary bipartite graph) because max-flow finds the true minimum
vertex cut, while minimum vertex cover is a 2-approximation for
general bipartite graphs (though König's theorem makes it exact for
bipartite — the key difference is that the flow formulation can
consider vertex weights, which minimum vertex cover does not
naturally handle).

### V-Cycle / F-Cycle Controller (`cycle.rs`)

```rust
pub(crate) enum CycleStrategy {
    /// Single pass (no cycling). Used by Fast mode.
    None,
    /// V-cycle: one re-coarsen/re-refine pass.
    VCycle,
    /// F-cycle: two recursive calls per level with different seeds.
    FCycle,
}

/// Run one cycle iteration: re-coarsen (preserving cut edges),
/// then re-refine.
pub(crate) fn run_cycle(
    graph: &Graph,
    where_: &mut [u8],
    strategy: CycleStrategy,
    refine_fn: &dyn Fn(&Graph, &mut [u8]),
);
```

**Key invariant:** When re-coarsening, edges that cross the current
partition boundary are NOT contracted. This means the coarsened graph
"remembers" the partition, and refinement at each level can only
maintain or improve quality. The partition is monotonically
non-worsening through cycles.

**V-cycle:** After initial multilevel pass, re-coarsen once (with
cut-edge preservation), re-refine. ~2× the initial cost.

**F-cycle:** At each coarsening level, make two recursive calls with
different random seeds for matching. Keep the better result. ~3-5×
the initial cost, but significantly better quality.

### Nested Dissection Driver (`node_nd.rs`)

```rust
/// Compute a fill-reducing nested dissection ordering using
/// KaHIP-style flow-based graph partitioning.
pub fn kahip_node_nd(
    pattern: &CscPattern,
    mode: KaHIPMode,
) -> Vec<usize>;

pub enum KaHIPMode {
    /// FM-only refinement + data reduction, single pass.
    /// **REVISED CLAIM**: comparable QUALITY to METIS at ~1.5–2× METIS time
    /// (data reduction is non-trivial overhead even when graph shrinks).
    Fast,
    /// FM + flow refinement (default `bnd_distance=3`), V-cycle. 2-3x
    /// slower than METIS, ≥5% fill reduction on KKT (geomean).
    Eco,
    /// FM + flow refinement (default `bnd_distance=5`), F-cycle with
    /// "branching factor 2 only at top few levels" tweak (else cost
    /// explodes 2^depth). 5-10× slower; up to 10% fill reduction
    /// on the optimistic end.
    Strong,
}

impl Default for KaHIPMode {
    fn default() -> Self { KaHIPMode::Eco }
}

pub struct KaHIPParams {
    pub mode: KaHIPMode,
    /// Override `bnd_distance` (Eco default 3, Strong default 5).
    /// Production KaHIP uses adaptive: start 2, grow until band > 5%|V|.
    pub bnd_distance: Option<usize>,
    /// Per-level imbalance tolerance (default 0.05).
    pub max_imbalance: f64,
    /// Per-level imbalance tightening with depth (default 0.95):
    /// at depth d, effective tolerance = max_imbalance * decay^d.
    /// Prevents wildly imbalanced leaves on deep recursion.
    pub imbalance_decay: f64,
    /// Deterministic seed for matching/RNG.
    pub seed: u64,
}
```

**Algorithm:**
```
function kahip_nd(graph, perm, offset, count, mode):
    // Step 0: data reduction (all modes)
    if let Some(reduced) = reduce_graph(graph, 0.7):
        kahip_nd(reduced.graph, reduced_perm, ...)
        perm = expand_permutation(reduced_perm, ...)
        return

    // Step 1: base case
    if graph.nvtxs < 120:
        order with amd_order
        return

    // Step 2: multilevel bisection
    levels = coarsen_graph(graph)         // shared with METIS
    initial_bisect(coarsest_level)        // shared with METIS

    // Step 3: uncoarsen with mode-specific refinement
    for level in levels.rev():
        project partition
        match mode:
            Fast   => fm_refine_bisection(...)
            Eco    => { fm_refine_bisection(...); flow_refine_bisection(..., d=3) }
            Strong => { fm_refine_bisection(...); flow_refine_bisection(..., d=5) }

    // Step 4: optional cycling
    match mode:
        Fast   => {}  // no cycling
        Eco    => run_cycle(VCycle, ...)
        Strong => run_cycle(FCycle, ...)

    // Step 5: flow-based node separator
    flow_node_separator(graph, where_, ...)

    // Step 6: standard nested dissection recursion
    number separator last
    recurse on left and right subgraphs
```

### Integration

In `src/symbolic/mod.rs`, extend the `OrderingMethod` enum:

```rust
pub enum OrderingMethod {
    Amd,
    MetisND,
    KaHIPND(KaHIPMode),
    ScotchND(ScotchParams),
}
```

Default remains `Amd`. For KKT matrices detected by structure
(presence of negative diagonal entries or saddle-point pattern),
auto-selection could choose `KaHIPND(Eco)`.

## Implementation Steps

### Phase K1: Data reduction (~250 lines)

- Degree-1 elimination
- Degree-2 path compression
- Twin detection (hash + verify)
- Neighborhood subset reduction
- Fixed-point loop
- Permutation expansion
- Tests: reduction ratio on known graphs, expanded permutation validity

### Phase K2: Push-relabel max-flow (~300 lines)

- Highest-label push-relabel
- Gap relabeling optimization
- Min-cut extraction via residual BFS
- Tests: known max-flow instances (Petersen graph, grid networks),
  correctness of min-cut partition

### Phase K3: Flow-based edge refinement (~250 lines)

- BFS band extraction around boundary
- Flow network construction (super-source/sink)
- Integration with push-relabel
- Balance-constrained cut application
- Tests: flow-refined cut ≤ FM-refined cut (probabilistic)

### Phase K4: Flow-based node separator (~200 lines)

- Boundary bipartite graph construction
- Vertex-capacitated flow network
- Min vertex cut → separator
- Tests: separator validity, weight comparison vs METIS approach

### Phase K5: V-cycle / F-cycle (~150 lines)

- Cut-edge-preserving re-coarsening
- Single-pass (V) and double-pass (F) strategies
- Monotone quality invariant
- Tests: cycled quality ≥ single-pass quality

### Phase K6: Driver and modes (~200 lines)

- `kahip_node_nd` with Fast/Eco/Strong dispatch
- Data reduction → multilevel → cycling → separator → recursion
- Integration into `symbolic_factorize` via `OrderingMethod`
- Tests: permutation validity, fill quality across modes

## Testing Plan

### Unit Tests

**T1. Data reduction correctness:**
- Path graph (n=20): degree-1/2 rules reduce to 2-3 vertices
- Regular 4×4 grid: twin detection merges corners
- Expanded permutation is a valid bijection of {0..n-1}
- Fill of reduced+expanded ordering ≤ fill of unreduced ordering

**T2. Push-relabel correctness:**
- Unit-capacity path: max flow = 1
- Unit-capacity grid (k×k): max flow = k (horizontal cut)
- Known textbook instances (Cormen et al. examples)
- Max-flow = min-cut verified on random graphs

**T3. Flow refinement quality:**
- 2D grid (7×7): flow-refined cut < FM-refined cut on ≥ 60% of
  trials (probabilistic, 50 trials with different seeds)
- Random sparse (n=200): flow refinement never worsens cut
- Balance constraint respected after refinement

**T4. Node separator quality:**
- Path (n=20): separator = 1 vertex (center)
- 2D grid (7×7): separator ≈ 7 vertices (a row)
- Flow separator weight ≤ METIS separator weight on ≥ 70% of trials

**T5. Cycle quality monotonicity:**
- For any graph: quality after V-cycle ≥ quality before
- For any graph: quality after F-cycle ≥ quality after V-cycle
  (probabilistic due to random seeds)

**T6. Mode hierarchy:**
- For any graph: fill(Strong) ≤ fill(Eco) ≤ fill(Fast)
  (on average over multiple seeds; may not hold per-instance)

**T7. Permutation validity** — `kahip_node_nd` output:
- `perm.len() == n`, bijection
- Works on all edge cases: n=0, n=1, diagonal, dense

### Integration Tests

**T8. Full pipeline correctness:**
- `tests/data/parity/` matrices with all three KaHIP modes
- Inertia matches MUMPS oracle
- Residuals within tolerance
- No regressions vs AMD ordering

**T9. Four-way ordering comparison:**
- For each parity matrix: fill with AMD, METIS, KaHIP-Eco, KaHIP-Strong
- Report distribution of fill ratios
- Identify matrices where KaHIP uniquely excels

### Benchmark Tests

**B1. Ordering time across modes:**
- Synthetic matrices: n ∈ {100, 500, 1k, 5k, 10k}
- Verify: Fast ≈ METIS time, Eco ≈ 2-3× METIS, Strong ≈ 5-10× METIS

**B2. Fill quality vs METIS:**
- For each parity matrix: `fill_kahip / fill_metis`
- **Gate (Eco):** geomean ≤ 0.95 (≥ 5% less fill than METIS) — defensible
  per Hogg-Scott 2013 reporting 5–15% fill reduction on indefinite KKT
- **Gate (Strong):** geomean ≤ 0.92 (≥ 8% less fill) — was 0.90 (10%)
  but audit calls 10% optimistic for KKT; road-network 15-30% gains do
  not transfer directly to KKT

**B3. Data reduction impact:**
- Compare ordering time and fill with/without data reduction
- Report: reduction ratio, speedup, fill change per matrix
- **Expected:** 1.5-3× speedup on structured matrices

**B4. End-to-end factorization:**
- `cargo run --bin bench --release` with AMD, METIS, KaHIP-Eco
- Compare total solve time (ordering + symbolic + numeric + solve)
- **Gate:** KaHIP-Eco total time ≤ METIS total time on matrices where
  KaHIP fill < 0.9 × METIS fill

**B5. Amortization analysis (problem-size-aware):**
- For matrices factored N times (same pattern, different values):
  N × factor_time_kahip + ordering_time_kahip vs
  N × factor_time_metis + ordering_time_metis
- Report crossover N **per problem-size band**:
  - Large (n > 10⁵): expected N = 1–2 (numeric factor dominates)
  - Medium (10⁴ ≤ n ≤ 10⁵): expected N = 2–5
  - Small (n < 10⁴): expected N ≥ 10 (ordering is large fraction
    of solve time at small n; amortization needs many refactorizations)
- IPM typically runs 10–50 iterations with same sparsity; Eco breaks
  even comfortably for medium/large KKT.

**B6. Comparison vs C KaHIP:**
- Run C KaHIP `node_ordering` on parity matrices
- Compare fill: feral KaHIP-Eco vs C KaHIP eco mode
- **Gate:** within 15% of C KaHIP fill on ≥ 80% of matrices

## Open-Source Reference Implementations

**Primary reference — KaHIP v3.16 (C++, MIT license):**
- Repository: https://github.com/KaHIP/KaHIP
- Key files for nested dissection:
  - `app/node_ordering.cpp` — nested dissection ordering entry point
  - `app/node_separator.cpp` — node separator computation
  - `lib/partition/coarsening/` — graph coarsening (shared concept with METIS)
  - `lib/partition/uncoarsening/refinement/kway_graph_refinement/` — FM refinement
  - `lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/` — flow-based refinement
  - `lib/tools/graph_extractor.cpp` — subgraph extraction for local flow
  - `lib/data_structure/priority_queues/` — bucket PQ and max-node-heap
  - `lib/algorithms/push_relabel.cpp` — push-relabel max-flow
- MIT license is ideal for FERAL's clean-room implementation

**Data reduction rules — integrated in KaHIP:**
- Paper: https://arxiv.org/abs/2004.11315
- Code in KaHIP's `lib/node_ordering/` directory

**Sanders & Schulz dissertation (comprehensive reference):**
- https://schulzchristian.github.io/dissertation_christian_schulz.pdf
- Chapters 3-5 cover KaFFPa, flow refinement, and V/F-cycles in detail

**Push-relabel reference implementations:**
- Goldberg-Tarjan: well-documented in Cormen et al. (CLRS) Chapter 26
- Rust crate `max-flow` (MIT): https://crates.io/crates/max-flow
  (for API reference; we implement our own for zero-dependency constraint)

**METIS (for shared infrastructure):**
- Repository: https://github.com/KarypisLab/METIS (Apache 2.0)
- Coarsening and FM modules shared via `src/ordering/metis/`

## Verification Checklist

- [ ] All existing `cargo test` pass (no regressions)
- [ ] `tests/parity.rs` — zero inertia regressions with KaHIP ordering
- [ ] Permutation is a valid bijection for all test matrices
- [ ] Max-flow solver correct on known instances
- [ ] Push-relabel includes gap relabeling (v1 gate, not optional)
- [ ] Push-relabel global relabeling implemented for Strong mode
- [ ] Push-relabel uses lowest-index admissible neighbor (determinism)
- [ ] Separator validity: removal disconnects subgraphs (BFS check)
- [ ] Data reduction: expanded permutation is valid bijection
- [ ] Degree-2 simplicial sub-case handled (zero-fill branch)
- [ ] Closed twins detected (not just open twins)
- [ ] Cascading degree-1 removals record order
- [ ] Reduction expansion uses operation stack, popped in reverse
- [ ] Data reduction NOT re-run inside V/F-cycle loop
- [ ] Flow refinement: source/sink at distance EXACTLY `bnd_distance`
      from boundary, pinned via super-source/super-sink with ∞ capacity
- [ ] Edge capacities modeled as anti-parallel directed edges
- [ ] Most Balanced Min Cut technique implemented (not naive reject)
- [ ] F-cycle uses "branching factor 2 only at top few levels" tweak
- [ ] Per-level imbalance tightening with depth (`imbalance_decay`)
- [ ] Fill quality: KaHIP-Eco ≤ 0.95 × METIS fill (geomean)
- [ ] Fill quality: KaHIP-Strong ≤ 0.92 × METIS fill (geomean)
- [ ] Mode hierarchy: Strong ≤ Eco ≤ Fast fill (on average)
- [ ] Fast mode benchmark: timing reported as 1.5–2× METIS (not "~METIS")
- [ ] Amortization crossover reported per size band (small/medium/large)
- [ ] No `unwrap()` or `expect()` in `src/ordering/kahip/`
- [ ] `cargo clippy -- -D warnings` clean
- [ ] Default ordering remains AMD (backward compatible)
- [ ] METIS implementation is prerequisite (shared infrastructure)
- [ ] Clean-room provenance note: derive from Sanders-Schulz 2011 +
      Ost-Schulz-Strash 2021 papers; KaHIP C++ used as behavioral
      reference only (not constants/structure copying)
