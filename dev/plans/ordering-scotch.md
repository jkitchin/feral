# Ordering Plan: SCOTCH-Style Nested Dissection

**Status:** Pre-implementation plan — audited 2026-04-16
**Date:** 2026-04-16
**Research note:** `.crucible/wiki/concepts/scotch-graph-ordering.org`
**Paper references:** `.crucible/wiki/summaries/pellegrini1996-scotch.org`,
`.crucible/wiki/summaries/chevalier2008-ptscotch.org`
**Code reference:** SCOTCH 7.0 `libscotch/` (CeCILL-C); MUMPS dispatches
SCOTCH via `mumps/src/mumps_scotch.c` and PORD via
`mumps/PORD/lib/interface.c`. **Clean-room provenance**: implement from
the paper algorithm descriptions (Pellegrini 1996 §3) only — do not copy
constants/structure from `vgraph_separate_fm.c`.
**Related:** `dev/plans/phase-2-planning.md`

---

## Audit Findings (mumps-expert, 2026-04-16)

The plan is structurally sound and the file decomposition is reasonable.
Two areas need pre-implementation revision; the strategy-string drop is fine
provided the enum encodes SCOTCH's actual defaults.

1. **Two-sided FM gain bookkeeping is incomplete (BUG).** Current wording
   "Net gain = vwgt[v] - Σ vwgt[u] for newly-added separator vertices"
   is ambiguous. Correct formula when moving separator vertex `v` to side 0:
   `gain = vwgt[v] - Σ_{u ∈ side1 ∩ N(v)} vwgt[u]` — the sum is restricted
   to neighbors **currently in the opposite part**, not abstract
   "newly-added." Implementations that miss this routinely produce buggy
   FM. Maintain an incremental "frontier load" per separator vertex,
   updated as moves are committed/rolled back (per `vgraph_separate_fm.c`
   semantics, not source).

2. **Band FM is missing anchor mechanism (BUG).** The plan's BFS-extract +
   project-back is correct in shape, but SCOTCH's `bgraph_bipart_bd.c`
   pins the band's outer boundary as fixed "anchor" vertices in two
   artificial supervertices (one per side); FM can't move them but their
   weight enters balance computations. Without anchors, balance accounting
   on the extracted subgraph is wrong, and projection back can violate
   global balance.

3. **Halo FM should be dynamic, not static.** Plan computes a one-hop halo
   once; SCOTCH updates the halo set as vertices move. A static halo
   underperforms on graphs where the boundary shifts substantially during
   a pass.

4. **Strategy enum defaults must match SCOTCH defaults**:
   - `cmin=20, cmax=100` for vertex-separation contexts (coarsening floor)
   - `n_sep_trials = 5` (plan said 3)
   - FM `move=200` per-pass cap, `pass=-1` unbounded passes-until-no-improvement
   - Switch to halo-AMD base case at `vert=120` (plan matches — keep)
   - `bal=0.05` imbalance tolerance
   - **Auto band-FM activation**: when `frontier_size / total_size < 0.1`
     (`bgraph_bipart_bd.c`). Recommend `RefineMethod::Auto` as default
     to preserve SCOTCH's adaptivity.

5. **Graph compression refinements**: edge weights must also sum on merge,
   not just vertex weights. Compression must be applied **once** at top of
   recursion (PORD style: `interface.c:84-99`) OR per-level. Plan does
   per-level which is more aggressive — keep but short-circuit on the
   first uncompressed level once attempted (re-attempting on graphs
   without supervariables wastes work).

6. **Initial separator construction is unspecified.** SCOTCH builds the
   initial separator from one side's boundary (typically the smaller part)
   then runs FM. Specify this; affects starting quality.

7. **Imbalance handling**: enforce `max_imbalance` as both
   (a) per-move (reject moves that exceed) AND (b) on the final accepted
   prefix. `vgraph_separate_fm.c` does both.

8. **MUMPS does not prefer SCOTCH over METIS for any matrix class.**
   Auto-selection (`mumps/src/ana_set_ordering.F:65-77`): METIS first,
   SCOTCH only as fallback when METIS isn't linked. PORD is lowest-priority
   fallback. There is no MUMPS evidence that SCOTCH outperforms METIS on
   KKT — the case for FERAL SCOTCH rests on the algorithmic claims
   (tighter vertex separators on structured meshes) plus diversity for
   cross-validation, not on MUMPS preference.

9. **MUMPS uses `esmumpsv` (vertex-weighted)**, not raw `SCOTCH_graphOrder`.
   Vertex weights feed compressed multiplicities back into the symbolic
   AMD step. FERAL's compression must be paired with a vertex-weighted
   AMD base case (or accept a quality gap vs. MUMPS+SCOTCH).

10. **Pivot threshold realism**: KKT systems need `CNTL(1) ≥ 0.01`. Heavy
    threshold pivoting promotes delayed pivots, inflating frontal matrices
    independent of ordering quality. The plan's 5–10% smaller-separator
    benefit can be entirely consumed by delayed-pivot growth. Report fill
    *with realistic pivot thresholds* in benchmarks, not symbolic-only fill.

11. **Determinism**: SCOTCH 7's default uses RNG-seeded coarsening. Expose
    `seed: u64` in `ScotchParams`; default to a fixed value for parity tests.

12. **PT-SCOTCH halo graphs**: confirmed defer entirely to Phase 4.

---

## Goal

Implement a pure-Rust SCOTCH-style nested dissection ordering in
`src/ordering/scotch/` that provides an alternative to METIS with:

- Direct vertex separator computation (tighter separators on
  structured meshes)
- Graph compression (supervariable merging before partitioning)
- Configurable strategy via Rust enums (not string parsing)
- Fill quality competitive with METIS and C SCOTCH

## Motivation

SCOTCH is lower priority than METIS for FERAL. The primary reasons
to implement it:

1. **Tighter vertex separators** — SCOTCH computes vertex separators
   directly via two-sided FM, rather than converting from edge
   bisections. On structured meshes this can produce 5-10% smaller
   separators and correspondingly less fill.

2. **Graph compression** — Merging indistinguishable vertices before
   partitioning can dramatically reduce graph size on structured
   problems, speeding up the multilevel pipeline.

3. **PT-SCOTCH path** — If FERAL adds distributed-memory support
   (Phase 4), having a SCOTCH-style framework makes PT-SCOTCH
   extension natural. PT-SCOTCH has better MPI scaling than ParMETIS.

4. **Diversity** — Having two independent nested dissection
   implementations enables cross-validation and allows choosing the
   better ordering per-matrix.

## Scope Decision: No Strategy Strings

SCOTCH's strategy string parser is ~2000 lines of C and adds
substantial complexity for marginal benefit in a solver where the
ordering method is chosen programmatically. Instead, we implement
SCOTCH's algorithmic contributions (direct vertex separator, graph
compression, halo FM) with a Rust enum-based configuration:

```rust
pub struct ScotchParams {
    /// Compress graph before partitioning.
    pub compress: bool,
    /// Compression ratio threshold (compress if ratio > threshold).
    /// PORD uses 0.75; we use 0.7 (slightly more aggressive — defensible).
    pub compress_ratio: f64,
    /// Base case: switch to AMD below this vertex count. SCOTCH default 120.
    pub amd_switch: usize,
    /// Coarsening floor: stop coarsening at this size. SCOTCH cmin=20, cmax=100.
    pub coarsen_floor: usize,
    /// Number of separator trials at each recursion level. SCOTCH default 5.
    pub n_sep_trials: usize,
    /// FM refinement: per-pass move cap (SCOTCH default 200).
    pub fm_move_cap: usize,
    /// FM refinement: per-call pass cap (SCOTCH default unbounded → ~32).
    pub fm_pass_cap: usize,
    /// Imbalance tolerance (SCOTCH default 0.05).
    pub max_imbalance: f64,
    /// FM refinement variant.
    pub refine: RefineMethod,
    /// Deterministic seed for coarsening matching.
    pub seed: u64,
}

pub enum RefineMethod {
    /// Auto: switch to BandFM when frontier_size/total_size < 0.1
    /// (matches SCOTCH's `bgraph_bipart_bd.c` activation criterion).
    /// THIS IS THE DEFAULT — preserves SCOTCH's adaptivity.
    Auto,
    /// Standard boundary FM.
    BoundaryFM,
    /// Halo FM (extends neighborhood one hop beyond boundary, DYNAMIC update
    /// per pass — not a static one-shot halo).
    HaloFM,
    /// Band FM (restrict to band of width w around separator).
    /// `width` defaults to 3 for first refinement; auto-grow with level
    /// noted as future work.
    BandFM { width: usize },
}

impl Default for ScotchParams {
    fn default() -> Self {
        ScotchParams {
            compress: true,
            compress_ratio: 0.7,
            amd_switch: 120,
            coarsen_floor: 100,
            n_sep_trials: 5,         // was 3 (audit: SCOTCH default is 5)
            fm_move_cap: 200,
            fm_pass_cap: 32,
            max_imbalance: 0.05,
            refine: RefineMethod::Auto,  // was BoundaryFM
            seed: 0xDEADBEEF,
        }
    }
}
```

This captures SCOTCH's key algorithmic ideas without the parsing
overhead. Users who need fine-grained control can construct custom
`ScotchParams`.

## Directory Structure

```
src/ordering/
├── mod.rs                    # add `pub mod scotch;`
├── amd.rs                    # UNCHANGED
├── metis/                    # METIS implementation (separate plan)
├── scotch/
│   ├── mod.rs                # public API: scotch_node_nd()
│   ├── compress.rs           # graph compression (~150 lines)
│   ├── vertex_separator.rs   # direct vertex separator FM (~350 lines)
│   ├── halo_fm.rs            # halo FM refinement variant (~200 lines)
│   ├── band_fm.rs            # band FM refinement variant (~150 lines)
│   └── node_nd.rs            # recursive nested dissection driver (~200 lines)
├── elimination_tree.rs       # UNCHANGED
└── postorder.rs              # UNCHANGED
```

SCOTCH reuses METIS's infrastructure where identical:
- `metis::graph::Graph` — CSR graph representation
- `metis::coarsen::coarsen_graph` — heavy-edge matching coarsening
- `metis::initial_partition::initial_bisect` — GGP initial partition
- `metis::fm_refine::BucketPQ` — gain bucket data structure

Only the following are SCOTCH-specific:
- Graph compression (supervariable merging)
- Direct vertex separator computation via two-sided FM
- Halo FM and band FM refinement variants
- The strategy dispatch (enum-based)

Estimated new code: ~1050-1200 lines (excluding shared METIS modules).

**Prerequisite:** The METIS implementation (`ordering-metis.md`)
must be completed first, as SCOTCH builds on its infrastructure.

## Design

### Graph Compression (`compress.rs`)

```rust
/// Compress a graph by merging indistinguishable vertices.
///
/// Two vertices u, v are indistinguishable if they have exactly the
/// same adjacency set (excluding each other). Merged vertices become
/// a single supervariable with summed weight.
///
/// Returns the compressed graph, a map from compressed vertex to
/// list of original vertices, and the compression ratio.
pub(crate) fn compress_graph(
    graph: &Graph,
    min_ratio: f64,
) -> Option<CompressedGraph>;

pub(crate) struct CompressedGraph {
    pub graph: Graph,
    /// compressed_vertex[c] = list of original vertices merged into c.
    pub vertex_map: Vec<Vec<usize>>,
    /// Ratio of compressed to original vertices.
    pub ratio: f64,
}
```

**Algorithm:**
1. Hash each vertex's sorted adjacency list
2. Group vertices with the same hash
3. Within each group, compare adjacency lists exactly
4. Merge confirmed matches:
   - **Vertex weight** = sum of original weights
   - **Edge weights to merged neighbors** also summed (was missing from plan)
5. Build compressed graph with merged adjacency
6. If compression ratio < min_ratio, return None (not worth it)

**Compression cadence**: applied per recursion level (PORD-style; see
`mumps/PORD/lib/interface.c:84-99`), but **short-circuit** on the first
uncompressed level once attempted — re-attempting on graphs without
supervariables (e.g. random matrices) wastes work.

This is the same algorithm as AMD's supervariable detection, applied
at the graph level before partitioning. Reuse the hash+verify pattern.

### Direct Vertex Separator (`vertex_separator.rs`)

The key algorithmic difference from METIS. Instead of:
1. Compute edge bisection → 2. Convert to node separator

SCOTCH directly computes vertex separators using two-sided FM:

```rust
/// Compute a vertex separator by direct two-sided FM.
///
/// Starting from an initial partition (where[v] ∈ {0, 1}),
/// identify a vertex separator S (where[v] = 2) by:
/// 1. Select boundary vertices as initial separator candidates
/// 2. Apply two-sided FM to minimize separator weight
///
/// Unlike METIS's convert-then-refine approach, this optimizes
/// separator size directly as the primary objective.
pub(crate) fn compute_vertex_separator(
    graph: &Graph,
    where_: &mut [u8],
    max_imbalance: f64,
    n_iter: usize,
);
```

**Two-sided FM for vertex separators (CORRECTED gain):**

State: each vertex is in part 0, part 1, or separator (2).
Invariant: removing separator vertices disconnects parts 0 and 1.

**Initial separator construction**: build from the *smaller side's* boundary
vertices (those with at least one neighbor in the other side). This yields
a non-minimal separator that FM then shrinks.

For each pass:
1. Build two priority queues: PQ_0 (separator vertices that could
   move to part 0) and PQ_1 (separator vertices that could move to
   part 1)
2. **Maintain incremental "frontier load" per separator vertex** =
   `Σ vwgt[u] for u ∈ N(v) currently in opposite side`. Updated as
   moves are committed/rolled back. Without this incremental update,
   FM bookkeeping silently drifts and produces buggy results.
3. Alternate between queues:
   a. Pop best vertex v from PQ_side
   b. Compute gain (CORRECTED):
      `gain(v→side) = vwgt[v] - Σ_{u ∈ N(v) ∩ opposite_side} vwgt[u]`
      i.e. only neighbors **currently in the opposite part**, not all
      "newly-added" abstractly.
   c. Reject the move if it would exceed `max_imbalance` per-move.
   d. Move v from separator to part `side`
   e. For each neighbor u of v in the opposite part:
      - Move u to separator (it's now adjacent to both parts)
      - Insert u into PQ_{opposite_side}
      - Update frontier loads of u's neighbors
   f. For neighbors of v already in `side`: re-check their "frontier"
      status — they may now have no neighbors in the opposite side
      and could be removed from any auxiliary frontier set.
4. Track best total separator weight during the pass; **also enforce
   final-prefix imbalance** at acceptance time (not just per-move).
5. Roll back to the best balanced state.

**Key insight:** This directly minimizes separator weight, whereas
METIS minimizes edge cut first and then converts — which may not
minimize the derived separator weight.

### Halo FM (`halo_fm.rs`)

Extension of boundary FM that considers vertices one hop beyond the
boundary as move candidates:

```rust
/// Halo FM: extends the FM candidate set to include a one-hop
/// halo around the partition boundary.
///
/// Standard boundary FM only considers vertices adjacent to the
/// other partition. Halo FM also considers their neighbors, which
/// can find moves that boundary FM misses (a non-boundary vertex
/// whose move would open up a chain of improving boundary moves).
pub(crate) fn halo_fm_refine(
    graph: &Graph,
    where_: &mut [u8],
    max_imbalance: f64,
    n_iter: usize,
);
```

**Implementation (DYNAMIC halo, not static):**
- Build boundary set B = {v : ∃ neighbor u with where[u] ≠ where[v]}
- Halo set H = {v : ∃ neighbor u ∈ B, v ∉ B}
- Candidate set = B ∪ H
- Run FM on candidate set with same gain/roll-back logic
- **Update halo as vertices move**: when v moves into the boundary, its
  one-hop neighbors enter the halo; when v leaves the boundary, re-check
  whether its neighbors should remain in the halo. SCOTCH does this
  dynamically; a static one-shot halo underperforms on graphs where the
  boundary shifts substantially during a pass.

Cost per pass: O(|B| + |H|) rather than O(|B|). On mesh-like graphs,
|H| ≈ 2|B|, so each pass is ~3× more expensive but finds better cuts.

### Band FM (`band_fm.rs`)

Restricts FM refinement to a narrow band around the separator:

```rust
/// Band FM: restrict refinement to a band of width `w` around
/// the current separator.
///
/// Builds a subgraph containing only vertices within distance w
/// of the separator, runs FM on this subgraph, and projects the
/// result back. Much cheaper than full FM when the separator is
/// small relative to the graph.
pub(crate) fn band_fm_refine(
    graph: &Graph,
    where_: &mut [u8],
    width: usize,
    max_imbalance: f64,
    n_iter: usize,
);
```

**Implementation (with ANCHOR vertices for correct balance accounting):**
1. BFS from separator vertices to distance `width`
2. Extract subgraph of vertices within the band
3. **Add two artificial anchor supervertices** (one per side) representing
   all vertices outside the band on each side. Anchor weight = total
   weight of unincluded vertices on that side. Anchors connect to the
   band's outer boundary with edges; FM cannot move anchors but their
   weight enters balance computations. **Without anchors, balance
   accounting is wrong**, and projection back can violate global balance
   (per `bgraph_bipart_bd.c`).
4. Run standard FM on the subgraph + anchors
5. Map refined partition back to full graph

Cost per pass: O(width × |separator|) instead of O(|E|). Useful when
the graph is very large but the separator is small.

### Nested Dissection Driver (`node_nd.rs`)

```rust
/// Compute a fill-reducing nested dissection ordering using
/// SCOTCH-style algorithms.
///
/// Returns a permutation vector `perm` (new-to-old mapping),
/// identical interface to `amd_order` and `metis_node_nd`.
pub fn scotch_node_nd(
    pattern: &CscPattern,
    params: &ScotchParams,
) -> Vec<usize>;
```

**Algorithm:**
```
function scotch_nd(graph, perm, offset, count, params):
    if graph.nvtxs < params.amd_switch:
        order with amd_order
        return

    // Optional: compress graph
    if params.compress:
        if let Some(compressed) = compress_graph(graph, params.compress_ratio):
            // recurse on compressed graph, then expand permutation
            scotch_nd(compressed.graph, ...)
            expand_permutation(compressed.vertex_map, ...)
            return

    // Multilevel nested dissection
    levels = coarsen_graph(graph)   // shared with METIS
    initial_bisect(coarsest_level)  // shared with METIS

    // Uncoarsen with chosen FM variant
    for level in levels.rev():
        project partition
        match params.refine:
            BoundaryFM => fm_refine_bisection(...)
            HaloFM     => halo_fm_refine(...)
            BandFM{w}  => band_fm_refine(..., w, ...)

    // Direct vertex separator (SCOTCH-specific)
    compute_vertex_separator(graph, where_, ...)

    // Standard recursion (shared logic with METIS)
    number separator last
    recurse on left and right subgraphs
```

### Public API and Integration

```rust
// In src/ordering/scotch/mod.rs:
pub fn scotch_node_nd(
    pattern: &CscPattern,
    params: &ScotchParams,
) -> Vec<usize>
```

Integration in `src/symbolic/mod.rs`:

```rust
pub enum OrderingMethod {
    Amd,
    MetisND,
    ScotchND(ScotchParams),
}
```

## Implementation Steps

### Phase S1: Graph compression (~150 lines)

- Hash-based indistinguishable vertex detection
- Compressed graph construction
- Expansion map for permutation recovery
- Tests: compression ratio on regular grids, correctness of expansion

### Phase S2: Direct vertex separator (~350 lines)

- Two-sided FM for direct vertex separator computation
- Priority queue management for two sides
- Separator validity verification
- Tests: separator disconnects graph, weight is reasonable

### Phase S3: Halo FM refinement (~200 lines)

- Halo set computation via BFS
- Extended candidate FM with halo vertices
- Tests: halo FM cut ≤ boundary FM cut (probabilistic)

### Phase S4: Band FM refinement (~150 lines)

- Band extraction via BFS from separator
- Subgraph construction and projection
- Tests: band FM produces valid partition, respects balance

### Phase S5: Driver and integration (~200 lines)

- `scotch_node_nd` with compression + recursion
- Strategy dispatch via `ScotchParams`
- Integration into `symbolic_factorize` via `OrderingMethod`
- Tests: permutation validity, fill quality

## Testing Plan

### Unit Tests

**T1. Graph compression:**
- Regular 4×4 grid: corners (degree 2) are indistinguishable →
  compression from 16 to ~12 vertices
- Block diagonal (3 identical 4×4 blocks): each block compresses
  identically
- Dense matrix: no compression possible (all vertices distinguishable)
- Verify expansion: compressed ordering expanded to full ordering
  produces valid bijection

**T2. Direct vertex separator:**
- Path graph (n=20): separator should be a single vertex near center
- 2D grid (7×7): separator should be a row/column of ~7 vertices
- Complete bipartite K_{5,5}: minimum separator is one full side (5)
- Validity: BFS from part 0 cannot reach part 1 after removing
  separator

**T3. Halo FM quality:**
- Compare halo FM vs boundary FM on 2D grids:
  halo_cut ≤ boundary_cut on ≥ 70% of trials (probabilistic)
- Verify partition balance maintained

**T4. Band FM correctness:**
- Compare band FM (width=3) vs full FM on 2D grid:
  cuts should be comparable (within 10%)
- Verify band FM is faster than full FM on large graphs

**T5. Permutation validity** — `scotch_node_nd` output:
- `perm.len() == n`, bijection
- Fill ≤ AMD fill on structured matrices
- Fill ≤ 1.1 × METIS fill (should be competitive)

### Integration Tests

**T6. Full pipeline with SCOTCH ordering:**
- `tests/data/parity/` matrices with `OrderingMethod::ScotchND`
- Inertia matches MUMPS oracle
- Residuals within tolerance
- No regressions vs AMD ordering on any matrix

**T7. Three-way ordering comparison:**
- For each parity matrix, compute fill with AMD, METIS, SCOTCH
- Report distribution of fill ratios
- Identify matrices where SCOTCH uniquely excels

### Benchmark Tests

**B1. Ordering time comparison:**
- AMD vs METIS vs SCOTCH on synthetic matrices: n ∈ {100, 1k, 5k, 10k}
- SCOTCH should be 1-2× METIS time (compression saves some, halo FM costs some)

**B2. Fill quality vs METIS:**
- For each parity matrix:
  `fill_scotch / fill_metis` distribution
- **Gate:** geomean within [0.90, 1.10] — SCOTCH should not be
  dramatically worse or better than METIS on average

**B3. Compression speedup:**
- Measure time with and without graph compression on structured matrices
- Report compression ratio and speedup factor
- **Expected:** 1.5-3× speedup on structured meshes with ratio > 0.7

**B4. End-to-end benchmark:**
- `cargo run --bin bench --release` with all three orderings
- Compare total solve time per-matrix
- Report best-of-three selection statistics (which ordering wins
  most often, and by how much)

**B5. Direct separator quality:**
- Compare SCOTCH separator weight vs METIS separator weight
  on 2D/3D grids
- **Expected:** SCOTCH 5-10% lighter separators on structured meshes

**B6. Realistic-pivot fill comparison** — measure fill **after numeric
factorization with realistic threshold pivoting** (`CNTL(1) ≥ 0.01`),
not just symbolic-only fill. Heavy pivoting promotes delayed pivots that
inflate frontal matrices independent of ordering quality. The 5–10%
separator-weight benefit can be entirely consumed by delayed-pivot
growth — confirm or refute on the parity corpus.

**Concrete oracle matrices for KKT validation** (from audit):
- `GHS_indef/c-big` (N=345k) — KKT, expected separator ≈ √N
- `GHS_indef/cont-300` (N=181k) — IPM KKT, separator ≈ 1500–1800
- `GHS_indef/stokes128` (N=49.5k) — separator ≈ 600–800
- `GHS_indef/copter2` (N=55.5k) — separator ≈ 1100–1400
- `Schenk_AFE/af_5_k101` (N=503k) — 3D structural KKT-like; sep ≈ 6000–7000
- `GHS_indef/qpband` (N=20k) — exact separator known: 2 vertices per band-cut
- `Boeing/bcsstk39` (N=46k) — plate KKT, separator ≈ 800–1000

**Per-matrix gates:**
- Top-level separator weight within 10% of MUMPS-via-SCOTCH on each
- Total `nnz(L)` within 5% on regular meshes, 15% on irregular KKT
- Operation count within 10%

## Open-Source Reference Implementations

**Primary reference — SCOTCH 7.0 (C, CeCILL-C ≈ LGPL):**
- Repository: https://gitlab.inria.fr/scotch/scotch
- Key files for ordering:
  - `src/libscotch/library_graph_order.c` — `SCOTCH_graphOrder` public API
  - `src/libscotch/graph_order.c` — ordering driver
  - `src/libscotch/vgraph_separate_fm.c` — vertex separator FM refinement
  - `src/libscotch/vgraph_separate_ml.c` — multilevel vertex separation
  - `src/libscotch/bgraph_bipart_fm.c` — edge bipartitioning FM
  - `src/libscotch/bgraph_bipart_bd.c` — band-graph bipartitioning
  - `src/libscotch/hgraph_order_nd.c` — nested dissection ordering
  - `src/libscotch/hgraph_order_hd.c` — halo AMD base case
  - `src/libscotch/graph_coarsen.c` — graph coarsening
  - `src/libscotch/library_strat.c` — strategy string parser
- CeCILL-C is LGPL-equivalent; dynamic linking OK, static linking
  problematic for MIT distribution. Clean-room implementation needed.

**PT-SCOTCH (C, CeCILL-C) — distributed memory:**
- Same repository, `src/libscotch/dgraph_*` files
- MPI-parallel nested dissection with distributed FM

**PORD (Fortran, bundled with MUMPS) — hybrid ND+AMD:**
- Bundled in MUMPS distribution: https://mumps-solver.org/
- `PORD/lib/` — implements tight ND+AMD coupling
- Reference for hybrid ordering strategy

**KaHIP (C++, MIT) — quality reference:**
- Repository: https://github.com/KaHIP/KaHIP
- `lib/partition/uncoarsening/refinement/` — flow-based refinement
  (produces higher-quality separators than FM-based approaches)
- Data reduction rules (supervariable-like graph simplification)
  in `lib/data_structure/` — applicable as preprocessing for any
  nested dissection implementation

**Pellegrini 2012 User Guide — strategy documentation:**
- Available from SCOTCH releases as `scotch_user7.0.pdf`
- Comprehensive reference for default strategy parameters and
  their effects on ordering quality

## Verification Checklist

- [ ] All existing `cargo test` pass with SCOTCH ordering available
- [ ] `tests/parity.rs` — zero inertia regressions with SCOTCH ordering
- [ ] Permutation is a valid bijection for all test matrices
- [ ] Separator validity verified for all test cases (BFS-disconnect)
- [ ] Two-sided FM gain uses CORRECTED formula
      (`Σ_{u ∈ N(v) ∩ opposite_side} vwgt[u]`, not abstract "newly added")
- [ ] FM maintains incremental "frontier load" per separator vertex
- [ ] Per-move AND final-prefix imbalance enforcement
- [ ] Halo FM updates the halo set DYNAMICALLY each pass
- [ ] Band FM uses anchor supervertices for balance accounting
- [ ] Graph compression sums BOTH vertex weights AND edge weights on merge
- [ ] Compression short-circuits after first uncompressed level attempt
- [ ] Initial separator built from smaller side's boundary
- [ ] Defaults match SCOTCH: `n_sep_trials=5`, `fm_move_cap=200`,
      `bal=0.05`, `coarsen_floor=100`, `amd_switch=120`
- [ ] `RefineMethod::Auto` activates BandFM when frontier_size/total < 0.1
- [ ] Deterministic RNG seed in `ScotchParams`
- [ ] Fill quality: SCOTCH competitive with METIS (within 10%)
- [ ] B6 realistic-pivot comparison reported (not just symbolic fill)
- [ ] No `unwrap()` or `expect()` in `src/ordering/scotch/`
- [ ] `cargo clippy -- -D warnings` clean
- [ ] Default ordering remains AMD (backward compatible)
- [ ] METIS implementation is prerequisite (shared infrastructure)
- [ ] Clean-room provenance: provenance note in module docstring stating
      implementation derives from Pellegrini 1996 §3, not SCOTCH source
