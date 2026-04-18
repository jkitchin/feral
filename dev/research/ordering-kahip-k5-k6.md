# KaHIP Phase K5+K6 — Multilevel Controller and ND Driver

## Context

K5 and K6 close the KaHIP pipeline. The K1-K4 pieces are all
standalone primitives; K5 orchestrates one multilevel node
bisection on a subgraph (coarsen → initial partition →
uncoarsen with refinement → separator), and K6 wraps K5 in the
recursive nested-dissection driver that emits a permutation
conforming to the FERAL ordering-crate contract.

References: Sanders & Schulz 2011 §3 (multilevel framework) and
§4.3 (V-cycle / F-cycle re-coarsening); George 1973 for ND.

## Architecture: reuse `feral-metis::internals`

`feral-scotch` demonstrates the precedent: SCOTCH's node-ND
driver reuses `feral_metis::internals::{coarsen, fm_refine,
graph, initial_partition, rng}` for the shared multilevel
plumbing and plugs in SCOTCH-specific refinement (halo FM) and
separator construction (direct two-sided FM). We follow the same
pattern for KaHIP:

- **Coarsen** and **initial bisect** — reused verbatim from
  `feral-metis` (SHEM + 2-hop matching, GGP / random BFS
  bootstrap).
- **Refine at each level** — use **flow-based refinement** (K3)
  instead of FM (METIS) or halo FM (SCOTCH). FM is still used
  as a cheap bootstrap at the coarsest level.
- **Separator** — use **flow-based node separator** (K4) via
  the bipartite vertex-cover reduction. This is König's theorem
  on the refined bisection.
- **Leaf fallback** — `feral_amd::amd_order` on subgraphs of at
  most `nd_to_amd_switch` vertices.

`UndirectedGraph` (K3 / K4's type, `usize`-indexed) and
`Graph` (feral-metis's type, `i32`-indexed) are bridged by a
small conversion helper: K3 and K4 both need `UndirectedGraph`,
so the multilevel pipeline converts the current `Graph` to an
`UndirectedGraph` at each level before calling refinement.

## K5 — multilevel node bisection

### Inputs
- `graph: &Graph` (feral-metis representation).
- `opts: &KahipOptions` (seed, mode).
- `rng: &mut SplitMix`.
- `stats: &mut KahipStats`.

### Algorithm
1. **Coarsen.** Call `feral_metis::internals::coarsen::coarsen`
   with a synthesized `MetisOptions` (matching seed,
   `coarsen_floor` from mode). Produces a `Vec<CoarseGraph>`
   ordered from finest to coarsest.
2. **Initial bisect at the coarsest level.** Best-of-N trials
   alternating GGP and random-BFS seeds, each refined by a
   short FM pass, score by cut weight.
3. **Uncoarsen with K3 flow refinement.** For each level from
   coarsest up to the finest:
   a. Project the coarser-level labels onto the finer level via
      `cmap`.
   b. Convert the fine-level `Graph` to `UndirectedGraph`.
   c. Apply a short FM pass (cheap local improvement) —
      `refine_bisection` from feral-metis.
   d. Apply one or more iterations of K3's
      `flow_refine_bisection` with `bnd_distance` and
      `max_imbalance` set from the KaHIP mode.
4. Return the labels at the finest level (`{PART_A, PART_B}`).

### KaHIP mode → K5 parameters
- **Fast:** 1 K3 iteration per level, only at the finest.
- **Eco:** 1 K3 iteration at every level.
- **Strong:** 2 K3 iterations at every level (approximates
  F-cycle quality with less code; full re-coarsening F-cycle is
  deferred to a follow-up).

### Out of scope (v1)
- **Full V-cycle / F-cycle re-coarsening** of Sanders-Schulz
  2011 §4.3. The "cut-edge-preserving re-coarsening" requires
  that the next coarsening respect the current cut; it improves
  quality on hard instances but adds ~150 lines and a second
  loop over the multilevel hierarchy. We ship v1 with a single
  pass (a "V-cycle in its trivial form") and document the
  follow-up.
- **Adaptive mode selection** (e.g., switch Eco → Strong on
  high-diameter graphs). Fixed mode per call for v1.

## K6 — recursive ND driver

### Entry point
```rust
pub fn kahip_order_full(
    pattern: &CscPattern<'_>,
    opts: &KahipOptions,
) -> Result<(Vec<i32>, OrderingStats, KahipStats), OrderingError>;
```

### Algorithm (per feral-scotch pattern)
1. **Build `Graph` from pattern.**
2. **Top-level connected components.** Each component is
   ordered independently and concatenated (same as SCOTCH /
   METIS).
3. **Per-component recursion**:
   - `n == 0`: no-op.
   - `n == 1`: single-vertex; place and return.
   - Subgraph disconnected (post-bisection can produce this):
     split by connected components, recurse.
   - `n <= opts.nd_to_amd_switch`: hand off to `feral_amd`.
   - Otherwise:
     a. Run K5 to get a bisection `labels ∈ {PART_A, PART_B}`.
     b. Convert `Graph` to `UndirectedGraph`, call K4's
        `flow_node_separator` to get separator labels
        `∈ {PART_A, PART_B, PART_SEP}`.
     c. If bisection is degenerate (one side empty), fall back
        to AMD on the whole subgraph.
     d. Number separator vertices last in the current window.
     e. Extract `A` and `B` induced subgraphs, recurse on each.
4. **Invert** the `iperm` (old → new) to a `perm` (new → old).

### KaHIP mode → K6 parameters
- **Fast:** `nd_to_amd_switch = 200`, `n_sep_trials = 3`,
  K5 in Fast mode.
- **Eco:** `nd_to_amd_switch = 120`, `n_sep_trials = 5`,
  K5 in Eco mode.
- **Strong:** `nd_to_amd_switch = 80`, `n_sep_trials = 7`,
  K5 in Strong mode.

Defaults mirror METIS / SCOTCH where sensible but trade
coarsen-floor depth for more flow refinement in Strong.

### Data reduction (K1) integration (v1 scope)
K1 is **deferred** from K6 v1. It wraps the pipeline above
(`kahip_order_full(reduce(pattern))` and expand the permutation
back) and should land after the base pipeline works. K6 v1
calls K5 directly on the input pattern without reduction;
opts.mode still controls K5/K6 parameters.

## Module layout

- `crates/feral-kahip/src/cycle.rs` — K5 multilevel pipeline
  (`multilevel_bisection`). ~200 LOC.
- `crates/feral-kahip/src/node_nd.rs` — K6 recursive ND driver
  (`kahip_nd_order`). ~300 LOC.
- `crates/feral-kahip/src/lib.rs` — wire `kahip_order_full` to
  call `node_nd::kahip_nd_order`; add `feral-metis` and
  `feral-amd` as dependencies in `Cargo.toml`.

## Test oracles

### K5 (cycle.rs)
1. **Trivial 10-vertex graph.** `multilevel_bisection` produces
   labels summing to two non-empty parts.
2. **Determinism.** Same seed, same input → same labels.
3. **Balance.** Part weights satisfy `max ≤ (1 + ε) · ⌈n/2⌉`.
4. **Non-worsening.** The cut after flow refinement is no
   larger than the cut after FM bootstrap at the finest level.

### K6 (node_nd.rs)
1. **Diagonal pattern, `n = 3`.** Valid permutation.
2. **Small grid (10×10, `n = 100`).** AMD leaf path hit
   (`n <= nd_to_amd_switch`).
3. **Large grid (16×16, `n = 256`).** Multilevel path hit;
   `n_separator_vertices > 0`, `n_levels > 0`.
4. **Determinism.** Same input + opts → same permutation.
5. **Disconnected components.** Two disjoint 8×8 grids handled
   correctly (`n_components == 2`).
6. **Fast / Eco / Strong produce valid permutations.**

### Contract conformance
7. **MalformedInput rejection** at the public entry when
   `col_ptr.len() != n + 1`.
8. **Empty graph** produces empty permutation.

## Out of scope (deferred)

- **Full V/F-cycle re-coarsening** per Sanders-Schulz §4.3.
- **K1 data reduction integration** with expansion pipeline.
- **Adaptive bnd_distance** per level.
- **OrderingMethod::KahipND dispatch in the main solver.** K6
  lands `kahip_order` / `kahip_order_full` as a callable library
  function; integrating into `symbolic_factorize` as a method
  choice is a separate task after benchmarking lands.
