# The node-separator defect was in all three ND backends

**Date:** 2026-09-18
**Machine:** Apple M2, `Mac14,2`, macOS.
**Context:** issue #203. Follows
`dev/research/feral-metis-node-separator-fm-2026-09-17.md`, which fixed
`feral-metis`.

## The defect, restated

All three nested-dissection crates refined a **2-way edge bisection** through
the uncoarsening hierarchy and built the **node separator once**, at the
finest level. Minimum edge cut and minimum vertex separator are different
objectives, so every level optimised the wrong thing.

| crate | through uncoarsening | separator built |
|---|---|---|
| `feral-metis` (pre-fix) | `refine_bisection` on `{A, B}` | König min-cover + one greedy pass, finest level |
| `feral-scotch` | `halo_fm_refine` on `{A, B}` | `compute_vertex_separator` (two-sided FM), finest level |
| `feral-kahip` | `refine_bisection` + `flow_refine_bisection` on `{A, B}` | `flow_node_separator` (max-flow vertex cover), finest level |

**Scotch corroborates the diagnosis from the other direction.** Its
finest-level step is *better* than pre-fix metis's — a proper two-sided FM
optimising separator weight directly, against a König cover plus a
positive-gain-only greedy pass — and it still measured slightly *worse*
(3.36x behind fixed metis, against pre-fix metis's 3.11x). The quality of
the final polish is not what decides this. The hierarchy is.

## The fix

Both crates gain `node_refine: bool`, defaulting to `true`, mirroring
`feral-metis`. Build the separator at the **coarsest** level, then project
the 3-way labels down and refine the separator at every level with
`refine_separator_fm` (reused from `feral_metis::internals::fm_refine` —
scotch and kahip already depend on metis for the shared coarsening).

One deliberate asymmetry: **kahip keeps its flow lift at the coarsest level
only.** `flow_node_separator` is a max-flow vertex-cover reduction, far too
expensive to rerun per level; the FM pass down the hierarchy is what fixes
the objective. This turned out to make kahip's *analysis* cheaper as well,
since the max-flow now runs on the smallest graph in the hierarchy rather
than the largest.

## Results — gaslib nfe=48 (n = 224,646), elimination flops

| backend | before | after | gain |
|---|---|---|---|
| `metis` | 2.095e10 | 6.737e9 | 3.11x |
| `scotch` | 2.264e10 | 1.017e10 | 2.23x |
| `kahip` | 2.577e10 | **7.184e9** | **3.59x** |
| real METIS (oracle) | — | 6.423e9 | — |

`max_front` falls with it: scotch 2080 -> 1783, kahip 2145 -> 1517. KaHIP's
symbolic time roughly halves (5,263 ms -> 2,274 ms) from moving the max-flow
to the coarsest graph.

## Results — the pounce corpus, wall-clock

`issue203_corpus_ab`, steady-state factor time against `Amf`, geomean over
all matrices (ratio > 1 means the ND backend is faster):

| backend | before | after |
|---|---|---|
| `scotch` | 1.060 | 1.045 |
| `kahip` | 1.118 | 1.102 |

**Neutral**, within run-to-run noise, exactly as the metis fix was. Per
family the shape is the same as metis's: a large win on `laptime` (the
collocation family — scotch 2.65x, kahip 2.58x against `Amf`), a win on
`poisson` (1.48x / 1.22x), and significant losses on `clnlbeam` and
`sparseqp` that the fix neither causes nor cures.

One symbolic regression worth naming: scotch's `corkscrw` flop proxy goes
8.887e6 -> 1.007e7, **13% worse**. It does not reach the clock — measured
wall-clock on that family is 1.002 before and 1.017 after, i.e. unchanged to
slightly better. Fill mispredicted again, in the direction that would have
scared us off a good change.

## What this does and does not buy

It buys consistency: there is no longer a known ND deficiency sitting in a
release, which is why it was done despite the narrow audience.

It does not change any default path. `choose_adaptive` routes every
would-be-`MetisND` decision to `Amf` and issue #50 deleted the `ScotchND`
route outright, so neither backend is reachable from `Auto`. Only a caller
naming `ScotchND` / `KahipND` explicitly sees any of this — or a caller
using `Solver::with_ordering_race` with those arms.

## Reproduction

```sh
cargo run --release -p feral-diagnostics --bin issue203_fill_probe -- gaslib.mtx
cargo run --release -p feral-diagnostics --bin issue203_corpus_ab -- \
    data/matrices/kkt-pounce amf scotch 3 6
```

Flip `node_refine` in `crates/feral-{scotch,kahip}/src/lib.rs` to compare
arms. Corpus built by `scripts/build-pounce-corpus.sh`.
