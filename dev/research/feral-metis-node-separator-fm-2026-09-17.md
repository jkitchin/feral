# feral-metis: refining the node separator, not the edge cut

**Date:** 2026-09-17
**Machine:** Apple M2, `Mac14,2`, macOS. CoinHSL v2023.11.17 (MA57 as the
real-METIS oracle).
**Context:** issue #203; see
`dev/research/issue-203-collocation-kkt-ordering-2026-09-17.md` for how the
question arrived here.

## The finding this note acts on

`feral-metis` produces nested-dissection orderings roughly 1.4x worse than
real METIS on canonical grid graphs and up to 3.3x worse on the collocation
KKT of issue #203. Measured as `sum_j (c_j-1)^2` over the column counts of L,
every permutation replayed through feral's own symbolic pipeline:

| graph | n | feral `Amf` | feral `MetisND` | real METIS |
|---|---|---|---|---|
| grid2d 300x300 | 90,000 | 3.02e8 | 4.69e8 | 3.33e8 |
| grid3d 40^3 | 64,000 | 2.33e10 | 2.31e10 | **1.67e10** |
| gaslib nfe=48 | 224,646 | 1.09e10 | 2.10e10 | **6.42e9** |
| gaslib nfe=96 | 449,286 | 5.06e10 | 6.89e10 | **1.75e10** |

It is a general deficit, not a pattern-specific one. Three candidate causes
were eliminated by measurement before this note:

* **Not the top-level separator.** Recovered from the permutation alone by
  `issue203_sep_probe`: feral 577 vertices against real METIS's 481 at
  nfe=48 — 1.20x, against a 3.26x flop gap.
* **Not tuning.** A sweep of `niparts`, `fm_passes`, `nd_to_amd_switch`,
  `coarsen_floor` and `max_imbalance` moves the result by at most 10%, and
  every arm is worse than the default on at least one metric.
* **Not graph compression.** Hashing each vertex's closed neighbourhood on
  the nfe=48 pattern gives 224,646 distinct classes out of 224,646 vertices,
  so METIS's `CompressGraph` would be a no-op here.

What the flop histogram shows instead (nfe=48, bucketed by column count):

| bucket | feral `Amf` | real METIS |
|---|---|---|
| 512..1024 | 3,890 cols, 20% of flops | 6,116 cols, 58% |
| 1024..2048 | 2,444 cols, 52% | 1,112 cols, 22% |
| >= 2048 | 447 cols, 18% | **0 cols, 0%** |

Real METIS caps its front width; feral does not. That is mid-tree separator
quality, and it points at the refinement, not the cut selection.

## The algorithmic difference

`crates/feral-metis/src/node_nd.rs::multilevel_node_bisection` does:

1. coarsen;
2. `niparts` initial **edge** bisections at the coarsest level, each followed
   by `refine_bisection` (edge-cut FM), keep the best cut;
3. uncoarsen, projecting the 2-way labels and running `refine_bisection`
   (edge-cut FM) at every level;
4. `construct_separator` — König minimum vertex cover of the boundary — once,
   at the finest level;
5. one `refine_separator` pass.

METIS 5.x (`libmetis/ometis.c`, `libmetis/refine.c`, `libmetis/sfm.c`) does:

1. coarsen;
2. `InitSeparator` at the **coarsest** level: edge bisection, then
   `ConstructSeparator` — so a *node separator* exists from the coarsest
   level onward;
3. `Refine2WayNode`: at every uncoarsening level, `Project2WayNodePartition`
   projects the 3-way labels down, `FM_2WayNodeBalance` restores balance, and
   `FM_2WayNodeRefine1Sided` (the NodeND default `rtype`) runs `ctrl->niter`
   passes of Fiduccia-Mattheyses **on the separator itself**.

Two gaps follow, and they compound:

**(a) Wrong objective during uncoarsening.** feral minimises the edge cut at
every level and only converts to a vertex separator at the end. Minimum edge
cut and minimum vertex separator are different objectives; the König cover of
a min-edge-cut bisection is a *bound*, not a minimiser, and on graphs whose
boundary vertices have unequal degrees the two diverge. Every level of
uncoarsening in feral optimises the wrong thing.

**(b) The one separator pass is greedy.** `fm_refine.rs:214`'s
`refine_separator` accepts a move only when `best_gain > 0`, with no
priority queue, no negative-gain moves and no rollback. It stops at the first
local minimum, which is exactly why `fm_passes=200` produces bit-identical
output to `fm_passes=10` in the sweep above. It is a hill-climber, not FM.

## The gain model to implement

State: `where[v] ∈ {A, B, SEP}`, part weights `pwgts[A]`, `pwgts[B]`, and for
each separator vertex `v` the two **external degrees**

```
edeg[v][A] = sum of vwgt[u] over neighbours u of v with where[u] == A
edeg[v][B] = likewise for B
```

Moving `v` out of the separator into part `to` forces every neighbour of `v`
in the *other* part into the separator. So

```
gain(v -> to) = vwgt[v] - edeg[v][other]
```

and the separator weight changes by `-gain`. A move is legal when
`pwgts[to] + vwgt[v] <= max_side`, with
`max_side = (1 + max_imbalance) * total / 2`.

Applying `v -> to`:

* `where[v] = to`; `pwgts[to] += vwgt[v]`; `sep_weight -= vwgt[v]`;
* for each neighbour `u` with `where[u] == other`: `where[u] = SEP`;
  `pwgts[other] -= vwgt[u]`; `sep_weight += vwgt[u]`; compute `edeg[u][*]`
  from scratch;
* for each neighbour `u` with `where[u] == SEP`: `edeg[u][to] += vwgt[v]`;
* every touched separator vertex has its key updated in the queue.

**One-sided passes.** METIS's default alternates the target side between
passes: a pass moves vertices only into `A`, the next only into `B`. That
keeps a single priority queue and is what `FM_2WayNodeRefine1Sided` does.
Two-sided uses two queues and picks the better of the two tops; it is
slightly better and noticeably slower. Start one-sided.

**Hill climbing.** Within a pass, keep extracting the best-gain vertex even
when the gain is negative, recording every move. Track the best
`(sep_weight, balance)` seen; stop the pass after `limit` consecutive moves
that fail to improve on the best, then **undo back to the best state**.
METIS uses `limit = min(3*nbnd, 300)` for the uncompressed case. Without the
rollback this is strictly worse than the current greedy, so the undo log is
not optional.

## Result (measured after implementing)

`refine_separator_fm` + the hierarchy restructure, measured the same way:

| graph | metis before | metis `node_refine` | real METIS | gain | still behind |
|---|---|---|---|---|---|
| gaslib nfe=6 | 4.25e8 | 1.23e8 | 8.99e7 | 3.45x | 1.37x |
| nfe=12 | 1.95e9 | 6.01e8 | 4.18e8 | 3.24x | 1.44x |
| nfe=24 | 7.71e9 | 3.05e9 | 1.72e9 | 2.52x | 1.77x |
| nfe=48 | 2.10e10 | **6.74e9** | 6.42e9 | 3.11x | **1.05x** |
| nfe=96 | 6.89e10 | 2.91e10 | 1.75e10 | 2.37x | 1.67x |
| grid2d 300x300 | 4.69e8 | 4.27e8 | 3.33e8 | 1.10x | 1.28x |
| grid3d 40^3 | 2.31e10 | 2.18e10 | 1.67e10 | 1.06x | 1.31x |

The prediction below was half right, and the half it got wrong is the
interesting one: the **KKTs moved far more than the grids**. At nfe=48 feral
is within 5% of real METIS and its top separator matches exactly (481 = 481);
the grids gained only 6-10% and are still 1.3x behind. So on grid Laplacians
something *other* than the refinement structure is giving away ~30% — the
next suspects, in order, are the coarsening (SHEM matching quality) and
METIS's two-level `MlevelNodeBisectionL2` coarsening schedule, neither of
which this note touched.

**Against the plan's acceptance table, 1 of 4 targets was met** (nfe=48 beat
its target; nfe=96, grid2d and grid3d all missed). The targets were set at
"match real METIS", which was the wrong bar for a gate on a change that is
never worse. The default was flipped anyway; see `dev/decisions.md`
(2026-09-17).

Guardrails: 37 of 38 `tests/data/parity` matrices produce a bit-identical
ordering, since they sit below `nd_to_amd_switch = 200` and never reach the
multilevel path; `tests/data/large` moves `nnz_L` 8700 -> 8707 (+0.08%);
`MetisND` symbolic time at nfe=96 goes 6.31 s -> 7.30 s (1.16x, guardrail
1.5x); determinism is covered by tests in both `fm_refine.rs` and
`node_nd.rs`.

**This does not reach `Auto`.** `choose_adaptive` reroutes every
would-be-`MetisND` decision to `Amf`, so the default path is unchanged and
still leaves 1.74x on the table at nfe=96. That override rests on wall-clock
evidence (#67/#73) and is not re-opened here; see the session checkpoint.

## Expected effect, and what would falsify it

If (a) and (b) are the whole story, `feral-metis` should reach real METIS's
flop counts within ~10% on the grids — `grid3d_40` from 2.31e10 to about
1.7e10 — and close most of the gap on the collocation KKTs. If it moves the
grids but not the KKTs, something else is specific to the KKT pattern
(the most likely candidate then being the `-dc I` dual block's effect on
boundary degrees) and this note is only half the answer. If it moves neither,
the diagnosis is wrong and the next suspect is the coarsening (SHEM matching
quality), which this note does not touch.

`grid3d_40` is the cheap canonical gate: n = 64,000, ~0.3 s per ordering,
2.31e10 now, 1.67e10 to match.

## Risks

* **Every METIS ordering in the repo changes.** `MetisND` is reachable from
  `Auto` via `pick_default_method`, so corpus fill numbers move. The change
  must be measured on the parity corpus, not only on the three graphs above.
* **Cost.** FM with an undo log at every uncoarsening level is more work than
  one greedy pass at the finest level. METIS pays it, but feral's symbolic
  time for `MetisND` is already 6.3 s at nfe=96 against 0.9 s for `Amf`; if
  the refinement doubles that, `pick_default_method`'s cost model needs a
  look. Measure symbolic time alongside fill.
* **Non-determinism.** The queue's tie-breaking must be deterministic or the
  ordering stops being reproducible, which the crate's contract requires
  (`seed` defaults to 1 and two runs must agree).

## References

- Karypis & Kumar (1998), *A Fast and High Quality Multilevel Scheme for
  Partitioning Irregular Graphs*, SIAM J. Sci. Comput. 20(1).
- Fiduccia & Mattheyses (1982), *A linear-time heuristic for improving
  network partitions*, DAC.
- METIS 5.2.0 sources: `libmetis/ometis.c` (`MlevelNodeBisectionL1/L2`),
  `libmetis/refine.c` (`Refine2WayNode`, `Project2WayNodePartition`),
  `libmetis/sfm.c` (`FM_2WayNodeRefine1Sided`, `FM_2WayNodeRefine2Sided`,
  `FM_2WayNodeBalance`), `libmetis/separator.c` (`ConstructSeparator`).

Full BibTeX in `dev/references.bib`.
