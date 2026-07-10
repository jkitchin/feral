# Issue #131 Gap B (parallel assembly) — measure-first — 2026-07-10

**Verdict: not justified.** Per-front assembly is a single-digit percent of the
factor, and — the decisive point — in the parallel driver the assembly of
independent fronts **already overlaps across threads** (each front's assembly is
part of its own tree task). The only assembly left on the critical path is the
root / near-root fronts', which is `O(nrow²)` behind the root's `O(nrow³)` dense
factor that intra-front parallelism (Lever 1.1) already targets. Column-
partitioned parallel assembly would chase <1–3% of the factor. Recorded per the
measure-first discipline (cf. #129/#130/#132/#133).

## Measurement (`probe_panel_frag`, phase_timing, sequential driver)

Assembly time and its split vs the dense factor it feeds:

```
grid220  (n=48400, bushy):   assembly 14.0 ms = 8.3% of assembly+densefactor
                             (buildrow 1.0 | scatter 1.3 | extendadd 7.5 ms)
                             densefactor 155.7 ms (schur 43.5% | panel 44.3%)
dense1400 (single front):    assembly  7.8 ms = 1.5% of assembly+densefactor
                             (buildrow 0.0 | scatter 3.2 | extendadd 0.0 ms)
                             densefactor 502.7 ms (schur 72.1% | panel 26.5%)
```

`extendadd` is 0 on dense1400 because a single front has no children; its
assembly is pure original-entry scatter. On grid220 the 7.5 ms of extend-add is
summed over **all 48 400 fronts** of a bushy tree.

## Why the parallel driver already hides most of it

The `probe_panel_frag` numbers are the *sequential* sum. The production driver
for these matrices is the rayon tree-parallel factor. There, each supernode's
assembly (build_row_indices + scatter + extend-add of its children) runs inside
that supernode's task, on whatever worker picks it up. Sibling fronts assemble
concurrently. So the *serial* assembly on the critical path is only the
root-path fronts' — dominated by the root, whose extend-add is `O(cdim²)` while
its dense factor is `O(cdim³)`. That root factor is exactly what Lever 1.1's
intra-front `par_chunks_mut` already parallelises. Column-partitioned assembly
would parallelise the root's `O(cdim²)` assembly — one part in `cdim` of the
work already being parallelised, i.e. deep in the noise.

## Interaction with #125

#125 (just landed) already removed the `build_row_indices` component of assembly
on the no-delay fast path (grid220 buildrow is now ~1.0 ms, down from the
pre-#125 scan+sort), for a measured ~8% per-supernode-loop / ~4.5% total warm
factor win on grid220 — the tractable, bit-exact part of the assembly cost.
What remains (scatter + extend-add) is the second-order part Gap B would target.

## Decision

Do not build column-partitioned parallel assembly. The bit-exact-safe machinery
(disjoint-destination-column partition, loop restructure) is real work for a
<1–3% ceiling that the parallel tree schedule already mostly captures. Re-open
only if a workload appears whose factor is dominated by a **single enormous
root front with many children** (so its serial extend-add is a large absolute
cost) AND the root dense factor is already fully thread-saturated — a narrow
corner not present in the current corpus.

Effort redirects to Gap A (the solve rewrite), the high-value gap: the solve is
100% serial today and IPM hosts run several solves per factor.
