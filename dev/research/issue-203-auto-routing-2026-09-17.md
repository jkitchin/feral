# Can `Auto` be taught to pick the new `MetisND`? Not yet, and here is why

**Date:** 2026-09-17
**Machine:** Apple M2, `Mac14,2`, 4P+4E, macOS.
**Status:** no code change. `choose_adaptive` is untouched.
**Context:** issue #203. Follows
`dev/research/feral-metis-node-separator-fm-2026-09-17.md`, which made
`MetisND` 2.4-3.5x better on collocation KKTs, and
`dev/research/issue-203-collocation-kkt-ordering-2026-09-17.md`.

`choose_adaptive` reroutes every would-be-`MetisND` decision to `Amf`
(issues #67/#73), so none of that improvement reaches the default path. This
note is the attempt to change that, and the reasons it stopped short.

## What the existing override actually rests on

Stronger than I had assumed before reading it.

* **#67**, `10_000 < n <= 100_000`: 36 of 54 families in the band are
  in-scope (MetisND-routed, non-arrow). **All 36 have `time_r >= 0.99`** —
  AMF wins or ties on factor+solve — with a median around 1.5x and a tail to
  4.5x. Critically, `dev/research/issue-67-thin-large-ordering.md:77` says
  the measurement ran "via the full `Solver` path (production parallel
  numeric)". There is no "they only measured the sequential path" loophole.
* **#73**, `n > 100_000`: five families on wall-clock, every one an AMF win —
  dtoc2 2.49x, pinene 1.18x, cont5_1_l 2.75x, nql180 2.05x, YATP1NE 2.13x.

Against that: three matrices, all from one reporter's model.

## The bridge that failed

The #67/#73 families are not on this machine. The plan was to build
geometry-matched stand-ins for the two whose structure is unambiguous —
`dtoc` (narrow discrete-time-control chain, matched to dtoc2's n=104k) and
`pde2d` (elliptic control on a grid, matched to cont5_1_l's n=181k) — and
re-run the comparison against the new `MetisND`.

**The proxies fail their own validity check.** Run against the *old* metis
(`node_refine: false`, byte-for-byte the code #67/#73 measured), production
parallel path, same paired protocol:

| proxy | old metis vs auto, here | #73 on the real family |
|---|---|---|
| `dtoc_wide`, n=103,992 | **1.129x, metis faster** | dtoc2: 2.49x, **AMF** faster |
| `pde2d`, n=181,548 | **1.037x, metis faster** | cont5_1_l: 2.75x, **AMF** faster |

The sign is wrong, not just the magnitude. A proxy that cannot reproduce the
result it exists to re-test says nothing about the new code, so the arm was
discarded. `external_benchmarks/chain_proxy/README.md` documents exactly this
failure mode; see `dev/tried-and-rejected.md` (2026-09-17).

As far as it was chased: `dtoc_wide`'s `max_front` is 31. At that width the
numeric phase is per-front overhead, not arithmetic, so the comparison
measures supernode count rather than flops. Widening `(nx, nu)` from `(4,2)`
to `(8,4)` moved `max_front` only 24 -> 31, so dtoc2's avg_deg of 17.5 is not
reachable by this construction at the right `n`.

## What the exercise did establish

On matrices that are not proxies — the issue #203 patterns with saddle-point
values.

### The 3.5x is two effects, and only half of it was predictable

nfe=48, `RAYON_NUM_THREADS=1` against the default parallel path, min over
pairs:

| arm | sequential factor | parallel factor | parallel speedup |
|---|---|---|---|
| `auto` (= `Amf`) | 1124 ms | 698 ms | 1.61x |
| `MetisND` | 694 ms | 198 ms | 3.51x |

`MetisND` is **1.62x faster sequentially** — close to its 1.37x `flop_proxy`
edge, i.e. genuine arithmetic — and a **further 2.2x** comes from the
ordering parallelising better. `max_front` says why: 2288 for `Amf` against
1293 for `MetisND`. Wide fronts serialise.

So the 3.52x headline decomposes as roughly 1.6x arithmetic times 2.2x
scheduling, and **only the first factor is visible in symbolic data at all**.

### No cheap guard metric survives

Every candidate, checked against all six matrices measured today plus #73's
`RDW2D51U` row:

* **`nnz_L`** — at nfe=48 `Amf` and `MetisND` are within 1.5% (17.64M vs
  17.37M) while wall-clock differs by 3.52x. Already rejected in 2026-05 for
  the fill-guarded race; this is a second, sharper counterexample.
* **`flop_proxy`** (`sum ncol * nrow^2`) — predicts 1.37x where the measured
  parallel answer is 3.52x, and has the **sign wrong** at nfe=24 (predicts
  `Amf` by 1.67x; `MetisND` measured 1.12x faster).
* **`max_front`** — picks the winner on 3 of 6.

A routing rule needs a predicate. There isn't one in the symbolic data,
because the dominant term at nfe=48 is a scheduling effect that no symbolic
quantity measured here captures.

## Verdict

**`choose_adaptive` is not changed.** Three matrices from one model against
41 real families on the production parallel path is not the evidence bar this
repo holds itself to, the proxies meant to bridge that gap are invalid, and
no cheap predicate survives contact with the data.

## Two ways forward, in order of preference

1. **Re-run the #67/#73 A/B on the real corpus against the new metis.** This
   is now more clearly required than it was this morning: those conclusions
   were measured against a `feral-metis` that has since improved 2.4-3.5x on
   at least one class, so the evidence base is stale for every class, not
   only for issue #203's. The probes exist (`probe_issue67_thin`,
   `probe_issue73_symbolic`) and `issue203_ab` will run any single matrix
   with a cross-arm correctness gate.

2. **Measure instead of guessing.** A one-time race that runs both symbolic
   analyses *and* one numeric factorization per arm, keeps the winner, and
   amortises over the ~100 IPM iterates that reuse the symbolic. This inverts
   the economics the 2026-05 fill-guarded race was rejected on: that race was
   priced as per-solve overhead and guarded on a metric now shown to be
   anti-correlated with speed, whereas a numeric race needs no predictor at
   all and pounce already reuses the symbolic across iterates
   (`pounce-feral/src/lib.rs:17`). It is a `Solver`-level API change, not a
   `choose_adaptive` tweak, and it needs its own plan.

There is also a loose end worth its own look, independent of METIS:
**`Amd` beat `Auto`/`Amf` by 1.63x** at nfe=96 and by 1.15x on the `pde2d`
proxy, in both cases while carrying *more* fill. `Auto`'s preference for
`Amf` over `Amd` at large `n` has not been examined on this class.
