# Does a measured race route per class? (issue #209)

**Date:** 2026-09-18
**Machine:** Apple M2, `Mac14,2`, macOS.
**Status:** measurement done; `choose_adaptive` **not** changed.

#209 observes that `OrderingMethod::Auto` never selects nested dissection —
the #67/#73 reroute sends every would-be-`MetisND` decision to `Amf` — and
that the evidence for that reroute was measured against the `MetisND` #203
has since fixed. It also states the constraint that matters: a blanket flip
back would trade the grid families for the collocation ones, so what is
needed is a *signal that separates them*.

Its first candidate direction names its own test:

> Opt-in `with_ordering_race`, ranking arms on the **second** factorization
> (steady state) … Would it pick `metis` on the three models above without
> regressing the grid QPs? That's the test.

## The answer: yes, on 29 of 31 matrices

`issue209_race_picks` times every fixed arm's steady state per matrix to
find the true best, then runs `Solver::with_ordering_race([Amd, MetisND])`
and reports what its pick costs relative to that best. The race is charged
nothing for setup here on purpose — the question is whether the *pick* is
right; setup cost is answered separately by `issue203_policy`.

| family | best arm | `amf` µs | `amd` µs | `metis` µs | race µs | race/best |
|---|---|---|---|---|---|---|
| gaslib (collocation) | **metis** | 337,643 | 340,053 | 131,421 | 132,657 | **1.008** |
| poisson450 | **metis** | 509,519 | 455,796 | 440,258 | 451,822 | 1.026 |
| poisson | **metis** | 74,605 | 66,367 | 64,741 | 66,452 | 1.026 |
| laptime (collocation) | **metis** | 84,993 | 68,724 | 42,685 | 53,354 | **1.198** |
| clnlbeam | amf | 17,540 | 17,550 | 21,945 | 17,523 | 1.004 |
| optcontrol | amf | 20,793 | 20,901 | 30,797 | 20,911 | 1.007 |
| sparseqp | amf | 23,781 | 24,486 | 33,103 | 24,135 | 1.018 |
| rosenbrock | amf | 204 | 202 | 209 | 199 | 1.001 |
| corkscrw | amd | 16,705 | 15,141 | 16,908 | 15,297 | 1.010 |
| bratu | amd | 1,857 | 1,651 | 1,711 | 1,566 | 0.976 |

**Geomean 1.030 against an oracle that always picks the best arm; within 5%
on 29 of 31 matrices.** It picks `metis` on the collocation and PDE-control
families and the local arm on the QP and thin families, which is exactly the
per-class behaviour #209 says is required — and it does it by measuring, so
it needs no predictor.

Two caveats, both real:

* **`laptime` is the weak spot, at 1.198.** The best arm there is `metis`
  (42,685 µs) and the race averages 53,354 µs, so on some of those iterates
  it kept `Amd`. `laptime` is the family with condition-3.4e20 iterates
  where timings are noisy and arms are close; the worst single matrix in the
  whole sweep is 1.936.
* **Setup is not free.** `issue203_policy` puts break-even at roughly 15
  factorizations of one pattern. Inside an IPM that is nothing (pounce runs
  50–400 iterates); for a one-shot solve it is a 2-3x loss.

## Why `choose_adaptive` is still not changed

The race answers "which arm" well. It does not answer "should `Auto` pay for
a race", and that is the actual routing question, because `Auto` is the
default path for callers who have expressed no opinion. Making the default
run `k` extra numeric factorizations on the first `factor()` of every pattern
is a large behavioural change to every consumer, including ones that factor
a pattern once.

The alternative — flipping the would-be-`MetisND` branch back — is what #209
itself rules out: `metis` is 5% slower geomean on the 138-problem
Maros-Meszaros QP set and loses 13-22% on `poisson`, `optcontrol` and
`sparseqp`. The table above reproduces that shape.

And no cheap predictor separates the two groups. Average degree, the obvious
candidate, does not: `metis` wins at 10.85 (`laptime`), 10.4 (`gaslib`) and
4.99 (`poisson`), and loses at 4.33 (`sparseqp`), 4.0 (`bratu`) and 3.0
(`optcontrol`) — `poisson` at 4.99 and `sparseqp` at 4.33 are on opposite
sides of the answer and adjacent in the metric. Fill is already ruled out
(#203's notes, five separate mispredictions).

## What this does change

The honest deliverable is guidance rather than routing:

1. `book/src/ordering.md` now states plainly that `Auto` selects no
   nested-dissection method, which it had claimed the opposite of.
2. A workload-shaped recommendation, backed by the table above: collocation
   / optimal-control / PDE-in-time should ask for `MetisND` by name, and a
   host that refactors one pattern many times and does not want to hardcode
   can use `Solver::with_ordering_race`.

## What would change the verdict

* A wall-clock A/B of the full #67/#73 corpus against the *fixed* `MetisND`
  — direction 3, and the one that would actually settle it. Nine families
  is not 41.
* A structural predictor that separates space-by-time products from thin
  chains and grids on an `O(nnz)` feature. Nobody has one; average degree is
  measured above not to be it.
* Evidence that the race's setup cost is acceptable as a *default*, which is
  a product decision as much as a measurement.
