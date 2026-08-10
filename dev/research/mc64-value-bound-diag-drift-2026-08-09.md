# The MC64 value-bound gate rejects at zero drift

**Date:** 2026-08-09
**Machine:** Apple M4 Pro (`Mac16,11`), 10P+4E, macOS. Sequential solver.
**Status:** Measured. Proposes a narrow fix to condition 3.

## The defect

`mc64_value_bound_passes` (`src/scaling/value_bound.rs:224`) decides whether a
cached MC64 scaling `D` may be reused on a new matrix. Three conditions:

```rust
let cond_ratio = stats.max_ratio    <= GROWTH_FACTOR * validity.r0;
let cond_count = n_off_dominant     <= GROWTH_COUNT  * validity.n_off_dominant_0;
let cond_diag  = stats.min_diag     >= EPS_DIAG      * validity.mean_diag_0;
```

Conditions 1 and 2 are *drift* measures: current statistic against the same
statistic on the baseline matrix. Condition 3 is not. It compares the current
**min** scaled diagonal against the baseline **mean**. Those are different
statistics, so `cond_diag` is an absolute property of the matrix — "does the
scaled diagonal have dynamic range wider than `1/EPS_DIAG`" — not a measure of
how far the matrix has moved.

The consequence is that a matrix can fail the gate **against itself**. Feeding
the baseline matrix back in gives `max_ratio == r0` and `n_off_dominant ==
n_off_dominant_0` exactly (conditions 1 and 2 pass by construction), yet
`cond_diag` still fails whenever `min_diag < 1e-12 * mean_diag`.

Instrumented run on `robot_1600_0001` (`FORCE_MC64`, temporary
`FERAL_VB_DEBUG` print, reverted):

```
vb: ratio true (6.111443e14 vs 1.222289e15) | count true (15410 vs 15410) | diag false
    min_diag 1.636275e-15, mean_diag 4.177246e-1  ->  threshold 4.177246e-13
```

Zero drift on the two drift conditions; condition 3 rejects anyway.

## How often it actually matters

`FERAL_VB_DEBUG` instrumentation on `precompute_mc64_validity` and
`mc64_value_bound_passes`, driven through `diag_trajectory_scaling` (one
`Solver`, iterates in order, as an IPM caller would). 24 families attempted;
`min_diag_0` recovered by pairing each `vb-chk` with the preceding `vb-pre`.

**Most of the corpus never reaches this gate.** 17 of the 24 families ran MC64
zero times — `pick_scaling_strategy` routes them to InfNorm, so no cache, no
gate. That includes every large-scaling% family from the phase profile:
`clnlbeam`, `dtoc2`, `rocket_12800`, `steering_12800`. Their scaling cost is
InfNorm, which this lever cannot touch.

Seven families do route to MC64, giving **53 gate evaluations**. Condition 3 is
the *sole* blocker on **three**:

| family | min_diag | min_diag_0 | drift | verdict |
|---|---|---|---|---|
| robot_1600 | 2.335e-15 | 2.335e-15 | 1.000 | false positive — zero drift |
| robot_1600 | 7.572e-15 | 7.667e-15 | 0.988 | false positive — zero drift |
| arki0003 | 1.984e-13 | 9.372e-06 | 2.1e-08 | **genuine collapse** |

The `arki0003` case is the control: an eight-order collapse of the minimum
scaled diagonal, which any correct form of condition 3 must reject. The two
`robot_1600` cases are pure false positives — the minimum scaled diagonal did
not move.

The remaining rejections come from conditions 1 and 2 and are untouched by this
note. `pinene_3200` rejects on condition 1 on all 8 of its checks, exactly as
`tried-and-rejected.md:2087` (2026-05-21) recorded.

## Proposed fix

Store `min_diag_0` in `Mc64CacheValidity` and make condition 3 pass if *either*
the existing absolute floor holds *or* the minimum has not drifted down:

```rust
let cond_diag = stats.min_diag >= EPS_DIAG * validity.mean_diag_0
             || stats.min_diag >= DIAG_SHRINK * validity.min_diag_0;
```

This is a strict widening of the accept set, and it is zero-drift-safe by
construction: re-checking the baseline matrix gives `min_diag == min_diag_0`
exactly, so the second clause holds for any `DIAG_SHRINK <= 1`.

`DIAG_SHRINK = 1.0 / GROWTH_FACTOR = 0.5` — symmetric with the existing
constant: the minimum diagonal may shrink by the same factor the worst
dominance ratio may grow.

**The threshold is not load-bearing.** Every value swept from 0.5 down to 1e-6
produces the same 25 accepts out of 53. The two `robot_1600` false positives
sit at drift 0.988 and 1.000; the `arki0003` genuine collapse sits at 2.1e-08.
Nothing in the corpus lies between. 0.5 is chosen because it is the tightest
defensible value, not because the data forces it.

## What it buys

`robot_1600` only, and only iterates 0004 and 0005:

```
trajectory total: factor 62084 us, scaling 20533 us (33.1%)
recovered:        4951 + 3605 = 8556 us  =  13.8% of the trajectory
```

No other family's gate decision changes at any swept threshold.

**This is much smaller than the 32.7% figure that motivated the work.** That
number was the total scaling share of the `robot_1600` trajectory, which
implicitly assumed the gate could be made to hit on all seven iterates. It
cannot: two of the four checks fail on conditions 1 and 2, which are working as
designed, and two iterates change pattern and never reach the gate.

## Verified after implementation

Same seven families, `diag_trajectory_scaling` built from the pre-fix and
post-fix sources, run back to back.

Hit-pattern diff across all 53 gate evaluations — the complete set of changes:

```
< robot_1600_0004 NO        > robot_1600_0004 yes
< robot_1600_0005 NO        > robot_1600_0005 yes
```

Nothing else moved. Inertia is byte-identical on every iterate of every family
(`14399/9601/0` on `robot_1600`, `38415/38392/0` on `marine_1600`, and so on).

`robot_1600` trajectory:

| | before | after |
|---|---|---|
| factor | 63,560 us | 54,445 us |
| scaling | 20,721 us (32.6%) | 12,479 us (22.9%) |

14.3% off the trajectory, 39.8% off its scaling — the 13.8% predicted from the
gate log, within run-to-run noise. Every other family's totals move by less
than the noise floor.

## Safety

Fresh-MC64-vs-reuse-`D0` comparison (`diag_scaling_reuse_correctness`), which is
*more* aggressive than this fix — it carries iterate 0's scaling across the
whole trajectory unconditionally, ignoring the gate entirely. 9 families, 32
iterates:

* inertia identical on **32 of 32**, no exceptions;
* refined residuals match within ~2x (worst `steering_12800_0002`, 1.2e-7 reuse
  vs 2.0e-9 fresh; `steering_12800_0001` reuse is *better*, 6.1e-9 vs 1.6e-7);
* **unrefined** residuals degrade materially — `marine_1600_0002` 7.3e-3 reuse
  vs 1.5e-6 fresh, ~4900x.

The unrefined degradation is a property of scaling reuse in general, not of this
fix, and it is already accepted by the existing gate on the 23 checks that pass
today. The gate exists to protect inertia (issue #38), and inertia is preserved
on every iterate measured under a far more aggressive policy than the one
proposed.

## Relation to the 2026-05-21 rejection

`tried-and-rejected.md:2087` rejected Track B2 because (a) the gate rejected
every warm iteration on `pinene_3200` via **condition 1**, confounded by the IPM
barrier trajectory, and (b) B2 targeted <2% of `pinene`'s cost.

This note does not revisit either finding. It addresses **condition 3**, leaves
condition 1 alone, and confirms `pinene_3200` still rejects 8/8 on condition 1.
The recorded lesson — "validate the cost model before building the
optimization... a per-factor profile of the *named target's full iteration
sequence*, not a single iteration, should precede the plan" — is why the 13.8%
figure above is a trajectory number and why the corrected sizing appears here
rather than after implementation.

## The bigger targets this measurement surfaced

**Condition 1, on the families where scaling actually dominates.** The
trajectory totals are lopsided:

| family | scaling share of trajectory | benefit from this fix |
|---|---|---|
| nql180 | 66.2% (11.2 s of 17.0 s) | none |
| pinene_3200 | 79.4% (1.19 s of 1.50 s) | none |
| marine_1600 | 50.6% | none |
| robot_1600 | 32.6% -> 22.9% | 14.3% |
| dtoc1nd | 10.4% | none |
| arki0003 | 3.5% | none |
| qcqp1000-2c | 0.5% | none |

`nql180` and `pinene_3200` spend two thirds to four fifths of their
factorization time in scaling and this fix does nothing for either — both
reject on **condition 1**, the growth-factor bound. That is precisely the
condition `tried-and-rejected.md:2087` found confounded by the IPM barrier
trajectory on `pinene_3200`. The prize is an order of magnitude larger than
what condition 3 was worth, and the 2026-05-21 entry says why it is hard.
Anything that revisits condition 1 must start from that entry.

**InfNorm, for the other 17 families.** They never run MC64 at all —
`pick_scaling_strategy` routes them to InfNorm, so there is no cache and no
gate. Their prologue scaling cost is 8.0% to 29.3% of numeric factorization in
the phase profile, largest on `rocket_12800`. No MC64 cache work can touch it.
