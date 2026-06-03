# Research Note: Thin-large ordering routing (issue #67)

**Status:** Decided — bounded reroute shipped (`AMF_BAND_MAX = 100_000`)
**Date:** 2026-06-03
**Author:** agent session 2026-06-03-04
**Related issues:** https://github.com/jkitchin/feral/issues/67
(found during #64, PR #66)
**Related code:** `src/symbolic/mod.rs` — `pick_default_method` (size-only
base rule), `choose_adaptive` (pattern-aware layer), `is_arrow_bordered`
(#64 catch).
**Prior art / precedent:** `dev/research/issue-50-metisnd-symbolic-cost.md`
(very-large-and-sparse → AMD; the powerflow-class guard this note must not
regress), `dev/research/issue-64-arrow-bordered-ordering.md` (the arrow
catch; #67 is the *non-arrow* residue of the same probe).

## Overview

`choose_adaptive` (the `OrderingMethod::Auto` resolver) routes:

```text
n == 0                      → Amd
n > 100_000 && avg_deg < 5  → Amd        (#50, powerflow-class)
n <= 10_000                 → Amf        (pick_default_method)
else (n > 10_000)           → MetisND    ← the rule #67 questions
MetisND && is_arrow_bordered → Amf       (#64)
```

The #67 fix inserts one more catch after #64's, raising the AMF band ceiling:

```text
MetisND && n <= 100_000      → Amf       (#67, AMF_BAND_MAX) ← new
```

Issue #67 is the **non-arrow** residue of the #64 calibration probe: on
*uniformly-thin* large matrices (no dense border, `is_arrow_bordered`
correctly does not fire, `heavy_count = 0`), the `n > 10_000 → MetisND`
default still loses to AMF on fill — bratu3d (n=27792) 1.59×, cont-201
(n=80595) 1.32×. These are 3-D-PDE-like discretization patterns where
nested dissection is "supposed" to win, yet AMF produces a smaller factor.

The issue sets a deliberately high evidence bar, for three reasons it
states explicitly:

1. **No structural signature to key on.** The degree distribution is flat
   (every column ≤ 7 on bratu3d/cont-201), so there is no arrow/border
   fingerprint like #64's. Any predicate must key on *low average degree*
   — the same axis #50 warned is dangerous.
2. **nnz_L is not the whole story.** MetisND trades fill for a shorter
   critical path / better parallelism; a fill-only comparison can be
   misleading. The decision must weigh **numeric factor + solve
   wall-time**, not nnz_L alone.
3. **The size rule is load-bearing.** `n > 10_000 → MetisND` protects
   genuinely large 3-D problems, and #50 showed how easily a broad
   low-avg-degree reroute regresses the corpus (it had to *delete* two
   such catches). Any change needs a corpus-wide A/B, not a two-matrix
   anecdote, and must be verified to not regress the powerflow-class.

## Scope of the in-scope population

The matrices #67 is about are exactly those that `choose_adaptive`
resolves to **MetisND** and that are **not** arrows:

- `n` in `(10_000, 100_000]` (any avg_deg), **or**
- `n > 100_000` with `avg_deg >= 5` (the `<5` ones already route to AMD).

The corpus inventory (`data/matrices/kkt-expansion`,
`data/matrices/kkt-mittelmann`, `tests/data/large`) yields 71 families
with n > 10_000; the n in (10k,100k] band has 54 representatives (one
`_0000` per family). The very-large-and-sparse families (BDRY2 n=501k,
PDE1/PDE2, cont5, nql180, QUADCOPTER, YATP*) all have avg_deg < 5 and
route to AMD — **not** in scope; forcing MetisND on them is both
out-of-scope and pathologically slow (the #50 powerflow lesson).

## Measurement

`src/bin/probe_issue67_thin.rs` factors each matrix under `Auto`, `Amf`,
and `MetisND` via the full `Solver` path (production parallel numeric,
Auto scaling), recording post-pivot `factor_nnz()` and median factor /
solve wall-time (RHS = ones). The `Auto` row records `resolved_method`,
so MetisND-routed non-arrow matrices are isolated as the in-scope set.
Inertia agreement between the two orderings is asserted as a sanity check.

### Results

**Band sweep (10k < n ≤ 100k), `probe_issue67_thin --reps 3`.** Of the 54
families in the band, 36 resolve to MetisND under `Auto` and are not arrows
(the in-scope #67 population; the other 18 are arrows already caught by #64
or otherwise route to AMF). `time_r = (MetisND factor+solve) / (AMF
factor+solve)`, so `> 1` means **AMF is faster**; `fill_r = nnz_L(MetisND) /
nnz_L(AMF)`, `> 1` means AMF's factor is smaller.

| matrix       |      n | avg_deg | time_r | fill_r | note                |
|--------------|-------:|--------:|-------:|-------:|---------------------|
| OSCIGRAD     |   ~15k |    flat |   4.48 |  large | biggest AMF win     |
| TABLE3       |        |         |   3.97 |        |                     |
| svanberg     |        |         |   2.88 |        |                     |
| ex1_160      |        |         |   2.56 |        |                     |
| CHAINWOONE   |        |         |   2.46 |        |                     |
| cont-201     |  80595 |    5.44 |   2.13 |   1.32 | issue exemplar      |
| bratu3d      |  27792 |    6.25 |   1.82 |   1.59 | issue exemplar      |
| …median…     |        |         | ~1.5   |        |                     |
| qcqp1500     |        |         |   1.01 |   ~1   | near-tie            |
| clnlbeam     |        |         |   0.99 |   ~1   | worst case (noise)  |

**All 36/36 in-scope matrices have `time_r ≥ 0.99`** — AMF wins or ties
MetisND on factor+solve wall-time across the entire band. The worst case
(clnlbeam 0.99) is within run-to-run noise; the median is ~1.5× and the tail
reaches 4.5×. `fill_r ≥ 1` everywhere (AMF's factor is never materially
larger). The issue's hypothesis — "MetisND trades fill for a shorter
critical path / better parallelism" — **never materialized** on this band:
MetisND is both larger *and* slower.

**Large guard (n > 100k, avg_deg ≥ 5), `--reps 1`.** Spot checks above the
band, to size how far the win extends and whether the `n > 100_000` boundary
is defensible:

| matrix         |      n | avg_deg | time_r | fill_r | outcome                       |
|----------------|-------:|--------:|-------:|-------:|-------------------------------|
| pinene_3200    | 127995 |    9.42 |   1.18 |   1.20 | AMF still wins just above band |
| RDW2D51U       | 195075 |     ~10 |    —   |    —   | did **not** complete in ~10min |

pinene (n≈128k) still favors AMF, but RDW2D51U (n≈195k) ran >10 min on a
single Auto+AMF+MetisND pass and was killed — the n > 100k regime is
qualitatively more expensive and under-sampled. There is not enough evidence
to reroute it, and #50's powerflow lesson warns against broad low-avg-degree
reroutes, so the band is bounded at `n ≤ 100_000`. pinene's win is left on
the table deliberately as the safety margin.

**AutoRace overhead, `probe_issue67_race --reps 3`.** An
`AutoRace(Amf, MetisND)` would run *both* symbolic analyses, keep the
smaller-fill one (always AMF here), then factor+solve once. Its overhead over
the bounded threshold is exactly MetisND's wasted symbolic time, as a
fraction of the bounded total `t_sym_amf + t_num_amf + t_solve_amf`:

| matrix    | race_ovh% |
|-----------|----------:|
| svanberg  |       255 |
| WOODSNE   |       184 |
| ex1_160   |       151 |
| cont-201  |       118 |
| arki0009  |        89 |
| HIER133A  |        58 |
| bratu3d   |        17 |

Median ~118%, up to 255% — MetisND's nested-dissection ordering is 2–5×
more expensive than AMF's, so racing it on every band matrix more than
*doubles* the analysis-dominated cases for zero benefit (AMF always wins the
fill race anyway). AutoRace is the wrong tool here: it pays the expensive
loser's cost on every solve.

**Decision: bounded reroute.** Raise the non-arrow AMF band to
`n ≤ AMF_BAND_MAX = 100_000` in `choose_adaptive`. This is a static `n`
threshold, not an avg-degree predicate (avoiding the #50 hazard) and not a
race (avoiding the AutoRace overhead). The change touches *only* the
would-be-MetisND decision in `(10_000, 100_000]`; the `n > 100_000 &&
avg_deg < 5 → Amd` (#50) and `n > 100_000 && avg_deg ≥ 5 → MetisND` paths
are untouched. The threshold is `n`, not the measured-sample identity:
every band matrix the corpus contains landed on the AMF side, and the
mechanism (AMF's lower fill + cheaper symbolic on thin patterns at this
scale) is size-bounded, not matrix-specific — so the rule generalizes to
unseen band matrices of the same character rather than memorizing these 36.

## Decision criteria

The change ships only if the corpus A/B shows AMF **does not lose on
factor+solve wall-time** across the in-scope MetisND population (not just
on fill), with no regression on the large-3-D / powerflow-class. If
MetisND wins on time anywhere meaningful despite higher fill, a blanket
reroute is wrong and the answer is either a narrower predicate or an
`AutoRace`-style nnz_L+time race for the borderline band — or no change.

## Plan

1. ✅ Band sweep (10k < n ≤ 100k) at reps=3 — fill + time A/B. 36/36
   in-scope favor AMF (`time_r ≥ 0.99`).
2. ✅ Targeted n>100k, avg≥5 cases — pinene_3200 favors AMF; RDW2D51U
   did not complete (band left bounded at n ≤ 100k).
3. ✅ Characterized: thin discretization patterns at this scale favor AMF
   on *both* fill and time; the parallelism trade-off never materialized.
4. ✅ Decided: bounded `n ≤ 100_000` reroute (not predicate — #50 hazard;
   not AutoRace — 50–255% overhead). Implemented tests-first.
5. ✅ No powerflow-class regression: the `n > 100_000 && avg_deg < 5 → Amd`
   catch and the `avg_deg ≥ 5 → MetisND` regime are untouched; routing
   unit tests (`choose_adaptive_rules` n=150_000 → MetisND) pin this.
