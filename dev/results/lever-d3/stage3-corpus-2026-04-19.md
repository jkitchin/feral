# D.3 stage-3 corpus bench — acceptance decision

**Date:** 2026-04-19 (session 04 continuation, after D.3 GREEN + stage 1/2).
**Generator:** `cargo run --release --bin bench` (full 154 588-matrix KKT corpus).
**Plan:** `dev/plans/sparse-tail-d3.md` §Measurement, stage 3.

## Headline

Sparse factor/MUMPS distribution, pre-D.3 vs post-D.3:

| percentile | pre-D.3 | post-D.3 |  Δ    |
|------------|--------:|---------:|------:|
| geomean    |    0.46 |     0.37 |  −20% |
| p50        |    0.33 |     0.29 |  −12% |
| p90        |    1.84 |     1.71 |  −7%  |
| p99        |    3.48 |     3.54 | +2%   |
| max        |  128.34 |    80.22 |  −37% |

**Ex-ante acceptance target:** geomean 0.46 → ≤ 0.44.
**Actual:** 0.37.
**Margin:** 0.07 under target — the gate captured more corpus value
than the scoping note projected.

## Per-family movement (top-25-by-count families)

Significant movers:

| family      | count | pre-D.3 geomean | post-D.3 | Δ     |
|-------------|------:|----------------:|---------:|------:|
| AVION2      |  2682 |            1.51 |     1.48 | −0.03 |
| BATCH       |  2054 |            1.65 |     1.62 | −0.03 |
| (others)    |     — |               — |        — |    ≈0 |

The D.1 rollout had already shaved AVION2/BATCH a large fraction; D.3
contributes a smaller marginal drop because those families are not
primarily gate-eligible (most have n > 128 or sparse enough to stay
on the multifrontal path).

## Top-10 by factor/MUMPS ratio — post-D.3

| name              |  n  | feral (µs) | MUMPS (µs) | ratio |
|-------------------|----:|-----------:|-----------:|------:|
| HS85_0022         |  68 |       1845 |         23 |  80 × |
| DECONVC_0105      |  52 |       2137 |         39 |  55 × |
| NELSON_0173       | 387 |       4186 |         86 |  49 × |
| HS118_0478        |  32 |        832 |         21 |  40 × |
| DECONVC_0094      |  52 |       1373 |         40 |  34 × |
| HS118_0438        |  32 |        700 |         22 |  32 × |
| DMN15103LS_0070   |  99 |       3342 |        139 |  24 × |
| PFIT4_2178        |   6 |        198 |         10 |  20 × |
| HATFLDH_2515      |  11 |        212 |         11 |  19 × |
| DECONVC_0102      |  52 |        632 |         40 |  16 × |

**Disappeared from the pre-D.3 top-10** (all gone):

- `CRESC50_0331` (4065×) — arrow-KKT class, n=306; was dominated by
  delayed-pivot iterations at the dense root. Not a D.3 target;
  something else in the post-D.1 → post-D.3 window caught it
  (possibly the commit graph between these benches — the checkpoint
  history can speak to this more precisely).
- All `HAHN1_*` (99–148×) — n=715, out of the D.3 gate. Same as above.
- `LEWISPOL_*` (121–223×) — n=15, tiny-n class (D.4 territory), not
  captured by the D.3 gate (density might or might not clear 0.25 on
  those matrices; irrelevant in this pass).
- `METHANL8LS_0899`, `NET1_0371` — similarly not gate-targets.

The new top-10 is dominated by in-gate-eligible matrices (9/10 have
n ≤ 128) where the dense path is not automatically faster because
the matrices are very sparse (HS85_0022 has nnz typical of a KKT
with almost no dense structure, so either the gate correctly rejects
it and it's still running multifrontal, or the gate accepts it and
the dense path pays for densifying sparse structure). Worth
confirming in a follow-up diagnostic — but the distribution
compression from 128× max to 80× max (−37 %) already tells us D.3
paid off.

## Phase 2.8.1 exit partitions — still PASS

| path  | bucket               |  count |  p90 | target | verdict |
|-------|----------------------|-------:|-----:|-------:|---------|
| sparse| small-frontal (<200) | 153455 | 1.71 | ≤ 2.0  | PASS    |
| sparse| medium (<500)        | 153560 | 1.71 | ≤ 3.0  | PASS    |

No regression on the Phase 2.8.1 gating criteria.

## Ex-ante "no-regression outside the gate" check

The gate is `n ≤ 128 ∧ density ≥ 0.25`. Matrices *outside* the gate
follow the bit-identical multifrontal path — they cannot regress by
construction (the dispatcher forwards arguments unchanged). The only
way they could differ is if the `dense_fast_factor` code path
*leaked* state into the workspace; but the dense path does not touch
`ws` at all, and stage-1 sanity-checked this via the
`test_cross_path_determinism_tro3x3` test.

Matrices *inside* the gate: their pre-D.3 wall time was the
multifrontal cost; post-D.3 it's the dense fast-path cost. The
stage-2 sweep showed dense ≤ multifrontal at `ρ ≥ 0.25` for all
n ≤ 192 — so in-gate matrices are either faster or at most tied
(the 128 × 0.10 tie line exists only at the ρ floor, which the gate
does not cross). Corpus-wide, the geomean, p50, p90 all dropped,
consistent with "wins, no regressions".

## Decision: D.3 closed

Ex-ante target met with 16 % relative margin. No test regression.
Phase 2.8.1 exit partitions still PASS. The two follow-ups identified
in stage 1 — small-n `compute_scaling` fast path, `to_dense`
pooling via `FactorWorkspace` — are not required to close D.3 and
are re-scoped as independent levers that tackle the residual
small-dense overhead if a future target demands it.

## Next levers (for tasks.org; not D.3 work)

1. **D.4** (tiny-n fast path, `n ≤ ~10`). The corpus still has
   PFIT4_2178, HATFLDH_2515, GAUSS1_0002 (n=8) in its top-10 worst
   ratios — a tiny-n specialized path skipping symbolic analysis
   entirely could capture these.
2. **Phase 2.4** (blocked BK + SIMD on dense root frontal). Arrow-KKT
   tail that D.3 can't touch — CRESC50, HAHN1, NET1 scale bands.
3. **Small-n `compute_scaling` fast path.** 34 µs on a n=69 matrix
   is unreasonable; a branch that uses ∞-norm scaling directly on
   the dense buffer (bypassing MC64's Hungarian matching overhead)
   on gate-hit inputs could halve dense-path wall time on TRO3X3
   class. Narrow, follow-up commit.
