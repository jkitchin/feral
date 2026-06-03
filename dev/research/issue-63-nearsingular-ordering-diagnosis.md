# Research Note: Near-singular KKT ordering-dependent IPM stall (issue #63) — diagnosis

**Status:** Investigation complete — no principled feral-side fix; recommend
pounce-side / joint. Issue left open as not-primarily-feral.
**Date:** 2026-06-03
**Author:** agent session 2026-06-03 (journal `2026-06-03-02.org`)
**Related issue:** https://github.com/jkitchin/feral/issues/63
**Repro:** `src/bin/probe_issue63_nearsingular.rs`; matrices regenerated from
`scrs8-2c-8.nl` via pounce `--dump kkt:24-28` + the mittelmann
`jsonl_to_mtx.py`.
**Code map:** ZeroPivotAction / 1×1 floor + delay/force-accept gate
`src/dense/factor.rs` (~616, ~3832–3980); BK alpha + 2×2 growth/det
`~2464–2660`; cascade-break `src/numeric/factorize.rs:2355–2386`
(`CASCADE_BREAK_MIN_N = 4096`); refinement caps `src/numeric/solve.rs`
`~1069–1153`; static pivot `Solver::with_static_pivot_threshold`.

## The claim (issue #63)

On `scrs8-2c-8` (an ill-conditioned LP), pounce's IPM stalls at "Acceptable"
(constraint violation `2.30e-8`, just over `tol=1e-8`) under `amd`/`amf`/`Auto`
but reaches "Optimal" (`1.00e-8`) under `metis`/`scotch`. The issue hypothesises
that FERAL force-accepts a collapsing 1×1 pivot at the working-precision floor
and the resulting **solve backward error** (~2e-8 under AMD vs ~1e-8 under
metis) leaks into the IPM feasibility residual; suggested fixes are pivot
stabilization / growth bounds / more refinement.

## What reproduces and what does not

**Empirical lever — confirmed** (current pounce 0.3.1-dev + feral):

| ordering | EXIT | constraint violation |
|---|---|---|
| default (Auto→AMF) / amd | Acceptable | `2.2999998941165188e-08` |
| metis / scotch | **Optimal** | `9.9999756785274591e-09` |

Values are **bit-identical across orderings within each class** — the outcome
is one of two discrete fixed points, not numerical noise.

**FERAL-level hypothesis — does NOT reproduce.** On the iter-26 KKT (μ collapse,
regularization first on; n=727), measuring `‖Ax−b‖/‖b‖` of FERAL factor+solve:

- On the δ_w-regularized system pounce actually steps with (full rank),
  backward error is **~1e-22 under every ordering**, both with Auto scaling
  (max_piv ~30–60) and with **no scaling** (max_piv ~6e16). A full-rank solve
  is ordering-independent in its *answer*; only fill/speed change.
- The issue's "AMD backward error ~2e-8 vs metis ~1e-8" and `max_abs_pivot ≈
  9.68e10` are not observed; the ~1e10 figure is an unscaled-pivot reading.

So **FERAL's linear-solve accuracy is not the bottleneck**, and the suggested
fixes target a backward error that is already ~1e-22.

## The actual mechanism

What *is* ordering-dependent is FERAL's **inertia on the pre-regularization
singular matrix** (force-accepted zero pivots):

| ordering | inertia (pos, neg, zero) |
|---|---|
| Amd | (453, 252, 22) |
| Amf | (457, 252, 18) |
| MetisND | (465, **255**, 7) — neg ≠ 252 expected |
| ScotchND | (410, 253, 64) |

pounce's δ_w-escalation loop terminates on this inertia, so a different ordering
→ different δ_w/iterate path. Trajectory comparison (iters 24–35):

- **AMF:** regularization on at iter 26; `inf_pr` 5.42e-8 → **freezes** at
  `2.30e-8` bit-identical for 27 iterations while `‖d‖` grows to 1.8. A frozen
  fixed point.
- **metis:** regularization on at iter 25 (one earlier); `inf_pr` jumps to
  `2.56e-4` (4 orders larger) then **steadily decreases** to `1e-8` (optimal).
- The δ_w *schedule* (`lg(rg)` −4.0, −4.5, …) is **nearly identical** between
  them — it is not a different δ_w magnitude.

**Paradox:** metis's singular-matrix inertia is *more wrong* (neg 255 ≠ 252) yet
it converges — by making pounce regularize earlier/harder, which escapes the
frozen point. The winning ordering wins by being *more pessimistic*, not more
accurate.

## Why there is no principled feral-side fix

1. The solve is already exact (~1e-22) under all orderings — nothing to fix in
   accuracy.
2. The MA57-style static pivot the issue names **backfires**: on the singular
   matrix, `with_static_pivot_threshold(1e-8)` makes the floor `1e-8·‖A‖∞`
   (with the μ→0 (1,1)-block blowup) perturb ~510–542 of 727 pivots → backward
   error **1.0** (solve destroyed), inertia scrambled. Force-accept-and-report-
   zeros is *useful* here: it correctly signals singularity so pounce
   escalates.
3. The lever that works (metis) is an *incorrect* inertia; there is no
   known-correct inertia change that fixes scrs8. Routing this KKT class to
   metis would "paper over" the symptom (issue's own words) and risks the
   don't-regress set (robot_1600, NARX_CFy, marine_1600, rocket_12800,
   pinene_3200).
4. cascade-break does not even fire here (`n=727 < CASCADE_BREAK_MIN_N=4096`);
   the force-accepted zeros come from the root front hitting `zero_tol`.

## Recommendation

The durable fix is in the **δ_w / inertia-acceptance interaction** (pounce-side
or joint): AMF's *correct* inertia leads pounce to under-regularize into a
frozen fixed point, while a more pessimistic regularization escapes it. That is
an IPM-regularization-strategy question, not a FERAL factorization-accuracy one.
Recommend tracking the fix in pounce (or as a joint investigation) and keeping
#63 open as not-primarily-feral. The ordering-class heuristic is only an
explicit, documented symptom mitigation and is not pursued.
