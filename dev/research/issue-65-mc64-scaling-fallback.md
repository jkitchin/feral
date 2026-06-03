# Research Note: Inertia-guided MC64 scaling fallback (issue #65)

**Status:** Pre-implementation
**Date:** 2026-06-03
**Author:** agent session 2026-06-03 (journal `2026-06-03-03.org`)
**Related issue:** https://github.com/jkitchin/feral/issues/65
**Related code:** `src/scaling/mod.rs:609` (`pick_scaling_strategy`),
`src/numeric/solver.rs:730` (`Solver::factor`),
`src/numeric/solver.rs:1063` (sticky-Auto pin).
**Repro:** `src/bin/probe_issue65_scaling.rs`; matrices regenerated from
`sawpath.nl` / `twirism1.nl` / `discs.nl` via pounce `--dump kkt:0`.

## Problem

On ill-conditioned symmetric-indefinite KKTs, the default `Auto` scaling
factors the matrix with **force-accepted zero pivots and a wrong inertia**,
where `Mc64Symmetric` recovers the correct full-rank inertia. The consuming IPM
(pounce) reads the wrong inertia as `Singular`, takes a bad regularized step at
iteration 0, and falsely declares infeasibility.

Reproduced (`probe_issue65_scaling`, iter-0 KKTs):

```
sawpath (n=1575, expected neg=786):
  Auto/InfNorm  (789,670,116) ✗  min|piv|=0.0      ← 116 spurious zeros
  Mc64Symmetric (789,786,0)  ✓   min|piv|=0.0296   ← matches MA27/numpy
  Identity      (797,712,66) ✗
```

## Why a structural router fix cannot work

`pick_scaling_strategy` routes to `Mc64Symmetric` iff `max_col_nnz > 32` (arrow
head) AND `diag_only/n >= 0.30` (nonzero-diagonal slack mass). sawpath has
`max_col_nnz = 586` but `diag_only = 0` — its (2,2) dual block is explicit-zero,
so there are no nonzero-diagonal slack columns — so it routes to InfNorm.

The decisive obstacle: **`twirism1` iter-0 has the *same* router signature**
(`diag_only = 0`, `max_col_nnz = 85 > 32`) but the **opposite** need:

```
twirism1 (n=745, expected neg=313):
  Auto/InfNorm  (432,313,0) ✓  min|piv|=8.9e-16   ← CORRECT
  Mc64Symmetric (433,311,1) ✗  min|piv|=1.6e-17   ← WRONG
```

So sawpath (needs MC64) and twirism1-iter0 (needs InfNorm) are
**structurally indistinguishable**. Any purely structural router that fixes
sawpath by routing its signature to MC64 regresses twirism1 iter-0. "Make MC64
the default" is likewise ruled out (it also regresses twirism1 and, per the
existing router docs, clnlbeam by 4.4× iters). The deciding factor is
**numerical** (does the factorization hit the floor?), not structural.

## Why feral's own `check_inertia` path doesn't help in production

pounce-feral calls `solver.factor(&matrix, None)` (`pounce-feral/src/lib.rs:451`)
— it passes `check_inertia = None` and compares the returned inertia itself. So
feral never learns the expected inertia, and any expected-inertia-driven retry
would not fire in production. The fix must be **self-contained in feral**, keyed
on a signal feral can see without the caller's expectation.

## The fix: inertia-guided MC64 fallback under `Auto`

The signal feral *can* see is **force-accepted zero pivots**: a 1×1 pivot that
collapses below `zero_tol` is force-accepted as a structural zero and counted in
`inertia.zero`. On sawpath InfNorm yields `zero = 116`; on twirism1 iter-0 it
yields `zero = 0`. That single bit separates them.

**Design.** In `Solver::factor`, after the numeric factorization, when

1. the user configured `ScalingStrategy::Auto` (explicit InfNorm/Identity/MC64
   are respected as-is — Auto means "feral chooses"), AND
2. the resolved scaling was *not* already `Mc64Symmetric`, AND
3. the factor reports `inertia.zero > 0` (the singular signature),

re-factor the same matrix with `Mc64Symmetric` and **adopt** the MC64 result iff
it strictly reduces the zero count (`mc64.zero < first.zero`). On adoption, pin
the sticky-Auto strategy (`auto_picked_strategy`) to `Mc64Symmetric` so every
subsequent refactor on this pattern uses MC64 directly — the fallback "learns"
and the retry cost is paid at most once per pattern.

**Behavior on the repro set:**
- sawpath: InfNorm `zero=116` → retry → MC64 `zero=0` (`0 < 116`) → adopt,
  inertia `(789,786,0)`. ✓
- twirism1 iter-0: InfNorm `zero=0` → no retry → stays InfNorm `(432,313,0)`. ✓
- discs iter-0: all strategies already agree `(79,74,0)`, `zero=0` → no retry.

## Correctness safety

MC64 is a diagonal/permutation rescaling — it **cannot change the rank** of the
matrix. On a *genuinely* rank-deficient KKT, the MC64 retry also force-accepts
zeros (`mc64.zero >= first.zero` typically), the strict-improvement gate fails,
and the original factor is kept — the only cost is one wasted factorization.
Conversely, when InfNorm reports spurious zeros on an *effectively full-rank*
matrix (sawpath: numpy rank 1572/1575, 0 eigenvalues below 1e-20), MC64 pulls
large entries onto the diagonal so the BK sequence never hits the floor, and the
recovered `zero=0` inertia is the correct one. So the fallback only moves feral
*toward* the MUMPS/SPRAL consensus (the inertia gate), never away from a true
singular classification.

## Scope and follow-up

- In scope: the **zero-trigger** fallback, which fixes the primary symptom
  (sawpath/discs-class false-infeasible at iter 0 → `Singular` misclassification).
- Out of scope / follow-up: `twirism1`'s **late-iteration** failure is a wrong
  *negative* count *without* zeros (`zero=0`, `neg` swings 310–347 vs true 313).
  feral returns `Success` and cannot detect it without the expected inertia,
  which pounce currently does not pass (`None`). Covering it requires either
  pounce passing expected inertia to `factor()` (then feral retries MC64 on
  `WrongInertia`), or a self-contained "suspicious neg count" heuristic. Record
  as a follow-up; do not block the sawpath fix on it.

## Risks / validation

- **Corpus inertia:** must not change any currently-correct inertia. The retry
  only fires on `zero > 0` and only adopts on strictly fewer zeros, so it can
  only reduce spurious zeros — aligned with the inertia gate. Validate with the
  consensus framework over the KKT corpus.
- **Cost:** one extra factorization when a singular-looking factor appears under
  Auto; amortized away by the sticky-pin on adoption. Measure retry frequency on
  the corpus — most healthy IPM factors have `zero = 0` and never retry.

## Test plan (external oracle = MA27/numpy inertia in the issue)

- Regression (skip-if-absent fixture, regenerate via
  `dev/scripts/regen_issue65_kkts.sh`): `Solver::new().factor(sawpath, None)`
  yields inertia `(789,786,0)` and `min|pivot| > 0` after the fallback (it is
  `(789,670,116)`, `min|piv|=0` today).
- Unit: the fallback fires only under `Auto`, only on `zero > 0`, and adopts
  only on strictly fewer zeros (synthetic or the real fixture).
- Keep all `pick_scaling_strategy_*` and inertia gate tests green.
