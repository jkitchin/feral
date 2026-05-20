# Research note: near-singularity signal (`min|λ(D)|`)

Date: 2026-05-19
Author: agent session 2026-05-19-01
Status: research → plan → implement

## 1. Problem

An interior-point method (IPM) backed by FERAL — `pounce`'s `FeralSolverInterface`
— cannot bump its Hessian perturbation `δ_w` on KKT systems that are
**ill-conditioned but have correct inertia**. On three Mittelmann-class
problems where Ipopt+MA57 converges in 100–291 iterations, the pounce+FERAL
backend stalls because the perturbation handler never fires.

Observed by the user:

> ipopt solves all three in 100–291 iters. Its linear solver (MA57) reports
> near-singularity, not just wrong inertia, and feeds that into the
> perturbation handler so `delta_w` gets bumped on ill-conditioned-but-
> correct-inertia systems. pounce's FERAL backend evidently only signals
> inertia, so the perturbation handler never fires on these steps.

### Root cause in FERAL

FERAL's default `ZeroPivotAction::ForceAccept` (`src/dense/factor.rs:444`,
`660`) **force-accepts** a near-singular 1×1 pivot `|d| <= zero_tol` (and the
F-01 band `(zero_tol, null_pivot_tol]`, `factor.rs:678`) — it counts the pivot,
sets `needs_refinement = true`, and returns `FactorStatus::Success`. The
factorization completes; inertia is reported; nothing else.

The *only* near-singularity-adjacent fact that survives to the IPM is
`needs_refinement` — and that flag is:
- internal to `SparseFactors` (`factorize.rs:745`), not on the C ABI;
- a coarse boolean, not a magnitude — it cannot be thresholded;
- already true on perfectly healthy KKT systems where cascade-break's
  L-perturbation fired, so it is not specific to near-singularity.

So a KKT system that is numerically rank-deficient to working precision but
happens to land on the *correct* inertia produces `FactorStatus::Success`
with the right negative-eigenvalue count — and pounce returns
`ESymSolverStatus::Success`. The perturbation handler sees a clean solve and
never escalates `δ_w`.

MA57, on the same matrix, applies its `CNTL(2)` small-pivot magnitude
threshold (default ≈ 1e-20, on the *scaled* matrix), and when a pivot falls
below it sets `INFO(1) == 4` (rank-deficient) / `INFO(25) < n` (rank). Ipopt's
`IpMa57TSolverInterface.cpp` maps `INFO(1)==4` to `SYMSOLVER_SINGULAR`. That
status is distinct from `SYMSOLVER_WRONG_INERTIA`, and it drives
`PDPerturbationHandler::PerturbForSingularity` — a *different* escalation
branch from `PerturbForWrongInertia`. It is the near-singularity report that
FERAL has no equivalent of.

(Ipopt has a *second*, independent near-singularity path: the
`PDFullSpaceSolver` iterative-refinement residual-ratio test → `pretend_singular`.
pounce already ports that one in `pd_full_space_solver.rs`. The gap addressed
here is only the *solver-reported* signal, the MA57 `CNTL(2)` analog.)

## 2. What MA57 actually reports, and the FERAL analog

| MA57 | meaning | FERAL today | FERAL after this note |
|------|---------|-------------|-----------------------|
| `INFO(24)` | # negative eigenvalues | `Inertia.negative`, `feral_num_neg` | unchanged |
| `INFO(1)==4`, `INFO(25)` | rank-deficient / rank | — (`ForceAccept` hides it) | `min|λ(D)|` below caller threshold |
| `CNTL(2)` | small-pivot magnitude threshold | — | caller-side threshold on `min|λ(D)|` |

MA57's `CNTL(2)` is an **absolute magnitude threshold on the pivot of the
scaled matrix**. The closest FERAL quantity is the **smallest accepted pivot
magnitude** of the D factor:

    min|λ(D)|  =  min over all eliminated pivots of:
                   |d|                     for a 1×1 pivot d
                   min(|λ₊|, |λ₋|)         for a 2×2 block

where the 2×2 block `[[d11,d21],[d21,d22]]` has eigenvalues
`λ± = (t ± √(t²−4Δ))/2`, `t = d11+d22`, `Δ = d11·d22 − d21²`.

A small `min|λ(D)|` means a pivot near the working-precision floor was
accepted — exactly the condition MA57's `CNTL(2)` flags. This is **not** the
same as `min_diagonal()` (`factorize.rs:882`), which returns the *signed
smallest* eigenvalue (the most-negative one) for Ipopt's unconstrained
inertia-correction shortcut. Near-singularity needs the *smallest in
magnitude*, regardless of sign. The two methods are complementary.

### Why `min|λ(D)|` and not the condition estimate

FERAL already has a Hager–Higham 1-norm condition estimator
(`src/numeric/condition.rs`, `Solver::estimate_condition_1norm`). κ₁(A) is a
*better* near-singularity measure, but it costs 3–5 extra solves per
factorization. `min|λ(D)|` is computed **for free** — it is a single `min`
reduction over the `d_diag`/`d_subdiag` arrays already stored in every
`FrontalFactors`, the same arrays `min_diagonal()` and `summary()` already
walk. The cheap signal is the right default; pounce can still call
`estimate_condition_1norm` as a second-stage confirmation when `min|λ(D)|`
is borderline.

### Scaling

The D blocks are eigenvalues of the **scaled** matrix `S·A·S`
(`SparseFactors.scaling`, plus the per-front equilibration in
`dense/factor.rs:601`). This is the correct space for an MA57 `CNTL(2)`
analog — MA57 likewise thresholds the *scaled* pivot. `min|λ(D)|` is therefore
a scaled-space quantity. To make the trigger scale-free without requiring the
caller to recompute `||S·A·S||`, FERAL also exposes `max|λ(D)|`; the ratio
`min|λ(D)| / max|λ(D)|` is a scale-free near-singularity proxy (≈ 1/κ(D), a
lower-bound proxy for 1/κ(A) when pivoting is well-behaved). The caller
thresholds whichever it prefers.

## 3. Chosen design

Additive only. No change to factorization, pivoting, inertia, or solve. Two
new aggregation methods over data that already exists, mirroring the existing
`min_diagonal()` precedent exactly.

### Rust API

- `SparseFactors::min_pivot_magnitude() -> Option<f64>` — `min|λ(D)|` over all
  eliminated pivots; `None` if nothing was eliminated. Scaled space.
- `SparseFactors::max_pivot_magnitude() -> Option<f64>` — `max|λ(D)|`, same
  domain. Lets the caller form the scale-free ratio.
- `Solver::min_pivot_magnitude() -> Option<f64>` and
  `Solver::max_pivot_magnitude() -> Option<f64>` — delegate to
  `last_factors`, `None` before any factor (matches `min_diagonal`).

### C ABI

- `feral_min_pivot(s) -> f64` — `min|λ(D)|`, or a negative sentinel
  (`-1.0`) when no factor / null handle.
- `feral_max_pivot(s) -> f64` — `max|λ(D)|`, same sentinel.

A negative sentinel is unambiguous: a magnitude is non-negative by
construction, so `< 0.0` means "no value".

### Out of scope

- The dense-direct `Factors` path (`src/dense/factor.rs`) is not on the C ABI
  and not used by pounce; adding the same accessor there is an optional
  consistency follow-up, not part of this change.
- No new `FactorStatus` variant. FERAL keeps returning `Success` on a
  force-accepted near-singular factor; the *caller* (pounce) decides, from
  `min|λ(D)|`, whether to treat it as singular. This keeps FERAL's
  policy-free contract intact and avoids a breaking ABI change.

## 4. How pounce consumes it (spec only — out of this repo)

In `pounce-feral`'s `factor()`, after a `FactorStatus::Success`:

```text
let min_piv = feral_min_pivot(handle);
let max_piv = feral_max_pivot(handle);
// MA57 CNTL(2) analog: scale-free ratio threshold.
if min_piv >= 0.0 && max_piv > 0.0
   && min_piv / max_piv < singular_pivot_ratio   // e.g. 1e-12
{
    return ESymSolverStatus::Singular;            // → PerturbForSingularity → δ_w
}
```

`ESymSolverStatus::Singular` routes into the IPM's `PerturbForSingularity`
branch — the same branch Ipopt reaches from MA57's `INFO(1)==4`. The
threshold `singular_pivot_ratio` is pounce's analog of `CNTL(2)`; a starting
value of `1e-12` is suggested (MA57 `CNTL(2)≈1e-20` is on the raw scaled
pivot; the ratio form is stricter and dimensionless — pounce tunes it on the
three regression problems). pounce keeps its existing residual-ratio
`pretend_singular` path unchanged; this adds the second, cheaper trigger.

## 5. Validation

Tests-first, oracle external to the implementation:

1. **1×1 hand oracle** — `diag(5, -2, 3, -7)`, identity scaling (forced, as in
   `min_diagonal_diagonal_matrix_one_by_one_pivots`): `min|λ(D)| = 2`,
   `max|λ(D)| = 7`. Pure hand calculation.
2. **2×2 hand oracle** — `[[0,1],[1,0]]`, identity scaling: BK forms one 2×2
   block, eigenvalues ±1, so `min|λ(D)| = max|λ(D)| = 1`. Verifies the
   smaller-*magnitude* eigenvalue is extracted (not `d_diag[0]=0`, and not
   the signed-min `-1` that `min_diagonal` returns).
3. **`None` before factor** — `Solver::new().min_pivot_magnitude()` is `None`.
4. **Near-singular regression** — a 2×2 (or small) matrix with one pivot at
   ≈ 1e-14: assert `min_pivot_magnitude()` is ≈ 1e-14, i.e. the signal is
   small and thresholdable, while inertia is still reported. Oracle is the
   hand-constructed pivot value.
5. **C ABI** — `feral_min_pivot` / `feral_max_pivot` return the sentinel
   before factor and the expected magnitudes after, on the 2×2 indefinite
   matrix already used by the `capi` tests.

Cross-check: on every test, `min_pivot_magnitude() <= max_pivot_magnitude()`
and both are `>= 0`, and `min_pivot_magnitude() >= |min_diagonal()|`-consistent
where the signed-min is itself the smallest-magnitude pivot.

## 6. References

- `dev/research/condition-estimate.md` — Hager–Higham κ₁ estimator (the
  expensive second-stage signal).
- `dev/research/static-pivot-perturbation-2026-05-17.md` — issue #38, the
  *rejected* "paper over it inside FERAL" approach; this note is the
  "report it, let the IPM decide" alternative.
- `dev/research/inertia-near-singular-certification.md` — prior analysis of
  near-singular inertia behavior.
- `dev/research/f01-rankdef-underreporting.md` — the F-01 band semantics that
  govern which near-singular pivots are accepted vs. zeroed.
- Ipopt 3.14: `IpMa57TSolverInterface.cpp` (`INFO(1)==4` →
  `SYMSOLVER_SINGULAR`), `IpPDPerturbationHandler.cpp`
  (`PerturbForSingularity`).
- `factorize.rs:882` `min_diagonal()` — the precedent this change mirrors.
