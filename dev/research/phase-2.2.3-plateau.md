## Phase 2.2.3 — CHWIRUT1 / CRESC100 / CRESC132 residual plateau

**Status:** Pre-implementation research note for Phase 2.2.3.
**Date:** 2026-04-13
**Related:**
- Prior validation: `dev/validation/phase-2.2.1-mc64-sweep.md`,
  `dev/validation/phase-2.2.2-pivot-rejection.md`.
- Prior research: `dev/research/mc64-scaling.md`,
  `dev/research/scaling-aware-pivot-rejection.md`.
- Prior plan: `dev/plans/scaling-aware-pivot-rejection.md`.
- Test file: `tests/mc64_regression.rs` (all `#[ignore]`'d).
- Solve path: `src/numeric/solve.rs:236` `solve_sparse_refined`.
- Factor path: `src/numeric/factorize.rs` + frontal assembly in
  `src/numeric/...` (the `D·A·D` scattering path from Phase 2.2.1).
**Key references:** citet:higham2002accuracy (ch. 12 iterative
refinement), citet:arioli1992residual (backward-error refinement
criterion), citet:duff2001mc64, citet:duff2005symmetric,
citet:amestoy2007muscaling.

---

### 1. Problem statement

After Phase 2.2.1 (MC64 symmetric scaling) and Phase 2.2.2
(column-relative pivot rejection), three of the four
`mc64_regression.rs` matrices still plateau at residuals orders of
magnitude above both target and canonical oracle:

| Matrix          |    n | Pre-fix | Post-2.2.1 | Post-2.2.2 | Target | Canonical MUMPS |
|-----------------|-----:|--------:|-----------:|-----------:|-------:|----------------:|
| CHWIRUT1_0000   |  645 | 1.41e+9 |    8.50e+2 |    8.50e+2 |  1e−8  |       9.51e−13  |
| CRESC100_0000   |  806 | 2.54e+4 |    1.43e+2 |    1.43e+2 |  1e−8  |       6.15e−15  |
| CRESC132_0000   | 5314 | 2.39e+8 |    1.37e+5 |    1.37e+5 |  1e−6  |       2.48e−11  |

Phase 2.2.2's column-relative threshold (`u = 0.01`) did **not**
change the residual on any of the three (`Δ < 0.5%`). Inertia is
exact vs MUMPS on CHWIRUT1 and CRESC100; CRESC132 is ±2. All three
tests remain `#[ignore]`'d with honest status comments.

ACOPP30 (the fourth regression test) recovered 47 orders in Phase
2.2.2 and now plateaus at `1.076e−1`, 7 orders above its `1e−8`
target — a different magnitude but plausibly the same bottleneck.

**Plateau in one line.** For matrices where column-relative pivot
rejection does **not** fire (because all Bunch-Kaufman pivots are
above the `u · max(RMAX,AMAX)` floor), the MC64-scaled sparse solve
loses between 8 and 16 orders of magnitude of accuracy somewhere
between `solve_sparse` and the iterative refinement loop. This note
enumerates the possible loss sites and proposes a diagnostic plan.

---

### 2. Hypotheses

The plateau is bounded below by the iterative refinement loop in
`solve_sparse_refined` (`src/numeric/solve.rs:236–309`). Either the
refinement is not firing, not converging, or converging to the wrong
answer. The five candidate root causes:

**H1 — Refinement is diverging and best-iterate is locking in a bad
initial guess.** `solve_sparse_refined` tracks the smallest
`||r||₂` across 3 steps and returns the corresponding `x`. If the
unrefined `solve_sparse` already produced a residual worse than any
subsequent correction can improve on, the returned `x` is the
unrefined output. CHWIRUT1's residual `8.50e+2` is oddly stable
across Phase 2.2.1 and 2.2.2 — suggestive of best-iterate lock-in.
**Evidence to collect:** per-iteration `||r||₂` and `||dx||₂/||x||₂`,
number of steps run before break, whether `divergence_factor` trips.

**H2 — Residual is computed in finite precision without compensated
summation.** Iterative refinement's theoretical guarantees
(citet:higham2002accuracy ch. 12) depend on `r = b − A·x` being
computed at higher precision than the factorization. Ipopt, MUMPS,
and SSIDS all offer an extended-precision residual option. feral
uses plain `f64` in `matrix.symv(&x, &mut ax)`. For a
well-conditioned residual ~`1e−14`, this is fine; for condition
number `κ(A) ~ 1e14+`, the residual is dominated by catastrophic
cancellation and refinement stalls. **Evidence to collect:**
`||A||·||x|| / ||b||` — if this is ~`1e16` or more, `r` has no
usable digits. Also check the MC64-scaled condition number, which
should be much lower than the raw one — if it still isn't low
enough, that's the story.

**H3 — The solve is applying the scaling in the wrong coordinate
frame.** Phase 2.2.1 wired scaling as `b' = D·b; y = core(b'); x =
D·y` where `core` factors `D·A·D`. This is correct **if** `D·A·D·x
= b` is the system being solved, which requires **the core to
return `y` such that `(D·A·D)·y = D·b`**, giving `x = D·y`
satisfying `A·(D·y) = D⁻¹·D·b = b`. Wait — that's wrong. `A·x = b`
with `x = D·y` gives `A·D·y = b`, not `D·A·D·y = D·b`. The correct
derivation is: let `A' = D·A·D` and solve `A'·y = D·b`, then `x =
D·y`. Check: `A·x = A·D·y`, and `A'·y = D·A·D·y = D·b ⇒ A·D·y =
D⁻¹·D·b = b`. ✓ OK so the math is right. But the iterative
refinement residual is `r = b − A·x` in the **original** frame,
and the correction solve is `dx = A⁻¹·r`. feral applies the
correction using the scaled factorization, which computes `dy =
A'⁻¹·(D·r)` and returns `dx = D·dy`. This is also correct. **So
H3 is probably not the bug** — but the compound `D·b ... D·dy`
arithmetic may introduce rounding floors worth measuring.
**Evidence to collect:** residual in both scaled and unscaled
coordinate frames; residual against the unrefined-scaled system
`A'·y − D·b`.

**H4 — Sparse-solve backsolve has a multi-supernode bug that the
bench single-supernode path masks.** The benchmark uses
`SupernodeParams { nemin: 10000, .. }` which collapses everything
into one supernode; `mc64_regression.rs` uses
`SupernodeParams::default()` with `nemin: 32`, which produces many
real supernodes. The reported 99.8% sparse residual pass rate
(bench) may therefore be an artifact of the nemin override. If
CHWIRUT1/CRESC100 pass cleanly under `nemin: 10000`, the plateau
is **structural** in the multi-supernode backsolve or assembly.
**Evidence to collect:** run each plateau matrix with both
`nemin: 32` and `nemin: 10000` and compare residuals.

**H5 — CRESC132 inertia mismatch (±2) is the direct cause of the
`1.37e+5` residual.** Two wrong inertia slots ≡ two wrongly-signed
diagonal entries in `D`, which flip-sign two components of the
solve. For a general KKT system this is a bounded perturbation,
but if the two slots correspond to structurally-important rows
(e.g., free-variable KKT blocks) the residual can explode by
`||A||·||D⁻¹·(sign-flipped-components)|| / ||b||`. The carry-over
deferred trace-based 2×2 inertia fix from session 01 should be
re-attempted here — it may *directly* close CRESC132's gap without
any refinement work. **Evidence to collect:** compare feral's
pivot type vector vs MUMPS; look for 2×2 pivots whose
trace-rule-inertia disagrees with `a00`-rule-inertia.

---

### 3. Diagnostic plan

Order of operations, cheapest first:

1. **H4 first — flip `nemin`.** One-line change to
   `tests/mc64_regression.rs` or a new example. If it changes
   residuals, the bug is in the multi-supernode backsolve/assembly
   and we need a separate investigation. If residuals are
   unchanged, H4 is ruled out.

2. **H1 — instrument the refinement loop.** Copy
   `solve_sparse_refined` into a debug example and log
   per-iteration `||r||₂`, `||r_new||₂`, `||dx||₂`, `||x||₂`,
   which break condition fires, and how many steps ran. Look for:
   (a) best_r_norm set only at step 0 (H1 confirmed), (b)
   monotone decrease that stops at `1e+2` (rounding floor, H2),
   (c) oscillation (H2 or numerical issue).

3. **H2 — measure the effective condition number and residual
   noise floor.** For CHWIRUT1 and CRESC100, compute `||A||₁`,
   `||x||∞`, `||b||∞`, and `κ(D·A·D)` via an LSMR or power-
   iteration proxy. Compare `||r||/(||A||·||x||)` — if this is
   already at machine epsilon, refinement has no room to work and
   the residual is exactly what the arithmetic permits. The
   canonical MUMPS `9.51e−13` on CHWIRUT1 suggests `κ ~ 1e14`,
   which in `f64` does leave ~2 digits — so feral losing ~14
   orders to MUMPS is **not** arithmetic unavoidability; there's a
   real bug.

4. **H5 — pivot-type diff vs canonical.** For CRESC132
   specifically, dump feral's pivot-type vector and compare
   against the MUMPS sidecar (if available) or against what SSIDS
   reports. Two off-diagonal 2×2 pivots being mis-classified as
   two 1×1 pivots of the wrong sign is the signature.

5. **H3 (rule-out pass).** Only if 1–4 are inconclusive: print
   residuals in both scaled and unscaled frames.

---

### 4. Prior art on iterative refinement

**citet:arioli1992residual** defines the backward-error-based stop
condition used by MUMPS and SSIDS:

```
ω₁ = max_i |r_i| / (|A|·|x| + |b|)_i
ω₂ = max_i |r_i| / (|A|·|x|)_i   (for components where |b_i| small)
stop when max(ω₁, ω₂) ≤ n·ε
```

feral currently uses `||dx||₂ / ||x||₂ < ε·√n` (norm-wise small
correction), which is weaker than the Arioli componentwise test and
specifically does **not** detect the "refinement has reached the
arithmetic floor but residual is still large in absolute terms"
failure mode — exactly the CHWIRUT1 symptom. This is a strong
candidate explanation for H1/H2 overlap: feral stops refining
because the correction is small, not because the residual is small.

**citet:duff2005symmetric** §6 notes that for MC64-scaled matrices,
iterative refinement must compute the residual in the **original**
coordinates (against un-scaled `A` and `b`), which feral already
does. Good.

**citet:higham2002accuracy** Theorem 12.3: fixed-precision iterative
refinement converges iff the infinity-norm condition number of the
refinement iteration matrix `(I − Â⁻¹·A)` is < 1. For feral's
scaled path, `Â = (D·A·D)⁻¹` applied as `x → D·A'⁻¹·D·b`, and the
iteration matrix norm depends on how accurately `A'⁻¹` is
represented by the LDL^T factors — which in turn depends on the
growth factor, the zero-tolerance clipping, and the inertia
correctness. This is consistent with "the plateau reflects an
LDL^T that isn't accurate enough", which would implicate either
pivoting (ruled out by Phase 2.2.2 for CHWIRUT1/CRESC100 whose
inertia is exact) or a numerical kernel bug.

---

### 5. What is *not* in scope for this phase

- Delayed pivoting (Phase 2.3). Already recognized as the fix for
  ACOPP30's remaining 7-order gap and CRESC132's ±2 inertia.
- Extended-precision residual (compensated summation / xblas).
  Track as a follow-up if H2 lands.
- Rewriting `solve_sparse_refined` from scratch. Any fix should be
  an Arioli stop-condition swap + maybe a step-count bump.

---

### 6. Exit criteria

The diagnostic is done when we can answer, with evidence:

1. Does flipping to `nemin: 10000` change the residuals? (H4)
2. How many refinement steps run, and what is the per-step
   `||r||₂` trajectory? (H1)
3. What is `||A||·||x|| / ||b||` and how does it compare to the
   observed residual? (H2)
4. (CRESC132 only) Do feral and MUMPS agree on the pivot-type
   sequence? (H5)

Answers to 1–4 determine the Phase 2.2.3 plan: either a small
solve-side fix lands this session, or we document the findings and
punt to a larger effort.
