# Static-pivot perturbation (issue #38)

Date: 2026-05-17
Status: implemented (see `dev/journal/2026-05-17-01.org` end-of-day)

## Problem

On rocket_12800 (Mittelmann optimal-control NLP), feral's BK factor
on IPM iters 1-4 returns 38402/38402/38402/38401 negative eigenvalues
whereas Ipopt expects 38400 (= m, constraint count). The journal
entries §09:45 / §16:30 established that this is the **true** inertia
of the dumped matrix (Sylvester's law of inertia, all 1055 2×2 blocks
are well-separated (+,−), smallest |1×1 pivot| ~1.4e-4). MA57 reports
the IPM-expected 38400 by perturbing two small negative pivots over
into the positive bucket; Ipopt's `IpMa57TSolverInterface.cpp` does
**not** explicitly set MA57's static-pivot control, so this happens
through MA57's internal small-pivot replacement when its delayed-pivot
mechanism collapses on small values.

Result of "wrong inertia" from feral:
- ipopt-feral: 8.8 s / 31 iters (vs MA57 1.7 s / 31 iters) because
  PDPerturbationHandler escalates δ_w 1-3 times per affected iter.
- pounce-feral: fails outright — pounce treats FERAL_WRONG_INERTIA
  as a restoration trigger.

## Design

Add a new `NumericParams::static_pivot_threshold: Option<f64>` knob.
When `Some(t)`, the solver computes `||A||_∞` once per `factor()` call
and propagates `static_pivot_floor = t * ||A||_∞` to
`BunchKaufmanParams.static_pivot_floor`. The dense BK pivot routines
then enforce a magnitude floor on every accepted pivot:

- **1×1 pivot `d`**: if `|d| < static_pivot_floor`, replace with
  `sign(d) * static_pivot_floor` (sign(0) → +). Inertia is counted from
  the new sign. `needs_refinement = true`.
- **2×2 pivot block `[[a,b],[b,c]]`**: if smaller `|eigenvalue|` is
  `< static_pivot_floor`, shift both diagonals by
  `sign(λ_min) * (static_pivot_floor - |λ_min|)` (i.e. apply
  `τ · I` where `τ = static_pivot_floor - |λ_min|`, sign matched to
  push `λ_min` away from zero). This keeps the block symmetric and
  bends `λ_min` to ±static_pivot_floor while leaving λ_max almost
  unchanged. Inertia is counted from the perturbed block.
  `needs_refinement = true`.

The "default sign" choice (preserve current sign of d, push λ_min
away from zero) means the perturbation magnitude is bounded by the
floor itself and the factor satisfies `LDL^T = A + Δ` with
`||Δ||_F ≤ static_pivot_floor` per perturbed pivot. Inertia is now
reported as the inertia of `A + Δ`, which on rocket_12800 matches
the IPM expectation when `static_pivot_floor` exceeds the two
nearly-zero negative eigenvalues.

## Why this is different from `PerturbToEps`

`ZeroPivotAction::PerturbToEps { abs_floor }` only activates when
`|d| ≤ zero_tol` (≈ `f64::EPSILON`). The new `static_pivot_floor` is
typically `1e-8 · ||A||_∞` — six orders of magnitude larger. The
existing `PerturbToEps` path catches strict zeros; the new path
catches "small but nonzero" pivots that an IPM driver would prefer
to see bent toward the expected inertia.

## MA57 / Ipopt reference

- `ref/Ipopt/src/Algorithm/LinearSolvers/IpMa57TSolverInterface.cpp:367-395`:
  Ipopt sets `cntl[0] = pivtol` (default 1e-8 via `ma57_pivtol`),
  leaves `cntl[1]` (numerical zero, default 1e-20) and `cntl[4]`
  (static pivoting, default 0.0) untouched. MA57's "static
  pivoting" knob is therefore not the driver of Ipopt's MA57
  inertia bending — instead MA57 collapses small candidate pivots
  via its delayed-pivot policy, eventually accepting them as 1×1
  with the column-relative `pivtol` threshold. The net effect is
  similar to what `static_pivot_threshold` provides feral.
- Ipopt's PDPerturbationHandler retries with escalated δ_w on
  WRONG_INERTIA: typically 1 retry per affected iter (cost ~ 1
  extra factor), up to ~5 retries before restoration.
- Residual quality impact: bounded by `floor * O(condition_number)`
  in the worst case; iterative refinement against unperturbed A
  recovers solve accuracy (already what `feral_solve` does by
  default).

## Default

`static_pivot_threshold: None` (off) — opt-in only.
Recommended starting value for IPM use: `1e-8` (matches
`feral_factor`'s `FERAL_PIVTOL` default and MA57 `cntl[0]`).
Wired through the C ABI as `FERAL_STATIC_PIVOT=<float>` env var.

## Files touched

- `src/dense/factor.rs` — `BunchKaufmanParams.static_pivot_floor`
  field, `do_1x1_pivot` / `do_2x2_pivot` floor application,
  helper `perturb_2x2_to_floor`.
- `src/numeric/factorize.rs` — `NumericParams.static_pivot_threshold`
  field.
- `src/numeric/solver.rs` — `with_static_pivot_threshold` builder,
  `||A||_∞` computation in `factor()`, propagation to
  `effective_params.bk.static_pivot_floor`.
- `src/capi.rs` — `FERAL_STATIC_PIVOT` env var in `feral_new`.

## Tests

- `dense::factor::static_pivot_tests` — unit tests on 3×3 / 4×4
  matrices verifying perturbation sign-bending vs. unperturbed
  inertia.
- `tests/issue_38_static_pivot.rs` — integration test on small
  symmetric-indefinite matrices, exercising the C ABI path with
  `FERAL_STATIC_PIVOT` env var.
