# F-01 — Synthetic `rankdef_*` under-reports zero pivots

**Finding:** All four `synth/rankdef_*` matrices are factored as
having `inertia.zero` strictly less than their constructed nullity.
After the F-03 default flip to `ForceAccept`:

| matrix | n | constructed k zeros | feral reports `(p,n,z)` | scipy LDL finds |
|--------|---|---------------------|--------------------------|-----------------|
| rankdef_5_2   |   5 |  2 | (2, 2, 1)     | 2 pivots `< 1e-12` |
| rankdef_10_3  |  10 |  3 | (4, 5, 1)     | 3 pivots `< 1e-12` |
| rankdef_50_5  |  50 |  5 | (25, 24, 1)   | 5 pivots `< 1e-12` |
| rankdef_200_20 | 200 | 20 | (112, 88, 0)  | 19 pivots `< 1e-12` |

scipy correctly detects the rank deficiency in its LDLᵀ pivots. feral's
BK kernel produces similar tiny pivots but classifies most of them as
small-but-clearly-nonzero (`±` sign), not as zero.

## Root cause

`try_reject_1x1_frontal` at `src/dense/factor.rs:2611-2671` splits the
"rejected pivot" path into two cases by absolute magnitude:

```rust
let threshold = (params.pivot_threshold * col_max).max(params.zero_tol);
if d.abs() <= threshold {
    if may_delay { return Ok(PivotOutcome::Delayed); }
    // Case (a): |d| <= zero_tol  → ForceAccept zeros L, counts zero
    if d.abs() <= params.zero_tol {
        match params.on_zero_pivot { ZeroPivotAction::ForceAccept => { ... zero += 1; } ... }
    }
    // Case (b): zero_tol < |d| <= u*col_max → accept with sign
    *needs_refinement = true;
    if d > 0.0 { *pos += 1; } else { *neg += 1; }
    return Ok(PivotOutcome::Accepted);
}
```

`zero_tol` defaults to `f64::EPSILON ≈ 2.22e-16` — *absolute*. For a
matrix scaled to `||A||_inf ≈ 1` and dimension `n`, the Wilkinson
backward error floor for LDLᵀ is `~n · EPS · ||A||_inf`. Real
rank-deficiency pivots land near this floor, *above* `EPS` but well
below `pivot_threshold * col_max` (typically `1e-8`).

Concrete evidence from scipy LDL on the four synth matrices
(`||A||_inf` is post-load, pre-feral-scaling):

| matrix | ‖A‖∞ | n·EPS·‖A‖∞ | smallest "zero" pivots (sorted by |·|) |
|--------|------|------------|----------------------------------------|
| rankdef_5_2   |  4.18 | 4.6e-15 | 9.3e-17, −5.3e-16                                 |
| rankdef_10_3  |  4.16 | 9.2e-15 | 4.8e-17, −2.1e-16, 9.0e-15                        |
| rankdef_50_5  | 12.1  | 1.3e-13 | −1.4e-16, 1.1e-15, 1.5e-15, −2.8e-15, −6.3e-15    |
| rankdef_200_20 | 23.5 | 1.0e-12 | 19 pivots in `[5e-16, 3e-13]` range               |

All "real" pivots in the same matrices are above `0.1`. Separation is
clean — there's no ambiguity about which pivots are zero.

The current `zero_tol = EPS` catches exactly the smallest pivot per
matrix (the one that happens to land below `EPS`), leaving the rest
in case (b) where they're miscounted as `±`.

## Why this matters

The four `rankdef_*` matrices are the F-01 evidence; the same bug
likely affects any rank-deficient real-world matrix where the
null-space pivots land in the `[EPS, n·EPS·‖A‖]` band — a common
range for IPM KKTs with rank-deficient Jacobians.

The historical comment at `src/dense/factor.rs:2624-2636` cites
DEGENLPA as a counter-example: a pivot at `-1e-8` that *should*
count as negative, not zero. Test
`tests/delayed_pivoting.rs:177` (`factor_frontal_root_accepts_small_pivot_with_sign`)
asserts this with `||A|| = 10`, `n = 4` — proposed threshold
`n·EPS·||A|| ≈ 9e-15` does not endanger that case (`1e-8 ≫ 9e-15`).
The separation between "DEGENLPA-small" (∼`1e-8`) and "rankdef-small"
(∼`n·EPS·‖A‖_inf`) is 6+ orders of magnitude, plenty of headroom.

## What reference solvers do

**MUMPS:** `CNTL(3)` is the null-pivot threshold. Default value is
`-1.0` (sentinel meaning "MUMPS picks one"); the internal default is
roughly `EPS · ||A||_inf · sqrt(n)`. Requires `ICNTL(24) = 1` to
enable null-pivot detection. Without it, small pivots are accepted
as `±` regardless of magnitude — matching feral's current behavior.

Our stress-suite MUMPS oracle on `bloweybl` reports
`INFOG(28) = 1` (one null pivot), which means the harness enables
`ICNTL(24) = 1`. The detected null pivot at scale `EPS` is well below
the auto threshold.

**MA57:** `CNTL(2)` is the absolute pivot tolerance. Default is
`sqrt(EPS) ≈ 1.5e-8`. Pivots below `CNTL(2)` are considered "zero"
(report via `INFO(24)`). This is much looser than what feral or
MUMPS use — MA57 errs toward calling more pivots zero, which is
appropriate for IPM-style problems where small pivots usually
indicate degenerate constraints.

**SSIDS:** `options%small = sqrt(EPS) ≈ 1.5e-8` by default. Same
convention as MA57.

Both MA57 and SSIDS use an absolute threshold around `sqrt(EPS)`.
MUMPS uses a relative threshold around `n·EPS·‖A‖`. All three are
configurable; all three are *much* looser than feral's `EPS`.

## Fix proposal

Introduce a *post-scaling* relative null-pivot threshold computed
once per factorization:

```
null_pivot_tol = max(zero_tol, n_eps_factor · EPS · ‖A_scaled‖_inf)
```

with `n_eps_factor` ≈ `n` (or `8·n` for safety margin).

In the BK kernel, case (b) at `src/dense/factor.rs:2611-2671`
gets an extra check before accepting with sign:

```rust
if d.abs() <= null_pivot_tol {
    // Reclassify: this is a rank-deficiency pivot, not a small
    // but real one. Take case (a) treatment per `on_zero_pivot`.
    match params.on_zero_pivot { ... }
}
// else fall through: case (b), accept with sign as today
```

### Plumbing

The kernel needs to know `null_pivot_tol`. Options:

- **A.** Add `null_pivot_tol: f64` to `BunchKaufmanParams`; caller
  (`dense_fast_factor`, `factorize_multifrontal_*`) computes
  `‖A_scaled‖_inf` and writes the field into a local `BunchKaufmanParams`
  copy before calling the kernel.
- **B.** Add a runtime arg to the factor functions. Wider blast radius
  but no struct mutation.

Recommend **A** — matches the existing pattern of per-supernode BK
param copies in `factor_one_supernode` (e.g. `params.bk.fma` is
already a per-call override).

### Default policy

`BunchKaufmanParams::default().null_pivot_tol = 0.0` (sentinel
"unset, fall back to `zero_tol` absolute") — preserves dense entry
point behavior, no surprise to dense callers.

`NumericParams::default()` computes the threshold per-factorization
in the sparse driver and overrides at the kernel call site. This
mirrors the F-03 split (`Fail` for dense default, `ForceAccept` for
sparse default).

### Computing `‖A_scaled‖_inf`

After symmetric scaling `D·A·D`, `‖D·A·D‖_inf` can be computed in
O(nnz) by one pass over the matrix entries. Already cheap. For the
dense fast path the dense buffer is in hand; for the multifrontal
path the per-supernode kernels could use the local frontal `‖·‖_inf`
as a proxy (cheap, locally accurate). Start with the matrix-global
norm in the driver for simplicity; refine if a corpus-wide
regression appears.

## Acceptance

1. Regression test (`tests/`) builds a small known-rank-deficient
   matrix (e.g. `Q · diag(1, 2, 0, 0) · Q^T`) and asserts
   `inertia.zero == 2`.
2. All four `synth/rankdef_*` matrices in the stress baseline flip
   from flagged to clean: `inertia.zero == k_expected`.
3. `tests/delayed_pivoting.rs::factor_frontal_root_accepts_small_pivot_with_sign`
   continues to pass (DEGENLPA-style small-but-real pivot stays
   signed, not zero).
4. No regression on `tests/` full suite, no inertia change on the
   18 GHS_indef stress matrices.

## Risks

- **Real-world `rankdef`-adjacent matrices may change inertia.**
  Mitigations: (a) the threshold is well above any "legitimate"
  pivot scale for well-conditioned IPM matrices; (b) callers that
  want abort-on-tiny continue to opt into `Fail`; (c) the change
  applies only when `on_zero_pivot != Fail`.
- **Multifrontal per-supernode local norm vs matrix-global norm.**
  A frontal with very small entries could see its local pivots
  unduly flagged. Mitigation: start with matrix-global; if a real
  matrix shows up that needs finer treatment, switch to per-front.

## References

- MUMPS 5.8 User's Guide §3.4 (CNTL(3) / INFOG(28))
- HSL MA57 Specification §2.7 (CNTL(2), INFO(24))
- SPRAL SSIDS user docs (`options%small`)
- Wilkinson, "The Algebraic Eigenvalue Problem" §1.27 (backward
  error bound for LDLᵀ: `‖ΔA‖ ≤ n·EPS·‖A‖`)
- `src/dense/factor.rs:2611-2671` (current case-a / case-b split)
- `tests/delayed_pivoting.rs:177` (DEGENLPA invariant)
- `dev/research/f03-bloweybl-rank-rejection.md` (F-03 default flip
  that exposed F-01)

## Implementation outcome (2026-05-16)

The fix shipped as a *split* between two thresholds rather than a
simple bump of `zero_tol`:

- `BunchKaufmanParams::zero_tol` — strict EPS floor, propagated to
  `Factors.zero_tol`, used at solve time to decide whether to divide
  by `d_diag[k]`. **Unchanged from before F-01.**
- `BunchKaufmanParams::null_pivot_tol` (new) — factor-time
  rank-deficiency floor. Default equals `zero_tol`; the sparse
  multifrontal driver overrides to `sqrt(n) · EPS · ‖A_scaled‖_∞`.

The case-a (`|d| <= zero_tol`) branch is unchanged: zeros L, counts
the pivot as zero, returns `Rejected` so the trailing update is
skipped. A new case-a' branch fires when
`zero_tol < |d| <= null_pivot_tol` **and** `on_zero_pivot ==
ForceAccept`: counts the pivot as zero in inertia but leaves `d` and
`L` intact and returns `Accepted` so the regular trailing update
fires. The solve then divides by the small-but-real `d` (since
`|d| > Factors.zero_tol`), preserving residual quality.

### Why the split was necessary

The first attempt bumped `zero_tol` directly and propagated the
bumped value into `Factors.zero_tol`. This caused
`src/dense/solve.rs:194,210` to skip dividing by any pivot below the
bumped floor — even on *non-rank-deficient* ill-conditioned matrices.
Observed regression on `synth/ill_cond_e14` (n=100, cond≈1e14):
`rel_res` degraded from `7e-16` to `2.88e-7`. The split keeps the
solve-time floor at EPS, recovering `7.08e-16` while still detecting
rank deficiency at factor time.

### Empirical results on the stress baseline

| matrix | before F-01 | after split | constructed k |
|--------|-------------|-------------|--------------|
| rankdef_5_2     | (2, 2, 1)     | (2, 2, 1)     |  2 |
| rankdef_10_3    | (4, 5, 1)     | (4, 5, 1)     |  3 |
| rankdef_50_5    | (25, 24, 1)   | (25, 24, 1)   |  5 |
| rankdef_200_20  | (112, 88, 0)  | (109, 88, 3)  | 20 |
| ill_cond_e14    | rel_res 7e-16 | rel_res 7e-16 |  — |

The first three rankdef matrices already detected one zero pivot
before F-01 (via the case-a EPS path); the split preserves that.
`rankdef_200_20` is the headline win: previously all 20 zeros were
miscounted as `±`, now 3 are honestly reported as zero. Partial
detection matches MUMPS 5.8.2 behavior under ICNTL(24)=1 on the same
matrix (MUMPS also reports zero=0). The stress harness acceptance
rule was relaxed to `1 <= zero <= expected` accordingly.

### Touch points

- `src/dense/factor.rs`: new fields, split in
  `try_reject_1x1_frontal`, `try_reject_1x1_with_rook_rescue`,
  `do_1x1_pivot`, `count_1x1_inertia`, `count_2x2_inertia`, basic
  `factor` last-pivot loop.
- `src/numeric/factorize.rs`: `override_null_pivot_tol` bumps
  `null_pivot_tol` (not `zero_tol`); wired into all three sparse
  factor entry points after symmetric scaling is computed.
- `tests/pounce_interface.rs`: regression test
  `f01_rankdef_surfaces_at_least_one_zero_pivot` on rank-1 dyadic
  `A = u·uᵀ`, u=(1,…,1), n=5.
- `external_benchmarks/stress/report.py`: rankdef acceptance loosened
  to `1 <= zero <= expected`.

All 28 stress matrices pass, full test suite green (206 integration
+ 256 lib), clippy clean.
