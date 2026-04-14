# Research: Porting SSIDS's scale-invariant 2×2 cancellation-aware det floor

**Date**: 2026-04-14
**Status**: Research complete. Implementation queued.
**Related**: `dev/research/task-19-dense-acopp30-expert-consultation.md` Step 2.

## Goal

Replace the absolute `|det| <= zero_tol_2x2` rejection test in
`factor_frontal`'s 2×2 pivot path with SSIDS's scale-invariant
cancellation-aware determinant floor. This hardens the sparse
production path (and the rerouted dense path) against catastrophic
cancellation in the determinant computation, without the regression
trap of absolute thresholds.

## Context: why absolute thresholds break

feral currently has at `src/dense/factor.rs:812`:

```rust
let det_floor_fail = absdet <= params.zero_tol_2x2;   // ≈ 4e-32
```

plus the Duff-Reid growth bound at `:809-810`. On the ACOPP30 KKT
block at k=58:

```
A_2x2 = [[ 0        , -4.16e-15 ],
         [ -4.16e-15, -6.08e-9  ]]
```

- `|det| = (4.16e-15)^2 ≈ 1.73e-29`. This passes the `4e-32` floor by
  a factor of ~350×, so the 2×2 is accepted.
- `|L21|` entries scale as ~1/|det| ≈ 10^29, destroying the trailing
  submatrix.

Two prior fix attempts (session 2026-04-13-05) used an `sqrt(eps)`
absolute floor. On the 154k-matrix bench they caused +6998 failures
(dense match 99.0% → 94.5%) because the corpus is not equilibrated:
a block with `||A||_∞ >> 1` has legitimately large-magnitude entries
that `sqrt(eps)` is far too coarse to gate against.

**The root fix is scale invariance, not a better absolute number.**

## SSIDS test_2x2 reference

Source: `src/ssids/cpu/kernels/ldlt_tpp.cxx:88-119`. Relevant block:

```cpp
bool test_2x2(int t, int p, double maxt, double maxp,
              double const* a, int lda, double u, double small, double* d) {
   double a11 = a[t*lda+t];
   double a21 = a[t*lda+p];
   double a22 = a[p*lda+p];
   double maxpiv = std::max(fabs(a11), std::max(fabs(a21), fabs(a22)));
   if (maxpiv < small) return false;                                  // (1)

   double detscale = 1/maxpiv;
   double detpiv0  = (a11*detscale)*a22;
   double detpiv1  = (a21*detscale)*a21;
   double detpiv   = detpiv0 - detpiv1;
   if (fabs(detpiv) <
       std::max(small,
                std::max(fabs(detpiv0/2),
                         fabs(detpiv1/2)))) return false;             // (2)

   d[0] = ( a22*detscale)/detpiv;
   d[1] = (-a21*detscale)/detpiv;
   d[3] = ( a11*detscale)/detpiv;
   if (std::max(maxp, maxt) < small) return true;
   double x1 = fabs(d[0])*maxt + fabs(d[1])*maxp;
   double x2 = fabs(d[1])*maxt + fabs(d[3])*maxp;
   return (u*std::max(x1, x2) < 1.0);                                 // (3)
}
```

Three guards applied in order, all ratios against the local block:

1. **Dead-zero block** (`small` = `1e-20` default, `datatypes.f90:260`).
2. **Cancellation-aware determinant floor** on the *scaled*
   determinant `detpiv = det/maxpiv`:
   ```
   reject iff  |detpiv| < max(small, |detpiv0|/2, |detpiv1|/2)
   ```
3. **Row-inf-norm L21 growth bound** — algebraically identical to the
   Duff-Reid `(|a22|*rmax + |a21|*tmax)*u > |det|` form after
   multiplying through by `|det|`.

## Scale invariance (walkthrough)

Let `a_ij → c·a_ij` for constant `c`:

| quantity     | transforms to             |
|--------------|---------------------------|
| `maxpiv`     | `|c|·maxpiv`              |
| `detscale`   | `detscale/|c|`            |
| `detpiv0`    | `sign(c)·c·detpiv0`, `|·|` scales by `|c|` |
| `detpiv1`    | same                      |
| `detpiv`     | same                      |
| `|detpiv|/|detpiv0|` | **invariant**     |
| `|detpiv|/|detpiv1|` | **invariant**     |

So guard (2) depends only on the relative cancellation in the
determinant computation, not on the absolute scale. Guard (1)'s
`small` is an absolute zero-detection floor only — *not* a stability
criterion — and sits near the underflow boundary (`1e-20`), not near
`sqrt(eps) ≈ 1.5e-8`.

Guard (3) is also invariant if the trailing submatrix scales with
the pivot block (full block scaling), because `|d|` shrinks by `1/c`
and `maxt, maxp` grow by `c`. When the trailing submatrix and the
pivot block scale independently (row/column equilibration), the
L21 bound correctly reflects the real growth — this is the property
we want.

## Cancellation theory: why `|detpiv0|/2, |detpiv1|/2`

For an indefinite pivot `det = a11·a22 − a21²`, the two products
`a11·a22` and `a21²` cancel catastrophically when near-equal. The
classical rule: if `|a − b| < |a|/2`, at least one bit of significance
is lost in the subtraction.

SSIDS rejects when `|detpiv|` drops below *half the larger* of the two
summands. Taking `max` (not `min`) is the strict choice: it ensures
the retained determinant has at least half the magnitude of the
dominant summand regardless of which one it is. Using `min` would
let a small `detpiv1` mask a ruined `detpiv0`.

The `small` term backstops the degenerate `detpiv0 == detpiv1 == 0`
case (literally-zero block) where half-of-max would demand
`|detpiv| >= 0` and accept anything.

## Relationship to feral's current growth bound

feral `src/dense/factor.rs:794-810` computes:
```rust
let mut rmax = 0.0f64;
let mut tmax = 0.0f64;
for i in (k + 2)..nrow {
    rmax = rmax.max(a[k * nrow + i].abs());
    tmax = tmax.max(a[(k + 1) * nrow + i].abs());
}
let amax = d21.abs();
let growth_fail = (d22.abs() * rmax + amax * tmax) * u > absdet
    || (d11.abs() * tmax + amax * rmax) * u > absdet;
```

This is **algebraically identical** to SSIDS's guard (3). Starting
from `x1 = |d11|*maxt + |d21|*maxp` with `d11 = a22/det`,
`d21 = -a21/det`:
```
x1 = (|a22|*maxt + |a21|*maxp) / |det|
x2 = (|a21|*maxt + |a11|*maxp) / |det|
```
and `u·max(x1,x2) < 1` ⇔ `u·(|a22|*maxt + |a21|*maxp) < |det|` AND
the twin — exactly feral's formula under `rmax=maxp`, `tmax=maxt`,
`amax=|a21|`.

So the growth bound in feral is already correct. **The only missing
guard is the cancellation-aware floor.** The absolute
`|det| <= zero_tol_2x2` that feral currently applies is a botched
attempt at guard (2) without the scaling normalization.

## Port design

Replace the absolute `det_floor_fail` at `src/dense/factor.rs:812`
with SSIDS's scaled cancellation test:

```rust
// SSIDS ldlt_tpp.cxx:101-106 — scale-invariant cancellation-aware
// determinant floor. `det_small` is a dead-zero absolute floor,
// NOT sqrt(eps). Default matches SSIDS's datatypes.f90:260 (1e-20).
let max_piv = d11.abs().max(d21.abs()).max(d22.abs());
let cancel_fail = if max_piv < params.det_small {
    true  // block is numerically zero
} else {
    let det_scale = 1.0 / max_piv;
    let detpiv0 = (d11 * det_scale) * d22;
    let detpiv1 = (d21 * det_scale) * d21;
    let detpiv  = detpiv0 - detpiv1;
    let cancel_floor = params
        .det_small
        .max(detpiv0.abs() * 0.5)
        .max(detpiv1.abs() * 0.5);
    detpiv.abs() < cancel_floor
};
```

Where `det_small = 1e-20` is added to `BunchKaufmanParams` as a new
field (default matches SSIDS). The existing `zero_tol_2x2` field
stays but is **no longer consulted in factor_frontal's 2×2 test**
— it remains relevant for `count_2x2_inertia` and friends where an
absolute floor is legitimate (inertia counts are integer-valued;
tiny |det| that escapes the scaled test still deserves an absolute
zero-vs-nonzero decision when tallying the signature).

On rejection, keep the existing `may_delay → break` / else
`try_reject_1x1_frontal` fallback unchanged — SSIDS does the same
structural thing (falls through to a 1×1 attempt on column `p`).

## Why this won't regress the bench

The failure mode that killed the `sqrt(eps)` patch was: a block like
`[[1e5, 1e-2], [1e-2, 1e5]]` in an unequilibrated `||A||_∞ ≈ 1e8`
matrix has `|det| ≈ 1e10` which easily passes `sqrt(eps) ≈ 1e-8`,
but a block like `[[10, 1e-8], [1e-8, 10]]` with legitimately small
off-diagonal has `|det| ≈ 100` — also easily passing — so it wasn't
absolute-magnitude that broke those matrices. It was that
`sqrt(eps)` was also applied to the *diagonal* as a "reducible
column" floor in the earlier patch, force-zeroing diagonals in
legitimately-large columns.

The SSIDS test is different:

1. It is pure rejection for the 2×2; it does *not* force-zero any
   diagonal. The post-rejection 1×1 fallback proceeds normally.
2. It compares `|detpiv|` against *fractions of its own summands*,
   so a well-conditioned large block `[[1e5, 1e-2], [1e-2, 1e5]]`
   has `detpiv0 = 1e5 * 1e5 / 1e5 = 1e5`, `detpiv1 = 1e-4 / 1e5 ≈ 1e-9`,
   `|detpiv| = detpiv0 − detpiv1 ≈ 1e5`, floor =
   `max(1e-20, 5e4, 5e-10) = 5e4`. Passes comfortably.
3. On the ACOPP30 block `[[0, -4e-15], [-4e-15, -6e-9]]`:
   `maxpiv ≈ 6e-9`, `detscale ≈ 1.7e8`,
   `detpiv0 = 0`, `detpiv1 = (4e-15)^2 * 1.7e8 ≈ 2.8e-21`,
   `|detpiv| ≈ 2.8e-21`, floor =
   `max(1e-20, 0, 1.4e-21)`. `|detpiv| = 2.8e-21 < 1e-20`.
   **Rejected.** The rejection routes to the 1×1 fallback (or delay
   for may_delay=true), which correctly handles the zero diagonal
   via `try_reject_1x1_frontal`'s column-relative rule.

The SSIDS threshold is tuned so well-conditioned indefinite blocks
pass and catastrophically-cancelled ones fail, independent of
absolute scale. This is the property we need.

## Implementation plan

1. Add `det_small: f64` field to `BunchKaufmanParams` with default
   `1e-20`. Document as "dead-zero floor for 2×2 pivot block
   entries". Place next to `zero_tol_2x2` in the struct definition.
2. In `factor_frontal` at `src/dense/factor.rs:812`, replace the
   absolute `det_floor_fail` with the scale-invariant cancellation
   test. Leave `growth_fail` untouched. The rejection branch stays
   the same (delay or `try_reject_1x1_frontal`).
3. Keep `zero_tol_2x2` field on `BunchKaufmanParams` and the
   `Factors` struct for now — it is consumed by inertia-counting
   code elsewhere (`count_2x2_inertia_val`, threshold tests outside
   the frontal kernel). Plan to revisit in a follow-up.
4. Unit test: construct a 2×2 block matching the ACOPP30 pattern
   at multiple absolute scales (1e-8, 1, 1e8) and verify the test
   rejects consistently. Construct a well-conditioned indefinite
   block `[[a, b], [b, -a]]` at the same scales and verify it
   always passes.
5. Run `examples/triage_dense_acopp30` to verify ACOPP30_{0026,
   0018,0000} still produce 1e-13..1e-14 residuals (should hold
   because `factor_frontal` already rejects via the growth bound +
   safe fallback for these; the scaled det floor is a tighter
   guard that should fire *before* the growth bound on the
   diagonal-zero case, giving the same rejection outcome).
6. Run full `cargo run --release --bin bench` and verify:
   - No regression on dense `inertia match` (99.0%) or
     `residual pass` (99.8%)
   - No regression on sparse (99.0% / 99.8%)
   - Worst residual either stable or improved
7. If any matrix moves from "pass" to "fail", investigate whether
   the scaled test is rejecting a legitimately stable block
   (unlikely per SSIDS's production track record, but the bench is
   the final arbiter).

## Files cited

- `src/ssids/cpu/kernels/ldlt_tpp.cxx:88-119` — `test_2x2`
- `src/ssids/cpu/kernels/ldlt_tpp.cxx:75-86` — `find_rc_abs_max_exclude`
- `src/ssids/cpu/kernels/ldlt_tpp.cxx:166-270` — caller with rejection path
- `src/ssids/datatypes.f90:260` — `small = 1e-20` default
- `src/ssids/datatypes.f90:262` — `u = 0.01` default

## Open questions

1. **Does the scaled det floor subsume the growth bound on the
   ACOPP30 block?** Yes: `maxpiv = 6e-9`, `|detpiv| = 2.8e-21 < small = 1e-20`.
   But the growth bound catches a different failure mode — large
   `rmax, tmax` with small |det|. Both guards are complementary and
   should stay.
2. **What about the 1×1 diagonal zero at k=58?** The scaled det
   floor rejects the 2×2 at k=58, then `try_reject_1x1_frontal`
   runs. That kernel has its own column-relative threshold — if
   the diagonal at k is below `pivot_threshold * gamma0`, the 1×1
   is also rejected, and either delay or force-accept kicks in.
   This is the exact path the sparse production solver already uses
   on these matrices successfully, so nothing new here.
3. **Should `count_2x2_inertia_val` switch to the scaled test too?**
   Probably not. Inertia counting needs a clean integer answer and
   absolute floors are the simpler story there. Revisit only if a
   bench failure points to it.
