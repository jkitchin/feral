# Plan: Pivot threshold consistency between factor and solve

## Goal

Fix the threshold mismatch where `factor` flags a pivot as numerically
zero (counted in `inertia.zero`, no division during factorization) but
`solve` then divides by the same pivot's stored value, producing garbage.
This is the root cause of POLAK6_0021's catastrophic 8.97e-1 residual
and is likely behind a meaningful fraction of the remaining ~400
residual failures on the 153k corpus.

## Evidence (from POLAK6_0021 triage)

```
=== POLAK6_0021 ===
n=9, condition number ~1e46

Default zero_tol (2.22e-14): inertia (4,1,4), residual 8.97e-1
Strict   zero_tol (1e-30):  inertia (5,4,0), residual 34.4
Loose    zero_tol (1e-3):   inertia (3,1,5), residual 4.6e-17  ← machine precision
```

The loose-tol experiment proves the matrix is solvable (residual at
machine precision is achievable). The default tol gets a *worse*
answer because pivots in the band [EPSILON·1e-10, zero_tol] are
flagged as zero by factor but **divided by** in solve.

## Bug

`src/dense/factor.rs:114`:
```rust
if d.abs() <= params.zero_tol { /* count zero, force accept */ }
// where zero_tol default = 100 * f64::EPSILON ≈ 2.22e-14
```

`src/dense/solve.rs:205`:
```rust
if d.abs() > f64::EPSILON * 1e-10 {  // ≈ 2.22e-26
    w[k] /= d;
}
```

Mismatch: pivots in `[2.22e-26, 2.22e-14]` are factored as zero
but solved as nonzero. Dividing a residual of magnitude `b ≈ 1e40`
by a pivot of `1e-14` produces a `1e54`-magnitude entry that then
contaminates the entire back-substitution.

`src/numeric/solve.rs:100`:
```rust
if ff.d_diag[k].abs() > 0.0 {
    w[k] /= ff.d_diag[k];
}
```

Sparse path is even worse — divides by any non-zero value, including
1e-300.

## Fix

Store `zero_tol` and `zero_tol_2x2` in both `Factors` and
`FrontalFactors` (populated from `BunchKaufmanParams` at factorization
time). Then both `solve` (dense) and `solve_sparse` (sparse) consult
the stored threshold:

```rust
// 1×1 block
let d = factors.d_diag[k];
if d.abs() > factors.zero_tol {
    w[k] /= d;
}
// else: leave w[k] alone — pivot was force-accepted as zero

// 2×2 block
let det = a*c - b*b;
if det.abs() > factors.zero_tol_2x2 {
    // normal 2×2 inverse
} else {
    // skip — block is near-singular
}
```

This is a strict superset of the current behavior:
- Well-conditioned matrices: nothing changes (no pivots fall below
  zero_tol).
- Force-accepted pivots: now correctly skipped in solve, producing
  the "least-squares-like" solution that the loose-tol experiment
  validated as machine-precision-correct on POLAK6_0021.

## Test-First

### Test 1: Threshold consistency invariant
For any matrix factored with ForceAccept, every pivot counted in
`inertia.zero` must satisfy `|d_diag[k]| <= factors.zero_tol`. If the
fix is implemented as "store zero_tol and skip in solve", this is
guaranteed by construction; the test serves as a regression guard.

### Test 2: POLAK6_0021 regression
Load POLAK6_0021, factor with default params, solve_refined,
assert residual < 1e-6. (Currently 8.97e-1.) Gated `#[ignore]`
because the data file isn't committed.

### Test 3: Hand-built singular matrix
A small singular KKT (e.g., 3×3 with rank 2) where the residual
should be at machine precision after the fix and was bad before.

## Implementation Steps

1. Read existing tests that exercise `verify_factorization` to confirm
   they don't break (none should — they test SPD/well-conditioned
   matrices that don't trigger ForceAccept).
2. Add `zero_tol: f64` and `zero_tol_2x2: f64` fields to `Factors`
   and `FrontalFactors`.
3. Populate the fields in `factor` and `factor_frontal` from `params`.
4. Update `dense::solve::d_block_solve` to use `factors.zero_tol` /
   `factors.zero_tol_2x2`.
5. Update `numeric::solve::solve_sparse` (Phase 2 D-block solve) to
   use `ff.zero_tol` / `ff.zero_tol_2x2`.
6. Run all tests (expect a few new field initializers in any mock
   `FrontalFactors`).
7. Run the POLAK6 triage example, expect residual ~ machine precision.
8. Run the bench, record delta.

## Acceptance

1. POLAK6_0021 residual < 1e-6 (was 8.97e-1).
2. Full `cargo test` passes.
3. `cargo clippy -- -D warnings` clean.
4. Bench numbers improve (residual pass count goes up, worst residual
   on dense path drops from 8.97e-1).
5. No worse on any matrix (best-iterate refinement guarantees this for
   any matrix where the unrefined solve was already correct).
