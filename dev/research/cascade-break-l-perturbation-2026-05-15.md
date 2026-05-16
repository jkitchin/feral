# Cascade-break L-factor perturbation forensics — 2026-05-15

Carried-forward item from session 2026-05-15-02. The original premise
was that `PerturbToEps`'s docstring promised a Weyl-bounded
perturbation (`||Δ||_∞ ≤ eps` per rejected pivot) and on
`robot_1600_0004` the unrefined solve was producing a relative
disagreement of `~1.4×10⁻⁵` vs the bound, suggesting a bug. This note
records the forensics that walked back that diagnosis. Conclusion:
the docstring bound was wrong, but the code was self-consistent. A
proposed code "fix" made the residual five orders of magnitude worse.
The feature is now **opt-in** rather than auto-armed by default.

## Mechanism — what the code actually does

The cascade-break trigger (`src/numeric/factorize.rs` ~line 1869-1890)
installs a per-supernode BK policy override when
`n_delayed_in / expanded_ncol ≥ cascade_break_ratio`:

```rust
let on_zero = match params.cascade_break_eps {
    Some(eps) => ZeroPivotAction::PerturbToEps { abs_floor: eps },
    None => ZeroPivotAction::ForceAccept,
};
```

The `ForceAccept` branch (`src/dense/factor.rs:2623-2630`) handles a
rejected pivot by zeroing the L column below diagonal, setting
`D[k,k] = 0`, and returning `PivotOutcome::Rejected`, which causes
`finish_1x1_outcome` to skip `do_1x1_update`. The Schur update is not
applied for this column. Solve later sees `D[k,k] = 0` and skips
position k.

The `PerturbToEps` branch (`src/dense/factor.rs:2633-2645`) instead:

1. Sets `D[k,k] = sign(d) · max(|d|, eps)`.
2. Returns `PivotOutcome::Accepted`.

`Accepted` causes `finish_1x1_outcome` to call `do_1x1_update`, which
reads the perturbed `d_new`, computes `inv_d = 1.0 / d_new`, and
multiplies the L column by it:

```rust
let inv_d = 1.0 / d_new;
for i in (k + 1)..n {
    a[k * n + i] *= inv_d;  // L[i,k] = A[i,k] / d_new
}
```

The Schur update then reduces the trailing matrix by
`A[i,k] · A[j,k] / d_new`.

## The "bug" that wasn't — and why the math is self-consistent

The naive Weyl-bound argument says: if `D[k,k]` shifts by at most
`eps`, then `||Δ||_∞ ≤ eps` and eigenvalues shift by at most
`||Δ||_2 ≤ eps`. The original docstring claimed exactly this.

That argument is wrong because `Δ` is *not* localised to entry
`(k,k)`. The factorization satisfies `L · D · L^T = A + Δ` exactly,
but in row/column `k` we have

```
(L·D·L^T)[i,k] = L[i,k] · D[k,k] · L[k,k]
             = (A[i,k]/d_new) · d_new · 1
             = A[i,k]
```

so `Δ[i,k] = 0` (off-diagonal column-k entries are preserved exactly,
modulo roundoff). And on the diagonal `Δ[k,k] = d_new − d_orig`,
bounded by `eps + |d_orig|`. So far so good.

The perturbation actually shows up in the *Schur update*. The trailing
submatrix is reduced by `L[i,k] · D[k,k] · L[j,k] = A[i,k] · A[j,k] /
d_new`, whereas the "true" factorization with `d_orig` would have
reduced by `A[i,k] · A[j,k] / d_orig`. The difference per off-diagonal
pair `(i,j)` is

```
Δ_schur[i,j] = A[i,k] · A[j,k] · (1/d_new − 1/d_orig)
```

For `d_orig ≪ d_new ≈ eps`, this is bounded by `||A[k+1:,k]||² ·
(1/eps − 1/d_orig)` in magnitude — *not* by `eps`. The bound on `Δ`
is `||A||² / eps`-scale in the worst case, the opposite of what the
docstring claimed.

However: the factorization is internally self-consistent. The
unrefined solve `LDL^T x = b` solves `(A + Δ) x = b` for the specific
`Δ` above. The `1/d_new` factor in `L[i,k]` cancels the `d_new` in
`D[k,k]` during forward/backward substitution as long as we walk the
same factors we built. So the unrefined residual `||Ax − b||` depends
on `cond(A+Δ) · ||Δ||/||A||` and stays small whenever the corpus-wide
`A[:,k]` entries are not pathologically large.

## Direct measurement — robot_1600_0004

Probe: `src/bin/probe_cascade_perturb.rs`. Three configurations of
`Solver`, each factored on `robot_1600_0004.mtx`, then solved against a
deterministic RHS (no iterative refinement):

| config | inertia | residual `||Ax−b||_∞/||b||_∞` |
| ---- | ---- | ---- |
| cb=off (no cascade-break) | (14399, 9601, 0) | 6.24e-7 |
| cb=default (Some(0.5), Some(1e-10)) | (14399, 9601, 0) | 1.06e-5 |
| cb=fa (Some(0.5), None — ForceAccept) | (14398, 9601, 1) | 2.10e+2 |

Conclusion: `cb=default` adds a single order of magnitude to the
unrefined residual on this matrix (`6e-7 → 1e-5`), well within range
for iterative refinement to recover. Inertia is preserved. `cb=fa`
(ForceAccept) is much worse — sign-loss in the (k,k) slot during
solve.

The original `~1.4×10⁻⁵` ratio from session 2026-05-15-02 does not
reproduce on the current tree under this probe; the actual relative
solve-diff vs `cb=off` is `~7×10⁻⁸`. The session-02 number may have
been from a different probe setup; doesn't matter, the residual is
the right metric.

## Tried-and-rejected: "zero L on PerturbToEps"

Initial fix attempt (drafted in this note's first revision): mirror
`ForceAccept`'s structure in `PerturbToEps` — zero the L column below
diagonal after writing the perturbed `D[k,k]`, return
`PivotOutcome::Rejected` so `do_1x1_update` is skipped. Predicted
post-fix residual: `~1e-14` (machine-precision LAPACK
static-pivoting bound).

Actual post-fix residual on `robot_1600_0004`: **2.13×10³**. Five
orders of magnitude *worse* than cb=default and seven orders worse
than cb=off.

Reason the fix failed: with L zeroed but `D[k,k] = d_new ≈ eps`, the
solve does `x[k] = (rhs[k] − Σ L[k,j]·x[j]) / d_new`. Without a live L
column to cancel, that `/d_new` factor goes straight into the
solution: `x[k]` blows up by `1/eps ≈ 10¹⁰`. The factorization
satisfies `LDL^T = A + Δ` with `||Δ||_∞ ≤ ||A[k+1:,k]||_∞` (the
static-pivoting bound), but `cond(A+Δ)` doesn't matter when the
diagonal contribution to `x[k]` is divided by `eps`.

Fix reverted. Recorded in `dev/tried-and-rejected.md`.

## Resolution

The premise — "fix the L perturbation bound to match the docstring" —
was wrong. The code is self-consistent; the docstring's bound was
wrong. Two changes landed instead:

1. **Docstring corrected** on `ZeroPivotAction::PerturbToEps`
   (`src/dense/factor.rs`) and `Solver::with_cascade_break_eps`
   (`src/numeric/solver.rs`). The new docstring describes the actual
   `Δ` structure (Schur-update perturbation, bounded in `||A||²/eps`
   rather than `eps`), references this note, and points at LAPACK
   static pivoting and MA57 `cntl(4)` as the closest published
   precedents.

2. **Cascade-break defaults flipped to off.** Previously
   `NumericParams::default()` armed `cascade_break_ratio = Some(0.5)`,
   `cascade_break_eps = Some(1e-10)` automatically. Now both are
   `None` by default. MUMPS and MA57 don't have an equivalent
   feature; auto-arming a non-standard mechanism by default was
   creating surprising behavior across the corpus and downstream
   tooling.

   Callers that want the `pinene_3200`-style cascade-absorption
   speedup (88.6s → 34ms on `_0009`) opt in explicitly via
   `Solver::with_cascade_break(0.5).with_cascade_break_eps(1e-10)`.

## What the win-case (pinene_3200_0009) needs now

Untested at the time of this note (the explicit opt-in API exists and
the underlying mechanism is unchanged, so the win should be
recoverable with one builder call). Suggested test for next session:

```rust
let mut s = Solver::new()
    .with_cascade_break(0.5)
    .with_cascade_break_eps(1e-10);
```

If pinene_3200_0009 factor time is still ~34 ms with this opt-in, the
feature is preserved. If not, there is regression in the mechanism
itself, separate from the default-arming change.

## Decisions

- Cascade-break is **opt-in** (NumericParams::default() returns
  `cascade_break_ratio: None, cascade_break_eps: None`).
- `PerturbToEps` documentation now honestly describes the
  perturbation structure (not a Weyl-localised `eps` bound).
- The proposed "zero L on PerturbToEps" code change is rejected.
