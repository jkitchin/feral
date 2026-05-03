# Research Note: KKT-augmented LS pivot_threshold default

**Status:** Pre-implementation
**Date:** 2026-05-02
**Issue:** https://github.com/jkitchin/feral/issues/2
**Related decisions:** `dev/decisions.md:212-269` (Phase 2.2.2,
2026-04-12), `dev/decisions.md:325-344` (Phase 2.3 dense/sparse split,
2026-04-13)
**Key references:**
- `src/dense/factor.rs:248-260` — `BunchKaufmanParams::default`
- `src/dense/factor.rs:2150-2316` — `factor_step_frontal` (1×1/2×2
  selection, det-floor rejection, rook fallback)
- `src/dense/factor.rs:2333-2403` — `try_reject_1x1_frontal`
- `src/dense/factor.rs:2422-2500` — `try_reject_1x1_with_rook_rescue`
- `src/numeric/factorize.rs:25-46` — `NumericParams` (currently
  `#[derive(Default)]`)
- `dev/research/scaling-aware-pivot-rejection.md` (MUMPS/SSIDS
  threshold semantics)
- `dev/research/rook-rescue.md` (rescue invariants)
- ripopt `src/linear_solver/feral_direct.rs:92-101` — KKT consumer
  config that surfaced the bug

## Overview

ripopt issue #2 reports that on the CUTEst/Mittelmann problem
`arki0003` (n=1872, m_eq=447, m_ineq=1691), feral's `FeralLdl`
returns `y_d = -0.0` on 58 of the 1691 inequality multipliers during
the iter-0 least-squares multiplier estimate. Ipopt 3.14 with MA27
(`cntl[1] = 1e-8`) returns `y_d ≈ 0.8` on the same rows on the same
matrix at the same iterate. The downstream effect is iter-0 dual
infeasibility 2.22 vs Ipopt's 1.486 (~50% worse). The 58 zero rows
are *not* structurally zero — every one of them has
`row_max(J_d) = 1.0` and `nnz(J_d) = 2`.

ripopt's `FeralLdl::new` configures:
- `bk.pivot_threshold = 0.0` (feral default)
- `bk.zero_tol = 1e-10`
- `bk.zero_tol_2x2 = 1e-20`
- `bk.on_zero_pivot = ForceAccept`
- `scaling = ScalingStrategy::Identity`

This note traces the mechanism, distinguishes which paths in the BK
kernel are actually firing, and motivates the default change in
`dev/plans/issue-2-kkt-pivot-default.md`.

## The augmented system

ripopt's `compute_ls_multiplier_estimate_augmented` builds the
saddle-point matrix matching Ipopt's `IpLeastSquareMults`:

```
A = [ I    J^T  ]
    [ J   diag ]
```

where `diag` has `0` on equality rows and `-1` on inequality rows.
For arki0003: `n = 1872`, `m = 447 + 1691 = 2138`, so the system is
4010×4010 with `m > n`. There are 1041 linear-constraint redundancies
in `J`, so the augmented matrix is **rank-deficient by at least
1041**. This is a normal regime — Ipopt/MA27 handles it fine via
threshold partial pivoting.

After eliminating the (1,1) identity block, the Schur complement on
the bottom block is `diag - J·J^T`. For an inequality row with `nnz(J_d) = 2,
row_max(J_d) = 1`, the Schur diagonal becomes
`-1 - 1² - 1² = -3` — well-conditioned in isolation. The 58 reported
zero rows do *not* arise from a small original diagonal.

## What rejection paths actually fire

The dense BK kernel inside the multifrontal driver has two
independent rejection mechanisms.

### Path A — 1×1 absolute floor

`try_reject_1x1_frontal` (`factor.rs:2333-2403`):

```rust
let threshold = (params.pivot_threshold * col_max).max(params.zero_tol);
if d.abs() <= threshold {
    if may_delay { return Ok(PivotOutcome::Delayed); }
    if d.abs() <= params.zero_tol {
        // case (a): zero L column AND diagonal — solve skips this position
        for i in (k+1)..nrow { a[k*nrow + i] = 0.0; }
        a[k*nrow + k] = 0.0;
        return Ok(PivotOutcome::Rejected);
    }
    // case (b): small but nonzero — accept with sign, flag refinement
    ...
}
```

With `pivot_threshold = 0.0`, the threshold collapses to `zero_tol`
(`1e-10` in ripopt's config). A pivot of magnitude `-3` is never
rejected here. **Path A alone does not explain the symptom.**

### Path B — 2×2 SSIDS scale-invariant det-floor

`factor_step_frontal` at `factor.rs:2150-2243` first applies BK's
α-test:

```rust
if akk * gamma_r >= alpha * gamma0 * gamma0 { /* 1×1 */ }
```

When the diagonal `|akk|` is dominated by an off-diagonal of
magnitude 1.0 (the J^T row coupling), the α-test routes the
candidate to the 2×2 path. The 2×2 block then faces two checks:

1. Duff-Reid growth (`growth_fail`, line 2210-2211): trivially
   satisfied at `pivot_threshold = 0.0` (`(... ) * 0 ≤ |det|`).
2. SSIDS scale-invariant det-floor (`det_floor_fail`,
   line 2232-2243): **fires regardless of `pivot_threshold`**. It
   rejects when `|detpiv| < max(SSIDS_DET_SMALL,
   |detpiv0|/2, |detpiv1|/2)`.

When the 2×2 is rejected, the code falls through to a 1×1 retry at
column `k` via `try_reject_1x1_with_rook_rescue`
(line 2265-2279).

### Why rook rescue is dead at threshold = 0

`try_reject_1x1_with_rook_rescue` (`factor.rs:2438-2500`):

```rust
let threshold = (params.pivot_threshold * col_max).max(params.zero_tol);
if d.abs() > threshold {
    // fast path: pivot clears threshold → delegate, NO rook attempt
    return try_reject_1x1_frontal(...);
}
// only here does rook_rescue get a chance
```

With `pivot_threshold = 0.0` and any non-zero `|d|`, the fast path
fires and rook is skipped. The 1×1 then succeeds (because
`|d| > 1e-10`). **Rook rescue is dead code at `pivot_threshold = 0`.**

This means the entire infrastructure built for handling exactly this
case (rank-deficient saddle structure, zero diagonals coupled to
unit-magnitude off-diagonals) — rook rescue, delayed pivoting at
parent supernodes — is gated behind a non-zero `pivot_threshold`.
Ripopt's `0.0` configuration never engages any of it.

## Where `y_d = -0.0` actually originates

For `y_d = -0.0` to come out of solve, some column in the factor
must have been hit by the case-(a) zero-out (lines 2380-2384) where
both the L column and the diagonal are set to 0. With
`pivot_threshold = 0.0`, this requires `|d| ≤ zero_tol = 1e-10`.

The 58 zeroed inequality rows cluster at `_scon[2052..2138]` — the
**very tail** of the elimination order. AMD pushes saddle/constraint
rows to the tail, so by the time the kernel reaches them their
diagonals have absorbed Schur updates from every earlier rejection
upstream. The 1041 redundancies in J guarantee 1041 columns
collapse to near-zero somewhere; the 58 reported zeros are the
downstream consequence of upstream 2×2 det-floor rejections feeding
case-(a) zero columns into the elimination tail.

The chain is:

1. Earlier in the elimination, a 2×2 block on a saddle pair is
   rejected by `det_floor_fail`.
2. The 1×1 fallback goes through the rook-rescue wrapper, but the
   fast path skips rook (because `|d|` is still above `1e-10` even
   though it's tiny relative to the column max).
3. The 1×1 is "accepted" with a tiny diagonal, producing a huge
   `1/d` rank-1 update.
4. That rank-1 update propagates large-magnitude noise into
   downstream columns. After enough such updates, downstream
   diagonals collapse below `1e-10` from cancellation.
5. Those collapsed diagonals hit case-(a) and zero out the L column
   plus the diagonal.
6. Solve sees zero L column → `y_d = -1 × 0 = -0.0`.

MA27 with `cntl[1] = 1e-8` avoids step 2-3 entirely: at threshold
`u = 1e-8`, a 1×1 candidate with `|d| < 1e-8 × col_max` is rejected
and MA27 swaps to find a stable off-diagonal pivot. The off-diagonal
J^T entry of magnitude 1.0 makes the 2×2 saddle block well-defined
even when both diagonals are tiny. **Threshold partial pivoting is
exactly the rescue mechanism for this configuration.**

## Why the dense vs sparse split (2026-04-13) is the right precedent

`dev/decisions.md:325-344` already documents that sparse multifrontal
callers should use `pivot_threshold = 0.01` and dense callers `0.0`.
The reasoning then was:

> The column-relative threshold test `|d| >= u·col_max` only pays
> off when rejected pivots have somewhere to go — delayed pivoting
> at non-root supernodes gives them a landing zone at the parent.
> The dense BK kernel has no delayed-pivoting machinery and runs
> under Knight-Ruiz ∞-norm equilibration, which handles column
> scaling at preprocess time.

That reasoning still holds. Sparse has rook rescue
(2026-04-23 Phase 2.4.3) and delayed pivoting (2026-04-15 Phase 2.3)
as landing zones; dense has neither.

The gap is that the **default** never moved to track the
sparse-caller convention. Every in-tree sparse caller — feral's own
`bench.rs`, `bench_solver_corpus.rs`, `parallel_corpus_parity.rs`,
all `tests/parity.rs`, `tests/delayed_pivoting.rs`, etc. — overrides
to `0.01`. ripopt is the one consumer that does
`NumericParams::default()` and inherits `0.0` because the default
is set on `BunchKaufmanParams::default()`, which is dense semantics.

## Decision

Override `NumericParams::default()` to set `bk.pivot_threshold =
1e-8` (MA27's `cntl[1]` default, also Ipopt's `ma27_pivtol`
default). Keep `BunchKaufmanParams::default()` at `0.0` so the
dense `factor()` entry point is unchanged. The dense benchmarks
and the `tests/threshold_consistency.rs` dense path continue to
construct `BunchKaufmanParams` directly and are unaffected.

Why `1e-8` and not the SSIDS/MUMPS canonical `0.01`:

- The reference solver in issue #2 (Ipopt + MA27) uses `cntl[1] =
  1e-8`, not `0.01`. ripopt is replacing Ipopt as a drop-in: matching
  Ipopt's reference choice is the most direct "this fixes the
  reported bug" justification.
- `0.01` is the SSIDS/MUMPS default validated on **MC64-equilibrated**
  matrices, where `|d| >= 0.01 * col_max` rejects pivots that are
  tiny *relative to a normalized column*. ripopt runs Identity
  scaling at the linear-solver layer (it owns scaling at a higher
  layer to preserve the inertia signal — see ripopt
  `feral_direct.rs:84-91`), so the threshold fires on raw-value
  ratios that have not been equilibrated. `0.01` on un-equilibrated
  matrices is more aggressive than its validation envelope; `1e-8`
  is the conservative MA27 baseline tested for exactly this
  configuration.
- Feral's in-tree sparse callers (`bench.rs`,
  `bench_solver_corpus.rs`, `parity.rs`, etc.) continue to override
  to `0.01` because they run with InfNorm/Auto scaling (their
  configuration matches the SSIDS validation envelope). The
  default-vs-explicit asymmetry is intentional: the default tracks
  the consumer profile (Identity-scaled, MA27-equivalent), the
  explicit overrides track the validation profile.
- `1e-8` activates the same rescue infrastructure (rook rescue,
  delayed pivoting) that `0.01` does — both are non-zero. The
  difference is rejection aggressiveness, not which paths fire.

The dense path stays at `0.0` because changing it requires the
Bratu3d analysis of 2026-04-25 and would touch the dense BK77
parity tests that compare against the original Bunch-Kaufman paper.
That is out of scope for issue #2.

## Test strategy

A new regression test must:

1. Construct a small synthetic saddle-point matrix
   `[I  J^T; J  diag]` with `m > n` and at least one redundant
   constraint, where the diagonal coupling on inequality rows is
   `-1` and the off-diagonal entries in J have magnitude 1.0.
2. Factor it with `NumericParams::default()` and verify the
   resulting solve does not return exact-zero entries on
   non-structurally-zero inequality rows.
3. (Negative control) Factor with
   `NumericParams::with_bk(BunchKaufmanParams::default())` and
   verify the `pivot_threshold = 0.0` baseline reproduces the
   zero-output symptom — this guards against the test silently
   passing if some other change later makes the rescue
   unnecessary.

The synthetic matrix should be small enough to be a unit test
(`n ≤ 20`, `m ≤ 30`) but structurally similar to arki0003's tail
58 rows: a row of J with `±1` in two columns, paired with a `-1`
diagonal in the (2,2) block, embedded in a system with linear
redundancies.

The acceptance gate is:

- All existing 146+ tests pass without tolerance changes.
- New test passes under the new default and reproduces the failure
  under the explicit `pivot_threshold = 0.0` override.
- `cargo run --bin bench --release` benchmark suite shows no
  residual regressions (it already runs at `0.01`, so the change
  is a no-op there — confirms no test/default skew).

## Out of scope

- Changing dense `BunchKaufmanParams::default()`. (Would touch
  Bratu3d, BK77 parity tests; tracked separately.)
- ripopt's existing `set_pivot_threshold(1e-8)` workaround for
  iter-0 LS init (referenced in `feral_direct.rs:128-131`). After
  this change ripopt can drop the explicit override; that's a
  ripopt-side cleanup tracked in ripopt's repo.
- Migrating the `inertia.zero` reinterpretation TODO at
  `feral_direct.rs:163-167` into feral. Logged but not addressed
  here.
