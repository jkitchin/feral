# Plan — MC64 catastrophic-spread guard (issue #45)

Date: 2026-05-20
Research note: `dev/research/kkt-mc64-scaling-blowup-2026-05-20.md`
Scope: correctness fix for #45. #46 (performance) is explicitly out of scope.

## Goal

`ScalingStrategy::Auto` must never apply an MC64 scaling vector whose own
spread `max|s| / min|s|` exceeds `1/EPS`. Such a scaling is degenerate
to working precision and silently corrupts the factorization. When MC64
returns such a vector, fall back to the InfNorm scaling already computed
in the same routine.

## Change set (all in `src/scaling/mod.rs`)

1. **New `Mc64FallbackReason` variant** — `Mc64ScalingDegenerate`, with a
   doc comment pointing at the research note. Append after the existing
   two variants (the enum is small and exhaustively matched in a few
   places — grep `Mc64FallbackReason::` and update any `match`).

2. **New constant** in `compute_scaling_auto_with_cache`:
   `const MC64_SPREAD_GUARD: f64 = 1.0 / f64::EPSILON;` (≈ 4.503e15),
   with a comment: corpus max is 3.27e15 (ssine); CHO catastrophe 3e82.

3. **Restructure the MC64 branch** of `compute_scaling_auto_with_cache`:
   - Compute `(mc_vec, mc_info) = mc64_from_cache(matrix)?` **once**,
     up front, immediately after the `IN_SPREAD_GUARD` early return.
   - Immediately check `scaling_spread(&mc_vec) > MC64_SPREAD_GUARD`; if
     so, `return Ok((in_vec, Mc64FallbackToInfnorm { reason:
     Mc64ScalingDegenerate }))`.
   - The existing `raw_diag_range` fast-path and `mc_off` Policy-4
     diagnostic then reuse the already-computed `mc_vec` / `mc_info`
     instead of recomputing. Behaviour for every non-degenerate matrix
     is unchanged (same vector returned, same `ScalingInfo`).

   Note `mc64_from_cache` is currently a closure used in two places; it
   stays, but is now called exactly once.

## Tests first (`src/scaling/mod.rs` `mod tests`)

Oracle policy: residual `‖A·x−b‖` is a mathematical identity (not a
fabricated oracle); the corpus spread numbers are empirical measurements
recorded in the research note. Both are admissible external oracles.

T1. `scaling_spread` sanity — hand-calculated: `[1e-3, 1.0, 4.0]` →
    spread 4000. (Guards the helper the new check depends on.)

T2. Catastrophic-spread fallback — construct a synthetic matrix whose
    MC64 scaling spread exceeds `MC64_SPREAD_GUARD`. A graded chain
    (entries stepping by a large constant factor so Hungarian dual
    potentials accumulate) reproduces the blow-up at small `n`. The
    construction is validated by `probe_mc64_spread`-style measurement
    *before* the test is written — the test asserts the measured fact.
    Assert: `compute_scaling(Auto)` returns the InfNorm vector and
    `ScalingInfo::Mc64FallbackToInfnorm { Mc64ScalingDegenerate }`.

T3. Non-regression — a well-conditioned arrow-KKT (`shape_csc`, the
    existing helper) still gets MC64 (spread far below the guard);
    assert the guard does **not** fire.

T4. End-to-end correctness — factor + solve T2's matrix through a
    default `Solver` (which uses `Auto`); assert `‖A·x−b‖ / ‖b‖` is at
    InfNorm quality (≤ 1e-6), i.e. the fallback produced a usable
    factorization rather than the garbage MC64 would have.

## Session verification (not committed tests)

- `probe_issue45_ordering` / `probe_issue46` against the real CHO KKT:
  confirm `Auto` now solves it (rel res ~2.46e-8) instead of garbage.
- Full `cargo test` — the existing scaling tests
  (`pick_scaling_strategy_*`, Policy-4 fallback tests, MSS1/ACOPP30
  regression tests) must stay green; their matrices have spread ≪ guard.
- `cargo run --bin bench --release` — record numbers in the checkpoint.

## Risk

Low. The guard adds one O(n) comparison on a vector already computed.
It can only change behaviour for a matrix whose MC64 spread exceeds
`1/EPS` — empirically none in the parity corpus. The fallback target
(InfNorm) is already computed unconditionally on this path.

## Out of scope (documented, not done)

- #46 performance (InfNorm 28M-nnz / 11 s cascade on the CHO KKT). Needs
  a saddle-point-stable matching scaling or an MC64 permutation-to-
  diagonal. Separate follow-up.
- Guarding the **explicit** `ScalingStrategy::Mc64Symmetric` path. The
  default `Solver` uses `Auto`; the explicit path is an informed user
  choice. Noted as a residual in the research note.

## Outcome (2026-05-20, executed)

Implemented as planned; one test-oracle divergence found and recorded.

`src/bin/probe_mc64_synth` built a parameter-estimation saddle KKT to
serve as a committed oracle. Two empirical findings (journal
2026-05-20-02 16:34):

1. MC64 symmetric spread scales as ≈ `base^(2·nx)` on this family —
   far steeper than the planned `base^(nx/2)` estimate.
2. A constant-ratio chain couples the MC64 blow-up to **genuine**
   ill-conditioning: by the time `mc_spread` crosses `1/EPS`
   (`base ≈ 1.3`) the bidiagonal `B` is already exponentially
   ill-conditioned and InfNorm's solve is garbage too. No synthetic
   `base` is simultaneously well-conditioned and MC64-degenerate.
   That combination (cond 1.4e15, MC64 spread 3e82) is specific to
   the real CHO KKT, where the blow-up is a saddle-point matching
   pathology, not raw ill-conditioning.

Consequently a committed end-to-end "guard fires → correct solve"
test on a synthetic matrix is not achievable. Final committed tests
(all in `src/scaling/mod.rs`):

- T1 `scaling_spread_hand_oracle` — hand calculation.
- T2 `auto_falls_back_on_catastrophic_mc64_spread` — `base=4.0`,
  MC64 spread 3.34e94 > guard → guard fires, `Mc64ScalingDegenerate`.
- T3 `auto_keeps_mc64_when_spread_below_guard` — `base=1.1`, MC64
  spread 9.31e6 < guard, MC64 branch genuinely reached → guard does
  NOT fire (was planned as `shape_csc`, but `shape_csc` triggers
  `IN_SPREAD_GUARD` first and never reaches the new guard).
- T4 `auto_solves_below_guard_matrix_correctly` — `base=1.1`,
  factor+solve via `Auto`, relres ≤ 1e-6 (residual identity oracle);
  proves the guard does not regress a matrix that reaches the new
  code path. (Planned as end-to-end on T2's matrix; T2's matrix is
  ill-conditioned so the relres oracle moved to the `base=1.1`
  matrix.)

The genuine "fallback rescues a real solve" evidence is the CHO KKT
via `probe_issue45_ordering` (Auto: relres 7.15e11 → 2.46e-8),
recorded in the session checkpoint and journal — not a committed
unit test, since the matrix is 43332×43332 and lives outside the
repo.
