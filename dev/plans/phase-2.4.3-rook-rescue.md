# Phase 2.4.3 — Rook Pivoting as Rescue Path

**Status:** Pre-implementation plan (pre-research-note)
**Date:** 2026-04-23
**Related:** `dev/plans/phase-2.4.1-blocked-ldlt.md`, `dev/plans/scaling-aware-pivot-rejection.md`
**Targets:** collapse the dense factor-ratio tail (CRESC100/GAUSS2 at 40–45×
vs MUMPS) without regressing the p90 gains from Phase 2.4.1b.

## Motivation

Current bench (head, 2026-04-23) has dense factor/MUMPS p90 = 1.83
(target ≤ 2.0 met), but the max = 45.14 and the top-10 worst list is
dominated by two families:

- CRESC100 (n≈806): 40–45×
- GAUSS2 (n≈758): 41–44×

These are almost certainly delay-heavy — BK-partial cannot find a pivot
that clears the column-relative threshold, so columns are delayed up
the etree, the parent supernode grows, AMD's fill prediction is wrong,
the parent becomes expensive. Rook pivoting searches row-then-column
for a local-max pivot and almost always clears the threshold on the
first try, breaking the cascade.

Corollary: FERAL's LDLT-aware ordering preprocessing (commit 1ffbd7c)
bets that factorization will follow the predicted fill pattern. With
BK-partial and delayed pivots, that bet partially fails. Rook is the
numerical half of the same bet.

## Design: Rescue, not Top-Level

Rook is **not** a top-level pivot strategy selected by a params flag.
It is a per-pivot rescue path spliced into the existing flow at the
rejection site. Rationale:

1. Well-conditioned matrices never hit rejection. They pay zero rook
   cost. This is ~99% of the 154k-matrix corpus.
2. Ill-conditioned matrices hit rejection often. Rook rescues those
   pivots instead of delaying them. The cost is paid exactly where
   the benefit accrues.
3. No dispatch policy or user flag needed. Auto by construction.
4. This matches how HSL MA57 operates (partial pivoting with rook
   fallback), which is the behavior Ipopt consumers expect.

**Splice point:** inside `try_reject_1x1_frontal` at
`src/dense/factor.rs:1520`, between the threshold test and the
delay/reject branch. Pseudocode:

```
fn try_reject_1x1_frontal(...) -> PivotOutcome {
    let d = a[k*nrow + k];
    let threshold = (pivot_threshold * col_max).max(zero_tol);

    if d.abs() <= threshold {
        // NEW: try rook rescue before giving up.
        if let Some(rook_result) = rook_rescue(a, nrow, ncol, k, params) {
            // Apply the (row, col) swap rook chose; the caller's do_1x1_update
            // (or do_2x2_update for 2x2 pivots) then runs normally.
            apply_rook_swap(a, nrow, k, rook_result, perm);
            return Ok(PivotOutcome::AcceptedAfterSwap(rook_result.kind));
        }

        // Existing behavior: delay or force-accept.
        if may_delay { return Ok(Delayed); }
        ...
    }
    ...
}
```

## Rook Search Algorithm

Standard textbook rook (Duff & Reid 1996, Ashcraft-Grimes-Lewis 1998)
over the trailing fully-summed submatrix `A[k..ncol, k..ncol]` with
off-front contributions from `A[ncol..nrow, k..ncol]` treated as ghost
rows for the column-max computation:

```
1. Start at (i, j) = (k, k).
2. Let gamma_col = max_{r > j, r in [k, nrow)} |A[r, j]|.
3. If |A[i, j]| >= alpha * gamma_col, accept (i, j) as 1x1.
4. Else let i' = argmax row selected.
5. Let gamma_row = max_{c != i', c in [k, ncol)} |A[i', c]|.
6. If |A[i', i']| >= alpha * gamma_row, accept i' as 1x1 (swap into k).
7. Else if |A[i', j_prev]| >= alpha * gamma_row, accept (i', j_prev) as 2x2.
8. Else set (i, j) := (j, i') and goto 2 (bounded iterations, empirically <= 4).
```

Bounded-iteration guarantee: the sequence `|A[i_0, j_0]| < |A[i_1, j_1]|
< ...` is strictly increasing and bounded by the submatrix max, so
terminates in O(n) worst case. Ashcraft-Grimes-Lewis report empirical
mean of ~1.5 iterations on KKT problems.

**Swap logic.** When rook picks (i, j) != (k, k):
- Swap columns `j` and `k` in `A` (symmetric swap — affects both
  column-major storage and implicit row data).
- Swap rows `i` and `k`.
- Record in `perm[k] <-> perm[i]` (and perm_inv).
- For 2×2 rook pivots, two column swaps and two row swaps are applied.

**Ghost rows.** The rook search scans rows `k..nrow` for column max
(matches BK's gamma0 domain) but restricts pivot candidates to fully-
summed rows `k..ncol` only. A pivot in rows `ncol..nrow` cannot be
eliminated at this front; those rows contribute to the threshold test
but not to pivot selection. This is a feral-specific adaptation of
the textbook algorithm — MA57 and SSIDS handle this via the
`nelim`/`nabove` split.

## Integration Points

1. **`scalar_pivot_step`** at `src/dense/factor.rs:1205`. Add
   `PivotStepResult::AdvancedAfterRookSwap(1 | 2)` variant so callers
   can distinguish rook-rescued pivots for logging/metrics. Control
   flow otherwise unchanged.
2. **`try_reject_1x1_frontal`** at `src/dense/factor.rs:1520`. Add
   the rescue call described above. Signature unchanged; outcomes
   include the new `AcceptedAfterSwap` variant.
3. **Blocked panel path** (`lblt_panel_frontal` at
   `src/dense/factor.rs:1022`). Rook rescue lives in `scalar_pivot_step`
   only. The panel fallback-on-rejection path (`PanelStatus::ScalarFallback`)
   already routes through `scalar_pivot_step`, so rook rescue fires
   transparently for blocked callers. The panel itself does not need
   rook awareness.
4. **`FrontalFactors`** does not change. Rook-rescued pivots produce
   the same `(L, D, perm, inertia)` shape as BK-partial pivots that
   happen to land at a different `k`; the only observable difference
   is that `perm` has more transpositions.
5. **Metrics.** Add a `n_rook_rescues: usize` field to `FrontalFactors`
   for diagnostics. This is non-API — test code and the bench harness
   consume it, production code ignores it.

## Parity Impact

Rook rescue **changes the output** on matrices that previously
delayed. That is the feature. Specifically:

- Matrices that never rejected: output unchanged bit-for-bit.
  (`test_spd_scalar_blocked_parity_size_sweep`, `test_frontal_ncol_lt_nrow_parity`,
  `test_kkt_regression_spot_checks` stay green unmodified.)
- Matrices that previously delayed: `nelim`, `n_delayed`, `perm`, and
  `L` change. Inertia **must not** change — rook is strictly more
  conservative about pivot size, so the sign pattern of D is stable.
- The existing `test_may_delay_rejection_parity` test will break by
  design. Replace with `test_may_delay_rejection_without_rescue` that
  constructs a matrix where even rook cannot find a pivot (singular
  column), plus a new `test_rook_rescues_delayed_pivot` that verifies
  the rescue fires and inertia matches.

## Test Plan

Six correctness tests, tests before code:

1. **Rook identity on SPD.** SPD matrices of the full size sweep never
   reject, so the rook-rescue code must be a no-op. Assert
   `n_rook_rescues == 0` and all outputs bit-identical to the pre-rook
   factor. This is the "zero cost on easy matrices" gate.
2. **Hand-computed rook example.** From Ashcraft-Grimes-Lewis 1998 or
   hand-traced: a 4×4 matrix where BK-partial delays but rook finds a
   valid 1×1 pivot. Assert rook rescue fires, pivot lands at the
   expected `(i, j)`, and the resulting L/D reproduce `A` exactly under
   `L·D·Lᵀ`.
3. **Rook 2×2 example.** Hand-traced 5×5 where rook picks a 2×2 block
   that BK-partial would have delayed. Verifies the 2×2 rook path.
4. **Inertia preservation under rescue.** Random indefinite matrices
   with `pivot_threshold = 0.1` (forces many BK rejections). Compare
   inertia against MUMPS oracle on each matrix. Hard gate: zero
   inertia mismatches.
5. **CRESC100 / GAUSS2 regression.** Spot-check 10 matrices from each
   family. Compare pre-rook and post-rook factor time, residual, and
   inertia. Assert: inertia matches MUMPS; residual does not worsen;
   factor time improves (one-sided test, `post <= pre * 1.1` tolerance
   for measurement noise).
6. **Bench tail gate.** Full corpus bench must show dense factor/MUMPS
   p90 ≤ 1.83 (current baseline) and max ≤ 20 (down from 45). geomean
   must not worsen by more than 2% (rook search overhead on matrices
   that don't need it).

## Implementation Order

One commit per step, tests before code.

- **Step 1.** Research note at `dev/research/rook-rescue.md`. Required
  by CLAUDE.md before any implementation. Covers: Duff-Reid 1996
  algorithm, Ashcraft-Grimes-Lewis 1998 bounded-iteration proof, MA57
  implementation reference, comparison to BK-partial on KKT growth.
- **Step 2.** Add `n_rook_rescues: usize` to `FrontalFactors`, wire
  through all constructors and tests. No behavior change. Byte-parity
  tests still pass unchanged (new field is 0 everywhere).
- **Step 3.** Write Tests 1–3 (rook identity, rook 1×1 hand-trace,
  rook 2×2 hand-trace) against a stubbed `rook_rescue` that always
  returns `None`. Test 1 passes; Tests 2–3 fail (RED).
- **Step 4.** Implement `rook_rescue` in a new file
  `src/dense/rook.rs`. Pure function — takes `&mut [f64]` trailing
  submatrix, returns `Option<RookPivot>` where `RookPivot` carries
  `{ kind: Pivot1x1 | Pivot2x2, row_swaps: ArrayVec<(usize, usize), 2>,
  col_swaps: ArrayVec<...> }`. Unit tests inside the module.
- **Step 5.** Splice the rescue into `try_reject_1x1_frontal`. Apply
  swaps via `a[]` rearrangement + `perm` update. Tests 2–3 flip to
  GREEN.
- **Step 6.** Add Test 4 (random indefinite inertia preservation).
  Iterate on rook edge cases until it passes. Also fix the existing
  `test_may_delay_rejection_parity` test — either update it to match
  the new behavior, or replace with a matrix that still delays.
- **Step 7.** Add Test 5 (CRESC100 / GAUSS2 regression). Requires the
  matrices from `data/matrices/kkt/`. Report factor-time improvements
  per matrix; commit only after all 20 matrices show improvement or
  no regression.
- **Step 8.** Full bench run. Test 6 gate. Write validation report to
  `dev/research/phase-2.4.3-rook-validation.md`.

## Risks

1. **2×2 rook logic.** The rook algorithm's 2×2 selection differs
   subtly from BK-partial's: BK compares `|a_kk| * gamma_r` against
   `alpha * gamma0^2` (Bunch-Parlett criterion), while rook uses the
   row-max of the off-column it settled on. Easy to get wrong at the
   boundary. Mitigation: Test 3 hand-traces a 2×2 case; Test 4 catches
   inertia mismatches.
2. **Swap bookkeeping.** Symmetric swaps in column-major lower-triangle
   storage are fiddly. The existing codebase has one symmetric swap
   helper (used in BK's row-exchange path); extend or reuse it.
   Mitigation: unit test swap primitives in isolation.
3. **Rook loop non-termination in degenerate cases.** Constant-valued
   submatrices make the `|A[i_k, j_k]|` sequence non-strictly-increasing.
   Mitigation: cap the rook loop at 8 iterations and fall through to
   delay/reject if it doesn't converge (Ashcraft-Grimes-Lewis use this
   safeguard).
4. **Blocked-panel interaction.** Rook rescue fires only inside
   `scalar_pivot_step`, which is called from the panel only on
   `ScalarFallback`. If rook forces a swap, the panel's `d_panel` state
   for columns `[k+c, k+n_elim)` may be stale relative to the new
   column ordering. Mitigation: on rook success, the panel caller must
   re-enter from the new `k` (treat rook rescue like a 2×2 pivot for
   control-flow purposes — full panel restart).
5. **Peek-ahead interaction during blocked path.** The blocked panel's
   peek-ahead (`peek_ahead_column`) reads staleness-corrected values
   for the pivot column but leaves the trailing matrix stale. Rook's
   row search would see stale rows. Mitigation: the panel never calls
   rook — it only falls back to scalar, which works on the materialized
   (non-stale) `a[]`. Explicit guard: rook rescue is a no-op if any
   panel state is pending (should never happen given the control flow,
   but assert for safety).

## Exit Criterion

Phase 2.4.3 is done when all of:

1. Tests 1–6 pass.
2. Full KKT bench passes with zero inertia regressions vs MUMPS
   oracle (hard gate, CLAUDE.md zero-tolerance rule).
3. Dense factor/MUMPS `max` ≤ 20 (down from 45.14). This is the main
   perf deliverable.
4. Dense factor/MUMPS `p90` ≤ 1.90 (no worse than 1.04 × current
   1.83). Ensures rook overhead doesn't poison the easy-matrix path.
5. Dense factor/MUMPS `geomean` within 2% of current (0.21).
6. Research note and validation report committed.

Items 1, 2, 3 are hard gates. If item 4 or 5 misses, rook rescue
ships gated behind `BunchKaufmanParams::enable_rook_rescue: bool`
(default true) so individual callers can opt out on perf-critical
paths. That fallback is the safety valve; the expectation is the
defaults work.

## Deferred to Phase 2.4.4 (if pursued)

- Rook in the **blocked panel** directly (not just via scalar
  fallback). Would require carrying panel state through a rook swap,
  which is non-trivial. Only worth doing if profiling shows the scalar
  fallback path is hot on rook-rescued matrices.
- Rook-aware **ordering bias** — if LDLT-aware ordering learns which
  columns need rook rescue, it could cluster them. Speculative; needs
  research.
