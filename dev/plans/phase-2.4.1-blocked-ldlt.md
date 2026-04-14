# Phase 2.4.1 — Blocked Dense LDLᵀ

**Status:** Pre-implementation plan (revised after mumps-expert + spral-expert + faer-expert consultation)
**Date:** 2026-04-14
**Related:** `dev/plans/phase-2-planning.md` §2.4.1, `dev/research/dense-ldlt.md`
**Baseline:** `dev/sessions/phase-2-baseline.md` (dense p90 vs MUMPS 2.27, sparse p90 3.18)
**Targets:** dense factor p90 ≤ 2.0 AND sparse factor p90 ≤ 3.0

## Split into 2.4.1a + 2.4.1b

Expert consultation (mumps-expert, spral-expert) showed that the
dense (factor_single_front, ncol=nrow) and sparse (factorize_multifrontal,
ncol<nrow) paths want *different* blocking strategies:

- **2.4.1a — MUMPS-style contribution-block deferral.** Only the
  contribution block `[ncol, nrow) × [ncol, nrow)` rank-1/rank-2 updates
  are deferred and applied as a single triangular rank-nelim update at
  the end of `factor_frontal`. Cross-strip
  `[ncol, nrow) × [k+1, ncol)` stays eager (needed for γ₀ of future
  pivots). Scalar pivot kernel unchanged; rejection logic unchanged;
  2×2 pivots unchanged. Targets **sparse p90 3.18 → 3.0**. Lower risk,
  smaller rewrite, useless on the dense KKT path (no contribution
  block when ncol=nrow).

- **2.4.1b — faer-style fully blocked kernel for factor_single_front.**
  W workspace + peek-ahead column update + panel termination on
  rejection + final triangular matmul. Targets **dense p90 2.27 → 2.0**
  via HAHN1 (n=715), ACOPR30 (n=564), AVION2. Higher risk, larger
  rewrite, pursued only after 2.4.1a ships.

Expert consensus on keeping 2×2 pivots: **retained.** KKTs with zero
diagonals need 2×2 blocks for inertia correctness; both MUMPS and
SSIDS's blocked kernels keep 2×2 machinery.

## Goal

Introduce a blocked Schur-complement update to `factor_frontal` so the
trailing submatrix is updated from accumulated panel columns in one
BLAS-3 sweep per block of `block_size = 64` pivots, instead of one
symmetric rank-1/rank-2 update per pivot. Must preserve:

- exact inertia correctness (Phase 2.8 hard rule)
- `try_reject_1x1_frontal` delayed-pivot semantics
- `may_delay` contract used by the multifrontal driver
- the SSIDS scale-invariant 2×2 det floor (added in session
  2026-04-14-01)
- the 2×2 normalized update (divide by `|a₁₀|`) already in the
  scalar kernel

## Reference: faer's blocked Bunch-Kaufman

From the spral-expert / faer-expert consultation in the companion
journal entry. faer's implementation lives at
`faer/src/linalg/cholesky/bunch_kaufman/factor.rs`:

1. **Right-looking panel with deferred trailing update** (xLASYF
   style). Outer driver is `lblt_blocked` at `factor.rs:700`; the
   per-panel routine is `lblt_blocked_step` at `factor.rs:490`.
2. **`W` workspace** of shape `(n-k) × block_size`, allocated once
   per panel at `factor.rs:718`. Column `j` of `W` holds the
   *unscaled* original column `j` of the trailing block; the
   *D-scaled* L column is written directly back into `A`. For 1×1
   pivots: `L_j = column_j / d_j`, `W_j = column_j`. For 2×2 pivots:
   `L_{2} = column_{2} · D_2x2^{-1}` (via the normalized formula),
   `W_{2} = column_{2}`.
3. **Peek-ahead update for pivot search**
   (`update_and_offdiag_argmax` at `factor.rs:461`). To find the
   pivot for column `i0`, reconstruct the would-be updated column
   via `dst = Ar[:,i0] − Al · Wl[i0,:]^H`, then argmax on that
   scratch column. Only the column needed for pivot selection is
   updated during the panel; the rest of `Ar` stays stale.
4. **Final BLAS-3 call:** `matmul(Ar, StrictTriangularLower,
   Accum::Add, W, Al^H, −1, par)` at `factor.rs:684`. Writes only
   the strict lower triangle. D is never formed explicitly — it is
   folded into the L-side of the product via the D-scaled `Al`.
5. **Panel boundary:** `kmax = block_size − 1` at `factor.rs:515` so
   a 2×2 pivot at the last legal column can still extend through
   `block_size − 1`. If a 1×1 pivot lands at that boundary, one
   panel column is left unused and the actual eliminated count is
   returned (`k` at `factor.rs:698`). faer does **not** carry partial
   2×2 state across panels.
6. **No pivot rejection.** faer treats near-zero pivots as "use
   whatever BK picks." feral's `try_reject_1x1_frontal` adds a
   rejection branch that is absent from the reference.
7. **Defaults:** `block_size = 64`, `par_threshold = 128 × 128`,
   crossover to scalar kernel if `n ≤ block_size`
   (`factor.rs:722`).

## Adaptation to `factor_frontal`

feral's frontal case has two extras not in faer's dense kernel:

1. **Partitioned matrix:** only the first `ncol` rows/columns are
   eliminated; rows `ncol..nrow` form the contribution block that
   goes to the parent node. Pivot search must restrict to column
   range `[k, ncol)` but gamma0 is the max over all rows
   `[k+1, nrow)`. The trailing update must hit both the
   fully-summed trailing block `[k+1, ncol) × [k+1, ncol)` and the
   contribution strip `[ncol, nrow) × [k+1, ncol)`.
2. **Pivot rejection:** `try_reject_1x1_frontal` can reject a 1×1
   pivot when its magnitude is below `zero_tol`, in which case the
   column is marked as "delayed" (if `may_delay`) or zeroed (if
   not). The blocked path must handle this.

### Structure

Introduce two new internal routines and one outer loop rewrite:

```
fn factor_frontal(...) -> FrontalFactors {
    // ... setup (unchanged) ...

    let bs = params.block_size; // new field, default 64
    let mut k = 0;

    while k < ncol {
        let remaining = ncol - k;

        // Scalar fallback for small remainders, rejection tail,
        // and the final 1×1 when remaining == 1.
        if remaining <= bs || /* rejection just happened */ {
            // existing scalar path for a single pivot step
            scalar_pivot_step(&mut a, nrow, ncol, k, ...)?;
            k += pivot_size; // 1 or 2
            continue;
        }

        // Panel path: try to eliminate up to bs columns.
        let mut w = vec![0.0; (nrow - k) * bs];
        let (n_elim, panel_status) = lblt_panel_frontal(
            &mut a, nrow, ncol, k, bs, &mut w, params,
            &mut pos, &mut neg, &mut zero, &mut needs_refinement,
            &mut perm, &mut subdiag, may_delay,
        )?;

        // Apply deferred trailing update:
        //   trailing_fully_summed[k+n_elim..ncol] -= W · L^H  (triangular)
        //   contribution[ncol..nrow]              -= W · L^H  (rectangular)
        apply_blocked_schur(&mut a, nrow, ncol, k, n_elim, &w);

        match panel_status {
            PanelStatus::Full => {}                // clean bs-eliminated panel
            PanelStatus::Rejected => { /* next iter handles rejection via scalar */ }
            PanelStatus::Delayed => break,
        }
        k += n_elim;
    }

    // ... finalization (unchanged) ...
}
```

### `lblt_panel_frontal`

The panel routine is a mild adaptation of `factor_frontal`'s
existing pivot-selection logic, with three changes:

1. The trailing update is *not* applied per pivot. Instead, after
   selecting pivot `p` at column `j` of the panel:
   - Compute the would-be updated column `j` via
     `col_j = a[j] − a[0..j] · W[j, 0..j]^H` into `w[*, j]`.
     This is the peek-ahead used for both gamma0 and the L column.
   - Divide `w[*, j]` by `d_j` (1×1) or by the normalized 2×2
     inverse to form the L column. Write back to `a`.
   - Leave the unscaled `w[*, j]` in `W` for the deferred update.
2. Pivot search for gamma0 uses the *fresh* column `j`, not the
   stale `a[j]`. Symmetric row search for gamma_r uses a
   recomputed row (`A[r, k:k+j] − A[r, k:k+j] · W[...]`) — this is
   where faer's peek-ahead gets expensive but it is correct.
3. On rejection (`try_reject_1x1_frontal` returns `Rejected` or
   `Delayed`), the panel terminates cleanly. Columns already
   eliminated in the panel keep their L / D values; the rejected
   column is *not* touched; `n_elim` reports how many pivots
   actually cleared. The caller applies the partial trailing
   update and falls back to scalar for the next step.

Because rejection and delay are rare on well-conditioned KKTs but
common on the pathological corner cases (ACOPP30, ERRINBAR,
FBRAIN3LS), this graceful degradation preserves correctness
without forcing the blocked path to replicate the full rejection
logic.

### `apply_blocked_schur`

Two matmul-style calls:

1. **Triangular fully-summed update:** strict lower triangle of
   `A[k+n_elim..ncol, k+n_elim..ncol]` gets `−= W · L^H` where
   `L` is the panel's D-scaled L columns
   `A[k+n_elim..ncol, k..k+n_elim]`. Use a hand-rolled triangular
   matmul that writes only `i > j` (analogous to faer's
   `StrictTriangularLower` mode — we cannot use faer, but the loop
   is ~30 lines).
2. **Rectangular contribution update:** full
   `A[ncol..nrow, k+n_elim..ncol] −= W[n_elim..nrow-k, :] ·
   L^H[..., k..k+n_elim]`. Dense gemm, no triangle exploitation.

Both calls hit the same `W` workspace. Neither spawns threads yet —
single-threaded rank-k update is the Phase 2.4.1 deliverable;
Rayon parallelism is deferred to Phase 2.5.2.

## Test plan

Correctness first. New tests go in `src/dense/factor.rs` test module:

1. **Scalar/blocked equivalence on SPD:** random SPD matrices of
   sizes 32, 64, 65, 100, 128, 129, 200, 256, 300 — both kernels
   must produce byte-identical `(L, D, perm, inertia)`. 32 and 64
   exercise the scalar fallback; 65, 129 exercise the
   block-boundary 1×1 1-column leftover; 128 is a clean 2-panel
   case.
2. **Scalar/blocked equivalence on symmetric indefinite:** use the
   BK77 worked examples and the synthetic KKT matrices from the
   existing test suite. All matrices must match scalar inertia
   exactly.
3. **Frontal `ncol < nrow`:** at least one test where the blocked
   kernel eliminates fewer than all rows, and the contribution
   block is compared against the scalar contribution.
4. **2×2 at the block boundary:** construct a matrix where a 2×2
   BK pivot lands exactly at `k = block_size − 1`, verify it
   extends through `k+1` and the panel returns `n_elim = bs − 1`
   (the last slot unused) on one iteration.
5. **Rejection fallback:** a matrix with a forced rejection at
   `k = block_size/2` — verify the panel returns early, the
   caller finishes that step in scalar mode, and re-enters the
   panel path.
6. **KKT regression:** run the full KKT test corpus sampler used
   by `factor_single_front` (examples/triage_errinbar,
   triage_acopp30_0004) and confirm inertia and residual match
   the Phase 2.1.8 baseline byte-for-byte.

Then and only then, benchmark.

## Implementation order

Step-by-step, one commit per step, tests before code:

- **Step 1.** Add `block_size: usize` field to `BunchKaufmanParams`,
  default 64, plus `BkConfig::block_size()` accessor. Existing
  call sites ignore it. Plumb through to `factor_frontal` but do
  not yet branch on it.
- **Step 2.** Extract the current per-pivot loop body of
  `factor_frontal` into an internal `scalar_pivot_step` helper
  that takes `&mut state` and advances one pivot (1×1 or 2×2).
  Pure refactor; byte-identical behavior verified by full KKT
  bench.
- **Step 3.** Write `lblt_panel_frontal` + `apply_blocked_schur`
  using the peek-ahead design, but wire it behind an env flag
  `FERAL_USE_BLOCKED_LDLT=1` so the default path stays scalar.
  Write all 6 correctness tests above. Verify they pass.
- **Step 4.** Flip the default so blocked is the new default path
  when `remaining > bs` and `may_delay == false`. Run full KKT
  bench, compare against Phase 2.1.8 baseline, write validation
  report.
- **Step 5.** Enable blocked path for `may_delay == true` after
  verifying the rejection/delay interaction on the sparse KKT
  path. Re-run sparse bench, update validation report.
- **Step 6.** (Optional, deferred) SIMD micro-kernel for the inner
  loop of `apply_blocked_schur`. This is Phase 2.4.2.

## Risks

1. **Peek-ahead symmetric row search cost.** faer's `gamma_r` is
   simpler because it only searches the updated column. feral's
   `symmetric_row_offdiag_max` walks both the column and the row;
   in blocked mode the row part is stale. Two options: (a)
   recompute the row slice from `W` on demand (expensive), or
   (b) restrict the panel to 1×1 pivots only on the row-search
   path and let 2×2 pivots force a panel boundary. Option (b) is
   simpler; measure first.
2. **Rejection tail.** If a matrix rejects many pivots, the
   blocked path degenerates to scalar plus overhead. This is
   acceptable per the baseline (rejection is rare on clean KKTs)
   but should not make rejection-heavy matrices *slower* than
   scalar. Track ERRINBAR_0824 and ACOPP30_0004 specifically as
   regression canaries.
3. **2×2 normalization in the peek-ahead.** The scalar kernel
   computes `d, d00, d11, d10` from the original trailing block;
   the blocked panel must compute them from the peeked (updated)
   column. Same formulas, but the arguments differ. Easy to get
   wrong at the boundary between "panel-k rows already done" and
   "panel-k+1 rows still stale."
4. **Contribution block fill-in.** The rectangular matmul into
   `A[ncol..nrow, :]` is pure dense — no triangle exploitation —
   so if `nrow ≫ ncol` the contribution strip dominates the cost
   and blocking may not help. Measure on AVION2 (typical frontal
   ~500, contribution ~0–200) before declaring victory.

## Exit criterion

Phase 2.4.1 is done when all of:

1. All 6 correctness tests pass.
2. Full KKT bench passes with zero inertia regressions vs the
   Phase 2.1.8 baseline (dense 152911/154481, sparse
   153009/154588).
3. Dense factor p90 vs MUMPS ≤ 2.0 (currently 2.27).
4. No individual matrix in the top-100 worst dense list has its
   ratio increase by more than 10%.
5. The scalar kernel is retained as the fallback path and all
   existing tests for it pass.

Items 1 and 2 are hard gates. Items 3–5 are the performance
target; if any of them miss, the blocked kernel ships as an
optional path (env flag) and Phase 2.4.2 (SIMD) is moved earlier
to close the gap.
