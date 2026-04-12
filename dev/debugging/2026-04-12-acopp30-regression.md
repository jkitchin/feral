# ACOPP30 MC64 regression triage (Phase 2.2.1 Step 8)

**Date:** 2026-04-12
**Head commit:** `8a95825` (Step 8 validation landing, MC64 + scaling wiring)
**Diagnostic binary:** `examples/debug_acopp30_mc64.rs`
**Status:** Root cause identified. No code changes in `src/`. Fix
deferred to Phase 2.2.2.

## Reproduction

```bash
cargo run --release --example debug_acopp30_mc64
```

Or, via the un-ignored regression test (ignored by default):

```bash
cargo test --release --test mc64_regression acopp30 -- --ignored --nocapture
```

Observed residual on `data/matrices/kkt/ACOPP30/ACOPP30_0000.mtx`
(n=209, nnz=765) via `solve_sparse_refined`:

| Path              | rel_res   | ‖x‖∞      | feral inertia | smallest accepted D |
|-------------------|-----------|-----------|---------------|---------------------|
| Identity (no MC64) | 2.842e+16 | 3.084e+17 | (62, 142, 5)  | 1.000e-08 (×5 tied) |
| MC64 Symmetric    | **2.274e+46** | **5.368e+47** | (62, 142, 5)  | **3.635e-10** |
| External([1;n])   | 2.842e+16 | 3.084e+17 | (62, 142, 5)  | 1.000e-08           |
| MC64 + zero_tol=1e-8 | 2.274e+46 | 5.368e+47 | (62, 140, 7)  | 3.635e-10 |

Matches the Step 8 sweep numbers exactly: pre-fix 2.84e+16 vs
post-MC64 2.27e+46. MUMPS canonical is 5.01e-14 with inertia
(71, 137, 1). IPOPT's own record (ACOPP30_0000.json) reports
`delta_c = delta_w = 0, inertia (72, 137, 0)` — i.e. the system is
structurally non-singular, iteration 0, no regularization applied.

## Instrumentation data

From `compute_scaling(matrix, Mc64Symmetric)` on ACOPP30_0000:

- `ScalingInfo::Applied` (full matching — n_matched == n)
- `scaling` length 209, all finite, no zeros, no negatives
- `min|s_i| = 5.289e-02` (index 23), `max|s_i| = 4.870e+00` (index 127)
- ratio max/min ≈ 92, geomean ≈ 0.49 — completely normal range
- 5 smallest: `[0.053, 0.054, 0.055, 0.060, 0.062]` at indices 23,49,53,55,66
- 5 largest:  `[4.87, 3.20, 3.16, 3.16, 3.16]` at indices 127,97,206,205,204
- MC64 clamp `LOG_HUGE = 709` never fires — all exponents are in
  [ln 0.053, ln 4.87] = [-2.94, +1.58]

After `symbolic_factorize` + `factorize_multifrontal`:

- Feral inertia `(62, 142, 5)` identical on all paths (Identity, MC64,
  External all-ones). So MC64 does NOT flip the pivot classification,
  it just shifts the *magnitudes* of the accepted pivots.
- `needs_refinement = true` on all paths (expected: ForceAccept fired).
- `|D|` range and quantiles diverge sharply between paths:

| Path      | min accepted |D| | p5      | p50      | p95      | max |D|   |
|-----------|-----------------:|--------:|---------:|---------:|----------:|
| Identity  | 1.000e-08        | 1.0e-08 | 1.78     | 80.0     | 261.0     |
| MC64      | **3.635e-10**    | 2.1e-09 | 1.00     | 1.55     | 23.4      |

MC64 equilibrates the matrix (p50 diagonal value drops from 1.78 to
1.00, p95 drops from 80 to 1.55 — the `D ≈ I` target), *but at the
cost of pushing the smallest accepted pivots from `1e-8` to `3.6e-10`*.

The solve then divides by `~3.6e-10`, amplifying rounding error by a
factor of `~2.8e9` per affected position. With 5 such pivots cascading
through both the forward L-solve, the D-inverse, and the backward
L^T-solve, the total magnitude compounds to `~1e30` — matching
the observed 30-order exponent gap between paths.

## Where the magnitude explodes

Tracing through the pipeline for the MC64 path:

1. **Scaling vector:** well-behaved, range [0.053, 4.87]. Not the source.
2. **Scaled RHS `D·b`:** ‖·‖∞ ≈ 100 × 4.87 ≈ 490, well-behaved.
3. **Factored M = D·A·D:** inertia identical to A, off-diagonals
   equilibrated as designed, but 5 non-zero pivots have magnitudes
   `~3.6e-10` to `~1e-9`. These are the amplifier.
4. **Core solve `y = M⁻¹·c`:** at the near-singular pivots, `w[k] /= d`
   with d ≈ 3.6e-10 amplifies any residual rounding into the 1e10
   range. This propagates up through the backward L^T-sweep
   multiplying each affected row's contribution back into all
   earlier positions. By the end of backward-sub, ‖y‖∞ ≈ 1e47.
5. **Unscale `x = D·y`:** ‖x‖∞ ≈ 5.4e47.
6. **Residual:** `‖A·x − b‖∞ ≈ 3.2e48`; rel_res = 2.27e46.

Iterative refinement's first correction step amplifies this further
to 1e81, then 1e115, etc — so `solve_sparse_refined` correctly
falls back to the best iterate, which IS step 0. Confirmed by
manual refinement trace in the debug binary.

## Hypothesis verdicts

| # | Hypothesis                                                 | Verdict         |
|---|------------------------------------------------------------|-----------------|
| 1 | MC64 produces non-finite / zero / NaN scaling entries      | **Ruled out**. All 209 entries finite, strictly positive, range [0.053, 4.87]. |
| 2 | MC64 produces extreme magnitudes (ratio > 1e10)            | **Ruled out**. Ratio is ~92. |
| 3 | Identity fallback not firing for unmatched rows            | **Ruled out**. `ScalingInfo::Applied`, matching is complete (n_matched == n). |
| 4 | ForceAccept × MC64 interaction shifts pivots into near-singular regime | **CONFIRMED as root cause**. Smallest non-zero pivots drop from 1e-8 to 3.6e-10 under MC64. Inertia classification is unchanged, but the non-zero entries of D are far closer to the zero threshold, and dividing by them in the solve is the amplifier. |
| 5 | Best-iterate refinement not running / tracking only final  | **Ruled out**. Best-iterate IS active (commit d954c73, `best_x`/`best_r_norm` in `solve_sparse_refined`). The returned residual is the step-0 result because every subsequent refinement step diverges by ~30 orders per step. This is working as designed — refinement correctly gives up. |
| 6 | Solve-side pre/post-scale direction wrong                  | **Ruled out**. Derivation re-checked: `c = D·b`, `y = M⁻¹·c`, `x = D·y` is correct for `M = D·A·D`. Both ends multiply by the same D, not its inverse. Code matches. `External([1;n])` path gives exactly the same result as `Identity`, confirming the wrapper is a no-op when s=1. |

## Root cause (one-paragraph)

MC64 symmetric scaling equilibrates `A ↦ D·A·D` so the factor's
median pivot magnitude collapses from ~1.78 (Identity) to ~1.0
(MC64, the designed target). The *scaled* off-diagonal magnitudes
drop in lockstep. What MC64 does NOT do is push the worst pivots
across `zero_tol`: the smallest non-zero pivots in the scaled factor
end up at `~3.6e-10`, which is ~28× smaller than their Identity
counterparts (`1e-8` from the KKT's constraint block) but still 6
orders of magnitude above `f64::EPSILON`. The Phase 1 `ForceAccept`
handler only zeros a pivot when `|d| <= zero_tol = f64::EPSILON`;
anything above that is inverted in the D-solve. So MC64 on this
matrix produces five "inverted near-singular" positions where
the Identity path had "exactly `delta_c = 1e-8`". The solve
then divides by `3.6e-10` five times, compounding rounding error
through the forward + backward sweeps to give a ‖x‖∞ of `5e47`
where the Identity path gave `3e17`. Best-iterate refinement
correctly keeps step 0 because each subsequent step diverges by
another 30 orders. The MC64 scaling vector itself is numerically
healthy; the compute_symmetric wrapper is correctly constructed;
the factorize assembly and solve wrappers are correctly wired.
The bug is in the interaction: MC64 assumes the downstream
pivoting is threshold-based and will reject the small pivots it
exposes, but feral's Phase 1 ForceAccept has a fixed `zero_tol`
at EPSILON that is orders of magnitude below MC64's "expected
reject" range.

## Proposed fix (Phase 2.2.2)

The canonical fix is **threshold-partial-pivoting with delayed
pivots** — the same mechanism MUMPS and SSIDS use to handle this
regime. Under threshold pivoting, any candidate pivot whose
magnitude is below `u · max_trailing_column_entry` (with
`u ≈ 0.01`) is refused and the column is deferred to the parent
frontal. MC64's equilibration makes this threshold meaningful
because after scaling the trailing column max is ~1, so the
pivot threshold effectively becomes "reject pivots with |d| < 0.01".
This is exactly what MC64 is designed to pair with.

A minimal Phase 2.2.2 change that captures the intent without
building the full delayed-pivot machinery: **raise `zero_tol`
adaptively when MC64 scaling is active**. Two options:

1. **Scaling-aware zero_tol:** when `scaling_info != NotApplied`,
   set `zero_tol` to a fraction of the scaled-matrix typical
   magnitude (e.g. `1e-12` instead of `eps`). This will cause
   the current 5 MC64 near-singular pivots at `3.6e-10` to be
   force-accepted as zero — identical-in-intent to the Identity
   path's behavior on the `1e-8` pivots. Expected outcome:
   roughly match the Identity path's `~1e16` residual for
   ACOPP30, which the sanity panel shows the *other* 6 matrices
   then improve on by 5–10 orders.

2. **Scaling-inverse pivot comparison:** compute threshold as
   `zero_tol · max(column)` rather than absolute, matching what
   the pivoting paper recommends. This is the textbook fix but
   requires touching `do_1x1_pivot` / `do_2x2_pivot` more
   carefully.

I recommend option 1 for Phase 2.2.2 Step 1 as the fastest path
to recovery, with option 2 queued for Phase 2.2.2 Step 2 as the
proper fix. Neither should touch MC64 itself — `compute_symmetric`
is correct.

## Bugs and quirks uncovered during investigation

- `solve_sparse_refined` best-iterate tracking is working
  correctly. No bug there. (Verified on the MC64 path by manual
  refinement trace in `debug_acopp30_mc64.rs`.)
- The 2×2 block determinants on the MC64 path are well-behaved
  (min |det| ≈ 0.084, well above `zero_tol_2x2 = eps²`). 2×2
  inversion is not the amplifier on this matrix.
- `External([1.0; n])` path produces numerically identical
  results to `Identity` (rel_res 2.842e+16 vs 2.842e+16). This
  is a useful sanity check for the pre/post-scale wrapper: when
  s ≡ 1, the wrapper is effectively a no-op as designed.
- The `capped MC64` experiment (rewrite any s with |s| > 1e6 or
  < 1e-6 to 1.0) flagged zero entries — no scaling value is
  close to extreme. MC64 is not producing outliers; the entire
  vector is equilibrated in a narrow 100× range. This is another
  confirmation that hypothesis 2 is ruled out.
- The `MC64 + zero_tol = 1e-8` path still produces the 2.27e+46
  residual even though two more pivots are force-accepted. This
  is because the 5 smallest non-zero pivots on the MC64 path
  are at `3.6e-10, 6.2e-10, 6.5e-10, 1.07e-9, 1.23e-9` — all
  below `1e-8`, but they're in 2×2 pivot blocks where the det
  is fine, so the pivot pair survives `zero_tol_2x2 = 1e-16`
  even though individual 1×1 magnitudes would fail the 1×1
  threshold. This suggests the Phase 2.2.2 fix must consider
  2×2 blocks, not just 1×1, and probably needs to tighten
  `zero_tol_2x2` in proportion (option 2 above handles this
  naturally via column-relative comparison).
- ACOPP30_0000's IPOPT sidecar reports iteration=0 and
  `delta_c = delta_w = 0`, which means **this matrix is meant
  to be factored cleanly with no regularization**. Feral's 5
  force-accepted pivots on the Identity path are themselves a
  pre-existing Phase 1 issue (not introduced by MC64, but
  exposed by it). This is consistent with the Phase 1
  retrospective's ACOPP30 residual-gap finding.

## What is committed

- `examples/debug_acopp30_mc64.rs` — the diagnostic binary used
  to produce the numbers in this report. Retained as a future
  regression probe: any fix for Phase 2.2.2 should drive its
  `rel_res` column down and the `smallest accepted |D|` column
  toward the Identity path's `1e-8` level.
- `dev/debugging/2026-04-12-acopp30-regression.md` — this report.
- `dev/journal/2026-04-12-06.org` — real-time journal entry.

Nothing in `src/` was modified.
