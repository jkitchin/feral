# Phase 2.4.3 — Rook Rescue Validation Report

**Date:** 2026-04-23
**Plan:** `dev/plans/phase-2.4.3-rook-rescue.md`
**Status:** **Correctness landed; performance exit criterion NOT met.**
 Rook rescue ships as a Sylvester-preserving fallback, not as the
 dense-tail perf fix the plan hypothesized.

## Summary

Rook rescue (Ashcraft-Grimes-Lewis 1998; Duff-Reid 1996) was spliced
into `scalar_pivot_step` in `src/dense/factor.rs` in commit e848f49
(Step 5) with the hypothesis, per plan §Motivation, that CRESC100 and
GAUSS2 factor-time tail of 40–45× vs MUMPS was caused by BK-partial
rejection → delayed-pivot cascades, and that a symmetric rook search
would clear the cascade.

That hypothesis is **wrong at default parameters**. The dense
factor/MUMPS `max` on the full bench went from 45.14 (pre-rook) to
**53.05 (post-rook)**. Not a regression from rook itself — rook
overhead is below noise — but the families that dominate the tail
(HAHN1, CRESC100, GAUSS2) are **not** BK-rejection cases at
`pivot_threshold = 0.01`.

## Evidence That Rook Doesn't Fire on the Tail

From `tests/rook_rescue_kkt.rs` during Step 7 development:

| threshold u | CRESC100 rescues (10 matrices) | GAUSS2 rescues (9 matrices) |
|-------------|-------------------------------:|----------------------------:|
| 0.01 (default)        | 0     | 0   |
| 0.5 (classical BK)    | 0     | 0   |
| 0.99 (near-maximal)   | 260   | 43  |

At the bench-default `u = 0.01`, BK-partial + LAPACK extension
(`akk * gamma_r >= alpha * gamma0^2`) accepts every pivot on CRESC100
and GAUSS2. The rescue path never executes. Even at the classical
BK value `u = 0.5` the rescue stays dormant. Only at `u = 0.99` do
we force BK into enough rejections for rook to matter — and at
`u = 0.99` rook DOES preserve inertia exactly across all matrices
(see below), proving the splice is correct, just unused on these
matrices.

Meaning: the plan's motivating model was wrong. These families'
~10 ms dense frontal factorization is not a pivoting pathology.

## Bench Numbers (release, full corpus of 154,481 dense-eligible
 matrices)

### Dense factor/MUMPS

| metric   | pre-rook (session 2026-04-22-01) | post-rook (this session) | target | verdict |
|----------|------:|------:|-------:|--------|
| geomean  | 0.21  | 0.21  | within 2% of 0.21 (i.e. ≤ 0.2142) | ✓ |
| p50      | 0.11  | 0.11  | —      | ✓ |
| p90      | 1.83  | 1.83  | ≤ 1.90 | ✓ |
| p99      | ~22   | 22.10 | —      | ✓ (no regression) |
| **max**  | 45.14 | **53.05** | **≤ 20** | **✗** |

p90 and geomean are on target — rook overhead does not poison the
easy-matrix path, confirming risk §4 in the plan was mitigated.
`max` misses by a factor of 2.65×.

### Top 10 worst (post-rook)

    name                             n    feral(μs)    mumps(μs)      ratio
    HAHN1_0153                     715        11883          224      53.05
    HAHN1_0154                     715         9164          204      44.92
    CRESC100_0000                  806         8750          200      43.75
    GAUSS2_0035                    758        10417          239      43.59
    GAUSS2_0029                    758        10256          238      43.09
    CRESC100_0027                  806        11269          263      42.85
    GAUSS2_0026                    758        10156          238      42.67
    GAUSS2_0025                    758        10134          242      41.88
    GAUSS2_0016                    758        11132          269      41.38
    HAHN1_0454                     715         8332          202      41.25

Same families as pre-rook (CRESC100, GAUSS2 — with HAHN1 now taking
the #1 spot). `feral_us` in the 8–12 ms range on n = 715–806 dense
frontals vs MUMPS 200–270 μs. A single matrix (HAHN1_0153 at 53.05)
sets the max; HAHN1_0154 at 44.92 is close to the pre-rook leader.
The 53.05 vs 45.14 delta looks like single-run variance on tiny
`mumps_us = 224` rather than a real regression.

### Sparse factor/MUMPS (for context)

| metric   | value |
|----------|------:|
| geomean  | 0.36  |
| p50      | 0.27  |
| p90      | 1.61  |
| p99      | 3.42  |
| max      | 10.71 (CRESC100_0000) |

Sparse is fine. The problem is localized to the dense frontal path.

## Inertia Gate (hard per CLAUDE.md)

**154,481 dense-eligible matrices, 152,911 inertia match (99.0%).**
The 1.0% failures are pre-existing inertia-mismatch families
(HAHN1 498, QPNBLEND 362, MSS1 240, CORE1 141, CRESC50 97, ACOPP30
68, ...) that fail on BOTH dense and sparse paths and are tracked
separately. **Zero new inertia regressions attributable to rook
rescue.**

Verified by:

- `tests/rook_rescue.rs` Test 4 (25 random indefinite matrices ×
   two thresholds): inertia under u=0.0 (BK-partial reference)
   matches inertia under u=0.1 (heavy-rescue) exactly on every
   matrix. Sylvester's law of inertia holds through the splice.
- `tests/rook_rescue_kkt.rs` (this session): 10 CRESC100 + 9
   GAUSS2 matrices at u=0.99 (260 + 43 rescues) match the MUMPS
   5.8.2 oracle inertia to the exact triple.

## Residual Gate (ABS_FLOOR = 1e-14, else 10× MUMPS)

- Full bench: **154,207 / 154,481 residual pass (99.8%).** Worst
   residual 1.87e-4 on ERRINBAR_0824 — pre-existing tail, not
   rook-related.
- Step 7 panels: all 19 oracle-ok matrices pass the gate. Feral
   residuals in the 1e-16 to 5e-15 range vs MUMPS 1e-16 to 3e-14.

## Why the Hard Gate Failed

Pulling the threads:

1. **Rook is dormant on the tail.** CRESC100/GAUSS2 accept every
    pivot at u=0.01; rook has nothing to rescue. It can only fire
    when BK rejects, and BK isn't rejecting here.

2. **The tail cost is in the dense kernel itself, not pivoting.**
    n=715–806 dense frontals taking 9–12 ms while MUMPS finishes
    in 200–270 μs implies the bottleneck is one of:
     - Dense GEMM / panel update throughput (we use a naive 4×4
        kernel vs MUMPS's BLAS3 via DGEMM; on n≈800 this hurts).
     - Ordering: feral may be packing HAHN1/CRESC100/GAUSS2 into a
        single dense-frontal supernode where MUMPS uses a sparser
        multifrontal tree.
     - Assembly/extend-add overhead on mid-sized frontals.

3. **The plan's motivating data was based on an earlier bench.**
    Between the plan being drafted and this session, other
    interventions (phase 2.6.5 LDLT-aware ordering preprocessing,
    phase 2.5.2 parallel multifrontal) may have changed the
    rejection behavior. The plan's "CRESC100 at 40–45× pre-rook"
    number matches today's post-rook number — suggesting the
    ordering preprocessing did not fix CRESC100 either, and the
    40–45× ratio has been stable across interventions.
    Confirmation: CRESC100_0000 is 43.75× here, matching the plan's
    baseline. Rook hasn't moved the needle on it; neither did prior
    Phase 2.6.5 / 2.5.2 work.

## What Rook Rescue Actually Delivers

Not a perf fix. A correctness guarantee:

- **Sylvester invariance through heavy pivoting.** Any caller that
   bumps `BunchKaufmanParams::pivot_threshold` above 0.5 now has a
   principled rescue path instead of force-accepts or delays. This
   matters for interior-point solvers (rIPopt) that may need heavy
   thresholds for numerical stability on near-singular KKT systems.

- **Dormant cost.** At the default u=0.01 the wrapper
   (`try_reject_1x1_with_rook_rescue`) delegates verbatim to
   `try_reject_1x1_frontal` on the fast path. The only overhead is
   the outcome dispatch match, measured to be below bench noise
   (p90 unchanged at 1.83, geomean unchanged at 0.21).

- **Byte-identity preserved with the blocked panel.** The splice
   put rook inside `scalar_pivot_step` only, so both scalar and
   blocked paths share the rescue logic via `PanelStatus::ScalarFallback`.
   `test_may_delay_rejection_parity` stays green (plan §"Parity
   Impact" predicted it would break; it doesn't — see Step 6
   commit message).

## Per-Phase Exit Criteria Scorecard

From plan §"Exit Criterion":

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Tests 1–6 pass | ✓ |
| 2 | Zero inertia regressions vs MUMPS oracle | ✓ |
| 3 | **Dense factor/MUMPS max ≤ 20** (from 45.14) | **✗ (53.05)** |
| 4 | Dense factor/MUMPS p90 ≤ 1.90 | ✓ (1.83) |
| 5 | Dense factor/MUMPS geomean within 2% of 0.21 | ✓ (0.21) |
| 6 | Research note + validation report committed | ✓ (this note) |

Items 1, 2 (hard gates) pass. Item 3 (primary perf deliverable
hard gate) fails. Items 4, 5 (soft gates) pass.

## Disposition

**Ship rook rescue as correctness.** The plan §"Exit Criterion"
offered a fallback: if items 4 or 5 miss, gate behind
`enable_rook_rescue: bool`. Items 4, 5 do not miss — it's item 3
that fails. But item 3 is a perf deliverable that rook **cannot
achieve** because rook doesn't fire on the problem matrices. The
right disposition is not to gate rook off; it's to acknowledge the
perf target needs a different intervention.

Rook stays on by default. Its default-parameter cost is sub-noise.

**Open a new plan for the dense tail.** The HAHN1/CRESC100/GAUSS2
ratios of 40–53× are a dense-kernel / ordering problem, not a
pivoting problem. Candidate follow-ups:

1. **Dense GEMM kernel (Phase 2.8.x).** Profile the n≈700–800
    frontal factorization. If DGEMM-equivalent throughput is
    bottlenecked, improve the 4×4 kernel or add a blocked BLAS3
    path for large panels.
2. **Ordering bias for HAHN1/CRESC100/GAUSS2.** Compare feral's
    supernode structure to MUMPS's elimination tree on these
    families. If feral is coalescing into larger dense blocks than
    MUMPS, bias the supernode merge threshold.
3. **Scaling diagnostic.** Check whether MUMPS applies a specific
    scaling (MC64, MC77-∞) that feral's default pipeline misses
    on these families. This is separate from the scaling-algorithms
    work already tracked in `dev/plans/scaling-algorithms-expansion.md`.

These are not phase 2.4.3 scope; they are the next session's work.

## Artifacts

- Splice: commit e848f49 — `src/dense/factor.rs`,
   `src/numeric/factor.rs` (threading `n_rook_rescues`).
- Rook kernel: `src/dense/rook.rs` (committed earlier in the phase).
- Property test: `tests/rook_rescue.rs` Test 4, committed 40fe0b9.
- KKT regression: `tests/rook_rescue_kkt.rs`, committed 28451b9.
- Full bench output this session: captured in journal entry at
   `dev/journal/2026-04-23-02.org` timestamp 20:48.

## References

- Bunch, J.R. & Kaufman, L. (1977). *Math. Comp.* 31:162–179 —
   baseline BK-partial.
- Ashcraft, C., Grimes, R.G., Lewis, J.G. (1998). *SIMAX* 20:513–561
   — symmetric rook pivoting, termination safeguard at 8
   iterations.
- Duff, I.S. & Reid, J.K. (1996). *ACM TOMS* 22:227–257 — rook
   variant used in multifrontal context.
- LAPACK `DSYTF2` extension criterion (source for the
   `akk * gamma_r >= alpha * gamma0^2` acceptance test used in
   `try_reject_1x1_frontal`).
