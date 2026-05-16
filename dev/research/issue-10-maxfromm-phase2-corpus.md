# Issue #10 Phase 2 — MAXFROMM corpus A/B (NEGATIVE result)

Status: Phase 1 correctness-complete (parity tests + clippy green,
commit 590bc50). Phase 2 corpus A/B on the 1D-banded Mittelmann panel
that motivated #10: **median speedup 0.997×, geomean 1.000×, panel
FAIL of the ≥2.0× success criterion**. Recommendation: keep MAXFROMM
as opt-in (`TppMethod::Maxfromm`), default stays `Plain`.

## A/B harness

`src/bin/diag_clnlbeam_maxfromm.rs`, modelled byte-for-byte on
`diag_clnlbeam_slb.rs` (issue #33 A/B). min-of-7 timings per config
per matrix, 4 families × 20 matrices.

```
=== PANEL TOTAL (n=20) ===
  geomean speedup: 1.000x
  median  speedup: 0.997x
  min:             0.939x
  max:             1.064x
  wins >=1.05x:    3
  losses <=0.95x:  1
```

Per-family medians: clnlbeam 0.97×, henon120 1.00×, lane_emden120
0.99×, dirichlet120 1.00×. All within ±5% measurement noise.

## Why the 2× prediction was wrong

The research note `issue-10-app-vs-maxfromm.md` predicted ≥2× based on
the assumption that the per-pivot AMAX scan dominated the small-front
hot path. The A/B falsifies that assumption.

**1. The scan was already cheap.** For a typical 1D-banded supernode
with ncol=5, nrow=10, the AMAX scan is ~9 abs+cmp per pivot. The
trailing update for the same pivot does ~9 `axpy_minus_unroll4` calls
each with a ~9-element inner loop — roughly an order of magnitude more
work. MAXFROMM saves at most ~10% of the per-pivot cost when it hits.

**2. The capture cost matches the scan cost.** MAXFROMM moves the
column scan from "before next pivot" to "after this pivot's axpy".
Total scan count per pivot loop is unchanged when the cache HITS, and
*increased* when the cache MISSES (we did a capture nobody read, plus
the full re-scan). On indefinite matrices with frequent 2×2 / rejection
cache-clears, the miss cost shows up.

**3. The cache-hot argument backfires on wide fronts.** Capture
happens after the inner trailing-update loop wrote every column k+1..n.
For wide n, column k+1 has been evicted from L1 by the time we scan it.
The pre-scan (Plain) reads column k+1 cold-from-L2 too, so the wash is
exact — no L1 advantage for capture.

**4. The Mittelmann panel is not 97% scalar-1×1-bound on the
*AMAX-scan* hot path.** #33's pounce-feral diagnostic said "97%
scalar 1×1" — meaning the pivot kind is 1×1. But the time inside each
1×1 is dominated by the rank-1 axpy, not the AMAX scan. MAXFROMM was
the wrong knob.

## What the SmallLeafBatch and MAXFROMM A/Bs jointly tell us

Both #33 (SLB) and #10 (MAXFROMM) targeted the same panel and both
failed the ≥10% / ≥2× criteria within noise. The bottleneck on
1D-banded Mittelmann is the rank-1 trailing-update axpy itself, not:

- per-supernode driver overhead (#33 ruled out)
- per-pivot AMAX scan (#10 ruled out)

The remaining levers for this corpus:

1. **APP-style block factor** (#10 original arm) — process several
   pivots at once with a small dense panel, hoist common loads, run
   the rank-1 updates as a level-2 BLAS-like kernel. Requires
   ncol ≥ ~16; on the narrow supernodes that dominate this corpus,
   APP cannot engage. So APP is also wrong for *this* corpus, but
   may help wider fronts (clnlbeam root + ACOPP30-style KKTs).

2. **Supernode amalgamation** (relax / merge thresholds) to widen the
   bottom-of-tree supernodes so APP / block kernels can engage.

3. **A SIMD-tightened scalar trailing-update kernel** for ncol ≤ ~10
   that fuses the L-column scale + axpy into one pass with explicit
   prefetch. This is implementation-only optimization in the existing
   scalar path.

(3) is the most direct attack on the actual hot path. (2) is a
symbolic-side restructure. (1) is still worth pursuing for the
wider-front regime (ACOPP30, vesuvia).

## What to do with the Phase 1 code

Keep it. The TppMethod::Maxfromm opt-in is:

- correct (5 byte-identity parity tests pass on scalar + blocked
  paths, SPD + indefinite + ncol<nrow)
- zero cost in default mode (`tpp_method: Plain` is the default; the
  hot path is unchanged)
- ~zero cost in opt-in mode on this corpus (within noise; we are not
  losing meaningfully)
- a useful primitive for later experimentation on wider-front
  workloads where the AMAX scan might actually be the bottleneck

Phase 4 (wiring MAXFROMM into `block_ldlt32`) is deferred — the
small-n=32 path needs its own investigation. No work on Phase 4 until
a corpus is identified where MAXFROMM measurably wins.

## Decision

- Default: `TppMethod::Plain` (no change to user-visible behavior).
- Opt-in: `TppMethod::Maxfromm` available.
- Issue #10 status: **defer the default-flip; keep the enum and
  capture/consume infrastructure in place**. The next blocker for the
  1D-banded Mittelmann panel is the rank-1 trailing-update kernel,
  not pivot selection.
- Issue #33: also stays open with the same blocker. Both #33 and #10
  unblock when (a) the scalar rank-1 axpy is tightened or (b) a
  supernode-amalgamation pass widens the narrow leaves enough to let
  APP / block_ldlt32 engage.

References: `diag_clnlbeam_maxfromm.rs`, `issue-10-app-vs-maxfromm.md`
(original research note), `issue-33-slb-ab.md` (parallel SLB ruling),
journal entry `2026-05-16-01.org` 12:00.
