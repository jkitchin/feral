# Panel fragmentation measurement (issue #129) — 2026-07-10

**Verdict: not justified. Close #129 with this measurement.** The in-panel
interchange (swapped-row replay) work is deferred; the data does not support it.

## Question

Issue #129 (measure-first per the 2026-05-13 decision): every swap-1×1 and
mid-panel swap-2×2 terminates the blocked dense panel via `ScalarFallback`,
flushing the deferred Schur update at a truncated `n_elim`. The hypothesis was
that on swap-heavy strongly-indefinite fronts this collapses effective panel
width toward 1–2 and re-traverses the trailing block `~bs/s̄` times more,
"directly diluting the measured 8–10× packed-kernel win." The issue's own
close criterion: **if fragmentation is <10–15% of dense-front time on the
corpus, close it.**

## Method

`src/bin/probe_panel_frag.rs`: sets `PANEL_DIAG_ENABLED` + `PHASE_TIMING_ENABLED`,
factors each matrix on the **sequential** driver (clean phase attribution),
and reports per matrix + aggregate:
- `panel_full` / `panel_partial` / `panel_delayed` (fragmentation rate),
- pivots handled inline vs pushed to the scalar path,
- `phase_timing` split of dense-front time: `SCHUR_NS`, `PANELFACTOR_NS`,
  `SCALARTAIL_NS` as a fraction of `DENSEFACTOR_NS`.

Corpus: synthetic saddle/rank-deficient (`saddle_rankdef_*`, `rankdef_*`),
ACOPP30 (indefinite power-flow KKT), AVION2, VESUVIO, CRESC132, and nuffield2
(26 649-row sparse KKT — dominates the aggregate by factor time).

## Results (aggregate, time-weighted)

```
panels: full=14 partial=24190 delayed=0        (frag = 99.9%)
pivots: inline=13299 scalar=25731              (scalar = 65.9%)
dense-front time: schur=4.1%  panel-factor=0.2%  scalar-tail=8.4%
                  (densefactor = 13.26 s)
```

Per-matrix `schur%` of dense-front time: nuffield2 4.1%, VESUVIO 0.8%,
ACOPP30_0005 19.0% (but only 324 µs total), saddle_rankdef_100_20_5 28.5%
(1977 µs total). The high-`schur%` matrices are all tiny; the two matrices
that actually cost anything (nuffield2 13.3 s, VESUVIO 1.2 s) have
`schur% = 4.1%` and `0.8%`.

## Interpretation

1. **Fragmentation is nearly universal (99.9%) but cheap.** The Schur
   trailing update — the *only* thing fragmentation inflates (by re-traversal
   at truncated width) — is **4.1%** of dense-front time on the corpus.
   Even if in-panel interchange eliminated fragmentation entirely and the
   re-traversal overhead were the *whole* of that 4.1% (it is not — most Schur
   time is the irreducible rank-`n_elim` update), the addressable saving is a
   few percent, **below the 10–15% close threshold**.

2. **Large fronts fragment least (in cost).** VESUVIO (n=3083, real dense
   fronts) has `schur% = 0.8%` and only 2 panels. The blocked kernel matters
   only for large dense fronts, and those do not fragment expensively here —
   the fragmentation count is dominated by nuffield2's many *tiny* fronts (a
   thin/path-like etree), where a "panel" is barely wider than scalar anyway,
   so collapsing its width costs almost nothing in absolute terms.

3. **Separate finding (not what #129 proposed to fix).** ~66% of pivots go
   through the scalar path and the scalar tail is 8.4% of dense-front time;
   the bulk of `DENSEFACTOR_NS` (~87%) is the small-front diagonal/scalar
   factor path, not the blocked panel/Schur machinery at all. Any future
   dense-kernel effort should target the small-front path, not in-panel
   interchange for Schur-flush reduction.

## Decision

Close #129. Fragmentation's Schur-re-traversal cost (4.1%) is below the
issue's own 10–15% justification bar. The swapped-row-replay primitive is not
worth its complexity and regression risk against this data. Re-measure with
this probe if the front-size distribution changes materially (e.g. after an
ordering or amalgamation change that produces larger dense fronts).
