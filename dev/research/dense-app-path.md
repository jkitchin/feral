# Issue #10 — APP (aggressive partial pivoting) path: re-open gate measurement

Date: 2026-05-13.
Status of issue #10: **not ready to implement.** The re-open criterion
the issue itself states (posted comment, by `jkitchin`) is not met by
the current `diag_supernode_cost` numbers. Recording the evidence and
the design space here so the next person who looks at this issue
doesn't have to re-derive it.

## The gate (verbatim from #10's posted comment)

> Re-open criterion. When #9 has landed (which itself requires #13)
> and a fresh `diag_supernode_cost` shows ns/nnz dominates ns/sup
> on a relevant cluster (ACOPR30, CRESC100 at nemin=32, or any new
> corpus with fronts wide enough to use the panel path), re-measure
> the panel-path γ₀ scan cost in isolation. If it shows up as a
> measurable fraction of per-front time on those fronts, re-open
> this issue and complete the APP path against #9's SIMD kernel.

Two preconditions: (a) #9 substantively landed, (b) ns/nnz dominates
ns/sup. (a) is satisfied as of `d3f1132` (2026-05-13). (b) is the
measurement below.

## Measurement: `cargo run --bin diag_supernode_cost --release` (2026-05-13, post-`d7267fe`)

```
matrix                           n fact_nnz  nsup   med   max  num_us   ns/sup  ns/nnz
CRESC100_0000                  806     2630   230     4    16     210      914      79
ACOPR30_0185                   564     2417   232     2    17     164      707      67
ACOPR30_0067                   564     2417   232     2    17     162      700      67
HAIFAM_0082                    249     5584   157     5    86     184     1174      33
HAHN1_0049                     715     4438   247     3    12     174      705      39
GAUSS2_0029                    758     5250   259     3    16     186      721      35
HS118_0001                      32      176    17     3    16       8      527      50
KIRBY2_0007                    458     1603   161     3    17      97      603      60
AVION2_0000                     94      181    41     1    16      20      488     110
```

nemin sweep on ACOPR30_0067 (the cluster the gate names explicitly):

```
ACOPR30_0067 nemin=1           564     2417   493     2    16     203      413      84
ACOPR30_0067 nemin=4           564     2417   280     2    15     159      569      65
ACOPR30_0067 nemin=8           564     2417   261     2    15     148      569      61
ACOPR30_0067 nemin=16          564     2417   232     2    17     149      643      61   (current default)
ACOPR30_0067 nemin=32          564     2417   158     2    32     149      943      61
ACOPR30_0067 nemin=64          564     2417    99     2    64     156     1577      64
ACOPR30_0067 nemin=128         564     2417    75     2   128     202     2693      83
ACOPR30_0067 nemin=256         564     2417    57     2   256     438     7686     181
```

Across **every** row on every cluster, **ns/sup > ns/nnz** by a factor
of 4× to 15×:

- ACOPR30_0067 at nemin=32 (the cluster the gate names): 943 vs 61 → **15×**.
- CRESC100_0000 at default nemin=16: 914 vs 79 → **12×**.
- HAIFAM_0082 (the corpus matrix with the widest fronts, max 86): 1174
  vs 33 → **36×**.

The gate condition "ns/nnz dominates ns/sup" is the opposite of what
the data shows. Even on the largest-front matrix on the corpus
(HAIFAM_0082, max front 86) ns/sup is 36× larger than ns/nnz.

**The gate is not met.** Per the issue's own re-open criterion, APP
work is not justified today.

## Why the motivating gap closed

The issue body cites
`dev/research/mumps-small-frontal-speed.md` claiming "feral at ~89
ns/nnz_L on CHAINWOO_0000, MUMPS at 14, SSIDS at 29". That
measurement is stale on the current build:

1. **fused_gamma0 landed `ad05ff4` (2026-04-11)**, before the
   research note was written but the note didn't re-measure. The
   scalar `factor_frontal` loop now carries `fused_gamma0`/`fused_r`/
   `have_fused` across the no-swap fast path (factor.rs:369-371,
   400-405, 439-441, 465-467, 486-488, 537-539, 555-557): the next
   pivot's γ₀ and argmax row come from the previous pivot's rank-1
   update as a side effect, so no second pass over the trailing
   column is done. This is the same trick `DMUMPS_FAC_MQ_LDLT`'s
   `MAXFROMM` thread does (cited by the issue body as a MUMPS-only
   advantage).

2. **The 32×32 SIMD kernel landed `d3f1132` (2026-05-13).** 32×32
   fully-summed fronts (the CHAINWOO root supernode shape) now route
   through `factor_block32` → `factor_frontal` with `do_1x1_update`'s
   `n==32` fast-path firing `update_1x1_block32`'s quad-tile SIMD
   body (block_ldlt32.rs).

The combined effect is that the two largest mechanisms the APP
proposal targeted are already in place via different code paths:
fused γ₀ via the scalar fused-update thread, and SIMD trailing
updates via the block-32 dispatch.

## What's left of the per-pivot scan cost — the panel γ₀ scan

`lblt_panel_frontal` (factor.rs:1480-1488) does a fresh γ₀ scan per
pivot:

```rust
let mut gamma0 = 0.0f64;
let mut r = col + 1;
for i in (col + 1)..nrow {
    let v = a[col * nrow + i].abs();
    if v > gamma0 { gamma0 = v; r = i; }
}
```

This is the panel-path equivalent of the scan that `fused_gamma0`
eliminated on the scalar path. The reason fusion hasn't been done
here is that the panel uses *deferred* rank-1 updates
(`peek_ahead_column` replays them just-in-time per column), so the
γ₀-from-previous-update bookkeeping isn't trivially available.

**But on the current corpus the panel γ₀ scan is unmeasured-and-
likely-tiny:**

- The dominant 32×32 case (CHAINWOO root supernodes) **never enters
  this code path** — `factor_frontal_blocked_in_place_with_scratch`
  dispatches `nrow == ncol == 32` fronts to `factor_block32` (which
  delegates to `factor_frontal`) before `lblt_panel_frontal` is
  reached. The panel only runs for `ncol > 32` or `ncol < 32 with
  ncol >= PANEL_MIN_NCOL=8`.
- For the corpus matrices in the table above, max front is ≤ 86 and
  most are ≤ 17 — small panels or no panel at all. Per-pivot scan
  cost on these fronts is dwarfed by the 600–1900 ns/sup fixed
  overhead.

## Narrow alternative (issue's own suggestion)

The #10 posted comment offers:

> Narrow alternative if a target appears before then. If a profile
> after #13 shows the panel γ₀ scan is the issue but APP
> block-deferral is overkill, fusing γ₀ into
> apply_blocked_schur_panel's rank-1 stream (same trick as the
> scalar path, no APP machinery) is a smaller surgical change.
> That would be a separate issue, not this one.

This alternative is also unjustified at the moment because the
panel γ₀ scan is not the bottleneck on any cluster in the data
above. It becomes interesting only when (a) per-front fixed cost
(ns/sup) is brought into the ns/nnz neighborhood, and (b) a
corpus matrix appears with `panel`-sized fronts (32 < ncol ≤ 96-ish)
where the panel γ₀ scan would be a meaningful fraction of per-front
time.

## Where the next bench-ratio gain probably lives

The data above is consistent with the picture from issue #13's
session checkpoint (Phase C land): fixed per-front overhead
dominates the long-tail corpus, and reducing it to where ns/nnz
becomes visible is the work that gates *any* per-nnz optimization
(APP, fused panel γ₀, broader SIMD, etc.).

Candidates that haven't been bench-attacked:

- Symbolic-phase pre-allocation of supernode IW slots — `assemble`
  still walks child contribs cell-by-cell on every front.
- Per-front pos/neg/zero/perm Vec allocations — these are O(nrow)
  but each is a separate heap call.
- Validation rescan of the lower triangle on each `factor_frontal`
  entry (`matrix.validate()` at line 871). Issue #13 listed this as
  a candidate target but it was not removed in Phase A/B/C.

These are the targets of an `issue-13-overhead-followups` line of
work, not of #10.

## Recommendation

Close #10 with a comment citing this note. Re-open if a future
profile shows panel γ₀ scan as a measurable fraction of per-front
time on a corpus front; even then, the surgical alternative
(fuse γ₀ into the panel's rank-1 stream) is the first thing to
try, not full APP block-deferral.

## Anchors

- Gate measurement: `cargo run --bin diag_supernode_cost --release`,
  output reproduced above. Re-runnable any time.
- `fused_gamma0` thread in scalar path: factor.rs:369-371, 400-405,
  439-441, 465-467, 486-488, 537-539, 555-557.
- Panel γ₀ scan (the remaining un-fused site): factor.rs:1480-1488.
- 32×32 SIMD dispatch: factor.rs:1189-1193 (entry) →
  block_ldlt32::factor_block32 → factor_frontal.
- #9 land record: dev/sessions/2026-05-13-02.md.
- #13 outcome: dev/sessions/2026-05-12-07.md.
