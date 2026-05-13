# Small-matrix perf retrospective: where issues #11, #12, #13 stand after #9/#13 land

Date: 2026-05-13. Companion to `dev/research/dense-app-path.md` (the
#10 closure note).

## Why this note exists

Issues #11, #12, #13 were a coupled set targeting the small-matrix
performance gap vs MUMPS (small-front bucket p90 ~1.74, geomean
~0.39 vs MUMPS on factor). #9 landed 2026-05-13; #13 Phases A+B+C
landed 2026-05-12/13; #10 closed 2026-05-13 (gate not met). With
three of the five sub-items resolved or closed, the original
small-matrix story no longer matches the data. This note records
what changed, why the original gating model was partly wrong, and
what's actually open as a measurable target.

## The original model

The model behind issues #11 + #12 was (paraphrased from #12 body
and #11 hypothesis):

1. Pivot-search architecture (#10) — TPP per-pivot scans are costly.
2. No 32×32 SIMD kernel (#9) — per-element trailing update is scalar.
3. Per-supernode driver overhead (#11) — allocations, row_indices
   build, extend_add, etc. SmallLeafBatch::On amortizes some of it.

Plus #13: reduce per-supernode fixed overhead independently of all
of the above. Once (1)+(2) shrink kernel cost, the (3) overhead
becomes visible and SmallLeafBatch should tip out of noise.

## What landed and what each landing actually moved

| Land | Issue | What it moved | Bench-ratio effect |
|---|---|---|---|
| `ad05ff4` 2026-04-11 | (precursor) | fused_gamma0 in scalar factor_frontal — eliminates per-pivot O(n) γ₀ scan on no-swap branches | Pre-baseline; in build before #12 was filed |
| `575f86a` 2026-05-12 | #13 Phase A | FactorScratch pool: subdiag, d_panel | ns/sup −16% to −54% on CRESC100/ACOPR30/KIRBY2; bench p90 +0.04 to +0.21 (slightly worse, within noise) |
| `1e26902` 2026-05-12 | #13 Phase B | extend_add direct slice writes; bypass SymmetricMatrix::set/get branches | ns/sup further down on multi-child fronts; bench p90 back to baseline ±0.02 |
| `fe2ca4d` 2026-05-12 | #13 Phase C | single-slot contrib pool (Option<Vec<f64>>) recycled across siblings | Bench neutral; multi-slot Vec<Vec<f64>> variant abandoned (regressed bench by ~+0.19 small, ~+0.30 medium) |
| `7356371` 2026-05-12 | #9 Step 1 | block_ldlt32 scaffold + bit-parity harness | Zero (scaffold only) |
| `585465a` 2026-05-12 | #9 Step 2a | update_1x1/2x2_block32 scalar primitives | Zero (scalar still) |
| `98ef545` 2026-05-13 | #9 Step 3 | SIMD body for update_1x1_block32 (quad/dual/single tiling) | Not yet wired |
| `d3f1132` 2026-05-13 | #9 Step 2 dispatch | n==32 fast-paths in do_*_update; nrow==ncol==32 dispatch in factor_frontal_blocked_in_place_with_scratch | Bench p90 small 1.36 → 1.33, medium 1.78 → 1.74 (modest, consistent across 3 runs) |
| `d3aa627` 2026-05-13 | #10 closure | Research-phase note + decisions entry; no code change | Zero |

**Total bench-ratio movement across all of this work**: small p90
~1.36 → 1.33, medium p90 ~1.78 → 1.74. About 0.04 absolute.

## What the current `diag_supernode_cost` shows (2026-05-13, post-`d7267fe`)

```
matrix                           n fact_nnz  nsup   med   max  num_us   ns/sup  ns/nnz
CRESC100_0000                  806     2630   230     4    16     210      914      79
ACOPR30_0067                   564     2417   232     2    17     162      700      67
HAIFAM_0082                    249     5584   157     5    86     184     1174      33
HAHN1_0049                     715     4438   247     3    12     174      705      39
GAUSS2_0029                    758     5250   259     3    16     186      721      35
KIRBY2_0007                    458     1603   161     3    17      97      603      60
AVION2_0000                     94      181    41     1    16      20      488     110
```

nemin sweep on ACOPR30_0067:

```
ACOPR30_0067 nemin=16          564     2417   232     2    17     149      643      61   (default)
ACOPR30_0067 nemin=32          564     2417   158     2    32     149      943      61   (gate cluster)
ACOPR30_0067 nemin=64          564     2417    99     2    64     156     1577      64
```

Across every row, ns/sup exceeds ns/nnz by 4× to 36×. The original
model treated the kernel layer (ns/nnz) as the dominant gap to close
first. The actual dominant gap is the per-front fixed cost
(ns/sup), and that layer has only been partly addressed.

## Where the original model was wrong

### (a) The kernel-cost-hiding-overhead hypothesis was partly false

The #11 plan said: "the per-front driver overhead that small-leaf
batching amortizes is currently *dwarfed* by per-front kernel cost
(TPP pivot search + scalar Schur update)." The 2026-04-25 SmallLeafBatch
flip attempt was supposed to tip out of noise once kernel cost dropped.

But the kernel cost wasn't actually dwarfing the overhead. The
2026-04-25 measurement on ACOPR30_0067 (Off vs On, 5 runs) showed
the *On* path moved `total_us` by ~6% mean, within noise. If kernel
cost had been the dominant share, the 6% reduction from skipping
build_row_indices would have shown as a measurable signal once the
other work landed. After #9 Step 2 + #13 Phases A+B+C landed, the
ACOPR30_0067 baseline dropped from ~158 µs to ~149 µs — a small
6% improvement, comparable to the per-leaf savings — but the
*relative* overhead share didn't shift, because both layers
shrank together.

The mistake is assuming a layer is "hiding" another when both are
the same order of magnitude. They're both visible; reducing either
just produces a proportional shrink without revealing a new floor.

### (b) "ns/nnz dominates ns/sup" was the wrong gate for #10/#11

The #10 posted comment and the #11 plan both implicitly assumed the
ns/sup ≈ ns/nnz crossover would be reached as kernel work landed.
The crossover has not happened on any cluster, and the levers that
might have driven it (kernel cost reduction via #9) are largely
spent — the 32×32 SIMD body is on the hot path now.

The gate should have been phrased in terms of *absolute* ns/sup
budget, not the ratio. SSIDS achieves ~150–300 ns/sup on the same
matrices (`dev/research/ssids-small-frontal-speed.md` §3
SmallLeafNumericSubtree path); feral is at 600–1200. A 4× absolute
gap in fixed overhead.

### (c) #13's "next lever is #9" prediction was wrong

The #13 re-scope comment said: "Per-front kernel cost (32×32 SIMD,
#9) is the next plausible lever for the bench-ratio gap, since it
touches the FLOP-dense inner loop directly rather than the
per-supernode bookkeeping overhead."

#9 Step 2 dispatch landed and bench p90 moved by 0.04 absolute
(small) and 0.04 (medium). Within noise on a 3-run sample, but
consistent — call it a real ~1.5% win. That's not the criterion
#2 jump (small <1.30 or medium <1.60); it's exactly the same
order of magnitude as #13's own bench-neutral pooling phases.

Both #13 pooling and #9 SIMD body landed real improvements on the
*ns/sup* and *ns/nnz* axes that don't show in the bench because:

1. The bench mix is dominated by matrices where dense factor time
   is a small fraction of total feral-vs-MUMPS time. Sparse
   path, symbolic phase, refinement loop also contribute.
2. The ns/sup-vs-ns/nnz ratio is preserved when both layers
   shrink proportionally.

The bench is the wrong instrument for measuring small-front kernel
work in isolation. `diag_supernode_cost` is the right instrument
for per-front kernel cost; bench captures end-to-end ratio which
includes everything else.

## What's actually open as a measurable target

### Validate() bypass on hot-path factor_frontal callers

`src/dense/matrix.rs:106-133` — `SymmetricMatrix::validate()` does
a full O(n²/2) lower-triangle scan for NaN/Inf on every call.
Hot-path call sites:

- `src/dense/factor.rs:336` — `factor()` entry. Outside the
  multifrontal path; called by direct dense callers and by
  `factor_single_front`.
- `src/dense/factor.rs:643` — `factor_single_front()`. Outside
  the multifrontal path.
- `src/dense/factor.rs:871` — `factor_frontal()`. **On the
  multifrontal hot path**, including via `factor_block32` for
  32×32 dispatch.
- `src/dense/factor.rs:1076` — `factor_frontal_blocked()`
  wrapper. The hot multifrontal path uses
  `factor_frontal_blocked_in_place_with_scratch` (no validate)
  but that *also* dispatches to `factor_frontal` for 32×32
  fronts, hitting the line 871 validate.

For a 32×32 front (the dominant CHAINWOO front shape and the
post-#9 SIMD-dispatched path), validate scans
`32*33/2 = 528` doubles. At an L1-resident 0.5–1.5 ns/double
that's **260–800 ns per front**, fully *inside* the ns/sup
budget of 600–1200. This is plausibly 30–60% of the current
per-front overhead on the SIMD-dispatched cluster.

The multifrontal driver assembles fronts from a value-checked
CSC (validation happens at the symbolic phase boundary), so the
per-front re-scan is redundant.

**Surgical fix.** Add a `validated: bool` parameter to
`factor_frontal_with_profile` (or split into a `_unchecked`
variant) and let the multifrontal driver call the unchecked
form. Direct dense callers keep the validated entry. Bit-
identical output by construction; the only behavioral
difference is the absent NaN/Inf scan.

**Expected bench movement.** If validate is 30% of the per-front
overhead on the 32×32 cluster, removing it drops ns/sup on that
cluster by ~30%, which is a similar shrink to #13 Phase A. By the
above analysis that does not necessarily move bench p90 — but the
*absolute* per-front time would drop closer to the SSIDS budget.

### What this does NOT close

- The 4× absolute fixed-cost gap vs SSIDS is structural. SSIDS
  bundles whole leaf subtrees into one task with one allocation,
  shared workspace, and no per-node loop overhead beyond the
  index walk. Feral's per-front dispatch always pays one
  function-call boundary, one row_indices build, one row_map
  populate/restore. Closing the rest of that gap is a real
  refactor (SmallLeafBatch::On done right, or column-renumbering
  per `dev/plans/phase-2.12-column-renumbering.md`), not a
  per-front pooling fix.

- The bench p90 small <1.30 / medium <1.60 criterion #2 from
  #13. Per the data above, bench p90 is dominated by factors
  outside the dense kernel + driver layer for these matrices.
  Closing those targets is its own line of work that wasn't
  scoped under any of #11/#12/#13.

## Recommended re-gating

- **#11**: re-gate. Original gate "after #9 + #10 land" is met
  but the underlying hypothesis didn't hold. New gate: ns/sup
  on the long-tail cluster (ACOPR30/CRESC100/KIRBY2) within
  2× of ns/nnz on that same cluster, *or* explicit measurement
  showing build_row_indices accounts for >15% of per-front
  cost on a representative matrix. If neither is met within a
  reasonable time horizon, close as "structural, requires the
  Phase 2.12 column-renumbering refactor."

- **#12**: keep open as a tracking issue but update the body to
  reflect that the "close 9/10/11" path doesn't match the data.
  The actual path to closing the bench gap on small fronts is:
  (a) finish per-front overhead reduction via the un-done
  validate-bypass target; (b) re-measure and assess what other
  layers (sparse path, refinement, scaling) account for the
  remaining ratio; (c) Phase 2.12 column-renumbering if the
  per-front floor needs to come down further.

- **#13**: post a follow-up comment acknowledging that "next
  lever is #9" was wrong, and listing the un-done candidates
  (validate bypass first). Don't re-open the issue — its scope
  was pooling and that scope was met.

## Anchors

- Gate measurement: `dev/research/dense-app-path.md` (full
  `diag_supernode_cost` output reproduced there).
- #13 pooling work: commits `575f86a`, `1e26902`, `fe2ca4d`.
- #9 dispatch land: commit `d3f1132`,
  `dev/sessions/2026-05-13-02.md`.
- Validate call sites: `src/dense/factor.rs:336, 643, 871, 1076`;
  `src/dense/matrix.rs:106-133`.
- SSIDS reference: `dev/research/ssids-small-frontal-speed.md`
  §3 (SmallLeafNumericSubtree).
- SmallLeafBatch flip attempt: `dev/tried-and-rejected.md`
  2026-04-25 entry (Phase 2.11 Option B).
