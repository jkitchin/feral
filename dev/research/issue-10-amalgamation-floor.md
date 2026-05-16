# Issue #10 final lever — forced supernode amalgamation (NEGATIVE)

Status: closes the last symbolic-side lever flagged at the bottom of
`issue-10-ordering-supernode-shape.md`. Hypothesis: raising
`SupernodeParams::nemin` above the Phase 2.13a default of 16 will
forcibly merge chain-link supernodes into wider fronts, re-engaging
MAXFROMM/APP-class kernels on the 1D-banded Mittelmann panel. The
A/B widens fronts in shape terms (`ncol_mean` doubles at nemin=64)
but factor time barely moves and `clnlbeam` regresses outright.
**Lever falsified.** Combined with the prior four negative
levers (SLB, MAXFROMM, axpy SIMD, ordering swap) this leaves only
the rank-1 axpy hardware floor as explanation. #10 closes as
"hardware floor reached on the 1D-banded panel"; the symbolic and
ordering knobs ship as opt-in but stay off by default.

## A/B harness

`src/bin/diag_nemin_amalgamation_panel.rs` runs the 4-family ×
20-matrix 1D-banded Mittelmann panel under
`nemin ∈ {16, 32, 64, 128}` and reports per-matrix shape
(`ncol_mean`, `ncol_p90`, `nrow_mean`, `snodes`), fill
(`factor_nnz`), and factor time (min-of-3, warm-up uncounted). The
per-family summary reports paired-by-matrix geomean ratios versus
the nemin=16 baseline.

### Why not nemin=256 / MAX

A pilot run with `nemin ∈ {16, 32, 64, 128, 256, MAX}` ran
overnight without finishing a single factor at nemin=MAX —
`clnlbeam_0000` collapsed into a near-dense supernode of order >n/2
and the dense LDL never returned. nemin=256 ran but cost 4.6× the
nemin=16 baseline on `clnlbeam_0000` with `factor_nnz` inflated 9.9×.
The binary is therefore capped at 128, which is the largest value
that still completes in seconds while preserving the monotonic-
regression signal.

## Result table

```
                     factor_us / nemin=16 (geomean, paired)
family              n=32    n=64    n=128
clnlbeam            1.032   1.356   1.989    (monotonic regression)
henon120            0.970   0.960   1.029    (within noise)
lane_emden120       0.953   0.903   0.909    (borderline; -10% at n=64)
dirichlet120        0.951   0.943   0.958    (within noise)

                     ncol_mean / nemin=16 (shape engagement)
family              n=32    n=64    n=128
clnlbeam            1.117   1.247   1.352
henon120            1.298   1.895   3.630
lane_emden120       1.364   1.975   3.819
dirichlet120        1.416   2.021   3.726

                     factor_nnz / nemin=16 (fill cost)
family              n=32    n=64    n=128
clnlbeam            1.533   2.585   4.567
henon120            1.095   1.333   1.875
lane_emden120       1.065   1.229   1.601
dirichlet120        1.065   1.235   1.620
```

Acceptance criteria from the binary's interpretation guide:

- `factor_us / nemin16 < 0.9` → direct win. **Only `lane_emden120`
  at nemin=64 (0.903) reaches the threshold, and only barely.**
- `ncol_mean / nemin16 > 1.5` → meaningful shape widening. Met at
  nemin=64 on three of four families (only clnlbeam falls short).
- `factor_nnz / nemin16 > 1.3` → fill inflated as the cost of forced
  merges. Triggered at nemin=64 across the board.

The shape lever therefore *does* engage — `ncol_mean` doubles at
nemin=64 — but the time profile does not respond. Fill grows
faster than the kernel speeds up. clnlbeam is the most extreme case:
the chain-like elimination tree forces merges of geometrically
unrelated columns, blowing fill 2.6× for a 36% regression in factor
time.

## Why the shape lever doesn't move time

The Phase 2.13a baseline already has `ncol_p90 = 10.08` across the
panel (per the ordering-swap note). The columns the amalgamation
sweep adds are by definition the *small leaves* of the elimination
tree — chain links of width 1..4 that AMD chose to keep separate
because their external degrees diverge. Merging them gives a
wider supernode in column count but *the trailing block stays
narrow* (most of the merged columns have very few off-diagonal
rows). The per-supernode work is then:

  - panel factor: `O(ncol^2 · nrow)` — grows with the merge
  - trailing update: still `O(ncol^2 · trail_size)` — but
    `trail_size` is now the *union* of the merged columns'
    trailing patterns, which inflates fill by the union size.

In practice the trailing fill grows faster than the dense kernel
can amortize. The crossover from "narrow rank-1 axpy" to "dense
panel factor with BLAS-3 reuse" only happens at ncol roughly equal
to the L2 register tile (16-32 columns on x86-v3, 8 on baseline).
On 1D-banded KKTs the *natural* ncol distribution doesn't reach
that tile boundary, and forcing it there inflates fill faster than
it widens the panel.

## Joint conclusion across all 5 levers (#10)

|  # | Lever                            | Verdict on 1D-banded panel                |
|----|----------------------------------|-------------------------------------------|
|  1 | SmallLeafBatch driver removal    | within noise                              |
|  2 | MAXFROMM AMAX-scan cache         | within noise                              |
|  3 | Manual axpy SIMD tightening      | pulp ties scalar within 1ns               |
|  4 | Ordering swap (Metis/Scotch ND)  | 1.3–2.3× slower, no shape widening        |
|  5 | Forced amalgamation (this note)  | shape widens 2×; time flat or worse       |

All five negative. The rank-1 axpy kernel on ncol=1..16 fronts is
bandwidth-bound; pulp is already saturating the vector ALU; and
the elimination tree shape is what AMD says it should be (any
restructure carries a fill penalty). No further per-pivot speedup
is available without changing the front structure in ways that
violate the nnz_L bound that motivated the ordering choice.

## Decision

Keep `SupernodeParams { nemin: 16, .. }` as the default. The
amalgamation knob ships *de facto* via the existing `nemin` field
on `SupernodeParams`; this note documents that raising it above 16
on the 1D-banded panel does not unblock #10. Future work that
*adds new front structure* (e.g., explicit children-of-children
amalgamation across non-adjacent tree levels, or an APP-class
kernel that handles ncol < tile-size differently) may revisit;
this lever as-is is exhausted.

`OrderingMethod::Amd` stays the default (consistent with
`pick_default_method`). `Solver::with_ordering(MetisND/ScotchND)`
remains the opt-in knob. No change to public API.

## Reproduction

```
cargo run --release --bin diag_nemin_amalgamation_panel
```

Full output captured at `/tmp/nemin_sweep_v2.log` during the run
that produced this note; the per-family summary tables above are
the verbatim "summary (paired vs nemin=16, geomean)" blocks from
that log.

## What this rules in

The default `nemin=16` is empirically defensible on the 1D-banded
KKT workload that motivates #10. The lever stays available for
workloads where the elimination tree genuinely has fusion
opportunities (wider PDE Jacobians, dense-fronted IPM iterations
on bordered systems) — this note only rules it out as a #10
unblocker on the specific 1D-banded panel.

## Open question for follow-up (NOT this issue)

The Phase 2.13a default `nemin = 16` was chosen by a separate
shape-bake-off and is preserved here without re-evaluation. The
present sweep starts from that default and only varies upward;
*lowering* nemin below 16 was not tested. If a future workload
demonstrates that the 16 threshold itself was tuned too aggressively
for one workload (say, the bordered-supernode regime), the
`diag_nemin_amalgamation_panel` binary can be extended to sweep
{1, 2, 4, 8, 16, 32}. This is not a #10 issue.
