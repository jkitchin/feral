# Issue #10 follow-up — supernode-shape thesis (NEGATIVE)

Status: closes the "untested unblocker" identified at the bottom of
`issue-10-maxfromm-phase2-corpus.md`. Hypothesis: a nested-dissection
ordering (Metis or Scotch) might widen the bottom-of-tree supernodes
on 1D-banded KKTs enough to re-engage the MAXFROMM or APP levers
against the Mittelmann panel. **Falsified.** No tested ordering
meaningfully widens supernodes; all ND alternatives are 1.30-2.30×
slower than AMD on this panel. #10 (and the underlying #33
clnlbeam-class blocker) is jointly blocked on supernode amalgamation
(a symbolic-side restructure) or acknowledging a hardware floor for
narrow sequential factorizations.

## A/B harness

`src/bin/diag_ordering_panel.rs` runs the 4-family × 20-matrix 1D-
banded Mittelmann panel under each ordering in `{Amd, MetisND,
ScotchND}` and reports for each `(matrix, method)` pair:

- supernode count
- mean / p50 / p90 / max eliminated-column width (`ncol`)
- mean frontal `nrow`
- factor time (min-of-5, warm-up uncounted)

Per-family summary reports the per-method geomeans and the
paired-by-matrix ratios `factor_us / Amd` and `ncol_mean / Amd`.

## Result table

```
                  ncol_mean/Amd    factor_us/Amd
family            MetisND  ScotchND  MetisND  ScotchND
clnlbeam          1.024    1.043     1.140    1.140
henon120          0.896    1.102     2.267    2.154
lane_emden120     1.128    1.213     2.297    2.099
dirichlet120      0.894    0.952     1.540    1.301
```

(`ncol_mean/Amd > 1.5` would have indicated meaningful shape
widening. `factor_us/Amd < 0.9` would have indicated a direct factor
win. Neither criterion is met in any cell.)

## Why no ordering widens

The p90 of `ncol` is **10.08 across every method on every family**.
The supernode-width distribution is essentially identical between
AMD, MetisND, and ScotchND. The reason is structural: 1D-banded KKTs
have a chain-like supervariable elimination tree where the only
amalgamation candidates are the chain links themselves. Both AMD's
exact external degree and ND's edge-separator splits choose the same
chain — there is no fill-reducing reordering that can fuse chain
links into rectangular fronts without simultaneously degrading the
nnz_L bound that motivated the ordering choice in the first place.

A few per-matrix anomalies are worth noting:

- `clnlbeam_0001`: ND reduces snode count (43_633 → 36_116) and
  raises ncol_mean (2.29 → 2.77). This is the closest any cell comes
  to shape widening, and it still doesn't translate to a factor win.
- `henon120` and `dirichlet120` `_0000` matrices: all three methods
  collapse to identical ncol distributions (mean ≈ 1.03, p50 = 1,
  p90 = 1). These are the densest IPM iterations and AMD's
  near-equivalent shape on them is invariant.

## What makes the factor times worse

ND orderings on 1D-banded matrices systematically inflate frontal
`nrow` (see `lane_emden120` `nrow_mean 22.73 → 50.65` for MetisND,
44.20 for ScotchND). The contribution-block assembly cost scales
with `nrow²`, so even when ncol stays flat the per-supernode rank-1
update count grows quadratically. The bench reflects this directly:
factor times track `nrow_mean` not `ncol_mean`.

## Joint implication for #10 and #33

Three architectural levers tried against this panel in May 2026:

1. SmallLeafBatch driver overhead removal (#33 SLB A/B) — within noise.
2. MAXFROMM AMAX-scan cache (#10 Phase 2 A/B) — within noise.
3. Manual axpy SIMD tightening (`bench_axpy_small` session 02) —
   pulp ties scalar within 1ns/call; manual unroll4 slower.

Now with this session's result added:

4. Ordering swap (this note) — no shape widening, factor times
   1.30-2.30× slower.

All four levers come up negative on the 1D-banded Mittelmann panel.
Remaining options:

- **Supernode amalgamation** — relax the `relax_*` thresholds in
  `SupernodeParams` to forcibly merge chain links into wider fronts.
  Will increase nnz_L and per-front work, so the question is
  whether the rectangularity gain (enabling APP / level-2 BLAS-like
  kernels) outpaces the nnz inflation. Independent research, not
  this session.
- **Hardware floor acknowledgement.** Sequential rank-1 axpy on
  ncol=1..16 fronts is bandwidth-bound; the kernel is already
  vectorized; no further per-pivot speedup is available without
  changing front structure.

Recommendation: keep `OrderingMethod::Amd` as default for the IPM
workload (consistent with the May-2026 `pick_default_method` rule).
`Solver::with_ordering(MetisND/ScotchND)` (shipped in session 02 as
the #33 §3 builder) stays as the opt-in knob; this note documents
that for the 1D-banded panel it does not unblock #10.

## What this rules in

ND orderings are still the right default on the dense / wide-class
matrices the IPM corpus also contains (per the existing
`pick_default_method` heuristics and the 41-matrix shape bake-off).
This note only rules out using ND as a #10 unblocker on the specific
1D-banded Mittelmann panel.

## Reproduction

```
cargo run --release --bin diag_ordering_panel
```

Full output captured at `/tmp/ordering_panel.log` during the run
that produced this note; raw per-matrix table above is taken from
that log.
