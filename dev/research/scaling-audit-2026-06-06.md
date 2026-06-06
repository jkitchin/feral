# Scaling audit — super-linear (O(n²)) phase hunt across the solver

**Status:** Investigation report (Steps 0–3 of the approved plan).
**Date:** 2026-06-06
**Author:** session 2026-06-06-02
**Related:**
- Issue #80 (MC64 preprocessor cost), commit 6699f09 (heap-reuse fix)
- `dev/research/per-factor-cost-cluster-2026-05-21.md` §10 (rocket = MC64)
- Tooling: `crates/feral-diagnostics/src/bin/scaling_sweep.rs`,
  `probe_mc64_hungarian.rs`; `src/scaling/hungarian.rs` (HungarianStats),
  `src/scaling/mod.rs::diagnose_mc64_matching`
- Journal `dev/journal/2026-06-06-01.org` (14:05–17:30)

---

## 1. Question

The #80 MC64 heap-reuse fix was the *second* discovery of the same O(n²)
trap (the first being rocket_12800's "99.5%-prologue" cost, root-caused to
MC64 in the 2026-05-21 note §10). Are there *other* latent super-linear
phases across ordering, scaling, and the numeric/symbolic prologue that
only bite at large n? And does the #80 fix actually hold?

Code-reading alone is insufficient — it found only *theoretical* O(n²)
(Hungarian phase-3, supervariable hash-chains) and missed the rocket cost
entirely. The detector here is **empirical, phase-attributed scaling
sweeps** with α (growth-exponent) fits per phase.

## 2. Method

- `scaling_sweep` runs `factor()` with profiling, forces a symbolic cache
  miss (so the normally-cached symbolic phase is timed), takes the
  per-field median of K cold factors, and emits a CSV with the full
  prologue breakdown, all 17 symbolic stages, and the `max_col_degree` /
  `sum_d_logd` structural control variates.
- α is fit by OLS log-log per phase (x = nnz), points above a 30 µs noise
  floor; **α > 1.3 with high R² flags a super-linear law**.
- `diagnose_mc64_matching` returns build-independent algorithmic counters
  (`augment_searches`, `touched_total`, `heap_init_slots`,
  `phase3_inner_iters`, `main_loop_edge_scans`) to localize *where* the
  MC64 matching time goes.
- **All absolute timings must be `--release`**: debug inflates ~12× (see §5).
  α-fits (ratios) are build-independent.

## 3. Headline result

**The MC64 Hungarian scaling is the single consistent super-linear phase
across KKT structure.** Everything else — symbolic analysis (excluding the
MC64 it invokes), the numeric prologue, the supernode loop, ordering — scales
α ≈ 1.0 on every ladder tested. No new O(n²) trap was found outside MC64.

| Ladder (release) | n span | scaling (MC64) α | other phases α |
|---|---|---|---|
| ACOPP/ACOPR power-flow KKT | 106–2460 | (sub-noise) | prologue 1.03, sym 1.00, total 0.90 |
| generated banded KKT | 1333–40000 | **1.26 (R²=1.00)** | prologue 1.14, loop 1.04, sym 1.02 |
| generated banded SPD | 300–10000 | 1.0 | all ≈1.0 |
| rocket_12800 (dense coupling) | 89601 | **severe** (see §4) | sym-ex-MC64 ≈ flat |

(The ACOPP `loop` α=1.31 is low-confidence: 4 points at <0.2 ms, dominated
by small-n per-front fixed costs, not an asymptotic law.)

## 4. MC64 on rocket_12800 — mechanism, localized

rocket_12800_0000: n=89601, cost_nnz=575985, **max_col_degree=38401** (one
coupling column touches 43% of all rows). Instrumented Hungarian:

```
searches=29416  touched_total=2.238e8  heap_init=2.239e8  phase3=63299  edge_scans=3.710e8
wall: DEBUG 44.0 s   RELEASE 3.69 s
```

- Cost is **O(searches × dense_deg)**: ~29k augmenting searches each
  repeatedly traversing the degree-38401 column, split between
  `main_loop_edge_scans` (3.7e8) and heap-reset/`touched` work (2.2e8).
- `heap_init_slots == n + touched_total` exactly — the #80 structural
  invariant holds (heap allocated once + incremental resets).
- **#80 helped, partially:** pre-fix heap work was `searches × m =
  29416 × 89601 ≈ 2.64e9`; post-fix it is 2.24e8 (~12× less). But the edge
  scans (untouched by #80) co-dominate, so the net rocket cost is roughly
  unchanged by #80. #80's big win (pf22, 170×) was on a *near-tree* matrix
  where heap-realloc was the whole cost; rocket is a *different cost class*
  (dense-column edge scans).
- This is a **one-time symbolic analysis** (LdltCompress runs MC64 once per
  pattern, cached across IPM iters). 3.7 s release on a 90k KKT is a real
  inefficiency but far less severe than the 38 s debug figure first
  suggested.

## 5. The debug/release correction (and the resolved discrepancy)

The 2026-05-21 §10 note measured this Hungarian at 4.1 s; the first sweep
here measured 38 s. **The difference is debug vs release (~12×), not a
regression** — §10 was release, the sweep was debug. Release re-measure:
3.69 s ≈ 4.1 s. Lesson baked into the tooling docs: run `scaling_sweep` /
`probe_mc64_hungarian` in `--release` for absolute timings; α-fits are fine
in either build.

## 6. What was NOT found super-linear

- **Symbolic analysis** (excluding the MC64 it invokes via LdltCompress):
  `sym_total` α≈1.00 on every ladder. The 17-stage breakdown shows
  `ldlt_compress` (= MC64) is the only large symbolic stage; ordering,
  col_counts, postorder, find_supernodes, renumber are all small and linear.
  The theoretical supervariable-hash-chain O(n²) (algo.rs:503-564) did **not**
  manifest — the lumped `ordering` stage never flagged.
- **Numeric prologue** (permute, from_triplets, symmetric_pattern, setup):
  α≈1.0–1.14; rocket's `permute_us` = 64 ms (exonerated, as the 2026-05-21
  note predicted).
- **Supernode loop:** α≈1.0 on banded ladders.
- **Hungarian phase-3** length-2 augmentation: linear in nnz on the random
  family (the Step-0 guard already pins this <16× over 8× n).

## 7. Deliverables landed

- **Deterministic MC64 regression guard** (commit 767b0d9): the structural
  invariant `heap_init_slots == n + touched_total` + a phase-3 sub-quadratic
  ratio. Teeth verified by fault injection. CI-noise-immune.
- **`scaling_sweep`** (1b2b21c) + **`probe_mc64_hungarian`** + the
  `diagnose_mc64_matching` instrumentation (this commit). Reusable for future
  per-phase scaling audits.

## 8. Recommendations / open work

1. **MC64 dense-column fast path (the one real super-linear lead).** The
   Hungarian cost is O(searches × dense_deg) when a near-dense coupling
   column exists. Options: (a) detect a near-dense column and handle it out
   of the augmenting-path inner loop; (b) in `LdltCompress`, skip the MC64
   matching when a near-dense column makes the compression leverage marginal
   (the existing `map.ncmp() == n` short-circuit is related). Priority:
   moderate — it is a one-time per-pattern cost (~3.7 s on a 90k KKT), not
   per-IPM-iter.
2. **Confirm the dense-column exponent.** Blocked here: diagonally-dominant
   generators give MC64 a trivial diagonal matching (searches=0). A
   KKT-style *hard*-matching generator (zero (2,2) block + dense coupling
   that defeats greedy) is needed to fit α in `max_col_degree` directly.
3. **Solve phase** (Step 4) was not instrumented and was not implicated —
   `total` is dominated by sym+loop, both linear. Deferred unless a
   multi-RHS / refinement-loop case surfaces.
4. **`ordering` sub-stage timers** (supervariable suspect) remain unbuilt —
   not needed, since the lumped `ordering` stage never flagged super-linear.
