# Plan — per-factor cost cluster

**Status:** B1 done (instrumentation landed). B2 implemented but
**pivoted off** 2026-05-21 — the cache is correct and tested but has
no measured corpus payoff (gate metric confounded by the IPM δ
trajectory; MC64 is <2 % of factor cost vs the iter 6-9 blowup). See
`decisions.md` / `tried-and-rejected.md` 2026-05-21. Effort moved to
**Track A**. A1 done 2026-05-21 (pinene cascade characterized — see
below). A2 diagnosed 2026-05-21 via scalar-path delay-cause
instrumentation — see `dev/research/kkt-cascade-amplifier-2026-05-21.md`.
The cascade is an **amplifier × two triggers**, NOT "the 2×2 stability
gate" (the earlier framing was an over-read; corrected below). The
recommended fix is **fine-grained delay** (swap-to-boundary), which
is the next implementation task — pending a tests-first plan.
**Date:** 2026-05-21
**Research note:** `dev/research/per-factor-cost-cluster-2026-05-21.md`
(see §10 for the B1 findings that revise this plan)
**Closes / advances:** #44 (NARX_CFy), #38 residual (rocket_12800),
touches #47.

The research note establishes two independent mechanisms behind the
Mittelmann per-factor loss cluster. This plan has one track per
mechanism plus a correctness side-task. **Track B first** — it is the
larger, clearer prize and the rocket/NARX timeouts are the
user-visible failures.

---

## Track B — prologue-dominated factor cost (Mechanism B)

Headline fact: on `rocket_12800` the numeric factor is 99.5% prologue
(3.4 s) and 0.5% supernode loop (10 ms). The prologue
(`src/numeric/factorize.rs:1668–1782`) reads as O(n)/O(nnz) code, so
the 3.4 s is unexplained by inspection and must be measured.

### B1 — instrument the prologue sub-phases (gating step) — DONE

Landed: `PrologueBreakdown` (8 fields) on `Profiler`/`ProfileReport`,
profiling-gated `tic`/`toc` around each prologue sub-phase in
`factorize_multifrontal_supernodal_with_workspace`, and the
`from_triplets` rebuild split out of `permute_csc_values` (signature
now `-> Result<(CscMatrix, u64), FeralError>`). Zero overhead when
`profiler.is_none()`. `probe_rocket_profile` takes an optional
problem-name arg and prints the breakdown + a `diagnose_scaling`
drill-down.

**Result (research note §10):** on `rocket_12800` the prologue is
**99.8% the `scaling` sub-phase** — and `diagnose_scaling` pins that
to the **MC64 Hungarian** (`compute_scaling(Mc64Symmetric)` = 4111 ms
vs `InfNorm` = 5.4 ms). The plan's prime suspect (`from_triplets`,
3.8 ms) is exonerated. `NARX_CFy` is *not* Mechanism B — it is
loop-dominated and value-dependent (reclassified A-adjacent).

### B2 — eliminate the per-call MC64 Hungarian on rocket_12800

Confirmed target: the MC64 Hungarian matching is rerun from scratch
on every IPM factor call. `cached_mc64` is cleared after the first
factor (the issue #38 fix `db20166`) because the iter-0 *scaling
values* go stale — but the **matching** (the permutation) is far more
value-stable than the scaling vector. Options, in preference order:

1. **Value-bounded MC64 cache**
   (`dev/research/mc64-value-bounded-cache-2026-05-17.md`). Keep the
   Hungarian *matching* across IPM iterations (pattern is
   bit-identical; the matching changes rarely) and recompute only the
   O(n) scaling vector from current values. This is the principled
   fix and sidesteps the #38 staleness trap — the matching is not the
   thing that went stale.
2. **Re-route rocket away from MC64.** `pick_scaling_strategy` picks
   `Mc64Symmetric` for rocket; InfNorm scaling is 5.4 ms. Check
   whether InfNorm gives an acceptable factor (residual + inertia) on
   the rocket corpus — if MC64's conditioning win is marginal here,
   the Policy-4 fallback threshold (`compute_scaling_auto_with_cache`)
   may just need widening. Cheap to test, but per-problem brittle.
3. **Speed up the Hungarian itself.** rocket's Hungarian is 4.1 s
   while `NARX_CFy`'s (same n≈90k, nnz≈340k) is 29 ms — a 140×
   structural gap. rocket's KKT must produce pathological augmenting
   paths. Worth a `diag`-style profile of `hungarian_match` before
   committing, but this is the deepest fix.

**Tests first.** This is a correctness-sensitive path (scaling feeds
the factor). External oracle: inertia + residual on the parity
corpus must be bit-unchanged for any matrix where the matching is
genuinely reused. The value-bounded cache needs a test that a reused
matching + recomputed scaling equals a from-scratch
`compute_symmetric` on a hand-built matrix whose values have drifted
within the bound.

**Exit:** `rocket_12800` replay factor time down from ~6.5 s toward
the MA57 band; re-run `probe_rocket_profile` and record the new
prologue/scaling split.

### B3 — end-to-end validation

Re-run ipopt-feral on `rocket_12800` and `NARX_CFy`; confirm `NARX_CFy`
finishes inside the 600 s cap (closes #44) and `rocket_12800` lands
within ~2× MA57. Re-run the Mittelmann sweep subset to check for
regressions. Update `REPORT-vs-plato.md`.

---

## Track A — pinene zero-(2,2)-block cascade (Mechanism A)

**Reframed 2026-05-21 by the A1/A2 investigation (session 2026-05-21-01).**
The `pinene_3200` iter 6-9 blowup — 493.9 s replay, 98 % of wall — is
the **issue #46 zero-(2,2)-block delayed-pivot cascade**, not a
generic spike that a proactive CB arm would fix.

### A1 — pinene cascade characterized — DONE

`diag_pinene_pivot_cliff` on iterates 0008/0009 (n=127995): the factor
builds a **fully dense ~17.5k×17.5k root front** (node 10486, ~14 % of
n; `nelim = nrow`), ~91 % of root columns in 2×2 pivots, fed by
133 648 delayed pivots. The three supernodes below the root are
~3.8k–18k columns wide but eliminate only 4 / 11 / 493 columns — pure
delay conduits. The cascade worsens monotonically as the IPM drives
δ_c→0 (root 15446→17538, `nnz_L` 128.5M→165.7M across two iters).

### A2 — fix the delayed-pivot cascade — FIX 1 DONE 2026-05-21-02

**Fix 1 (fine-grained delay / swap-to-boundary) shipped session
2026-05-21-02.** Both BK driver loops in `src/dense/factor.rs` now
swap a stuck column to the fully-summed boundary and decrement
`ncol_eff` instead of breaking; the new `delay_swap_to_boundary`
helper does the symmetric swap. On `pinene_3200_0009`: `n_delayed`
133648→11309, factor_nnz ~165.7M→3.6M (blowup 69×→1.51×), factor
time ~183 s→78 ms, inertia exact and unchanged (64000,63995,0).
Bench: all four exit-partition buckets PASS, no regression. The
residual 11309 delays are the genuine (un-amplified) count from
Triggers A/B — at 1.51× blowup **Fix 2/3 are not needed**. Plan:
`dev/plans/kkt-cascade-fix1-fine-grained-delay.md` (status DONE).

Original diagnosis follows.

### A2 — fix the delayed-pivot cascade — DIAGNOSED 2026-05-21

Full diagnosis: `dev/research/kkt-cascade-amplifier-2026-05-21.md`.
Scalar-path delay-cause instrumentation (new `panel_diag` counters)
+ `probe_issue46_supernode pinene_3200_0009.mtx` localized the
cascade as an **amplifier × two triggers**:

- **Amplifier — break-on-first-delay.** The BK driver loop
  (`factor.rs:1719-1849`) does `Delayed => break` then
  `n_delayed = ncol - nelim`: one delayed pivot forfeits the whole
  remaining tail of the supernode. 3936 delay events →
  `n_delayed = 133648` (~34 columns/event). Config 2 (static
  pivoting, `n_delayed=0`, healthy 1.25× factor) proves the
  forfeited columns are pivotable — the forfeit throws away work.
- **Trigger A — split MC64 pairs (2840 events).** 1×1 delays whose
  matched partner is not co-located (`a[k][k+1]==0` →
  `partner=None`). Lines up with the 3781 split-across-supernodes
  pairs — the residual #46 co-location gap.
- **Trigger B — growth-bound rejection (1096 events).** Co-located
  saddle 2×2 candidates that failed the Duff-Reid growth bound (816
  with `det<0`, genuine indefinite saddles). The SSIDS det floor
  fires 0 times — not involved.

**Correction to the earlier framing.** Prior plan/journal text said
A2 is "fix the 2×2 stability gate" and pointed at
`fallback_2x2_need_swap_or_bound=31542`. That was an over-read:
`need_swap_or_bound` is the *panel delegating* a swap-2×2 case to
`scalar_pivot_step`, not a rejection. The true delay split is
2840 (1×1) / 1096 (2×2 growth) / 0 (det floor).

**Recommended fix (research note §6), in order:**

1. **Fix 1 — fine-grained delay (swap-to-boundary).** PRIMARY.
   Replace `Delayed => break` with: swap the stuck column to the
   fully-summed boundary, decrement `ncol_eff`, keep eliminating.
   Forfeits 1 column per stuck pivot, not ~34. Inertia-exact by
   construction (real delayed pivoting; no force-accept, no
   perturbation). Largest single lever, independent of the
   matching machinery. The panel path (`PanelStatus::Delayed`
   break, `factor.rs:1844`) needs the same treatment for
   generality — though `pinene` has `PANEL_DELAYED=0`.
2. **Fix 2 — matching-aware growth-bound exemption** for co-located
   MC64-matched saddle 2×2s (≤1096 events). Only if a residual
   cascade remains after Fix 1.
3. **Fix 3 — tighter co-location** for split pairs (analysis-phase,
   `ldlt_compress.rs`). Deepest; only if triggers A still dominate.

**Correctness-critical** — Fix 1 touches the pivot/inertia path.
Needs a tests-first plan; oracle = saddle-point inertia theorem
(Benzi/Golub/Liesen 2005 §3.4), exactly as #46 did. Do **not** ship
`cascade_break` as the production fix (100× speedup but corrupts
iter-9 inertia — same inadmissibility as `allow_delayed_pivots=false`).
The bounded-Δ CB repair stays a fallback only.

Cross-check: `dev/research/kkt-zero-2x2-block-cascade-2026-05-20.md`
(the #46 fix), `dev/research/cascade-break.md`.

### A3 — validation

`robot_1600` and `marine_1600` replay totals should drop to the
`CB=on` numbers (0.199 s, 10.5 s) under the *default* config. Re-test
end-to-end; confirm no regression on problems where CB currently
hurts (the auto-CB spot-check in 2026-05-17-01 showed `bearing_400`
and `rocket_12800` are CB-neutral-to-slightly-negative).

---

## Track C — marine_1600 WrongInertia drift (correctness)

Research note §7: `marine_1600` returns `WrongInertia` at iters
10/14/16/17 under both CB modes (spurious zero eigenvalues). This is a
correctness defect, not perf. Filed as **#48**; triage
separately — likely related to the 2×2 near-singular classification
work (#39, `dev/research/fbrain3ls-2x2-stability.md`). Do not fold it
into the perf tracks.

---

## Sequencing

1. **B1** (instrument prologue) — gates everything in Track B.
2. **B2** (fix dominant sub-phase) — the headline win.
3. **A1 + A2** (proactive cascade-break) — can run in parallel with B2;
   independent code paths.
4. **B3 + A3** (end-to-end validation, sweep, REPORT update).
5. **Track C** — file the issue now; schedule independently.

## Issue mapping

| issue | track | note |
|-------|-------|------|
| #44 NARX_CFy timeout       | A   | reclassified by B1 — loop-dominated, value-dependent large fronts; needs the A1 delayed-pivot trace, not the prologue fix |
| #38 residual rocket_12800  | B   | B2/B3 — the closed-issue residual; confirmed = per-call MC64 Hungarian |
| #47 explicit-zero fast path| B-adjacent | re-evaluate after B2 |
| marine_1600 WrongInertia   | C   | filed as #48 |
