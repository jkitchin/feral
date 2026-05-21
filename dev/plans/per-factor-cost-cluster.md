# Plan — per-factor cost cluster

**Status:** B1 done (instrumentation landed). B2 implemented but
**pivoted off** 2026-05-21 — the cache is correct and tested but has
no measured corpus payoff (gate metric confounded by the IPM δ
trajectory; MC64 is <2 % of factor cost vs the iter 6-9 blowup). See
`decisions.md` / `tried-and-rejected.md` 2026-05-21. Effort moved to
**Track A**. A1 done 2026-05-21 (pinene cascade characterized — see
below). A2 reframed 2026-05-21: it is the issue #46 zero-(2,2)-block
cascade, and the fix is extending the matching-aware kernel, not
arming CB (CB gives 100× but corrupts inertia). A2 is the next
implementation task.
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

### A2 — make bounded-Δ cascade-break inertia-exact — THE REAL FIX

`CB=on` (forced `cascade_break(0.5, eps=1e-10)` every factor) gives a
**100× speedup** on pinene — 493.9 s → 4.86 s, iter 6-9 fronts from
64-208 s to ~45 ms. **But iter 9 returns `WrongInertia`** (got
63999/63995/**1**, want .../0): the `PerturbToEps` absolute floor
1e-10 is below the inertia zero-tolerance, so a force-accepted
near-zero pivot still classifies as zero on the most-converged
iterate. This is the *same* inadmissibility issue #46 already recorded
for its `allow_delayed_pivots=false` control ("breaks the cascade but
gets the inertia wrong … not an admissible fix").

So A2 is **not** "arm CB by default" (the symbolic arm was already
disproved in `warm-state-cascade-amplification-2026-05-17.md`; and an
armed CB that corrupts inertia is inadmissible regardless of *when* it
arms). The #46 kernel fix (matching-aware 2×2 partner in
`scalar_pivot_step`) is already in pinene's production path
(`preproc=LdltCompress` confirmed) yet pinene still cascades — so #46
did not fully cover pinene. A2 must:

1. **Localized — DONE 2026-05-21.** `probe_issue46_supernode
   pinene_3200_0009.mtx`: co-location is good (93.9 % of MC64 pairs
   adjacent, symbolic estimate 2.4M), so #46's gaps A/B are clear.
   The production path cascades anyway — `panel_diag` dominant
   fallback **`fallback_2x2_need_swap_or_bound = 31542`** (the *same*
   fallback that was #46's CHO culprit). #46 widened the 2×2
   *search* so the kernel finds the co-located partner `k+1`; on
   pinene the partner *is* found but the `{k,k+1}` 2×2 candidate is
   then **rejected at the numerical stability gate** → delayed →
   cascade. Static pivoting force-accepts the same columns as 1×1,
   which is fast (52 ms, 1.25×) but wrong by one sign
   (inertia 64001/63994 vs 64000/63995).
2. **Fix the 2×2 stability gate for saddle pivots.** Read the
   `fallback_2x2_need_swap_or_bound` path in `src/dense/factor.rs`
   (`scalar_pivot_step` and the panel inline-2×2). A genuine saddle
   2×2 `[[0,b],[b,a]]` has `det = -b² < 0` — a legitimate indefinite
   pivot (one +, one −), inertia-exact by construction. Find why
   pinene's saddle 2×2s fail the swap-or-bound check and admit them.
   This is the correct fix vs. CB's perturb-and-hope. **Correctness-
   critical** — write a research note first, tests-first, oracle from
   the saddle-point inertia theorem (Benzi/Golub/Liesen 2005 §3.4),
   exactly as #46 did.
3. **If a residual cascade remains**, make bounded-Δ CB inertia-exact:
   scale `cascade_break_eps` by `||A||_∞` (the factorize.rs:2206-2209
   comment already prescribes this and neither `CB=1` nor the auto-CB
   path at solver.rs:638 does it) so the perturbation clears the
   zero-tolerance, and sign the perturbation to preserve inertia.

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
