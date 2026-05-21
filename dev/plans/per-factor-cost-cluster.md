# Plan — per-factor cost cluster

**Status:** B1 done (instrumentation landed). B2 implemented but
**pivoted off** 2026-05-21 — the cache is correct and tested but has
no measured corpus payoff (gate metric confounded by the IPM δ
trajectory; MC64 is <2 % of factor cost vs the iter 6-9 blowup). See
`decisions.md` / `tried-and-rejected.md` 2026-05-21. Effort moves to
**Track A** (the delayed-pivot cascade — the actual 98 % cost).
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

## Track A — proactive cascade-break (Mechanism A)

`CB=on` already collapses the `robot`/`marine` spikes ~10×. The defect
is that `Solver::with_auto_cascade_break` is *reactive* — it arms
factor N+1 from factor N's delayed count, so the first cascade in a
trajectory is never prevented (`robot` iter 1, `marine` iter 9).

### A1 — delayed-pivot trace (instrumentation, also serves arki + NARX)

Expose per-factor `sum_delayed` / `max_delayed` from the `Solver`
result (today only `capi.rs`'s `FERAL_FACTOR_TRACE` env path sees it).
This both (a) lets a proactive arm decide and (b) gives the
delayed-pivot count needed to finish classifying `arki0003` (research
note §6) and `NARX_CFy` (#44 — research note §10.3: loop-dominated,
449 fronts with nrow>128 up to nrow=1877 ncol=77, value-dependent
across IPM iters). NARX's large fronts are consistent with
delayed-pivot accumulation; A1's trace confirms or refutes that
before A2's lever is chosen.

### A2 — make the arm proactive

Options, in preference order:

1. **Symbolic arm** — `dev/research/issue-15-cascade-break-symbolic-arm.md`
   already explored arming cascade-break from symbolic structure
   (arrow/saddle KKT shape) rather than from a prior factor. Pick this
   up: if the symbolic phase flags a cascade-prone structure, arm CB
   from iter 0.
2. **Arm-from-iter-0 heuristic** — start armed, disarm once a factor
   completes with low delayed count. Cheaper to implement, looser.

Cross-check against `dev/research/cascade-break.md` and
`warm-state-cascade-amplification-2026-05-17.md` before choosing.

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
