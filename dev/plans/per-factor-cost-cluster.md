# Plan — per-factor cost cluster

**Status:** Proposed. No code written yet.
**Date:** 2026-05-21
**Research note:** `dev/research/per-factor-cost-cluster-2026-05-21.md`
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

### B1 — instrument the prologue sub-phases (gating step)

Add per-sub-phase timers inside `factorize_numeric` between the
existing `t_prologue` start and the loop, attributing wall to:

1. `row_map` resize
2. `compute_scaling_with_cache`
3. `scaling_pivot_order` build
4. `permute_csc_values` (split out the `from_triplets` rebuild
   inside it separately — that is the prime suspect)
5. `symmetric_pattern()`
6. `scaled_matrix_infnorm` / `override_null_pivot_tol`
7. `is_root` + `contrib_blocks` + `node_factors` allocation

Plumb the split through `Profiler` (it already carries
`prologue_us`; add a `prologue_breakdown` field, gated on
`profiler.is_some()` so the default path is untouched).

**Exit:** a sub-phase breakdown of the 3.4 s on `rocket_12800` iter 0
and a second large-n problem (`NARX_CFy`). One sub-phase is expected
to dominate.

### B2 — fix the dominant sub-phase

Branch on B1's result. Likely candidates and their fixes:

- **`from_triplets` rebuild dominates** — `permute_csc_values` builds
  three throwaway `Vec`s of length nnz and calls `CscMatrix::from_triplets`,
  which re-sorts. Replace with a direct counting-sort permutation that
  writes the permuted CSC in place (the permutation is known; no
  general triplet dedup is needed). Likely the single biggest win.
- **`symmetric_pattern()` dominates** — it is rebuilt every numeric
  call although the pattern is value-independent. Cache it on
  `SymbolicFactorization` (it is a pure function of the permuted
  pattern, which is fixed once symbolic is done). Watch the #38-class
  trap: this cache *is* value-independent, unlike `cached_mc64`, so it
  is safe to persist — document that distinction.
- **scaling dominates** — fold into Track-B-adjacent MC64 work
  (`dev/research/mc64-value-bounded-cache-2026-05-17.md`).
- **per-supernode setup scales with `n_snodes`** — then small-front
  amalgamation (`dev/research/phase-2.11-small-front-amalgamation.md`,
  `phase-2.13a-amalgamation-auto.md`) becomes the lever; 16406
  supernodes for n=89601 is heavy fragmentation.

**Tests first.** Per CLAUDE.md, the oracle must be external: the
permuted-CSC correctness oracle is the existing factorization result
(inertia + residual on the parity corpus must be bit-unchanged — this
is a pure perf refactor). Add a focused unit test that
`permute_csc_values`' replacement produces a CSC equal to the current
implementation's output on a hand-built matrix.

**Exit:** `rocket_12800` replay factor time down from 6.5 s toward the
MA57 band; re-run the probe and record the new prologue/loop split.

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

### A1 — delayed-pivot trace (instrumentation, also serves arki)

Expose per-factor `sum_delayed` / `max_delayed` from the `Solver`
result (today only `capi.rs`'s `FERAL_FACTOR_TRACE` env path sees it).
This both (a) lets a proactive arm decide and (b) gives the
delayed-pivot count needed to finish classifying `arki0003` (research
note §6).

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
| #44 NARX_CFy timeout       | B   | B2 fix → B3 should close it |
| #38 residual rocket_12800  | B   | B2/B3 — the closed-issue residual |
| #47 explicit-zero fast path| B-adjacent | re-evaluate after B2; may interact with the `from_triplets`/pattern path |
| marine_1600 WrongInertia   | C   | filed as #48 |
