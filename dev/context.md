# FERAL Context (auto-generated)

Generated: 2026-05-21T22:14:27Z

## Latest Session
File: dev/sessions/2026-05-21-03.md
```
# Session 2026-05-21-03

## Goal

Track A3 — validate Fix 1 (fine-grained delayed pivoting, `42434a5`)
end-to-end via `probe_kkt_replay` on `pinene_3200`, `robot_1600`,
`marine_1600`; confirm the pinene iter 6–9 factor-time explosion is
gone and per-iter inertia stays exact.

A3 surfaced a **correctness regression**; per the human's "fix forward
this session" instruction the goal expanded to: keep Fix 1, diagnose
and fix the residual so pinene is both fast *and* inertia-exact.

## Accomplished

### A3 validation — found a regression

Fix 1 broke the pinene delayed-pivot cascade (456 s → 4.7 s) but
returned `WrongInertia` on the borderline near-singular iterates 8/9
(δ_c ≈ 1e-11): a spurious `inertia.zero`. Pre-Fix-1 warm replay was
all-exact (456 s, worktree at `ef5fb7e`) — so this is a Fix 1
regression. It violated the hard rule "inertia must be exactly correct
on non-singular matrices."

### Root cause — pre-existing 2×2-inertia cancellation bug

Fix 1 did not *cause* the regression; it *exposed* one. The pre-Fix-1
break-on-first cascade dumped ~116k–133k columns to a dense root front
whose full BK pivoting gave Sylvester-exact inertia — that cascade was
silently buying correctness. The latent bug: `count_2x2_inertia` /
`count_2x2_inertia_val` classified signs from `λ = 0.5·(tr ∓ s)`;
although `s` is cancellation-free, the *final* subtraction `0.5·(tr∓s)`
cancels — a genuine non-singular 2×2 whose small eigenvalue is below
`ULP(0.5·tr)` IEEE-rounds to *exactly 0.0*, counted as a `zero`.

### Fix 2 — cancellation-free 2×2 inertia classification

Lifecycle: research (journal §17:40/§18:05) → plan
(`dev/plans/kkt-cascade-fix2-2x2-inertia-cancellation.md`) →
tests-first → implement → verify → benchmark.

- Added `det_sym2x2` — Kahan fused difference-of-products
  (`w=fl(d21²)`, `e=fma(d21,d21,-w)`, `det=fma(d11,d22,-w)+e`),
  relative error ≤ 2·u for any inputs.
- Added `classify_2x2_inertia` — classifies from `sign(det)` +
  `sign(tr)`: `det<0`→(1,1,0); `det>0`→(2,0,0)/(0,2,0); `det==0`
  exactly→(1,0,1)/(0,1,1)/(0,0,2).
- `count_2x2_inertia_val` delegates to it; `count_2x2_inertia`'s three
  branches reclassified through it (force-accept bands fold a genuine
  zero into `neg`, preserving #42 Option A; non-singular branch reports
```

## Git Status
```
12585d9 docs(journal): record #47 root cause and #44 assessment
2eab12f test(issue-44): add NARX_CFy per-factor cost probe
6185415 test(issue-47): add explicit-zeros warm-refactor probe
787315f docs(session): checkpoint 2026-05-21-03 — Track A3 + Fix 2 (2×2 inertia)
80c05f5 fix(inertia): classify 2×2 blocks from cancellation-free sign(det) (#48)
```

## Test Status
```
