# FERAL Context (auto-generated)

Generated: 2026-05-21T15:13:30Z

## Latest Session
File: dev/sessions/2026-05-20-03.md
```
# Session 2026-05-20-03

## Goal

Fix issue #46 — FERAL's LDLᵀ ~160× slower than MA57 on the POUNCE CHO
`parmest` IPM KKT (n=43332): a delayed-pivot cascade on a saddle-point
KKT with a structurally-zero (2,2) block (28M factor-nnz, ~17 s
factor). Post a status comment on #46, then fix it correctly.

## Benchmark Results

No regression. Bench numbers are flat vs the prior session
(2026-05-20-02). The single trivially-worse number — dense medium p90
1.70 → 1.71 (+0.01) — is benchmark noise: an interim run this session
read 1.74, and the final run below reverted to baseline. The #46 fix
is a numeric-kernel change that does not touch ordering, and none of
the bench corpus is a zero-(2,2)-block saddle KKT, so no real effect is
expected.

```
--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.32     <= 2.0     PASS
medium (<500)            152145     1.71     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.57     <= 2.0     PASS
medium (<500)            153560     1.57     <= 3.0     PASS

Top 10 worst factor-ratio vs MUMPS:
MUONSINE_0000  18.96   ACOPR30_0001  12.31   ACOPR14_0211  9.93
ACOPR30_0039    8.14   KIRBY2_0007    7.92   ACOPR14_0128  7.03
ACOPR14_0365    6.93   KIRBY2_0006    6.91   ACOPR14_0187  6.75
ACOPR14_0472    6.66
```

(Prior session 2026-05-20-02: Dense 1.32 / 1.70, Sparse 1.57 / 1.57.)

## Accomplished

**Diagnosed #46 — and overturned the three-agent research diagnosis.**
The session opened with a three-agent research phase (feral source,
MUMPS 5.8.2, SPRAL SSIDS) that converged on "analysis-phase ordering
failure, fix = broaden `pick_ordering_preprocess`'s activation
predicate". Two ground-truth probes on the real CHO KKT refuted every
load-bearing claim:

- `probe_issue46_preprocess` — feral stores only the lower triangle, so
  KKT constraint columns are stored-degree 0/1, *not* high-degree. The
```

## Git Status
```
c990def feat(scaling): value-bounded MC64 scaling cache (B2) + fix External 10× bug
9512c0a perf(profiler): instrument numeric prologue sub-phases (Track B1)
8a481b0 docs(plan): reference filed issue #48 for Track C
60febc5 docs(research): profile the per-factor cost cluster — two mechanisms
c898f71 fix(ci): commit arch-unstable rankdef synth fixtures to fix stress-smoke (#46)
```

## Test Status
```
