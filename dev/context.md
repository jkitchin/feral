# FERAL Context (auto-generated)

Generated: 2026-05-15T12:26:37Z

## Latest Session
File: dev/sessions/2026-05-15-01.md
```
# Session 2026-05-15-01

## Bench vs. prior session

Phase 2.8.1 corpus gates unchanged — no `src/` code on the hot
path moved this session (only a new diagnostic bin was added).

- dense small <200: p90 = **1.21** (prior 1.21 in `2026-05-14-01`)
- dense medium <500: p90 = **1.54** (prior 1.54)
- sparse small <200: p90 = **1.46** (prior 1.46)
- sparse medium <500: p90 = **1.46** (prior 1.56)

All four gates PASS.

## Goal

Investigate feral issue #17: pounce-feral with default
`cascade_break_ratio = Some(0.5)` fails on `robot_1600.nl` with
WrongInertia → MaxIter @ 200, 53 s. The issue hypothesises that
cascade-break force-accepts small pivots → wrong inertia count →
IPM over-regularization.

## Accomplished

### 1. Reproduced the failure and confirmed cascade-break is the trigger

| config                              | result      | iters | wall  |
|-------------------------------------|-------------|-------|-------|
| `cascade_break_ratio = Some(0.5)`   | MaxIter     | 200   | 53 s  |
| `cascade_break_ratio = None`        | **Optimal** | 40    | 6.1 s |

### 2. Diagnostic patch in pounce-feral

Wired `POUNCE_FERAL_CASCADE_BREAK=off` env-var into
`pounce-feral/src/lib.rs:50-65` so we can A/B compare without
rebuilding feral. Pattern matches existing `FERAL_PARALLEL` and
`POUNCE_DUMP_KKT`. Tests: 6/6 passing on pounce-feral.

Committed upstream on pounce `main` as `84add74`. Push to
`origin/main` denied by auto-mode classifier (default-branch push
not authorized); sits local pending user.

### 3. Direct inertia comparison vs MA57 — **overturns the issue's hypothesis**

Dumped 4 KKT matrices from a `cb=default` run via
`POUNCE_DUMP_KKT` (iter004, 010, 043, 046). MUMPS failed all 4 with
INFOG(1) = -9 (memory allocation); MA57 succeeded on all 4 →
used MA57 as reference oracle.

feral inertia under `cb=default` matches MA57 **exactly** on all 4:
```

## Git Status
```
921cb23 diag(issue-17): bin to compare cb=off vs cb=default per-matrix
6e95b82 docs(CLAUDE.md): note core.hooksPath workaround for pre-commit
e8dab31 style: cargo fmt on src/numeric/solve.rs
1f25a54 feat(bench): synthetic-matrix scaling bench vs MUMPS + MA57
a534897 chore(context): make corpus bench opt-in in assemble-context.sh
```

## Test Status
```
