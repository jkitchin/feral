# FERAL Context (auto-generated)

Generated: 2026-05-16T16:37:14Z

## Latest Session
File: dev/sessions/2026-05-16-02.md
```
# Session 2026-05-16-02

## Goal
Continue the issue-10 "remaining lever" investigation started end of
session 01 (axpy SIMD kernel microbench), and clear independent
single-shot issues from the open queue.

When the user reported every push breaking on GH Actions, scope
expanded to a root-cause CI-hooks fix, then to merging the parallel-
agent deliveries on #25 and #24, and to closing out #33 §3
(`Solver::with_ordering`) since it was the smallest open lever
adjacent to #10's blocker.

## Accomplished

### "Remaining lever" axpy microbench — negative (commit `05722a3` co-bundled)
Built `src/bin/bench_axpy_small.rs` comparing `pulp` /
`scalar` / `unroll4` at lengths [3..128] with 50M iters/measure.
Result: pulp SIMD dispatch ties with plain scalar within 1ns/call
quantization at all small lengths; manual unroll4 is slower. The
compiler auto-vectorizes the scalar form as well as the explicit
SIMD dispatch. *Rules out kernel-call overhead as the bottleneck for
clnlbeam.*  Combined with the prior negative #33 SLB A/B and the
negative #10 MAXFROMM Phase 2 A/B, all three architectural levers
tried against the 1D-banded Mittelmann panel come up within noise.

### CI hooks self-heal (commit `05722a3`)
Root cause of "every push breaks on GH Actions": `core.hooksPath`
was set to `/Users/jkitchin/Dropbox/projects/feral/.git/hooks`
(stale from a prior clone location), pointing at a directory that
does not exist on this machine — so git silently bypassed every
local pre-commit hook. CI caught the fmt drift on every push.

Fix: added a self-healing guard at the top of
`dev/assemble-context.sh` that detects a `core.hooksPath` pointing
nowhere, auto-unsets it, and reinstalls pre-commit. Verified the
guard fires on a synthetic broken state. CLAUDE.md already
documented this exact failure mode as a doc note; the guard
promotes the doc into automation so future sessions cannot
inherit the broken state silently.

### Issue #25 — cascade-break defaults research note (commit `7f096c1`)
Worktree-isolated agent wrote `dev/research/cascade-break.md` (392
lines) deriving (or not deriving) the defaults `ratio = 0.5` and
`eps = 1e-10` from the published literature.

**Conclusion: empirical, not derivable.** `ratio = 0.5` was
calibrated on `pinene_3200_0009` in #8 and cross-validated against
the bimodal `n_delayed_in / expanded_ncol` distribution measured in
#15. Wächter & Biegler 2006 uses `κ⁻_w = 1/3` not `1/2`;
```

## Git Status
```
fa62918 test(scaling): relax MSS1_0009 reason check to either fallback variant
e789ec3 Merge branch 'worktree-agent-ab2727cb5b91921b5'
2efa315 feat(solver): Solver::with_ordering builder (#33 §3)
02e699a feat(scaling): surface MC64 -> InfNorm silent fallback (#24)
7f096c1 docs(cascade-break): document derivation status of ratio=0.5, eps=1e-10 (#25)
```

## Test Status
```
