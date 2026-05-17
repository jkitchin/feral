# FERAL Context (auto-generated)

Generated: 2026-05-17T16:17:25Z

## Latest Session
File: dev/sessions/2026-05-17-01.md
```
# Session 2026-05-17-01

## Goal

Localize and remediate the per-iter factor cost gap vs MA57 on the
ipopt-feral Mittelmann sweep. Going in, the suspect was MC64 cost
(98% of warm wall on `probe_rocket_slow.rs` against the pounce
corpus dumps).

## Accomplished

### Investigation pivot: MC64 is rocket-specific, cascade is general

1. **MC64 live trace.** Added `MC64_RECOMPUTE_COUNT` +
   `FERAL_MC64_TRACE=1` to `src/scaling/mc64.rs`. Live ipopt-feral
   run on rocket_12800: 30 MC64 recomputes (one per warm factor —
   confirms #38 `db20166` invalidation), MC64 = **55%** of total
   Ipopt wall (not 98% as the dumped corpus suggested). Per-call
   wall ranges 14–2482 ms.

2. **Per-supernode profile on robot_1600** (`probe_robot_profile.rs`).
   Found MC64 is only **2.6%** of robot_1600 warm wall (350 ms of
   13.7 s). The 32× MA57 gap there lives elsewhere.

3. **Factor trace** (`FERAL_FACTOR_TRACE=1` in `src/capi.rs`).
   Per-factor wall + `sum_delayed` + `max_delayed` exposed
   smoking-gun: robot_1600 late-IPM factors have
   `sum_delayed = 30k–60k` on n=24000 — classic delayed-pivot
   cascade (the same mechanism issue #38 fixed for pinene_3200).

4. **Auto-CB was dead code.** `Solver::with_auto_cascade_break(β)`
   (the warm cascade-break auto-arm from #38) was never wired into
   the capi, so ipopt-feral never benefited. Wired it as the
   default with `FERAL_AUTO_CB_BETA` env (default 0.05).

Spot-check on 10 Mittelmann problems (`benchmarks/mittelmann_ipopt`):

| problem        | CB=off    | auto-CB   | Δ        |
|----------------|-----------|-----------|----------|
| robot_1600     | 13.81 s   |  3.58 s   |  -74 %   |
| marine_1600    | 470.87 s  | 58.13 s   |  -88 %   |
| clnlbeam       | 361.26 s  | 47.73 s   |  -87 %   |
| corkscrw       | 53.89 s   | 15.43 s   |  -71 %   |
| camshape_6400  |  6.67 s   |  2.09 s   |  -69 %   |
| dtoc2          | timeout   | 78.0 s    | rescued  |
| bearing_400    |  6.21 s   |  4.86 s   |  -22 %   |
| rocket_12800   |  8.73 s   | 10.55 s   |  +21 %   |
| arki0003       |  3.10 s   |  3.17 s   |   ~0 %   |
| pinene_3200    | timeout   | timeout   | needs CB=on |

```

## Git Status
```
ad48ab2 chore(dev-tools): add Makefile wrapper + pounce KKT replay/time probes
f1da854 feat(capi): auto-arm cascade-break by default to rescue delayed-pivot cascades
a28cfec fix(scaling): require dense arrow head, not just diag-only mass, for MC64 routing (#68)
0d874f5 feat(numeric): MA57-style static-pivot perturbation knob (#38)
1c36f2d feat(dense): closed-form 2x2 eigenvalue inertia classifier (#38)
```

## Test Status
```
