# FERAL Context (auto-generated)

Generated: 2026-05-15T23:58:31Z

## Latest Session
File: dev/sessions/2026-05-15-06.md
```
# Session 2026-05-15-06

## Bench vs. prior session

Synthetic-only bench (corpus matrices live elsewhere). The hot path
touched this session is the `should_parallelize_assembly` flop gate
const; the synthetic bench all sits well below 1e7 flops so the gate
decision is unchanged on these problems (sequential everywhere).
Numbers within run-to-run noise of session 05.

```
spd_10             10           56            0     (10, 0, 0)
spd_50             50           27            3     (50, 0, 0)
spd_100           100           89            5    (100, 0, 0)
spd_200           200          423           20    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           25            1    (30, 10, 0)
kkt_50_15          65           55            2    (50, 15, 0)
kkt_100_30        130          223            7   (100, 30, 0)
```

Corpus partition (Phase 2.8.1 gate ratios vs MUMPS, from this session's
end-of-session bench):

```
Dense:
small-frontal (<200)     147982     p90 1.32   PASS
medium (<500)            152145     p90 1.70   PASS

Sparse:
small-frontal (<200)     153455     p90 1.54   PASS
medium (<500)            153560     p90 1.54   PASS
```

No regression vs session 05.

## Goal

Issue #19 follow-up task 1 from session 05: cross-hardware probe
data on issue #19, then a default-policy decision on `PAR_MIN_FLOPS`.

## Accomplished

### 1. Cross-hardware Poisson-KKT calibration on feral-home

Ran `cargo run --release --bin calibrate_par_min_flops -- --reps 10`
on feral-home (Apple M4 Pro, 14 rayon threads). Numbers match
session-05's M4 Pro within ~10% across every row (worst case K=80:
par/seq 0.55 vs 0.61). Break-even 6×10⁶, ≥1.2× win at 1.2×10⁷,
≥1.35× win at 2.3×10⁷. Cross-hardware verification within the M4
```

## Git Status
```
b12e03c perf(factor): lower PAR_MIN_FLOPS from 1e8 to 1e7 (#19 closeout)
25926cc feat(bench): probe_issue_19 binary
30a30fc chore(session): 2026-05-15-05 -- PAR_MIN_FLOPS calibration
db7b761 feat(bench): calibrate_par_min_flops probe + research note (#19 follow-up)
0d0412e fix(test): make_supernodes leaves row_indices empty (CI OOM on 8a2a8e1)
```

## Test Status
```
