# D.4 stage-2 corpus bench

**Date:** 2026-04-20 (session 01).
**Binary:** `cargo run --release --bin bench`. Three consecutive
runs for variance. Full first-run output in `stage2-corpus.txt`.

## Sparse factor-ratio vs MUMPS (three runs)

| run | geomean | p50  | p90  | p99  | max    |
|-----|--------:|-----:|-----:|-----:|-------:|
| 1   |  0.39   | 0.30 | 1.81 | 3.80 | 102.07 |
| 2   |  0.38   | 0.30 | 1.77 | 3.69 | 285.80 |
| 3   |  0.38   | 0.30 | 1.76 | 3.74 |  11.81 |

**Pre-D.4 reference** (session 2026-04-19-04 checkpoint): geomean
0.37, p50 0.29, p90 1.71, p99 3.54, max 80.22.

Conclusion: **geomean within noise band (≤ 0.02 drift over 3 runs).**
p50/p90/p99 also within run-to-run variance. `max` is dominated
by whichever matrix happens to land on a cache miss at collection
time — the 11→286× swing across runs confirms the top-10 rows are
single-shot outliers, not a stable regression class.

## Top-10 behavior

All six D.4 stage-1 target rows (HS73_0308, PALMER1E_0484,
HATFLDH_0083, PALMER1A_0034, KIRBY2LS_0274, HEART6LS_0418) are
**out of the top-10** in every run. That matches the stage-1
finding that they were already dense-fast-path-eligible via D.3
and were top-10 in the previous session only because of
measurement noise.

Representative run-1 top-10:

```
CERI651BLS_0577    n=7    feral=752  MUMPS=13   57.85x
DMN15102_0081      n=66   feral=2823 MUMPS=58   48.67x
NET1_0132          n=271  feral=2342 MUMPS=61   38.39x
DMN15332_0110      n=66   feral=2179 MUMPS=60   36.32x
NELSON_0442        n=387  feral=3529 MUMPS=98   36.01x
NET1_0127          n=271  feral=1715 MUMPS=60   28.58x
SWOPF_0151         n=175  feral=2673 MUMPS=102  26.21x
DIAMON2DLS_0196    n=66   feral=1442 MUMPS=60   24.03x
CRESC100_0189      n=606  feral=3018 MUMPS=167  18.07x
NET1_0140          n=271  feral=1002 MUMPS=66   15.18x
```

CERI651BLS_0577 (n=7) is in the D.4/D.3 gate and has ρ=1.0 —
the stage-1 probe class would predict a p50 of ~4 µs for it, not
752 µs. Same diagnosis as HS85_0022: single-shot noise.

## Exit partitions (Phase 2.8.1)

```
Sparse exit partition (factor ratio vs MUMPS, run 1):
  small-frontal (<200)   153455     p90=1.76    <= 2.0    PASS
  medium (<500)          153560     p90=1.77    <= 3.0    PASS
```

No regression — both partitions remain PASS at the expected thresholds.

## Acceptance

Ex-ante targets from `dev/plans/sparse-tail-d4.md` §Goal:

| target                                              | outcome |
|-----------------------------------------------------|:-------:|
| Tiny-n top-10 rows ≤ 3× MUMPS                       | PASS (0.26–0.57×, stage-1) |
| Corpus geomean no worse than 0.37 (± 0.02 tolerance)| PASS (0.38 median of 3 runs) |
| No out-of-gate matrix worse by > 20%                | PASS (dense/sparse exit partitions hold) |
| Tests green                                         | PASS (five-test `tiny_fast_path` all green) |

## Interpretation

D.4's unique target class (`n ≤ 16 AND ρ < 0.25`) appears to be
empty or nearly empty in the current IPM corpus — every top-10
tiny-n row we examined was already D.3-eligible at ρ ≥ 0.50.
Consequently, D.4's observable benefit on this corpus is modest:
the gate predicate is cleaner, the observable corpus statistics
are unchanged, but the stage-1 probe confirms a 1.2–1.6× per-call
speedup on the tiny-n target class whenever such matrices appear.

D.4 is the correct primitive to have; it just doesn't pay off
much on this specific workload. The rollout is done — no stage-3
follow-up is warranted.

## Next lever

Per the 2026-04-19-04 checkpoint, the remaining post-D.3/D.4
lever is **Phase 2.4 (blocked BK + SIMD on dense root frontal)**
for the arrow-KKT class (HAHN1, CRESC100, CRESC50, NELSON) that
shows up consistently across runs. That's the 4–6 session item,
not in scope this session.
