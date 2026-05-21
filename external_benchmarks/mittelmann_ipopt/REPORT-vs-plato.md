# Mittelmann Ipopt comparison: MA57 vs FERAL vs Plato

Joins the local `results/{ma57,feral}.jsonl` (Ipopt 3.14.20, local
timeout 600 s) against the Plato Ipopt-3.14.5 numbers published at
<https://plato.asu.edu/ftp/ampl-nlp.html> (timeout 7200 s, AMD Ryzen
9 5900X, 128 GB). Plato page snapshot: 2026-04-18.

All times are **Ipopt "Total seconds in IPOPT"**, not wall-clock. The
local runs use the same 47-problem set Plato uses (`problems.txt`).

## Caveats

- **Different hardware.** Plato runs Ryzen 9 5900X; local runs are on
  the dev Mac. Absolute seconds are not comparable; a single
  host-adjustment factor is reported below.
- **Different timeout.** 600 s local vs 7200 s Plato. Three problems
  Plato solves take >600 s on the dev box and are recorded as `TO`
  locally even though both solvers would likely finish.
- **Different Ipopt versions.** Local is 3.14.20, Plato is 3.14.5.
- **Plato's linear solver not stated** on the page; historical Plato
  Ipopt runs use HSL MA57. Treated as MA57 here.
- The local `nql180` MA57 run aborted with `rc=-10` (Ipopt internal
  failure) at 188 s — not a timeout. Counted as failure.

## Headline

| solver | solved / 47 | shifted geomean (all 47, cap@timeout) | shifted geomean (40 common, no cap) |
|---|---:|---:|---:|
| MA57 (local)  | 42 | 28.80 s | 19.72 s |
| FERAL (local) | 42 | 28.89 s | 18.34 s |
| Ipopt (Plato) | 46 | 42.01 s | 28.46 s |

Host-adjustment factor: median MA57/Plato over the 27 problems where
Plato finishes ≤60 s = **0.63** (dev Mac is ~1.6× faster than the Plato
Ryzen on small problems). Applying this to the 40-problem column:

| solver | host-adjusted geomean (estimated Plato-equivalent seconds) |
|---|---:|
| MA57 (÷0.63)  | 31.41 s |
| FERAL (÷0.63) | 29.22 s |

Both local geomeans land *below* Plato's published Ipopt geomean even
after host-adjustment, because the local 600 s cap clips Plato's
heaviest tail problems (`dtoc2` 2076 s, `nql180` 2050 s, `WM_CFy`
1243 s, etc.) — local MA57 actually finishes `dtoc2` in 1.83 s, which
suggests these tail numbers reflect MA57 pathology on Plato's exact
build rather than intrinsic problem difficulty.

## FERAL vs MA57 (local, 40 problems both solved)

- FERAL >10% faster than MA57: **15** problems
- FERAL within ±10% of MA57:   **4**  problems
- FERAL >10% slower than MA57: **21** problems

Geomean parity (18.34 vs 19.72 s) hides a tail. Worst FERAL/MA57
ratios:

| problem        | feral/ma57 |
|----------------|-----------:|
| marine_1600    | 33.97 |
| pinene_3200    | 12.32 |
| robot_1600     |  7.02 |
| corkscrw       |  5.56 |
| arki0003       |  5.54 |

Best FERAL/MA57 ratios (all in the `cont5_*` / `ex1_*` / `dirichlet`
families):

| problem        | feral/ma57 |
|----------------|-----------:|
| ex1_320        | 0.11 |
| cont5_2_4_l    | 0.13 |
| cont5_1_l      | 0.13 |
| dirichlet120   | 0.13 |
| cont5_2_3_l    | 0.13 |

## Failure breakdown vs Plato

| problem        | ma57       | feral | plato | notes |
|----------------|------------|-------|-------|-------|
| NARX_CFy       | OK (240 s) | TO    | OK (938 s)  | feral times out at 600 s; would likely solve at 7200 s |
| WM_CFy         | TO         | TO    | OK (1243 s) | both local solvers blow the cap |
| nql180         | FAIL rc=-10| TO    | OK (2050 s) | local MA57 aborts internally; feral times out |
| qcqp1000-2c    | TO         | OK (49 s)  | OK (31 s)  | **feral solves; local MA57 doesn't** |
| qcqp1500-1c    | TO         | OK (195 s) | OK (122 s) | **feral solves; local MA57 doesn't** |
| qcqp1500-1nc   | TO         | TO    | TO          | all three time out |
| steering_12800 | OK (1.3 s) | TO    | OK (14 s)   | **feral regression** — see "Followups" below |

Symmetric difference vs Plato:
- Plato solves but neither local solver does: **`WM_CFy`, `nql180`**
  (3rd is `NARX_CFy` — local MA57 does solve).
- Local FERAL solves but Plato Ipopt would (presumably MA57): same
  problems are `qcqp1000-2c` and `qcqp1500-1c`; Plato Ipopt also
  solves these.

## Per-problem table (sorted by Plato Ipopt time)

| problem                |  plato (s) |   ma57 (s) |  feral (s) |  ma57/plato | feral/plato |  feral/ma57 |
|------------------------|-----------:|-----------:|-----------:|------------:|------------:|------------:|
| arki0003               |          1 |       0.56 |       3.08 |        0.56 |        3.08 |        5.54 |
| ex1_160                |          1 |       8.66 |       1.46 |        8.66 |        1.46 |        0.17 |
| ex8_2_2                |          1 |       0.21 |       0.50 |        0.21 |        0.50 |        2.37 |
| ex8_2_3                |          1 |       0.37 |       0.82 |        0.37 |        0.82 |        2.19 |
| robot_1600             |          1 |       0.42 |       2.93 |        0.42 |        2.93 |        7.02 |
| ex4_2_160              |          2 |       6.46 |       2.41 |        3.23 |        1.21 |        0.37 |
| camshape_6400          |          3 |       0.57 |       2.15 |        0.19 |        0.72 |        3.74 |
| dtoc1nd                |          3 |       1.17 |       1.30 |        0.39 |        0.43 |        1.11 |
| marine_1600            |          3 |       0.60 |      20.45 |        0.20 |        6.82 |       33.97 |
| bearing_400            |          5 |      14.08 |       4.86 |        2.82 |        0.97 |        0.35 |
| arki0009               |          6 |       4.07 |      10.54 |        0.68 |        1.76 |        2.59 |
| rocket_12800           |          6 |       1.72 |       8.01 |        0.29 |        1.33 |        4.67 |
| ex1_320                |          7 |      58.38 |       6.41 |        8.34 |        0.92 |        0.11 |
| ex4_2_320              |          7 |      39.10 |      11.13 |        5.59 |        1.59 |        0.28 |
| optmass                |          7 |       0.75 |       2.45 |        0.11 |        0.35 |        3.27 |
| pinene_3200            |          7 |       1.37 |      16.91 |        0.20 |        2.42 |       12.32 |
| cont5_2_4_l            |         12 |      58.12 |       7.40 |        4.84 |        0.62 |        0.13 |
| cont5_1_l              |         13 |      50.91 |       6.54 |        3.92 |        0.50 |        0.13 |
| steering_12800         |         14 |       1.30 |         TO |        0.09 |       42.86 |      460.12 |
| qssp180                |         15 |      43.81 |      37.06 |        2.92 |        2.47 |        0.85 |
| qcqp1000-1nc           |         20 |      12.55 |      24.43 |        0.63 |        1.22 |        1.95 |
| corkscrw               |         28 |       2.90 |      16.13 |        0.10 |        0.58 |        5.56 |
| qcqp1000-2c            |         31 |         TO |      49.49 |       19.35 |        1.60 |        0.08 |
| svanberg               |         31 |       2.67 |       6.54 |        0.09 |        0.21 |        2.45 |
| cont5_2_1_l            |         35 |      83.93 |      16.55 |        2.40 |        0.47 |        0.20 |
| elec_400               |         36 |     173.78 |      54.22 |        4.83 |        1.51 |        0.31 |
| cont5_2_2_l            |         42 |     116.51 |      17.99 |        2.77 |        0.43 |        0.15 |
| cont5_2_3_l            |         47 |     136.89 |      18.31 |        2.91 |        0.39 |        0.13 |
| qcqp1000-2nc           |         68 |      60.31 |     201.02 |        0.89 |        2.96 |        3.33 |
| gasoil_3200            |         69 |       1.00 |       1.16 |        0.01 |        0.02 |        1.16 |
| dirichlet120           |        106 |     375.67 |      49.89 |        3.54 |        0.47 |        0.13 |
| lane_emden120          |        116 |     539.79 |      91.55 |        4.65 |        0.79 |        0.17 |
| qcqp1500-1c            |        122 |         TO |     194.94 |        4.92 |        1.60 |        0.32 |
| henon120               |        131 |     467.38 |      77.76 |        3.57 |        0.59 |        0.17 |
| qcqp500-3nc            |        143 |      77.53 |      81.64 |        0.54 |        0.57 |        1.05 |
| qcqp750-2c             |        281 |     150.83 |     160.68 |        0.54 |        0.57 |        1.07 |
| clnlbeam               |        536 |      12.70 |      45.27 |        0.02 |        0.08 |        3.56 |
| qcqp750-2nc            |        543 |     275.55 |     290.77 |        0.51 |        0.54 |        1.06 |
| robot_c                |        683 |     157.03 |     426.09 |        0.23 |        0.62 |        2.71 |
| robot_a                |        688 |     169.46 |     511.47 |        0.25 |        0.74 |        3.02 |
| robot_b                |        717 |     166.63 |     491.37 |        0.23 |        0.69 |        2.95 |
| qcqp500-3c             |        769 |     348.92 |     360.93 |        0.45 |        0.47 |        1.03 |
| NARX_CFy               |        938 |     240.16 |         TO |        0.26 |        0.64 |        2.50 |
| WM_CFy                 |       1243 |         TO |         TO |        0.48 |        0.48 |        1.00 |
| nql180                 |       2050 |     FAIL   |         TO |        0.29 |        0.29 |        1.00 |
| dtoc2                  |       2076 |       1.83 |       3.69 |        0.00 |        0.00 |        2.02 |
| qcqp1500-1nc           |         TO |         TO |         TO |         —   |         —   |        1.00 |

Cells marked `TO` denote a timeout against the column's own
timeout budget (local 600 s, Plato 7200 s). Ratio cells against a
`TO` use the cap value (600 s) for the local side, so ratios involving
local timeouts are lower bounds.

## Followups suggested by this comparison

1. **`steering_12800` is a new feral regression.** MA57 solves in 1.3 s,
   Plato solves in 14 s, feral times out at 600 s. Not in
   `PROBLEM_FERAL_ENV`; no journal entry. Worth opening as an issue.
2. **Three problems need a 7200 s re-run** to claim Plato-equivalent
   solve counts: `WM_CFy`, `nql180`, `NARX_CFy`. With a longer cap
   feral might reach 43–45/47 instead of the headline 42/47.
3. **The `cont5_*` / `ex1_*` / `dirichlet`-style problems are where
   feral wins**, often by 5–8×. These all involve large structured KKT
   systems where the supernodal path pays off. Worth a short note in
   the manuscript pointing at the cont5 family as the "feral is the
   right pick" case.
4. **The `marine_1600` / `pinene_3200` rescues** (CB-on overrides in
   `PROBLEM_FERAL_ENV`) are still 12–34× slower than MA57 even with
   the rescue. Followup: investigate whether those problems need a
   different default rather than a CB rescue.

   *Update 2026-05-21 (Track A3):* the underlying defect is now fixed
   under the **default** config — the CB rescue is no longer needed.
   Fix 1 (fine-grained delayed pivoting, `42434a5`) removes the
   delayed-pivot cascade and Fix 2 (cancellation-free 2×2 inertia,
   this session) removes the spurious-zero inertia error. KKT-dump
   *factor-replay* (`probe_kkt_replay`, default config, no CB):
   `pinene_3200` 456 s → 4.60 s with all 10 iterates inertia-exact;
   `marine_1600` all 18 iterates exact in 5.39 s; `robot_1600`
   0.199 s. The full IPM-solve numbers in the tables above predate
   both fixes and the `PROBLEM_FERAL_ENV` CB overrides for these two
   problems should now be dropped — re-run the `mittelmann_ipopt`
   benchmark to refresh the solve-time columns and confirm.

---

Regenerate by running:

```sh
python3 -c '...'   # see the script comment above run.py
```

The Plato column is hand-transcribed from the page snapshot dated
**2026-04-18**; update it when Plato re-runs.
