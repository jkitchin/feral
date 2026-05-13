# Ipopt 3.14.20 × {feral, MUMPS, MA57} on the Scalable NLP suite

Total problems: **35** from `ref/Ipopt/examples/ScalableProblems` (LuksanVlcek 1–7 in equality + inequality flavors, Mittelmann boundary/distributed/3D control suites, and MittelmannParaCntrl).  

Each problem is fed to **three different Ipopt 3.14.20 binaries**, each linked to a single sparse direct linear solver and otherwise identical:

| Binary | Linear solver | Build dir | Version |
|---|---|---|---|
| `build-mumps`  | MUMPS 5.8.2 (sequential) | `ref/Ipopt/build-mumps` | `--with-mumps` against vendored `ref/mumps` |
| `build-ma57`   | HSL MA57 (CoinHSL 2023.11.17, sequential) | `ref/Ipopt/build-ma57` | `--with-hsl` against `libcoinhsl.a` |
| `build-feral`  | feral 0.2.0 (Rust, multifrontal parallel) | `ref/Ipopt/build-feral` | `feral-ipopt-shim` C-ABI patch onto Ipopt |

All three binaries use Ipopt's stock defaults — only `linear_solver` is overridden, written to an `ipopt.opt` in the working directory. Each run is fresh (separate cwd) so no state leaks between solvers.

**Host**: macOS-26.3.1-arm64-arm-64bit, arm64.

## 1. Success / failure summary

| Solver | Optimal | Acceptable | Iter-limit | Infeasible | Restoration-failed | Timeout | Other |
|---|---:|---:|---:|---:|---:|---:|---:|
| mumps | 35 | 0 | 0 | 0 | 0 | 0 | 0 |
| ma57 | 34 | 0 | 0 | 0 | 0 | 1 | 0 |
| feral | 34 | 0 | 0 | 0 | 0 | 1 | 0 |

## 2. IPM iteration counts (where they differ)

Same NLP → same KKT system at each iterate → expect identical iteration counts when each solver delivers a comparable Newton step. Discrepancies indicate one of the linear solvers is giving subtly different residuals (scaling, inertia, refinement).

**29/35** problems have identical iteration count across solvers.

| Problem | N | MUMPS | MA57 | feral |
|---|---:|---:|---:|---:|
| MBndryCntrl5 | 128 | 27 | 18 | 19 |
| MBndryCntrl6 | 128 | 129 | 20 | 20 |
| MBndryCntrl7 | 128 | 24 | 19 | 19 |
| MBndryCntrl8 | 128 | 21 | 24 | 21 |
| MDistCntrl5 | 128 | 53 | 57 | 46 |
| MBndryCntrl_3D | 24 | 19 | 17 | 17 |

## 3. Total Ipopt time (geomean, only over problems where ALL solvers reached *Optimal Solution Found*)

(34/35 problems converge on all three solvers — only these contribute to the geometric mean.)

| Solver | geomean Ipopt seconds |
|---|---:|
| mumps | 138.6ms |
| ma57 | 161.9ms |
| feral | 158.1ms |

Speedup feral vs MUMPS: **0.88×** (geomean over triple-optimal subset).

Speedup feral vs MA57: **1.02×** (geomean over triple-optimal subset).

## 4. Per-problem detail

Status legend: ✓ = Optimal Solution Found; ~ = Solved to Acceptable Level; ✗ = other (see exit_status column).

| Problem | N | n_vars | MUMPS iter / sec / status | MA57 iter / sec / status | feral iter / sec / status | Objective |
|---|---:|---:|---|---|---|---|
| LukVlE1 | 1000 | 1000 | 6 / 4.0ms / ✓ | 6 / 3.0ms / ✓ | 6 / 6.0ms / ✓ | 6.2325e+00 |
| LukVlE2 | 1000 | 1002 | 20 / 22.0ms / ✓ | 20 / 29.0ms / ✓ | 20 / 43.0ms / ✓ | 2.8122e+04 |
| LukVlE3 | 1002 | 1004 | 10 / 3.0ms / ✓ | 10 / 2.0ms / ✓ | 10 / 5.0ms / ✓ | 6.5121e+01 |
| LukVlE4 | 1002 | 1004 | 14 / 7.0ms / ✓ | 14 / 6.0ms / ✓ | 14 / 14.0ms / ✓ | 4.8457e+03 |
| LukVlE5 | 1000 | 1000 | 18 / 12.0ms / ✓ | 18 / 16.0ms / ✓ | 18 / 19.0ms / ✓ | 2.6393e+00 |
| LukVlE6 | 1000 | 1001 | 14 / 8.0ms / ✓ | 14 / 8.0ms / ✓ | 14 / 15.0ms / ✓ | 6.2752e+04 |
| LukVlE7 | 1000 | 1002 | 11 / 8.0ms / ✓ | 11 / 2.0ms / ✓ | 11 / 6.0ms / ✓ | -1.3108e+03 |
| LukVlI1 | 1000 | 1000 | 26 / 16.0ms / ✓ | 26 / 12.0ms / ✓ | 26 / 52.0ms / ✓ | 5.7153e+00 |
| LukVlI2 | 1000 | 1002 | 18 / 16.0ms / ✓ | 18 / 11.0ms / ✓ | 18 / 54.0ms / ✓ | 2.6471e+04 |
| LukVlI3 | 1002 | 1004 | 13 / 4.0ms / ✓ | 13 / 3.0ms / ✓ | 13 / 6.0ms / ✓ | 3.0606e+01 |
| LukVlI4 | 1002 | 1004 | 42 / 26.0ms / ✓ | 42 / 20.0ms / ✓ | 42 / 82.0ms / ✓ | 1.1918e+03 |
| LukVlI5 | 1000 | 1000 | 37 / 29.0ms / ✓ | 37 / 23.0ms / ✓ | 37 / 101.0ms / ✓ | 2.5523e+00 |
| LukVlI6 | 1000 | 1001 | 28 / 20.0ms / ✓ | 28 / 14.0ms / ✓ | 28 / 65.0ms / ✓ | 1.2534e+04 |
| LukVlI7 | 1000 | 1002 | 13 / 10.0ms / ✓ | 13 / 2.0ms / ✓ | 13 / 7.0ms / ✓ | -1.3495e+03 |
| MBndryCntrl1 | 128 | 16896 | 15 / 369.0ms / ✓ | 15 / 2.02s / ✓ | 15 / 454.0ms / ✓ | 1.9839e-01 |
| MBndryCntrl2 | 128 | 16896 | 16 / 392.0ms / ✓ | 16 / 2.25s / ✓ | 16 / 481.0ms / ✓ | 9.8329e-02 |
| MBndryCntrl3 | 128 | 16896 | 26 / 606.0ms / ✓ | 26 / 3.34s / ✓ | 26 / 731.0ms / ✓ | 3.2410e-01 |
| MBndryCntrl4 | 128 | 16896 | 26 / 603.0ms / ✓ | 26 / 3.29s / ✓ | 26 / 728.0ms / ✓ | 2.5207e-01 |
| MBndryCntrl5 | 128 | 17408 | 27 / 4.18s / ✓ | 18 / 2.38s / ✓ | 19 / 542.0ms / ✓ | 5.5320e-01 |
| MBndryCntrl6 | 128 | 17408 | 129 / 22.03s / ✓ | 20 / 2.68s / ✓ | 20 / 572.0ms / ✓ | 1.5314e-02 |
| MBndryCntrl7 | 128 | 17408 | 24 / 2.43s / ✓ | 19 / 2.52s / ✓ | 19 / 556.0ms / ✓ | 2.6557e-01 |
| MBndryCntrl8 | 128 | 17408 | 21 / 2.08s / ✓ | 24 / 3.46s / ✓ | 21 / 607.0ms / ✓ | 1.6654e-01 |
| MDistCntrl1 | 128 | 32768 | 16 / 541.0ms / ✓ | 16 / 764.0ms / ✓ | 16 / 810.0ms / ✓ | 6.3184e-02 |
| MDistCntrl2 | 128 | 32768 | 17 / 572.0ms / ✓ | 17 / 780.0ms / ✓ | 17 / 834.0ms / ✓ | 5.7492e-02 |
| MDistCntrl3 | 128 | 32768 | 14 / 478.0ms / ✓ | 14 / 612.0ms / ✓ | 14 / 721.0ms / ✓ | 1.1028e-01 |
| MDistCntrl4 | 128 | 33280 | 13 / 449.0ms / ✓ | 13 / 587.0ms / ✓ | 13 / 704.0ms / ✓ | 7.8232e-02 |
| MDistCntrl5 | 128 | 33280 | 53 / 1.67s / ✓ | 57 / 1.86s / ✓ | 46 / 2.06s / ✓ | 5.2824e-02 |
| MDistCntrl6 | 128 | 33280 | 19 / 949.0ms / ✓ | 19 / 574.0ms / ✓ | 19 / 970.0ms / ✓ | -4.2943e+00 |
| MBndryCntrl_3D | 24 | 17280 | 19 / 22.86s / ✓ | 17 / 13.08s / ✓ | 17 / 7.74s / ✓ | 1.2922e-01 |
| MBndryCntrl_3D_27 | 24 | 17280 | 12 / 5.77s / ✓ | 12 / 3.97s / ✓ | 12 / 18.19s / ✓ | 2.4821e-01 |
| MBndryCntrl_3Dsin | 24 | 17576 | 163 / 1.1m / ✓ | - / - / timeout | - / - / timeout | 7.0268e-01 |
| MPara5_1 | 64 | 4224 | 13 / 64.0ms / ✓ | 13 / 333.0ms / ✓ | 13 / 93.0ms / ✓ | 2.7252e+00 |
| MPara5_2_1 | 64 | 4224 | 15 / 77.0ms / ✓ | 15 / 316.0ms / ✓ | 15 / 105.0ms / ✓ | 6.5468e-04 |
| MPara5_2_2 | 64 | 4224 | 19 / 100.0ms / ✓ | 19 / 467.0ms / ✓ | 19 / 128.0ms / ✓ | 5.1435e-04 |
| MPara5_2_3 | 64 | 4224 | 34 / 179.0ms / ✓ | 34 / 590.0ms / ✓ | 34 / 202.0ms / ✓ | 5.1444e-04 |

## 5. Objective-value cross-check

For problems where all three solvers reach Optimal, the final objective should agree to several digits. Large spreads indicate one solver found a different local minimum or terminated early.

Top 10 by relative objective spread:

| Problem | N | rel spread | MUMPS obj | MA57 obj | feral obj |
|---|---:|---:|---|---|---|
| MBndryCntrl7 | 128 | 2.25e-06 | 2.6557e-01 | 2.6556e-01 | 2.6556e-01 |
| MBndryCntrl5 | 128 | 1.55e-06 | 5.5320e-01 | 5.5320e-01 | 5.5320e-01 |
| MDistCntrl5 | 128 | 5.25e-07 | 5.2824e-02 | 5.2824e-02 | 5.2824e-02 |
| MBndryCntrl_3D | 24 | 7.88e-08 | 1.2922e-01 | 1.2922e-01 | 1.2922e-01 |
| MBndryCntrl6 | 128 | 7.34e-08 | 1.5314e-02 | 1.5314e-02 | 1.5314e-02 |
| MBndryCntrl8 | 128 | 3.33e-08 | 1.6654e-01 | 1.6654e-01 | 1.6654e-01 |
| MPara5_2_1 | 64 | 3.80e-11 | 6.5468e-04 | 6.5468e-04 | 6.5468e-04 |
| MPara5_2_2 | 64 | 5.08e-12 | 5.1435e-04 | 5.1435e-04 | 5.1435e-04 |
| MBndryCntrl2 | 128 | 7.28e-14 | 9.8329e-02 | 9.8329e-02 | 9.8329e-02 |
| MBndryCntrl3 | 128 | 3.70e-14 | 3.2410e-01 | 3.2410e-01 | 3.2410e-01 |
