# FERAL vs MUMPS vs HSL MA57 vs HSL MA97 — KKT solver comparison

Total matrices: **66**, drawn from the FERAL CUTEst
KKT corpus and Mittelmann large-scale KKT corpus.
Sampling spans 5 size buckets and 63 distinct CUTEst/Mittelmann
families. RHS is synthetic: `b = A · x_true` with
`x_true[i] = 1 + i/n`. Same RHS is fed to all four solvers.

**Solvers** — each is configured with its recommended
high-accuracy defaults: shape-aware scaling on, iterative
refinement on. The goal is an apples-to-apples comparison
of what a real consumer would get from each library when
accuracy matters; the bare back-substitution defaults are
*not* what these tables show.

| Solver | Version | Driver | Configuration |
|---|---|---|---|
| feral | feral-0.2.0 | `factorize_multifrontal_parallel` + `solve_sparse_refined` | `ScalingStrategy::Auto` (MC64-symmetric or inf-norm by shape, threshold tuned 2026-04-19); BK pivot threshold `1e-8` (MA27 default); refinement loop runs up to 10 steps with stagnation-based exit. rayon-parallel multifrontal; falls through to sequential below 32 supernodes. |
| MUMPS | mumps-5.8.2 | `dmumps SYM=2` | `ICNTL(10) = 2` (two iterative-refinement steps; MUMPS default is `0` = no refinement); `ICNTL(11) = 1` (full error analysis); `ICNTL(24) = 1` (null pivot detection). Sequential build, no MC64 scaling by default. |
| MA57  | ma57-2023.11.17 | `ma57bd` + `ma57dd` | `ICNTL(6) = 5` (auto AMD / METIS); `ICNTL(7) = 1` (numerical BK-style pivoting); `ICNTL(15) = 1` (automatic scaling on); `CNTL(1) = 1e-8` pivot threshold (matches Ipopt's MA57 default). Solve uses `MA57DD JOB=1` with `ICNTL(9) = 1` — single solve with no iterative refinement, since with refinement on (`ICNTL(9) = 10`) MA57 produced non-deterministic NaN solutions on several borderline-conditioned KKT systems. Residual is computed externally by the driver. CoinHSL 2023.11.17, sequential. |
| MA97  | ma97-2.8.1 | `ma97_factor matrix_type=4` + Richardson loop around `ma97_solve_d` | `scaling = 1` (MC64 enabled, the recommended HSL default); `ordering = 5` (auto AMD/METIS); `action = 1` (continue past singular pivots). MA97 has no built-in residual-based refinement entry point, so the driver wraps `ma97_solve_d` in a 4-step Richardson loop (stagnation exit) to match what MUMPS+ICNTL(10) and feral+`solve_sparse_refined` deliver. CoinHSL 2023.11.17, OpenMP. |

> Change any of these settings and the timing/accuracy
> columns will move. The bench captures each library's
> *best-effort* mode, not its raw defaults.

**Host**: macOS-26.3.1-arm64-arm-64bit, arm64, Python 3.12.11.

## Sample composition

| Bucket | Count |
|---|---:|
| tiny (n<100) | 20 |
| small (100-1k) | 20 |
| medium (1k-10k) | 8 |
| large (10k-100k) | 10 |
| xl (>=100k) | 8 |

## Status summary

| Solver | OK | Fail | Missing |
|---|---:|---:|---:|
| feral | 64 | 2 | 0 |
| mumps | 62 | 4 | 0 |
| ma57 | 66 | 0 | 0 |
| ma97 | 66 | 0 | 0 |

## Factor time by size bucket (geomean μs)

| Bucket | n range |  feral |  MUMPS |  MA57  |  MA97  | feral/MUMPS | feral/MA57 | feral/MA97 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tiny (n<100) | 4–66 | 4 | 36 | 5 | 13 | 0.11× | 0.76× | 0.30× |
| small (100-1k) | 102–888 | 472 | 228 | 111 | 200 | 2.07× | 4.23× | 2.36× |
| medium (1k-10k) | 1154–9685 | 7,307 | 9,744 | 5,487 | 18,937 | 0.75× | 1.33× | 0.39× |
| large (10k-100k) | 11214–92229 | 82,406 | 351,010 | 90,850 | 1,146,258 | 0.23× | 0.91× | 0.07× |
| xl (>=100k) | 103920–259681 | 96,957 | 624,244 | 142,803 | 1,097,360 | 0.16× | 0.68× | 0.09× |

> Ratios < 1.0 mean **feral is faster**. Geomean is over the
> matrices in the bucket where the named solver succeeded.

## Accuracy: ‖Ax − b‖₂ / ‖b‖₂ distribution

| Solver | min | median | p90 | max | # > 1e-8 |
|---|---:|---:|---:|---:|---:|
| feral | 1.5e-17 | 2.3e-16 | 1.2e-15 | 1.2e-14 | 0 |
| mumps | 8.1e-18 | 3.4e-16 | 5.2e-15 | 4.3e-09 | 0 |
| ma57 | 1.2e-54 | 1.4e-16 | 4.8e-16 | 1.8e-14 | 0 |
| ma97 | 3.9e-17 | 1.6e-16 | 7.5e-16 | 8.2e-15 | 0 |

## Behavior on ill-conditioned KKTs

MUMPS emits a componentwise condition-number estimate
(`RINFOG(10)` / COND1, computed under `ICNTL(11) = 1`).
Pulling the matrices with the highest COND1 and showing
what each solver does on the *same* system answers the
question "does feral hold up when the matrix is hard?".

Selection: top matrices in the sample by MUMPS-reported
COND1, with a floor of 1e8 (below that the system is well-
enough conditioned that all three solvers reach machine ε).

| Matrix | n | MUMPS COND1 | feral res / inertia | MUMPS res / inertia | MA57 res / inertia | MA97 res / inertia |
|---|---:|---:|---|---|---|---|
| cont5_2_1_l/cont5_2_1_l_0002 | 180,900 | 1.7e+14 | 6.9e-16 / 90600+90300+0 | 9.8e-16 / 90600+90300+0 | 1.2e-16 / 90600+90300+0 | 7.3e-16 / 90600+90300+0 |
| cont5_2_2_l/cont5_2_2_l_0002 | 180,900 | 1.7e+14 | 7.1e-16 / 90600+90300+0 | 9.8e-16 / 90600+90300+0 | 1.3e-16 / 90600+90300+0 | 9.5e-16 / 90600+90300+0 |
| cont5_2_3_l/cont5_2_3_l_0002 | 180,900 | 1.7e+14 | 7.1e-16 / 90600+90300+0 | 9.8e-16 / 90600+90300+0 | 1.3e-16 / 90600+90300+0 | 9.5e-16 / 90600+90300+0 |
| HEART6_pounce_diag/heart6_iter_b | 12 | 2.8e+13 | 1.4e-16 / 4+8+0 | 6.9e-16 / 4+8+0 | 7.7e-17 / 4+8+0 | 7.7e-16 / 4+8+0 |
| HEART6_pounce_diag/heart6_iter_a | 12 | 9.5e+11 | 2.3e-16 / 6+6+0 | 4.0e-15 / 6+6+0 | 1.7e-16 / 6+6+0 | 6.9e-17 / 6+6+0 |
| cont5_1_l/cont5_1_l_0002 | 180,900 | 9.3e+11 | 7.0e-16 / 90600+90300+0 | 9.7e-16 / 90600+90300+0 | 1.1e-16 / 90600+90300+0 | 4.7e-17 / 90600+90300+0 |
| qcqp1000-1nc/qcqp1000-1nc_0043 | 1,154 | 6.3e+11 | 8.9e-17 / 1000+154+0 | 1.2e-16 / 1000+154+0 | 1.2e-54 / 1000+154+0 | 2.1e-16 / 1000+154+0 |
| ex4_2_160/ex4_2_160_0009 | 77,115 | 3.5e+11 | 3.2e-16 / 51198+25917+0 | 3.7e-16 / 51198+25917+0 | 9.3e-17 / 51198+25917+0 | 1.5e-16 / 51198+25917+0 |
| arki0009/arki0009_0033 | 12,144 | 3.0e+11 | 1.3e-16 / 6220+5924+0 | 1.4e-16 / 6220+5924+0 | 6.7e-37 / 6220+5924+0 | 1.8e-16 / 6220+5924+0 |
| NARX_CFy/NARX_CFy_0001 | 92,229 | 1.5e+11 | 2.0e-16 / 43973+48256+0 | 1.4e-16 / 43973+48256+0 | 2.2e-17 / 43973+48256+0 | 2.2e-16 / 43973+48256+0 |

Interpretation: a residual ≈ ε·COND1 is the best a
linear solve can theoretically achieve. When COND1 is
1e14, machine-ε factors give ~1e-2 forward error; what
matters in this regime is whether the solver (a) detects
the conditioning rather than silently returning garbage,
(b) agrees with the reference on inertia, and (c) gets
a residual close to the others on the same system.
Disagreements on inertia for ill-conditioned matrices
are surfaced in the next section.

**HEART6 (pounce-filed report, 2026-05-10).** Three specific KKT iterations from the CUTEst HEART6 IPM run (`dev/debugging/2026-05-10-pounce-heart6-residual.md`) were filed against feral as silent correctness regressions on ill-conditioned KKTs. With the refinement-on Solver wired in, all three are now unanimous across feral, MUMPS, and MA97:
- iter_a (cond ≈ 1e12): feral residual 2.3e-16, MUMPS 4.0e-15, MA57 1.7e-16, MA97 6.9e-17; inertia 6+6+0.
- iter_b (cond ≈ 3e13): feral residual 1.4e-16, MUMPS 6.9e-16, MA57 7.7e-17, MA97 7.7e-16; inertia 4+8+0.
- iter_c (cond ≈ 500): feral residual 2.4e-16, MUMPS 5.0e-15, MA57 1.6e-16, MA97 1.9e-16; inertia 6+6+0.
The pounce report observed feral residual ≈ 1e11 on iter_a, **silent wrong-inertia (reported 6 instead of true 8) on iter_b**, and residual ≈ 1e4 on iter_c at a modest cond ≈ 500. The 1.4e-16 residual on iter_b plus the matching 4+/8+/0 inertia is the headline: feral no longer hides the conditioning when refinement is on.

**MSS1 (issue #5).** Triage subject for the BK 1×1/2×2 inertia-monotonicity investigation. MUMPS `fail`s on this matrix (`INFOG(1) = -9`, insufficient symbolic-phase integer workspace — a known MUMPS-side limitation that doesn't reflect the matrix's analytic conditioning). feral, MA57, and MA97 all succeed with inertia 89+74+0 and residuals feral=2.6e-16, MA57=2.2e-16, MA97=8.7e-16. Feral is strictly more robust than MUMPS here.

**Caveats on the HEART6 numbers.**

- *Synthetic vs original RHS.* The bench RHS is
  `b = A · x_true` with `x_true[i] = 1 + i/n`, which
  is a consistent system — the easy case for iterative
  refinement. The original pounce failure used IPM step
  RHSes (computed from the optimization residual and
  not guaranteed consistent with the factorization's
  working precision). Those original RHSes are on disk
  at `data/matrices/kkt/HEART6_pounce_diag/heart6_iter_*_rhs.mtx`;
  replaying them is the stronger correctness test and
  is a follow-up the bench harness doesn't yet automate.
- *COND1 discrepancy.* MUMPS COND1 ≈ 3e13 on iter_b
  while the pounce report quoted cond ≈ 4e24 from a
  hand estimate. MUMPS's COND1 is a componentwise
  estimate from `RINFOG(10)` (Arioli/Demmel/Duff
  1989); the pounce number was a normwise hand
  estimate. Different estimators, both confirm the
  same qualitative point: the matrix is hard.


## Inertia agreement

All four solvers report identical inertia on **60** of 66 matrices.

## Failures

| Matrix | n | nnz | Solver | Reason |
|---|---:|---:|---|---|
| SPANHYD/SPANHYD_0291 | 114 | 561 | feral | `factor_InvalidInput("matrix_contains_NaN_or_Inf_at_index_(0,0)")` |
| MSS1/MSS1_0165 | 163 | 2521 | mumps | `fail` |
| LHAIFAM/LHAIFAM_0410 | 249 | 960 | mumps | `fail` |
| BENNETT5/BENNETT5_0128 | 465 | 1238 | mumps | `fail` |
| qcqp1500-1nc/qcqp1500-1nc_0000 | 12008 | 191476 | mumps | `fail` |
| dtoc2/dtoc2_0001 | 103920 | 992230 | feral | `factor_InvalidInput("matrix_contains_NaN_or_Inf_at_index_(0,0)")` |

## Notable matrices (feral / MUMPS factor-time ratio)

### Top 10 feral wins vs MUMPS

| Matrix | n | nnz | feral | MUMPS | MA57 | MA97 | feral/MUMPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| gasoil_3200/gasoil_3200_0007 | 63,999 | 425,766 | 12.5ms | 2.45s | 3.71s | 6.18s | 0.01× |
| BT2/BT2_0006 | 4 | 9 | 12μs | 861μs | 7μs | 87μs | 0.01× |
| pinene_3200/pinene_3200_0005 | 127,995 | 732,976 | 28.9ms | 1.91s | 138.5ms | 110.8ms | 0.02× |
| HS17/HS17_0006 | 4 | 9 | 1μs | 31μs | 2μs | 13μs | 0.03× |
| PALMER4NE/PALMER4NE_0009 | 4 | 10 | 1μs | 21μs | 1μs | 3μs | 0.05× |
| LANCZOS1/LANCZOS1_0029 | 6 | 21 | 1μs | 21μs | 2μs | 4μs | 0.05× |
| POLAK4/POLAK4_0066 | 6 | 18 | 1μs | 21μs | 1μs | 7μs | 0.05× |
| 3PK/3PK_0005 | 30 | 230 | 14μs | 276μs | 21μs | 31μs | 0.05× |
| CERI651B/CERI651B_0978 | 7 | 28 | 1μs | 16μs | 2μs | 6μs | 0.06× |
| HS43/HS43_0004 | 7 | 19 | 1μs | 16μs | 3μs | 5μs | 0.06× |

### Top 10 feral losses vs MUMPS

| Matrix | n | nnz | feral | MUMPS | MA57 | MA97 | feral/MUMPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| NELSON/NELSON_0250 | 387 | 1,027 | 423μs | 100μs | 45μs | 73μs | 4.23× |
| ACOPP14/ACOPP14_0010 | 106 | 586 | 302μs | 82μs | 48μs | 50μs | 3.68× |
| DISCS/DISCS_0320 | 102 | 496 | 446μs | 127μs | 32μs | 57μs | 3.51× |
| CORE1/CORE1_0250 | 242 | 516 | 316μs | 102μs | 70μs | 77μs | 3.10× |
| CRESC50/CRESC50_0250 | 306 | 1,067 | 274μs | 109μs | 43μs | 51μs | 2.51× |
| elec_400/elec_400_0006 | 1,600 | 722,200 | 83.3ms | 33.3ms | 139.8ms | 2.65s | 2.50× |
| AIRPORT/AIRPORT_0214 | 126 | 1,932 | 320μs | 131μs | 69μs | 79μs | 2.44× |
| KOEBHELBNE/KOEBHELBNE_0250 | 471 | 1,248 | 415μs | 176μs | 45μs | 50μs | 2.36× |
| ACOPR14/ACOPR14_0250 | 284 | 953 | 302μs | 132μs | 69μs | 69μs | 2.29× |
| QPCBLEND/QPCBLEND_0210 | 157 | 648 | 247μs | 122μs | 67μs | 69μs | 2.02× |

## Full per-matrix table

| Matrix | n | nnz | density | feral factor / rel\_res | MUMPS factor / rel\_res | MA57 factor / rel\_res | MA97 factor / rel\_res |
|---|---:|---:|---:|---|---|---|---|
| BT2/BT2_0006 | 4 | 9 | 90.0% | 12μs / 3.8e-16 | 861μs / 4.9e-16 | 7μs / 4.8e-16 | 87μs / 1.0e-16 |
| HS17/HS17_0006 | 4 | 9 | 90.0% | 1μs / 1.5e-17 | 31μs / 3.3e-17 | 2μs / 1.0e-17 | 13μs / 5.4e-17 |
| PALMER4NE/PALMER4NE_0009 | 4 | 10 | 100.0% | 1μs / 3.7e-16 | 21μs / 2.8e-16 | 1μs / 1.4e-16 | 3μs / 1.4e-16 |
| OSBORNE1/OSBORNE1_0041 | 5 | 15 | 100.0% | 2μs / 2.8e-16 | 19μs / 2.8e-16 | 2μs / 2.8e-16 | 8μs / 1.1e-16 |
| LANCZOS1/LANCZOS1_0029 | 6 | 21 | 100.0% | 1μs / 1.1e-16 | 21μs / 7.1e-17 | 2μs / 1.6e-16 | 4μs / 1.4e-16 |
| POLAK4/POLAK4_0066 | 6 | 18 | 85.7% | 1μs / 1.4e-16 | 21μs / 1.1e-16 | 1μs / 1.6e-16 | 7μs / 5.6e-17 |
| CERI651B/CERI651B_0978 | 7 | 28 | 100.0% | 1μs / 2.5e-16 | 16μs / 4.0e-16 | 2μs / 2.0e-16 | 6μs / 1.3e-16 |
| HS43/HS43_0004 | 7 | 19 | 67.9% | 1μs / 3.9e-16 | 16μs / 7.2e-17 | 3μs / 3.7e-17 | 5μs / 4.9e-16 |
| HS76I/HS76I_0005 | 7 | 19 | 67.9% | 1μs / 1.8e-16 | 15μs / 1.8e-16 | 2μs / 1.9e-16 | 6μs / 1.6e-16 |
| VESUVIA/VESUVIA_0040 | 8 | 36 | 100.0% | 2μs / 1.5e-17 | 17μs / 2.5e-16 | 3μs / 1.2e-16 | 10μs / 2.4e-16 |
| VESUVIO/VESUVIO_0043 | 8 | 36 | 100.0% | 3μs / 7.0e-17 | 15μs / 1.5e-16 | 2μs / 1.5e-16 | 4μs / 2.8e-16 |
| OSBORNEB/OSBORNEB_0008 | 11 | 66 | 100.0% | 3μs / 5.9e-16 | 18μs / 1.3e-16 | 4μs / 3.4e-16 | 5μs / 5.2e-16 |
| HEART6_pounce_diag/heart6_iter_a | 12 | 58 | 74.4% | 14μs / 2.3e-16 | 75μs / 4.0e-15 | 7μs / 1.7e-16 | 49μs / 6.9e-17 |
| HEART6_pounce_diag/heart6_iter_b | 12 | 58 | 74.4% | 3μs / 1.4e-16 | 23μs / 6.9e-16 | 12μs / 7.7e-17 | 18μs / 7.7e-16 |
| HEART6_pounce_diag/heart6_iter_c | 12 | 58 | 74.4% | 3μs / 2.4e-16 | 18μs / 5.0e-15 | 4μs / 1.6e-16 | 17μs / 1.9e-16 |
| 3PK/3PK_0005 | 30 | 230 | 49.5% | 14μs / 1.5e-16 | 276μs / 2.3e-16 | 21μs / 2.4e-16 | 31μs / 1.1e-16 |
| METHANB8LS/METHANB8LS_0004 | 31 | 256 | 51.6% | 21μs / 5.1e-16 | 46μs / 5.3e-16 | 18μs / 3.6e-16 | 23μs / 8.5e-16 |
| DECONVB/DECONVB_0039 | 51 | 891 | 67.2% | 23μs / 3.7e-16 | 65μs / 2.4e-16 | 47μs / 2.4e-16 | 28μs / 3.0e-16 |
| HIMMELBJ/HIMMELBJ_0023 | 57 | 358 | 21.7% | 46μs / 5.9e-17 | 65μs / 5.8e-17 | 28μs / 5.8e-17 | 31μs / 1.3e-16 |
| DMN37142/DMN37142_0122 | 66 | 2,211 | 100.0% | 56μs / 5.4e-16 | 82μs / 5.6e-16 | 93μs / 2.9e-16 | 92μs / 3.1e-16 |
| DISCS/DISCS_0320 | 102 | 496 | 9.4% | 446μs / 1.2e-16 | 127μs / 5.0e-14 | 32μs / 1.1e-16 | 57μs / 1.0e-16 |
| ACOPP14/ACOPP14_0010 | 106 | 586 | 10.3% | 302μs / 1.6e-16 | 82μs / 1.8e-16 | 48μs / 1.2e-16 | 50μs / 2.1e-16 |
| SPANHYD/SPANHYD_0291 | 114 | 561 | 8.6% | `fail` | 95μs / — | 117μs / — | 31μs / — |
| AIRPORT/AIRPORT_0214 | 126 | 1,932 | 24.1% | 320μs / 3.2e-16 | 131μs / 1.4e-15 | 69μs / 3.2e-16 | 79μs / 7.5e-16 |
| QPCBLEND/QPCBLEND_0210 | 157 | 648 | 5.2% | 247μs / 1.6e-16 | 122μs / 2.1e-15 | 67μs / 1.5e-16 | 69μs / 4.1e-16 |
| MSS1/MSS1_0165 | 163 | 2,521 | 18.9% | 207μs / 2.6e-16 | `fail` | 103μs / 2.2e-16 | 173μs / 8.7e-16 |
| HYDCAR20/HYDCAR20_0110 | 198 | 1,071 | 5.4% | 216μs / 4.7e-16 | 181μs / 4.2e-16 | 90μs / 1.6e-16 | 94μs / 1.7e-16 |
| GROUPING/GROUPING_0190 | 225 | 1,475 | 5.8% | 274μs / 1.9e-16 | 159μs / 8.8e-15 | 189μs / 1.9e-16 | 179μs / 5.8e-17 |
| CORE1/CORE1_0250 | 242 | 516 | 1.8% | 316μs / 9.4e-17 | 102μs / 1.4e-14 | 70μs / 8.2e-17 | 77μs / 7.3e-17 |
| HAIFAM/HAIFAM_0370 | 249 | 1,303 | 4.2% | 367μs / 9.7e-17 | 211μs / 3.1e-16 | 85μs / 7.2e-16 | 62μs / 2.0e-16 |
| LHAIFAM/LHAIFAM_0410 | 249 | 960 | 3.1% | 314μs / 4.7e-16 | `fail` | 90μs / 8.2e-17 | 66μs / 6.3e-16 |
| ACOPR14/ACOPR14_0250 | 284 | 953 | 2.4% | 302μs / 3.9e-16 | 132μs / 5.2e-15 | 69μs / 1.3e-16 | 69μs / 1.4e-16 |
| CRESC50/CRESC50_0250 | 306 | 1,067 | 2.3% | 274μs / 1.1e-16 | 109μs / 1.5e-15 | 43μs / 2.6e-16 | 51μs / 8.9e-17 |
| NELSON/NELSON_0250 | 387 | 1,027 | 1.4% | 423μs / 1.9e-16 | 100μs / 6.5e-16 | 45μs / 1.7e-17 | 73μs / 3.9e-17 |
| BENNETT5/BENNETT5_0128 | 465 | 1,238 | 1.1% | 356μs / 1.2e-15 | `fail` | 63μs / 1.8e-14 | 122μs / 5.3e-16 |
| KOEBHELBNE/KOEBHELBNE_0250 | 471 | 1,248 | 1.1% | 415μs / 4.5e-15 | 176μs / 1.7e-16 | 45μs / 9.2e-17 | 50μs / 7.0e-17 |
| qcqp500-3c/qcqp500-3c_0003 | 620 | 130,891 | 68.0% | 7.3ms / 1.7e-16 | 5.7ms / 1.3e-16 | 3.7ms / 4.1e-37 | 46.6ms / 1.0e-16 |
| CHWIRUT1/CHWIRUT1_0195 | 645 | 1,718 | 0.8% | 384μs / 9.5e-16 | 268μs / 1.5e-15 | 75μs / 6.7e-17 | 1.0ms / 4.7e-17 |
| GAUSS2/GAUSS2_0018 | 758 | 3,265 | 1.1% | 461μs / 1.6e-16 | 283μs / 1.7e-16 | 116μs / 6.2e-17 | 304μs / 1.3e-16 |
| qcqp750-2c/qcqp750-2c_0000 | 888 | 281,604 | 71.3% | 20.2ms / 1.7e-16 | 11.3ms / 1.7e-16 | 9.4ms / 1.6e-16 | 1.05s / 1.9e-16 |
| qcqp1000-1nc/qcqp1000-1nc_0043 | 1,154 | 11,868 | 1.8% | 10.3ms / 8.9e-17 | 9.0ms / 1.2e-16 | 18.0ms / 1.2e-54 | 785.2ms / 2.1e-16 |
| elec_400/elec_400_0006 | 1,600 | 722,200 | 56.4% | 83.3ms / 1.6e-15 | 33.3ms / 3.2e-14 | 139.8ms / 1.6e-15 | 2.65s / 4.0e-16 |
| VESUVIOU/VESUVIOU_0028 | 3,083 | 12,813 | 0.3% | 1.6ms / 9.0e-17 | 2.6ms / 8.1e-18 | 482μs / 3.0e-18 | 558μs / 1.5e-16 |
| arki0003/arki0003_0033 | 4,010 | 15,359 | 0.2% | 1.3ms / 9.2e-17 | 1.5ms / 1.1e-16 | 4.8ms / 3.4e-49 | 484μs / 9.2e-17 |
| CRESC132/CRESC132_0003 | 5,314 | 22,576 | 0.2% | 3.5ms / 1.4e-16 | 11.5ms / 4.6e-17 | 1.4ms / 2.5e-18 | 2.2ms / 2.6e-16 |
| qcqp1000-2c/qcqp1000-2c_0010 | 6,107 | 210,614 | 1.1% | 46.9ms / 2.5e-16 | 73.2ms / 2.2e-16 | 23.3ms / 8.2e-40 | 2.96s / 1.8e-16 |
| ex8_2_2/ex8_2_2_0054 | 9,453 | 32,089 | 0.1% | 3.1ms / 6.2e-17 | 7.3ms / 8.0e-17 | 856μs / 3.6e-53 | 776μs / 1.2e-16 |
| dtoc1nd/dtoc1nd_0010 | 9,685 | 217,270 | 0.5% | 8.8ms / 7.5e-16 | 11.6ms / 2.8e-15 | 4.9ms / 2.7e-16 | 5.9ms / 4.9e-17 |
| qcqp1000-2nc/qcqp1000-2nc_0006 | 11,214 | 209,574 | 0.3% | 237.1ms / 7.3e-17 | 179.6ms / 1.4e-16 | 27.6ms / 1.7e-17 | 2.87s / 9.3e-17 |
| qcqp1500-1nc/qcqp1500-1nc_0000 | 12,008 | 191,476 | 0.3% | 831.0ms / 2.0e-16 | `fail` | 139.5ms / 2.1e-16 | 5.72s / 6.7e-17 |
| arki0009/arki0009_0033 | 12,144 | 37,147 | 0.1% | 5.7ms / 1.3e-16 | 50.0ms / 1.4e-16 | 8.5ms / 6.7e-37 | 229.3ms / 1.8e-16 |
| robot_a/robot_a_0001 | 53,008 | 252,774 | 0.0% | 51.6ms / 1.5e-16 | 214.8ms / 1.5e-16 | 7.7ms / 1.2e-16 | 10.6ms / 1.7e-16 |
| robot_c/robot_c_0000 | 53,014 | 248,794 | 0.0% | 57.0ms / 4.2e-16 | 216.7ms / 1.6e-12 | 12.1ms / 3.3e-16 | 10.0ms / 3.6e-16 |
| dirichlet120/dirichlet120_0003 | 54,122 | 422,193 | 0.0% | 291.1ms / 1.6e-15 | 802.8ms / 1.3e-15 | 1.74s / 8.1e-16 | 28.66s / 5.6e-17 |
| lane_emden120/lane_emden120_0003 | 57,962 | 396,481 | 0.0% | 327.3ms / 1.8e-15 | 710.1ms / 1.2e-15 | 585.2ms / 1.5e-15 | 15.20s / 8.8e-17 |
| gasoil_3200/gasoil_3200_0007 | 63,999 | 425,766 | 0.0% | 12.5ms / 1.1e-14 | 2.45s / 2.3e-15 | 3.71s / 9.9e-15 | 6.18s / 5.6e-16 |
| ex4_2_160/ex4_2_160_0009 | 77,115 | 255,354 | 0.0% | 43.1ms / 3.2e-16 | 290.6ms / 3.7e-16 | 66.1ms / 9.3e-17 | 2.41s / 1.5e-16 |
| NARX_CFy/NARX_CFy_0001 | 92,229 | 400,808 | 0.0% | 85.4ms / 2.0e-16 | 476.3ms / 1.4e-16 | 50.3ms / 2.2e-17 | 1.51s / 2.2e-16 |
| dtoc2/dtoc2_0001 | 103,920 | 992,230 | 0.0% | `fail` | 594.3ms / — | 60.2ms / — | 47.9ms / — |
| optmass/optmass_0007 | 110,011 | 260,020 | 0.0% | 47.5ms / 1.2e-16 | 242.9ms / 1.4e-16 | 8.6ms / 8.3e-17 | 13.2ms / 1.3e-16 |
| pinene_3200/pinene_3200_0005 | 127,995 | 732,976 | 0.0% | 28.9ms / 1.2e-14 | 1.91s / 4.3e-09 | 138.5ms / 8.1e-15 | 110.8ms / 8.2e-15 |
| cont5_1_l/cont5_1_l_0002 | 180,900 | 720,303 | 0.0% | 126.0ms / 7.0e-16 | 565.3ms / 9.7e-16 | 153.1ms / 1.1e-16 | 8.91s / 4.7e-17 |
| cont5_2_1_l/cont5_2_1_l_0002 | 180,900 | 720,303 | 0.0% | 103.5ms / 6.9e-16 | 551.2ms / 9.8e-16 | 132.0ms / 1.2e-16 | 8.00s / 7.3e-16 |
| cont5_2_2_l/cont5_2_2_l_0002 | 180,900 | 720,303 | 0.0% | 121.9ms / 7.1e-16 | 551.0ms / 9.8e-16 | 166.8ms / 1.3e-16 | 7.75s / 9.5e-16 |
| cont5_2_3_l/cont5_2_3_l_0002 | 180,900 | 720,303 | 0.0% | 124.9ms / 7.1e-16 | 601.7ms / 9.8e-16 | 165.9ms / 1.3e-16 | 7.54s / 9.5e-16 |
| nql180/nql180_0003 | 259,681 | 939,300 | 0.0% | 295.3ms / 1.1e-14 | 810.9ms / 5.2e-15 | 4.32s / 1.3e-16 | 7.19s / 9.2e-16 |

---

*Generated by `external_benchmarks/comparison/report.py` from `comparison.json`. Reproduce with `python3 run.py && python3 aggregate.py && python3 report.py`.*
