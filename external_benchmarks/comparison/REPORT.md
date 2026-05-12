# FERAL vs MUMPS vs HSL MA97 — KKT solver comparison

Total matrices: **63**, drawn from the FERAL CUTEst
KKT corpus and Mittelmann large-scale KKT corpus.
Sampling spans 5 size buckets and 63 distinct CUTEst/Mittelmann
families. RHS is synthetic: `b = A · x_true` with
`x_true[i] = 1 + i/n`. Same RHS is fed to all three solvers.

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
| MA97  | ma97-2.8.1 | `ma97_factor matrix_type=4` + Richardson loop around `ma97_solve_d` | `scaling = 1` (MC64 enabled, the recommended HSL default); `ordering = 5` (auto AMD/METIS); `action = 1` (continue past singular pivots). MA97 has no built-in residual-based refinement entry point, so the driver wraps `ma97_solve_d` in a 4-step Richardson loop (stagnation exit) to match what MUMPS+ICNTL(10) and feral+`solve_sparse_refined` deliver. CoinHSL 2023.11.17, OpenMP. |

> Change any of these settings and the timing/accuracy
> columns will move. The bench captures each library's
> *best-effort* mode, not its raw defaults.

**Host**: macOS-26.3.1-arm64-arm-64bit, arm64, Python 3.12.11.

## Sample composition

| Bucket | Count |
|---|---:|
| tiny (n<100) | 17 |
| small (100-1k) | 20 |
| medium (1k-10k) | 8 |
| large (10k-100k) | 10 |
| xl (>=100k) | 8 |

## Status summary

| Solver | OK | Fail | Missing |
|---|---:|---:|---:|
| feral | 61 | 2 | 0 |
| mumps | 59 | 4 | 0 |
| ma97 | 63 | 0 | 0 |

## Factor time by size bucket (geomean μs)

| Bucket | n range |  feral |  MUMPS |  MA97  | feral/MUMPS | feral/MA97 |
|---|---|---:|---:|---:|---:|---:|
| tiny (n<100) | 4–66 | 4 | 32 | 37 | 0.12× | 0.10× |
| small (100-1k) | 102–888 | 455 | 239 | 168 | 1.90× | 2.71× |
| medium (1k-10k) | 1154–9685 | 6,655 | 9,555 | 6,113 | 0.70× | 1.09× |
| large (10k-100k) | 11214–92229 | 77,759 | 342,640 | 307,957 | 0.23× | 0.25× |
| xl (>=100k) | 103920–259681 | 93,399 | 601,558 | 477,336 | 0.16× | 0.20× |

> Ratios < 1.0 mean **feral is faster**. Geomean is over the
> matrices in the bucket where the named solver succeeded.

## Accuracy: ‖Ax − b‖₂ / ‖b‖₂ distribution

| Solver | min | median | p90 | max | # > 1e-8 |
|---|---:|---:|---:|---:|---:|
| feral | 1.5e-17 | 2.5e-16 | 1.6e-15 | 1.2e-14 | 0 |
| mumps | 8.1e-18 | 2.8e-16 | 5.2e-15 | 4.3e-09 | 0 |
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

| Matrix | n | MUMPS COND1 | feral res / inertia | MUMPS res / inertia | MA97 res / inertia |
|---|---:|---:|---|---|---|
| cont5_2_1_l/cont5_2_1_l_0002 | 180,900 | 1.7e+14 | 6.9e-16 / 90600+90300+0 | 9.8e-16 / 90600+90300+0 | 7.3e-16 / 90600+90300+0 |
| cont5_2_2_l/cont5_2_2_l_0002 | 180,900 | 1.7e+14 | 7.1e-16 / 90600+90300+0 | 9.8e-16 / 90600+90300+0 | 9.5e-16 / 90600+90300+0 |
| cont5_2_3_l/cont5_2_3_l_0002 | 180,900 | 1.7e+14 | 7.1e-16 / 90600+90300+0 | 9.8e-16 / 90600+90300+0 | 9.5e-16 / 90600+90300+0 |
| cont5_1_l/cont5_1_l_0002 | 180,900 | 9.3e+11 | 7.0e-16 / 90600+90300+0 | 9.7e-16 / 90600+90300+0 | 4.7e-17 / 90600+90300+0 |
| qcqp1000-1nc/qcqp1000-1nc_0043 | 1,154 | 6.3e+11 | 8.9e-17 / 1000+154+0 | 1.2e-16 / 1000+154+0 | 2.1e-16 / 1000+154+0 |
| ex4_2_160/ex4_2_160_0009 | 77,115 | 3.5e+11 | 3.2e-16 / 51198+25917+0 | 3.7e-16 / 51198+25917+0 | 1.5e-16 / 51198+25917+0 |
| arki0009/arki0009_0033 | 12,144 | 3.0e+11 | 1.3e-16 / 6220+5924+0 | 1.4e-16 / 6220+5924+0 | 1.8e-16 / 6220+5924+0 |
| NARX_CFy/NARX_CFy_0001 | 92,229 | 1.5e+11 | 2.0e-16 / 43973+48256+0 | 1.4e-16 / 43973+48256+0 | 2.2e-16 / 43973+48256+0 |
| dtoc1nd/dtoc1nd_0010 | 9,685 | 5.6e+10 | 7.5e-16 / 5960+3725+0 | 2.8e-15 / 5960+3725+0 | 4.9e-17 / 5960+3725+0 |
| ex8_2_2/ex8_2_2_0054 | 9,453 | 2.0e+10 | 6.2e-17 / 7510+1943+0 | 8.0e-17 / 7510+1943+0 | 1.2e-16 / 7510+1943+0 |

Interpretation: a residual ≈ ε·COND1 is the best a
linear solve can theoretically achieve. When COND1 is
1e14, machine-ε factors give ~1e-2 forward error; what
matters in this regime is whether the solver (a) detects
the conditioning rather than silently returning garbage,
(b) agrees with the reference on inertia, and (c) gets
a residual close to the others on the same system.
Disagreements on inertia for ill-conditioned matrices
are surfaced in the next section.

## Inertia agreement

All three solvers report identical inertia on **57** of 63 matrices.

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

| Matrix | n | nnz | feral | MUMPS | MA97 | feral/MUMPS |
|---|---:|---:|---:|---:|---:|---:|
| gasoil_3200/gasoil_3200_0007 | 63,999 | 425,766 | 12.0ms | 2.41s | 6.15s | 0.00× |
| pinene_3200/pinene_3200_0005 | 127,995 | 732,976 | 27.3ms | 1.87s | 109.5ms | 0.01× |
| BT2/BT2_0006 | 4 | 9 | 10μs | 544μs | 4.9ms | 0.02× |
| HS17/HS17_0006 | 4 | 9 | 1μs | 21μs | 38μs | 0.05× |
| LANCZOS1/LANCZOS1_0029 | 6 | 21 | 1μs | 21μs | 13μs | 0.05× |
| PALMER4NE/PALMER4NE_0009 | 4 | 10 | 1μs | 19μs | 11μs | 0.05× |
| POLAK4/POLAK4_0066 | 6 | 18 | 1μs | 17μs | 950μs | 0.06× |
| HS43/HS43_0004 | 7 | 19 | 1μs | 16μs | 9μs | 0.06× |
| HS76I/HS76I_0005 | 7 | 19 | 1μs | 16μs | 8μs | 0.06× |
| OSBORNE1/OSBORNE1_0041 | 5 | 15 | 2μs | 21μs | 13μs | 0.10× |

### Top 10 feral losses vs MUMPS

| Matrix | n | nnz | feral | MUMPS | MA97 | feral/MUMPS |
|---|---:|---:|---:|---:|---:|---:|
| DISCS/DISCS_0320 | 102 | 496 | 583μs | 127μs | 60μs | 4.59× |
| NELSON/NELSON_0250 | 387 | 1,027 | 361μs | 109μs | 72μs | 3.31× |
| CORE1/CORE1_0250 | 242 | 516 | 291μs | 103μs | 76μs | 2.83× |
| elec_400/elec_400_0006 | 1,600 | 722,200 | 80.7ms | 29.9ms | 106.4ms | 2.70× |
| ACOPR14/ACOPR14_0250 | 284 | 953 | 365μs | 157μs | 69μs | 2.32× |
| AIRPORT/AIRPORT_0214 | 126 | 1,932 | 317μs | 140μs | 78μs | 2.26× |
| GROUPING/GROUPING_0190 | 225 | 1,475 | 352μs | 157μs | 337μs | 2.24× |
| ACOPP14/ACOPP14_0010 | 106 | 586 | 230μs | 105μs | 49μs | 2.19× |
| KOEBHELBNE/KOEBHELBNE_0250 | 471 | 1,248 | 334μs | 174μs | 52μs | 1.92× |
| GAUSS2/GAUSS2_0018 | 758 | 3,265 | 527μs | 293μs | 343μs | 1.80× |

## Full per-matrix table

| Matrix | n | nnz | density | feral factor / rel\_res | MUMPS factor / rel\_res | MA97 factor / rel\_res |
|---|---:|---:|---:|---|---|---|
| BT2/BT2_0006 | 4 | 9 | 90.0% | 10μs / 3.8e-16 | 544μs / 4.9e-16 | 4.9ms / 1.0e-16 |
| HS17/HS17_0006 | 4 | 9 | 90.0% | 1μs / 1.5e-17 | 21μs / 3.3e-17 | 38μs / 5.4e-17 |
| PALMER4NE/PALMER4NE_0009 | 4 | 10 | 100.0% | 1μs / 3.7e-16 | 19μs / 2.8e-16 | 11μs / 1.4e-16 |
| OSBORNE1/OSBORNE1_0041 | 5 | 15 | 100.0% | 2μs / 2.8e-16 | 21μs / 2.8e-16 | 13μs / 1.1e-16 |
| LANCZOS1/LANCZOS1_0029 | 6 | 21 | 100.0% | 1μs / 1.1e-16 | 21μs / 7.1e-17 | 13μs / 1.4e-16 |
| POLAK4/POLAK4_0066 | 6 | 18 | 85.7% | 1μs / 1.4e-16 | 17μs / 1.1e-16 | 950μs / 5.6e-17 |
| CERI651B/CERI651B_0978 | 7 | 28 | 100.0% | 2μs / 2.5e-16 | 16μs / 4.0e-16 | 11μs / 1.3e-16 |
| HS43/HS43_0004 | 7 | 19 | 67.9% | 1μs / 3.9e-16 | 16μs / 7.2e-17 | 9μs / 4.9e-16 |
| HS76I/HS76I_0005 | 7 | 19 | 67.9% | 1μs / 1.8e-16 | 16μs / 1.8e-16 | 8μs / 1.6e-16 |
| VESUVIA/VESUVIA_0040 | 8 | 36 | 100.0% | 2μs / 1.5e-17 | 17μs / 2.5e-16 | 19μs / 2.4e-16 |
| VESUVIO/VESUVIO_0043 | 8 | 36 | 100.0% | 3μs / 7.0e-17 | 13μs / 1.5e-16 | 19μs / 2.8e-16 |
| OSBORNEB/OSBORNEB_0008 | 11 | 66 | 100.0% | 2μs / 5.9e-16 | 18μs / 1.3e-16 | 11μs / 5.2e-16 |
| 3PK/3PK_0005 | 30 | 230 | 49.5% | 14μs / 1.5e-16 | 87μs / 2.3e-16 | 207μs / 1.1e-16 |
| METHANB8LS/METHANB8LS_0004 | 31 | 256 | 51.6% | 17μs / 5.1e-16 | 48μs / 5.3e-16 | 26μs / 8.5e-16 |
| DECONVB/DECONVB_0039 | 51 | 891 | 67.2% | 21μs / 3.7e-16 | 65μs / 2.4e-16 | 28μs / 3.0e-16 |
| HIMMELBJ/HIMMELBJ_0023 | 57 | 358 | 21.7% | 45μs / 5.9e-17 | 76μs / 5.8e-17 | 32μs / 1.3e-16 |
| DMN37142/DMN37142_0122 | 66 | 2,211 | 100.0% | 58μs / 5.4e-16 | 89μs / 5.6e-16 | 96μs / 3.1e-16 |
| DISCS/DISCS_0320 | 102 | 496 | 9.4% | 583μs / 1.2e-16 | 127μs / 5.0e-14 | 60μs / 1.0e-16 |
| ACOPP14/ACOPP14_0010 | 106 | 586 | 10.3% | 230μs / 1.6e-16 | 105μs / 1.8e-16 | 49μs / 2.1e-16 |
| SPANHYD/SPANHYD_0291 | 114 | 561 | 8.6% | `fail` | 93μs / — | 30μs / — |
| AIRPORT/AIRPORT_0214 | 126 | 1,932 | 24.1% | 317μs / 3.2e-16 | 140μs / 1.4e-15 | 78μs / 7.5e-16 |
| QPCBLEND/QPCBLEND_0210 | 157 | 648 | 5.2% | 165μs / 1.6e-16 | 127μs / 2.1e-15 | 67μs / 4.1e-16 |
| MSS1/MSS1_0165 | 163 | 2,521 | 18.9% | 223μs / 2.6e-16 | `fail` | 170μs / 8.7e-16 |
| HYDCAR20/HYDCAR20_0110 | 198 | 1,071 | 5.4% | 186μs / 4.7e-16 | 182μs / 4.2e-16 | 100μs / 1.7e-16 |
| GROUPING/GROUPING_0190 | 225 | 1,475 | 5.8% | 352μs / 1.9e-16 | 157μs / 8.8e-15 | 337μs / 5.8e-17 |
| CORE1/CORE1_0250 | 242 | 516 | 1.8% | 291μs / 9.4e-17 | 103μs / 1.4e-14 | 76μs / 7.3e-17 |
| HAIFAM/HAIFAM_0370 | 249 | 1,303 | 4.2% | 309μs / 9.7e-17 | 218μs / 3.1e-16 | 60μs / 2.0e-16 |
| LHAIFAM/LHAIFAM_0410 | 249 | 960 | 3.1% | 286μs / 4.7e-16 | `fail` | 66μs / 6.3e-16 |
| ACOPR14/ACOPR14_0250 | 284 | 953 | 2.4% | 365μs / 3.9e-16 | 157μs / 5.2e-15 | 69μs / 1.4e-16 |
| CRESC50/CRESC50_0250 | 306 | 1,067 | 2.3% | 244μs / 1.1e-16 | 139μs / 1.5e-15 | 52μs / 8.9e-17 |
| NELSON/NELSON_0250 | 387 | 1,027 | 1.4% | 361μs / 1.9e-16 | 109μs / 6.5e-16 | 72μs / 3.9e-17 |
| BENNETT5/BENNETT5_0128 | 465 | 1,238 | 1.1% | 362μs / 1.2e-15 | `fail` | 85μs / 5.3e-16 |
| KOEBHELBNE/KOEBHELBNE_0250 | 471 | 1,248 | 1.1% | 334μs / 4.5e-15 | 174μs / 1.7e-16 | 52μs / 7.0e-17 |
| qcqp500-3c/qcqp500-3c_0003 | 620 | 130,891 | 68.0% | 6.8ms / 1.7e-16 | 5.3ms / 1.3e-16 | 33.2ms / 1.0e-16 |
| CHWIRUT1/CHWIRUT1_0195 | 645 | 1,718 | 0.8% | 452μs / 9.5e-16 | 276μs / 1.5e-15 | 186μs / 4.7e-17 |
| GAUSS2/GAUSS2_0018 | 758 | 3,265 | 1.1% | 527μs / 1.6e-16 | 293μs / 1.7e-16 | 343μs / 1.3e-16 |
| qcqp750-2c/qcqp750-2c_0000 | 888 | 281,604 | 71.3% | 19.5ms / 1.7e-16 | 11.1ms / 1.7e-16 | 172.1ms / 1.9e-16 |
| qcqp1000-1nc/qcqp1000-1nc_0043 | 1,154 | 11,868 | 1.8% | 9.4ms / 8.9e-17 | 9.7ms / 1.2e-16 | 56.1ms / 2.1e-16 |
| elec_400/elec_400_0006 | 1,600 | 722,200 | 56.4% | 80.7ms / 1.6e-15 | 29.9ms / 3.2e-14 | 106.4ms / 4.0e-16 |
| VESUVIOU/VESUVIOU_0028 | 3,083 | 12,813 | 0.3% | 1.3ms / 9.0e-17 | 2.5ms / 8.1e-18 | 517μs / 1.5e-16 |
| arki0003/arki0003_0033 | 4,010 | 15,359 | 0.2% | 1.1ms / 9.2e-17 | 1.4ms / 1.1e-16 | 472μs / 9.2e-17 |
| CRESC132/CRESC132_0003 | 5,314 | 22,576 | 0.2% | 3.3ms / 1.4e-16 | 11.4ms / 4.6e-17 | 2.1ms / 2.6e-16 |
| qcqp1000-2c/qcqp1000-2c_0010 | 6,107 | 210,614 | 1.1% | 41.8ms / 2.5e-16 | 73.0ms / 2.2e-16 | 142.4ms / 1.8e-16 |
| ex8_2_2/ex8_2_2_0054 | 9,453 | 32,089 | 0.1% | 3.0ms / 6.2e-17 | 7.0ms / 8.0e-17 | 758μs / 1.2e-16 |
| dtoc1nd/dtoc1nd_0010 | 9,685 | 217,270 | 0.5% | 8.8ms / 7.5e-16 | 11.7ms / 2.8e-15 | 5.9ms / 4.9e-17 |
| qcqp1000-2nc/qcqp1000-2nc_0006 | 11,214 | 209,574 | 0.3% | 207.8ms / 7.3e-17 | 193.0ms / 1.4e-16 | 176.2ms / 9.3e-17 |
| qcqp1500-1nc/qcqp1500-1nc_0000 | 12,008 | 191,476 | 0.3% | 792.1ms / 2.0e-16 | `fail` | 347.4ms / 6.7e-17 |
| arki0009/arki0009_0033 | 12,144 | 37,147 | 0.1% | 5.1ms / 1.3e-16 | 48.8ms / 1.4e-16 | 19.7ms / 1.8e-16 |
| robot_a/robot_a_0001 | 53,008 | 252,774 | 0.0% | 55.6ms / 1.5e-16 | 207.2ms / 1.5e-16 | 10.3ms / 1.7e-16 |
| robot_c/robot_c_0000 | 53,014 | 248,794 | 0.0% | 55.1ms / 4.2e-16 | 205.4ms / 1.6e-12 | 9.9ms / 3.6e-16 |
| dirichlet120/dirichlet120_0003 | 54,122 | 422,193 | 0.0% | 261.5ms / 1.6e-15 | 747.1ms / 1.3e-15 | 5.39s / 5.6e-17 |
| lane_emden120/lane_emden120_0003 | 57,962 | 396,481 | 0.0% | 297.1ms / 1.8e-15 | 688.0ms / 1.2e-15 | 6.29s / 8.8e-17 |
| gasoil_3200/gasoil_3200_0007 | 63,999 | 425,766 | 0.0% | 12.0ms / 1.1e-14 | 2.41s / 2.3e-15 | 6.15s / 5.6e-16 |
| ex4_2_160/ex4_2_160_0009 | 77,115 | 255,354 | 0.0% | 42.1ms / 3.2e-16 | 281.2ms / 3.7e-16 | 672.2ms / 1.5e-16 |
| NARX_CFy/NARX_CFy_0001 | 92,229 | 400,808 | 0.0% | 79.5ms / 2.0e-16 | 465.4ms / 1.4e-16 | 447.3ms / 2.2e-16 |
| dtoc2/dtoc2_0001 | 103,920 | 992,230 | 0.0% | `fail` | 590.0ms / — | 48.3ms / — |
| optmass/optmass_0007 | 110,011 | 260,020 | 0.0% | 46.9ms / 1.2e-16 | 231.5ms / 1.4e-16 | 13.5ms / 1.3e-16 |
| pinene_3200/pinene_3200_0005 | 127,995 | 732,976 | 0.0% | 27.3ms / 1.2e-14 | 1.87s / 4.3e-09 | 109.5ms / 8.2e-15 |
| cont5_1_l/cont5_1_l_0002 | 180,900 | 720,303 | 0.0% | 121.6ms / 7.0e-16 | 549.1ms / 9.7e-16 | 1.90s / 4.7e-17 |
| cont5_2_1_l/cont5_2_1_l_0002 | 180,900 | 720,303 | 0.0% | 102.8ms / 6.9e-16 | 550.6ms / 9.8e-16 | 2.13s / 7.3e-16 |
| cont5_2_2_l/cont5_2_2_l_0002 | 180,900 | 720,303 | 0.0% | 116.9ms / 7.1e-16 | 527.5ms / 9.8e-16 | 1.87s / 9.5e-16 |
| cont5_2_3_l/cont5_2_3_l_0002 | 180,900 | 720,303 | 0.0% | 120.2ms / 7.1e-16 | 529.8ms / 9.8e-16 | 1.69s / 9.5e-16 |
| nql180/nql180_0003 | 259,681 | 939,300 | 0.0% | 275.8ms / 1.1e-14 | 792.8ms / 5.2e-15 | 2.96s / 9.2e-16 |

---

*Generated by `external_benchmarks/comparison/report.py` from `comparison.json`. Reproduce with `python3 run.py && python3 aggregate.py && python3 report.py`.*
