# FERAL Benchmark History

Append-only log of benchmark results. Each entry records the state of the solver
at a point in time so regressions can be detected.

Comparison note: "inertia match" and "residual pass" compare FERAL against MUMPS
(via ripopt sidecar). Disagreements are not necessarily FERAL errors — see
eigenvalue analysis in session 2026-04-11-03 showing both solvers can be wrong
on ill-conditioned matrices (condition > 1e11).

---

## 2026-04-11 — Baseline (dense BK, real KKT matrices)

- **Commit:** post-1fbe813 (uncommitted: io module, KKT loading, bench harness)
- **Solver:** Dense LDL^T with Bunch-Kaufman pivoting (scalar/unblocked)
- **Toolchain:** rustc 1.94.1, release mode, Apple M-series
- **Test count:** 54 (17 exact + 12 MTX/sidecar parser + 6 property + 8 stress + 8 KKT + 3 sidecar)

### Built-in matrices (random, deterministic seed=42)

| name | n | factor(μs) | solve(μs) | inertia |
|------|---|------------|-----------|---------|
| spd_10 | 10 | 3 | 10 | (10, 0, 0) |
| spd_50 | 50 | 20 | 3 | (50, 0, 0) |
| spd_100 | 100 | 75 | 5 | (100, 0, 0) |
| spd_200 | 200 | 385 | 16 | (200, 0, 0) |
| kkt_10_3 | 13 | 3 | 0 | (10, 3, 0) |
| kkt_30_10 | 40 | 28 | 1 | (30, 10, 0) |
| kkt_50_15 | 65 | 53 | 2 | (50, 15, 0) |
| kkt_100_30 | 130 | 199 | 7 | (100, 30, 0) |

### Real KKT matrices (from collect_kkt, dim <= 500)

| Metric | Value |
|--------|-------|
| Matrices loaded | 153,151 |
| Matrices skipped (NaN/Inf) | ~16,400 |
| Inertia match vs MUMPS | 151,930 / 153,151 (99.2%) |
| Residual pass | 152,489 / 153,151 (99.6%) |
| Worst residual | 8.97e-1 (POLAK6_0021) |
| Factor failures | 0 |
| Problems collected | ~148 / 727 (partial, dim>500 still available for sparse phase) |

### Eigenvalue analysis of disagreements

Spot-checked ACOPP14_0001 (dim=106, cond=1e11) and ACOPP30_0000 (dim=209, cond=1e18):
- ACOPP14: eigenvalue truth (37,69,0), MUMPS says (38,68,0), FERAL says (37,68,1) — both wrong
- ACOPP30: smallest eigenvalue 2.2e-14 (~100*eps), MUMPS says positive, FERAL says zero — debatable
- Conclusion: disagreements cluster on ill-conditioned matrices where inertia is ambiguous
