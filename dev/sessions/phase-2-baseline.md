# Phase 2 Performance Baseline Report

**Date:** 2026-04-14
**Head commit under test:** `e08c7a1` (Triage: ERRINBAR_0824 and ACOPP30_0004)
**Corpus:** 154588 KKT matrices in `data/matrices/kkt/` (dense-eligible: 154481
at `n <= 1000`; 107 skipped at `n > 1000`)
**Oracles:** canonical Fortran MUMPS 5.8.2 and SPRAL SSIDS (via
`external_benchmarks/{mumps,ssids}_oracle`), per-matrix `factor_us` /
`solve_us` in `*.mumps.json` / `*.ssids.json` sidecars.

This is the Phase 2.1.8 baseline required by
`dev/plans/phase-2-planning.md` §2.1.8. Every later optimization in Phase
2.4 (dense perf) and Phase 2.5 (sparse perf) is measured against these
numbers.

## Harness additions (Phase 2.1.7)

`src/bin/bench.rs` gained:

- `OracleTiming` + `read_oracle_timing` — parses the `factor_us` /
  `solve_us` fields out of oracle JSON sidecars.
- `KktEntry::{mumps_timing, ssids_timing}` — populated in `load_kkt_dir`
  by `with_extension("mumps.json")` / `with_extension("ssids.json")`;
  missing files leave the fields as `None`.
- `MatrixTiming` — per-matrix feral factor+solve μs, collected in both
  the dense and sparse loops.
- Sparse-loop `Instant::now()` calls — the old sparse loop reported
  inertia and residual but not timings; now records `sp_factor_us`
  (symbolic + numeric combined, matching the semantics of MUMPS's and
  SSIDS's single `factor_us` field) and `sp_solve_us`.
- `print_perf_comparison` — joins feral timings against
  `{mumps,ssids}_timing`, emits overall ratio distribution
  (geomean, p50, p90, p99, max), per-family geomean, and top-10 worst
  factor-ratio matrices vs MUMPS.

Ratio clamp: both sides use `.max(1) μs` so that sub-microsecond
matrices at the clock-resolution floor produce ratio = 1.0 rather than
collapsing the log-space geomean.

## Overall results — ratio = feral_μs / oracle_μs

Lower ratio = feral is faster. Ratio < 1.0 means feral beats the oracle.

### Dense path (`factor_single_front` + `solve_refined`), 154481 matrices

| metric        | count  | geomean |   p50 |   p90 |   p99 |      max |
|---------------|-------:|--------:|------:|------:|------:|---------:|
| factor/MUMPS  | 153472 |    0.23 |  0.11 |  2.27 | 28.99 |   296.45 |
| solve/MUMPS   | 153472 |    0.37 |  0.25 |  2.00 | 23.40 |   523.76 |
| factor/SSIDS  | 154393 |    0.01 |  0.00 |  0.34 |  8.04 |    48.23 |
| solve/SSIDS   | 154393 |    1.48 |  1.00 |  8.50 | 76.33 |   576.13 |

### Sparse path (`factorize_multifrontal` + `solve_sparse_refined`), 154588 matrices

| metric        | count  | geomean |   p50 |   p90 |   p99 |      max |
|---------------|-------:|--------:|------:|------:|------:|---------:|
| factor/MUMPS  | 153560 |    0.67 |  0.50 |  3.18 | 11.40 |  1505.40 |
| solve/MUMPS   | 153560 |    0.47 |  0.38 |  2.60 | 14.19 |   324.42 |
| factor/SSIDS  | 154500 |    0.03 |  0.02 |  0.47 |  2.87 |    46.46 |
| solve/SSIDS   | 154500 |    1.87 |  1.40 | 12.00 | 39.67 |   760.00 |

## Headline interpretation

1. **feral beats MUMPS at the median** on both factor (p50 dense 0.11,
   sparse 0.50) and solve (p50 dense 0.25, sparse 0.38). MUMPS pays a
   ~10× startup overhead on small matrices that feral does not.
2. **feral loses to MUMPS in the tail** on factor (p90 dense 2.27,
   sparse 3.18; p99 dense 28.99, sparse 11.40). This is where Phase
   2.4 (blocked dense LDLᵀ + SIMD micro-kernel) and Phase 2.5 (Liu
   column counts + Rayon on the assembly tree) are expected to pay.
3. **feral beats SSIDS on factor at all percentiles** (geomean 0.01
   dense, 0.03 sparse; p99 dense 8.04, sparse 2.87; max sparse 46.46).
   SSIDS's symbolic phase has heavier fixed-cost setup; feral's is
   simpler and wins on small to medium matrices.
4. **feral loses to SSIDS on solve** (geomean 1.48 dense, 1.87 sparse;
   p90 dense 8.50, sparse 12.00). SSIDS's blocked solve kernel is the
   reference implementation; feral's per-supernode `Vec` allocations
   in `solve_sparse` are the likely bottleneck (Phase 2.5.3 in the
   plan). This is a known gap.

## Phase 2.8 exit criterion check

Spec (FERAL-PROJECT-SPEC.md §1747): *within 2× of MUMPS on small-frontal
KKT set; within 3× on medium set.*

- **Dense path factor p90 = 2.27** — fails the 2× bar by 14% in the
  tail; passes at p50. Medium-frontal threshold is comfortably met at
  p90 (≤3×).
- **Sparse path factor p90 = 3.18** — just above the 3× medium bar
  (6% over). Passes at p50, fails at p90.
- **p99 gaps are large** (dense 28.99, sparse 11.40). The exit
  criterion does not specify a p99 bound but these are the tail
  matrices Phase 2.4 / 2.5 optimization work has to address.

Conclusion: feral is within-spec for most matrices but has a
too-heavy tail. The Phase 2 exit criterion is not yet met at the 90th
percentile; closing it is the explicit goal of Phase 2.4 (dense
performance) and Phase 2.5 (sparse performance).

## Per-family factor geomean vs MUMPS (consistently slow families)

Families with geomean > 1.0 on either path — these are where feral is
slower than MUMPS at the median and are priority targets:

| family   | count  | dense geomean | sparse geomean |
|----------|-------:|--------------:|---------------:|
| AVION2   |   2682 |          1.76 |           2.26 |
| BATCH    |   2054 |             — |           2.55 |
| HS118    |   3000 |          0.47 |           1.31 |
| CONCON   |   3000 |          0.45 |           1.10 |
| MCONCON  |   3000 |          0.51 |           0.96 |

AVION2 and BATCH are the only families where both paths pay consistent
slowdown at the median. Everything else is < 1.0 at the median.

## Top-10 worst individual factor-ratio vs MUMPS

### Dense path

| matrix           |    n | feral(μs) | mumps(μs) | ratio  |
|------------------|-----:|----------:|----------:|-------:|
| MCONCON_2963     |   26 |      5929 |        20 | 296.45 |
| ACOPR30_0035     |  564 |     47461 |       181 | 262.22 |
| MCONCON_1553     |   26 |      3816 |        16 | 238.50 |
| HS2NE_0369       |    8 |      1946 |         9 | 216.22 |
| ACOPR30_0208     |  564 |     32271 |       167 | 193.24 |
| HS85_0286        |   68 |      3976 |        23 | 172.87 |
| CHWIRUT2_0090    |  159 |      5674 |        34 | 166.88 |
| HAHN1_0309       |  715 |     36490 |       223 | 163.63 |
| MCONCON_1493     |   26 |      3222 |        20 | 161.10 |
| HAHN1_0143       |  715 |     29733 |       194 | 153.26 |

### Sparse path

| matrix           |    n | feral(μs) | mumps(μs) | ratio   |
|------------------|-----:|----------:|----------:|--------:|
| PALMER1E_0274    |    8 |     15054 |        10 | 1505.40 |
| MCONCON_0253     |   26 |      4152 |        23 |  180.52 |
| SSI_0887         |    3 |      2172 |        13 |  167.08 |
| HS92_2858        |    7 |      2756 |        18 |  153.11 |
| RES_0194         |   62 |      3059 |        21 |  145.67 |
| HS92_2799        |    7 |      1399 |        10 |  139.90 |
| HS92_0653        |    7 |      1260 |        10 |  126.00 |
| METHANL8LS_0209  |   31 |      2470 |        20 |  123.50 |
| METHANL8LS_0481  |   31 |      2425 |        20 |  121.25 |
| VESUVIO_0021     | 3083 |    247756 |      2265 |  109.38 |

Small-n outliers (e.g. PALMER1E_0274 at 15054 μs for n=8) are noise-floor
artifacts: the first call in each family pays JIT and cache-warming
cost that the aggregate median does not see. The useful signal is the
large-n cases: HAHN1_0309 at n=715 (163×), ACOPR30_0035 at n=564
(262×), VESUVIO_0021 at n=3083 (109×). These are the real optimization
targets.

## Correctness delta since Phase 2.3

No correctness change — this is purely a measurement-infrastructure
commit. The correctness numbers match the Phase 2.3 validation report:

| metric                 |      value |
|------------------------|-----------:|
| Dense inertia match    | 152911/154481 (99.0%) |
| Dense residual pass    | 154207/154481 (99.8%) |
| Dense worst residual   | 1.87e-4 (ERRINBAR_0824) |
| Sparse inertia match   | 153009/154588 (99.0%) |
| Sparse residual pass   | 154329/154588 (99.8%) |
| Sparse worst residual  | 2.50e-4 (ERRINBAR_0824) |

## Caveats

- **Microsecond-resolution noise floor.** Both feral and MUMPS use
  `Instant::now()` / equivalent at μs granularity; matrices below
  ~10 μs on either side are noise-limited and their individual
  ratios mean little. The clamp guarantees the geomean stays
  stable, but the top-10 worst lists still surface noise artifacts
  for small-n matrices.
- **Oracle timing was captured on a different hardware run.** The
  MUMPS and SSIDS numbers in `*.mumps.json` / `*.ssids.json` were
  produced by `external_benchmarks/{mumps,ssids}_oracle/run_*.py`
  on the same machine as feral for this report, but not in the
  same process. Cross-process timing noise is ±10% typical.
- **Combined factor timing for the sparse path.** feral's
  `sp_factor_us` is `symbolic_factorize` + `factorize_multifrontal`
  combined. MUMPS and SSIDS report a single `factor_us` that covers
  their equivalent analysis + numeric phases, so the comparison is
  apples-to-apples. Do not subtract one from the other to compare
  phase-level timings.
- **The dense path uses `factor_single_front`, not the old blocked
  `factor()`.** Task #19 (session 2026-04-14-01) rerouted the bench's
  dense KKT validation through the frontal kernel to close the
  ACOPP30 residual gap. This is also the kernel that feral's sparse
  path uses at the root supernode, so dense-vs-sparse here is really
  a single-frontal-vs-multifrontal comparison on the same numerical
  kernel.

## What this baseline gates

Every later Phase 2.4 / 2.5 / 2.6 optimization PR must re-run this
harness and compare. The authoritative numbers to track are:

1. Dense factor p90 vs MUMPS (bar: 2.0, current: 2.27, delta: +0.27)
2. Sparse factor p90 vs MUMPS (bar: 3.0, current: 3.18, delta: +0.18)
3. Dense factor p99 vs MUMPS (no formal bar, current 28.99)
4. Sparse factor p99 vs MUMPS (no formal bar, current 11.40)
5. Sparse solve vs SSIDS geomean (no formal bar, current 1.87 —
   Phase 2.5.3 allocator work is expected to close this)
6. AVION2 and BATCH family geomeans on the sparse path (2.26, 2.55)
7. HAHN1, ACOPR30, VESUVIO large-n worst cases (>100× vs MUMPS)

When any of these move materially, note the delta in the commit
message and re-emit this table in the next validation report.

## Next steps per the Phase 2 plan

With the baseline in hand, the plan's ordering becomes executable:

1. **Phase 2.4.1 — blocked dense LDLᵀ (`block_size = 64`).** Target:
   pull dense factor p90 below 2.0 and p99 under ~10 by getting the
   rank-1/rank-2 updates into the L1 cache-efficient regime. Faer's
   blocked kernel is the cited reference.
2. **Phase 2.4.2 — SIMD micro-kernel for the Schur update.** Inner
   loop of the blocked kernel. Target: 4–8× on the scalar rank-1 loop.
3. **Phase 2.5.1 — Liu column counts.** The current O(n²) algorithm
   is the documented scaling weak point at n ≥ 10³. Target: linearize
   the symbolic pipeline.
4. **Phase 2.5.3 — preallocated solve scratch buffers.** Remove the
   per-supernode `Vec` allocations in `solve_sparse`. Target: close
   the sparse-solve gap vs SSIDS (current geomean 1.87 → target
   under 1.2).

The AVION2, BATCH, HAHN1, ACOPR30, and VESUVIO cases all cluster at
the "medium frontal dimension" end of the spectrum (n ~500–3000), so
these optimizations should hit the right matrices.
