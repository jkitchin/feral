# FERAL Context (auto-generated)

Generated: 2026-04-13T11:19:00Z

## Latest Session
File: dev/sessions/2026-04-13-01.md
```
# Session 2026-04-13-01 — Phase 2.2.3 supernode adjacency fix

## Goal

Debug the CHWIRUT1 / CRESC100 / CRESC132 plateau (Phase 2.2.3 per
the prior session's "Next Session Should" list). Three matrices
that remained `#[ignore]`'d in `tests/mc64_regression.rs` with
residuals 8 to 16 orders above both target and canonical MUMPS
oracle, attributed in Phase 2.2.2 to "solve-side rounding" or
"iterative refinement stagnation". Strategy agreed with the user:
build a minimal repro first, then use expert agents for
independent references on the canonical supernodal solve.

## Accomplished

**Important context:** this session's bench stats are not directly
comparable to last session's. The old numbers were an artifact of
a `nemin=10000` override in `bench.rs` that accidentally masked
the bug being fixed today. See §"Benchmark Results" below.

### Phase 2.2.3 Steps 1–2: research + diagnostic (`cccd640`)

1. **Research note** `dev/research/phase-2.2.3-plateau.md` — five
   hypotheses: H1 refinement best-iterate lock-in, H2 residual at
   arithmetic noise floor, H3 scaling frame error, H4
   multi-supernode solve bug, H5 CRESC132 ±2 inertia. Covers
   `solve_sparse_refined`, Arioli ω₁ stop criterion, and the
   SSIDS/MUMPS iterative-refinement conventions.

2. **Diagnostic binary** `examples/triage_plateau.rs` — instruments
   each matrix under both `nemin=32` and `nemin=10000`, logs
   per-iteration `||r||₂`, `||dx||₂/||x||₂`, Arioli ω₁,
   `||A||₁·||x||∞/||b||∞`, refinement break reason. Immediate
   finding: **all four matrices converge under `nemin=10000` and
   diverge under `nemin=32`**, including ACOPP30 and CRESC132.

   Under `nemin=10000` (single-supernode-forcing):
   ```
   ACOPP30_0000   residual 1.02e-14 (was 1.08e-1)  INERTIA MATCH
   CHWIRUT1_0000  residual 8.69e-14 (was 8.50e+2)  INERTIA MATCH
   CRESC100_0000  residual 2.15e-16 (was 1.43e+2)  INERTIA MATCH
   CRESC132_0000  residual 3.65e-15 (was 1.37e+5)  INERTIA MATCH
   ```

   Three matrices **beat the canonical MUMPS oracle**. CRESC132's
   ±2 inertia mismatch also disappears. The plateau was a single
   root cause, not three; the "solve-side stagnation" and
   "trace-rule inertia" diagnoses from Phase 2.2.2 were wrong.

### Phase 2.2.3 Step 3: minimal repro + expert consultation
```

## Git Status
```
d4f5c9c Phase 2.2.3: validation report
fcf3c57 Phase 2.2.3 Step 3: update mc64_regression comments; drop bench nemin override
91e808b Phase 2.2.3: add adjacency check to supernode amalgamation
cccd640 Phase 2.2.3 Step 1-2: plateau research note + diagnostic
c7664cb Session 2026-04-12-02 checkpoint: Phase 2.2.1 + Phase 2.2.2
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s

     Running tests/threshold_consistency.rs (target/debug/deps/threshold_consistency-c660296127e8afca)

running 6 tests
test polak6_0021_residual_after_threshold_fix ... ignored
test factors_carry_zero_tol_from_params ... ok
test factor_inertia_force_accept_implies_solve_skip_invariant ... ok
test dense_solve_skips_zero_pivots_rank_deficient ... ok
test refinement_does_not_amplify_error_on_rank_deficient_matrix ... ok
test sparse_solve_skips_zero_pivots_rank_deficient ... ok

test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
FERAL benchmark harness
Loading matrices from data/benchmark-config.toml ... not found

name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10           59            1     (10, 0, 0)
spd_50             50           74            9     (50, 0, 0)
spd_100           100          248           11    (100, 0, 0)
spd_200           200         1332           38    (200, 0, 0)
kkt_10_3           13            9            1     (10, 3, 0)
kkt_30_10          40           66            2    (30, 10, 0)
kkt_50_15          65          138            4    (50, 15, 0)
kkt_100_30        130          694           18   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 154588 matrices loaded

KKT summary: 154588 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000)
  Inertia match: 150094/154481 (97.2%)
  Residual pass: 151212/154481 (97.9%)
  Worst residual: 1.57e7 (DEGENLPA_0061)

--- Sparse solver validation ---
Sparse solver: 154588/154588 total
  Inertia match vs MUMPS: 114695/154588 (74.2%)
  Residual pass: 120465/154588 (77.9%)
  Worst residual: 2.32e12 (HYDCAR20_0000)

--- Dense failure analysis (4742 failures) ---

family                    total    inertia   residual      worst_res
BATCH                      1589       1589       1588         2.34e1
SWOPF                      1185       1185       1185         1.19e1
HAHN1                       498        498         24         6.26e0
QPNBLEND                    362        362          0       7.56e-16
MSS1                        240        240          0       1.40e-15
CORE1                       141        141          0       3.56e-16
CRESC50                      97         97          0       2.08e-16
ACOPP30                      68         68         68         1.14e5
FBRAIN3LS                    59          6         57        4.11e-7
CERI651DLS                   51          3         48        1.69e-7
CERI651A                     42         42          7         8.72e0
HS46                         42          0         42        7.51e-8
PFIT4                        38         38          0       3.27e-14
DEVGLA2                      37          0         37        2.77e-6
PFIT2                        24          0         24        8.14e-6
CERI651ALS                   24          2         22        1.79e-7
PALMER1ENE                   23          0         23        1.79e-8
CERI651CLS                   23          1         22        2.78e-7
CRESC100                     19         19          0       9.89e-16
MISTAKE                      16          6         16         5.58e5
ALLINITA                     15          8         13        8.10e-4
KIRBY2                       12         12          0       1.76e-13
HATFLDFL                     12          0         12        2.49e-9
SNAKE                         9          0          9        6.99e-9
HS114                         9          7          9        7.21e-1
  ... and 52 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
DEGENLPA_0061                   35       1.57e7    (20, 15, 0)    (18, 15, 2)
MISTAKE_0101                    22       5.58e5     (9, 13, 0)     (8, 13, 1)
MISTAKE_0102                    22       4.81e5     (9, 13, 0)     (8, 13, 1)
DEGENLPB_0045                   35       2.15e5    (20, 15, 0)    (19, 15, 1)
ACOPP30_0001                   209       1.14e5   (72, 137, 0)   (71, 137, 1)
ACOPP30_0000                   209       1.08e5   (72, 137, 0)   (71, 137, 1)
ACOPP30_0002                   209       3.36e4   (72, 137, 0)   (71, 137, 1)
ACOPP30_0067                   209       6.84e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0066                   209       6.74e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0065                   209       6.64e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0064                   209       6.53e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0063                   209       6.43e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0062                   209       6.33e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0061                   209       6.22e2   (72, 137, 0)   (71, 137, 1)
ACOPP30_0060                   209       6.12e2   (72, 137, 0)   (71, 137, 1)

--- Sparse failure analysis (40210 failures) ---

family                    total    inertia   residual      worst_res
HATFLDBNE                  2504       2504       1004        5.33e-1
HS118                      2472       2471       2472         1.80e3
AVION2                     2453       2453       1330         5.58e3
PFIT4                      2278       2278       2240         5.51e1
CONCON                     2217       2217       2217         6.91e0
MCONCON                    2113       2113       2113         6.91e0
ALLINITA                   2085       2077         67         1.52e1
BATCH                      2059       2059       2059         3.97e0
HS99EXP                    1880       1880       1880         4.00e8
SWOPF                      1190       1190       1190        2.40e11
DISCS                       619        619        618         3.61e2
HIMMELBI                    605        605        601         3.96e5
HS73                        503        503        503         2.80e2
PALMER3ENE                  500        500        500        3.11e-1
PALMER7ANE                  500        500        499         2.46e3
PALMER5BNE                  500        500        500         7.11e1
HAHN1                       498        498         64        1.84e-5
NELSON                      498        498        498         1.66e0
HIMMELBFNE                  495        495        495         1.27e3
CORE1                       493        493        352         7.95e5
HS2NE                       492        492        492         2.01e0
MISRA1C                     490        490        490         1.91e0
CHWIRUT2                    485        485        484        5.51e-1
LEWISPOL                    484        484         26         3.17e3
PALMER2NE                   477        477        476        1.67e-1
  ... and 227 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
HYDCAR20_0000                  198      2.32e12    (99, 99, 0)  (23, 23, 152)
SWOPF_0005                     175      2.40e11    (83, 92, 0)   (63, 68, 44)
SWOPF_0004                     175      2.22e11    (83, 92, 0)   (63, 68, 44)
SWOPF_0006                     175      1.67e11    (83, 92, 0)   (63, 68, 44)
SWOPF_0007                     175      1.22e11    (83, 92, 0)   (63, 68, 44)
SWOPF_0003                     175      7.83e10    (83, 92, 0)   (63, 68, 44)
SWOPF_0009                     175      7.41e10    (83, 92, 0)   (63, 68, 44)
SWOPF_0008                     175      6.94e10    (83, 92, 0)   (63, 68, 44)
SWOPF_0010                     175      6.00e10    (83, 92, 0)   (64, 69, 42)
SWOPF_0011                     175      4.96e10    (83, 92, 0)   (63, 68, 44)
SWOPF_0002                     175      4.59e10    (83, 92, 0)   (63, 68, 44)
SWOPF_0012                     175      3.08e10    (83, 92, 0)   (64, 68, 43)
POWELLBS_0005                    4      2.74e10      (2, 2, 0)      (1, 2, 1)
SWOPF_0015                     175      2.67e10    (83, 92, 0)   (63, 68, 44)
SWOPF_0013                     175      2.63e10    (83, 92, 0)   (64, 69, 42)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  4667
Failed in dense only:             75
Failed in sparse only:            35543
```

## Recent Decisions
so much amalgamation under the buggy loop that the claimed column
range became `[0, n)` — trivially contiguous — producing a
degenerate configuration where the sparse path reduced to a
dense LDLᵀ wrapped in sparse plumbing. That configuration is what
produced the historical 99.8% sparse residual pass rate on the
153k–154k KKT corpus. **The 99.8% rate is obsolete and should
never be cited again.** The honest Phase 2.2.3 rate under the
default `nemin=32` is 74.2% inertia match / 77.9% residual pass,
with a worst residual of 2.32e+12 on HYDCAR20_0000. The 22-point
drop reveals the real surface area of the multi-supernode code
path and defines the correctness-closing work for Phases 2.3–2.4.

**Why the minimal fix over the SSIDS-style renumbering.** SSIDS
handles non-adjacent sibling merging by emitting a permutation
`sperm` that renumbers columns so every amalgamated supernode is
contiguous by construction (`src/core_analyse.f90:644-685`). This
is strictly better for fill and flops on arrow-like trees and
would probably close the ACOPP30 regression this session
introduced. But it is a substantially larger refactor touching
the symbolic analysis pipeline end-to-end, and shipping a
correct-but-slower supernode amalgamation today unblocks three
plateau matrices (CHWIRUT1, CRESC100, CRESC132) that now all
beat the canonical MUMPS oracle. Logged as follow-up.

**Commitment.** The README and any future user-facing documents
should cite the post-Phase-2.2.3 numbers, not the historical
99.8%. Phase 2.3 (delayed pivoting) remains on the roadmap and
is expected to help ACOPP30; the SSIDS-style renumbering is
logged as Phase 2.2.4 or as prerequisite work for Phase 2.3. No
test tolerances were loosened. All 146 non-ignored tests pass.

## Recent Tried-and-Rejected
   trace fix is more correct in absolute terms but moves feral
   away from the current oracle.

**Decision.** Revert and re-attempt after canonical Fortran MUMPS becomes
available as a second oracle (per `dev/plans/phase-1b-consensus-exit.md`).
At that point we can verify whether canonical MUMPS uses trace-based or
a00-based inertia counting on the 16 regressed matrices and reapply the
fix in the direction that the canonical solver agrees with.

**Code state.** A `KNOWN BUG` comment is left in
`src/dense/factor.rs::count_2x2_inertia` documenting the issue and
linking back here. The function signature remains unchanged so we don't
need `#[allow(clippy::too_many_arguments)]` for code that we know will
need to change again.

**Symptoms.** Inertia error pattern `(p+1, n+1, 0) → (p, n, +1)` on
matrices with zero-diagonal Hessian rows. The "lost positive" appears
as a "gained zero" in feral's output. Most visible on the ACOPP30
family (68 matrices, all with the same `(72,137,0) → (71,137,1)`
mismatch).

## Source Files
```
src/bin/bench.rs
src/dense/equilibrate.rs
src/dense/factor.rs
src/dense/matrix.rs
src/dense/mod.rs
src/dense/solve.rs
src/error.rs
src/inertia.rs
src/io/mod.rs
src/io/mtx.rs
src/io/sidecar.rs
src/lib.rs
src/numeric/factorize.rs
src/numeric/mod.rs
src/numeric/solve.rs
src/ordering/amd.rs
src/ordering/elimination_tree.rs
src/ordering/mod.rs
src/ordering/postorder.rs
src/scaling/hungarian.rs
src/scaling/mc64.rs
src/scaling/mod.rs
src/sparse/csc.rs
src/sparse/mod.rs
src/symbolic/column_counts.rs
src/symbolic/mod.rs
src/symbolic/supernode.rs
```

## Test Files
```
tests/dense_ldlt.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/mc64_end_to_end.rs
tests/mc64_regression.rs
tests/mc64_scaling.rs
tests/pivot_rejection.rs
tests/property_tests.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/stress_tests.rs
tests/threshold_consistency.rs
```
