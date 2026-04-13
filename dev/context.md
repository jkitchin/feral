# FERAL Context (auto-generated)

Generated: 2026-04-13T16:51:37Z

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
c84b7c9 Phase 2.3 Step 4 fixup: revert may_delay flag flip
7fb3779 Phase 2.3 Step 4: wire may_delay through factorize_multifrontal
29ccf83 Phase 2.3 Steps 1-3: delayed-pivoting kernel plumbing
bd1c6e4 Phase 2.3 setup: research note + implementation plan
b87bff1 Add InfNorm scaling strategy; make it the new default
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s

     Running tests/threshold_consistency.rs (target/debug/deps/threshold_consistency-c660296127e8afca)

running 6 tests
test polak6_0021_residual_after_threshold_fix ... ignored
test factor_inertia_force_accept_implies_solve_skip_invariant ... ok
test factors_carry_zero_tol_from_params ... ok
test refinement_does_not_amplify_error_on_rank_deficient_matrix ... ok
test dense_solve_skips_zero_pivots_rank_deficient ... ok
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
spd_10             10           53            1     (10, 0, 0)
spd_50             50           46            5     (50, 0, 0)
spd_100           100          144            9    (100, 0, 0)
spd_200           200          747           33    (200, 0, 0)
kkt_10_3           13            6            1     (10, 3, 0)
kkt_30_10          40           37            2    (30, 10, 0)
kkt_50_15          65           82            2    (50, 15, 0)
kkt_100_30        130          293           10   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 154588 matrices loaded

KKT summary: 154588 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000)
  Inertia match: 152979/154481 (99.0%)
  Residual pass: 154051/154481 (99.7%)
  Worst residual: 3.99e-2 (ACOPP30_0002)

--- Sparse solver validation ---
Sparse solver: 154588/154588 total
  Inertia match vs MUMPS: 149820/154588 (96.9%)
  Residual pass: 152453/154588 (98.6%)
  Worst residual: 2.31e11 (SWOPF_0827)

--- Dense failure analysis (1928 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       1.76e-12
QPNBLEND                    362        362          0       7.56e-16
MSS1                        240        240          0       1.40e-15
CORE1                       141        141          0       3.56e-16
CRESC50                      97         97          0       2.08e-16
ACOPP30                      68          0         68        3.99e-2
FBRAIN3LS                    59          6         57        4.11e-7
CERI651DLS                   51          3         48        1.69e-7
HS46                         42          0         42        7.51e-8
PFIT4                        38         38          0       3.27e-14
CERI651A                     37         37          0       9.15e-14
DEVGLA2                      37          0         37        2.77e-6
CERI651ALS                   24          2         22        1.79e-7
PFIT2                        24          0         24        8.14e-6
PALMER1ENE                   23          0         23        1.79e-8
CERI651CLS                   23          1         22        2.78e-7
CRESC100                     19         19          0       9.89e-16
MISTAKE                      12          0         12        1.83e-6
HATFLDFL                     12          0         12        2.49e-9
KIRBY2                       12         12          0       1.76e-13
ALLINITA                      9          2          7        5.43e-7
SNAKE                         9          0          9        6.99e-9
DISCS                         8          8          0       6.31e-16
BENNETT5                      8          8          0       1.70e-13
DJTL                          7          0          7        1.01e-6
  ... and 44 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
ACOPP30_0002                   209      3.99e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0037                   209      2.92e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0026                   209      2.80e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0018                   209      2.76e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0045                   209      2.76e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0000                   209      2.74e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0012                   209      2.69e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0065                   209      2.64e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0046                   209      2.64e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0036                   209      2.63e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0051                   209      2.58e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0024                   209      2.54e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0057                   209      2.53e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0055                   209      2.49e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0013                   209      2.46e-2   (72, 137, 0)   (72, 137, 0)

--- Sparse failure analysis (5197 failures) ---

family                    total    inertia   residual      worst_res
HATFLDBNE                  1500       1500          0       4.02e-10
SWOPF                      1190       1190       1190        2.31e11
HAHN1                       498        498          0       3.23e-13
QPNBLEND                    362        362          0       5.01e-16
MSS1                        240        240          0       1.38e-15
CORE1                       141        141          0       3.52e-16
HATFLDF                     126        126        126         1.55e9
CRESC50                      97         97          0       1.77e-16
ACOPP30                      68         68         68         5.14e6
FBRAIN3LS                    61          3         59        2.79e-7
HAIFAM                       59          3         59         1.19e0
CERI651DLS                   51          3         48        1.94e-7
HS46                         44          0         44        9.97e-8
PFIT4                        38         38          0       5.84e-14
BATCH                        37         36          1        2.64e-7
CERI651A                     37         37          0        2.34e-8
POLAK5                       28         28         28        1.57e-1
DEVGLA2                      26          0         26        1.58e-6
CERI651CLS                   25          1         24        3.20e-7
PFIT2                        24          0         24        1.22e-5
CERI651ALS                   24          2         22        1.45e-7
PALMER1ENE                   23          0         23        1.79e-8
DENSCHNDNE                   23         23         23         5.91e9
CONGIGMZ                     19         19         17         5.28e6
SPIRAL                       19         19         19         6.73e3
  ... and 109 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
SWOPF_0827                     175      2.31e11    (83, 92, 0)   (68, 90, 17)
SWOPF_0941                     175      2.28e11    (83, 92, 0)   (68, 90, 17)
SWOPF_0880                     175      5.28e10    (83, 92, 0)   (66, 92, 17)
SWOPF_0685                     175      5.14e10    (83, 92, 0)   (66, 92, 17)
SWOPF_1044                     175      5.13e10    (83, 92, 0)   (66, 92, 17)
SWOPF_1018                     175      5.04e10    (83, 92, 0)   (66, 92, 17)
SWOPF_0682                     175      5.01e10    (83, 92, 0)   (66, 92, 17)
SWOPF_0867                     175      5.00e10    (83, 92, 0)   (66, 92, 17)
SWOPF_0838                     175      5.00e10    (83, 92, 0)   (66, 92, 17)
SWOPF_0785                     175      5.00e10    (83, 92, 0)   (66, 92, 17)
SWOPF_0814                     175      5.00e10    (83, 92, 0)   (66, 92, 17)
SWOPF_0817                     175      4.99e10    (83, 92, 0)   (66, 92, 17)
SWOPF_0759                     175      4.99e10    (83, 92, 0)   (66, 92, 17)
SWOPF_0728                     175      4.99e10    (83, 92, 0)   (66, 92, 17)
SWOPF_0673                     175      4.99e10    (83, 92, 0)   (66, 92, 17)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  1869
Failed in dense only:             59
Failed in sparse only:            3328
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
src/scaling/infnorm.rs
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
tests/delayed_pivoting.rs
tests/dense_ldlt.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/mc64_end_to_end.rs
tests/mc64_scaling.rs
tests/parity.rs
tests/pivot_rejection.rs
tests/property_tests.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/stress_tests.rs
tests/threshold_consistency.rs
```
