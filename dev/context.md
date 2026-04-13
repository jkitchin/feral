# FERAL Context (auto-generated)

Generated: 2026-04-13T22:16:21Z

## Latest Session
File: dev/sessions/2026-04-13-03.md
```
# Session 2026-04-13-03 — Phase 2.3 Steps 5+6: parent-side delay assembly + solve

## Goal

Land Phase 2.3 Step 5 (parent-side delay assembly) and Step 6
(solve-side nelim fix) per `dev/plans/phase-2.3-delayed-pivoting.md`.
Steps 1-4 (kernel plumbing, `PivotOutcome::Delayed`, `n_delayed`
field, `NodeFactors.n_delayed_in`) landed in session 2026-04-13-02
(commits `bd1c6e4` → `bd5e0e2`). That session explicitly deferred
the `may_delay = !is_root` flag flip to Step 5 because the 2×2
Duff-Reid det-floor rejection fires independently of
`pivot_threshold` and would otherwise trip the
`debug_assert(n_delayed == 0)` trip-wire on SWOPF / HAIFAM et al.

Target for this session:

- `build_row_indices` learns to expand the fully-summed column count
  by `sum(child.n_delayed)` and place each child's delayed columns
  in the parent's fully-summed region.
- Contrib store site unpermutes through `ff.perm` to recover each
  delayed column's global index.
- `is_root` is computed once at the top of the loop, and the
  factor_frontal call uses `may_delay = !is_root[snode_idx]`.
- `src/numeric/solve.rs` switches from `ff.ncol` to `ff.nelim` in
  all three phases so the solve bounds match `L.cols()`.
- Two new integration tests (plan Steps 1.4 and 1.5) pin the
  parent-ward delay propagation and an end-to-end inertia match
  against a dense LDLᵀ oracle.

## Accomplished

**Phase 2.3 Steps 5+6 landed atomically** (one feature commit for
the kernel+assembly+solve changes; a second commit for the integration
tests). No regressions on any existing test; massive improvement in
the sparse residual tail.

### Code changes

- `src/numeric/factorize.rs`:
  - Added `is_root: Vec<bool>` computed once at the top of
    `factorize_multifrontal` by walking
    `symbolic.supernodes[*].children`. Handles disconnected matrices
    (multi-root forests) uniformly.
  - Renamed the loop-local `ncol` binding to `own_ncol = snode.ncol()`
    to distinguish the supernode's native column count from the
    expanded fully-summed count after absorbing delayed children.
  - Computed `n_delayed_in` and `expanded_ncol = own_ncol + n_delayed_in`
    from the children's live `ContribBlock`s before the row-indices
    build.
  - Step 1 assembly still iterates `row_indices[..own_ncol]` — only
```

## Git Status
```
a630977 Phase 2.3 sign-preservation fix: preserve pivot sign at root fallback
6245952 Phase 2.3 Step 7: restore pivot_threshold = 0.01 for sparse callers
28ff3b1 Session 2026-04-13-03 checkpoint: Phase 2.3 Steps 5+6 landed
0364e6d Phase 2.3 Steps 5+6: parent-side delay assembly + solve nelim fix
bd5e0e2 Session 2026-04-13-02 checkpoint: Phase 2.3 Steps 1-4 landed
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

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
spd_10             10           19            0     (10, 0, 0)
spd_50             50           20            3     (50, 0, 0)
spd_100           100           76            5    (100, 0, 0)
spd_200           200          393           16    (200, 0, 0)
kkt_10_3           13            2            0     (10, 3, 0)
kkt_30_10          40           20            1    (30, 10, 0)
kkt_50_15          65           49            2    (50, 15, 0)
kkt_100_30        130          199            7   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 154588 matrices loaded

KKT summary: 154588 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000)
  Inertia match: 152979/154481 (99.0%)
  Residual pass: 154051/154481 (99.7%)
  Worst residual: 3.99e-2 (ACOPP30_0002)

--- Sparse solver validation ---
Sparse solver: 154588/154588 total
  Inertia match vs MUMPS: 153009/154588 (99.0%)
  Residual pass: 154237/154588 (99.8%)
  Worst residual: 3.22e-4 (ERRINBAR_0824)

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
DEVGLA2                      37          0         37        2.77e-6
CERI651A                     37         37          0       9.15e-14
CERI651ALS                   24          2         22        1.79e-7
PFIT2                        24          0         24        8.14e-6
CERI651CLS                   23          1         22        2.78e-7
PALMER1ENE                   23          0         23        1.79e-8
CRESC100                     19         19          0       9.89e-16
HATFLDFL                     12          0         12        2.49e-9
MISTAKE                      12          0         12        1.83e-6
KIRBY2                       12         12          0       1.76e-13
ALLINITA                      9          2          7        5.43e-7
SNAKE                         9          0          9        6.99e-9
BENNETT5                      8          8          0       1.70e-13
DISCS                         8          8          0       6.31e-16
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

--- Sparse failure analysis (1929 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       8.06e-13
QPNBLEND                    362        362          0       6.78e-16
MSS1                        240        240          0       1.32e-15
CORE1                       141        141          0       3.52e-16
CRESC50                      97         97          0       1.77e-16
ACOPP30                      67         67          0       2.80e-14
FBRAIN3LS                    61          3         59        2.79e-7
CERI651DLS                   51          3         48        1.94e-7
HS46                         40          0         40        7.91e-8
PFIT4                        38         38          0       5.84e-14
CERI651A                     37         37          0       1.01e-13
DEVGLA2                      26          0         26        1.58e-6
CERI651CLS                   25          1         24        3.20e-7
CERI651ALS                   24          2         22        1.45e-7
PFIT2                        24          0         24        8.16e-6
PALMER1ENE                   23          0         23        1.79e-8
CRESC100                     19         19          0       8.49e-16
SNAKE                        12          0         12        3.34e-9
HATFLDFL                     12          0         12        2.97e-9
KIRBY2                       12         12          0       1.68e-13
MISTAKE                      11          0         11        1.50e-6
ALLINITA                     10          2          8        9.30e-7
VESUVIO                      10         10          0       1.84e-13
DISCS                         8          8          0       7.33e-16
BENNETT5                      8          8          0       8.69e-14
  ... and 48 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
ERRINBAR_0824                   27      3.22e-4     (18, 9, 0)     (18, 9, 0)
PFIT2_0590                       6      8.16e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0588                       6      8.16e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0589                       6      8.16e-6      (3, 3, 0)      (3, 3, 0)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)
PFIT2_0341                       6      7.17e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0329                       6      4.07e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0327                       6      4.07e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0328                       6      4.07e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0547                       6      4.06e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0545                       6      4.06e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0546                       6      4.06e-6      (3, 3, 0)      (3, 3, 0)
FLETCHER_0002                   16      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0300                       6      3.55e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0390                       6      2.42e-6      (3, 3, 0)      (3, 3, 0)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  1865
Failed in dense only:             63
Failed in sparse only:            64
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
