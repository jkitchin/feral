# FERAL Context (auto-generated)

Generated: 2026-04-14T02:20:16Z

## Latest Session
File: dev/sessions/2026-04-13-05.md
```
# Session 2026-04-13-05 — Dense ACOPP30 fix attempt (rejected)

## Goal

Continue Task #19 from the 2026-04-13-04 checkpoint: close the dense
ACOPP30 residual gap. 67 ACOPP30 variants produce inertia `(72, 137, 0)`
with residual ~2.8e-2 on the dense path while sparse (and MUMPS/SSIDS)
produce `(71, 137, 1)` with residual ~1e-14. The goal was to land a
dense factor.rs fix, commit the in-flight triage harness and research
note, and re-validate the full bench.

## Accomplished

### Committed in-flight harness/research (c55bacf, 555b579)

- `c55bacf` — Shared bench-failure triage tooling + research note.
  Extends `src/bin/bench.rs::print_cross_comparison` to join dense and
  sparse failure records by name and bucket the intersection by failure
  mode (inertia/residual/mixed), size class, top families, and top-15
  worst shared residuals. Adds `dev/research/shared-failure-triage.md`
  which documents the 1809 shared dense+sparse failures in the full KKT
  bench — headline: 1499/1809 (83%) are corpus data-quality artifacts
  where the sidecar disagrees with both MUMPS 5.8.2 and SPRAL SSIDS.
- `555b579` — Dense ACOPP30 triage example.
  `examples/triage_dense_acopp30.rs` runs three ACOPP30 cases under four
  parameter configs (A/A' threshold=0 plain/refined, B/C threshold=0.01
  plain/refined) and prints inertia + relative residual + ||b|| alongside
  the MUMPS oracle residual from `.mumps.json`.

Both commits caught `cargo fmt --check` failures in pre-commit and were
re-committed after running `cargo fmt`. No test or clippy warnings.

### Dense factor.rs root-cause tracing

Added `FERAL_DENSE_TRACE=1` env-gated prints to `src/dense/factor.rs`
and ran against ACOPP30_0026. Smoking gun at k=58: the 2×2 block

```
[[ 0       , -4.16e-15 ],
 [ -4.16e-15, -6.08e-9 ]]
```

has `|det| = 1.73e-29`, which passes `count_2x2_inertia`'s `eps²` floor
(4.93e-32) by 350×. At `u = params.pivot_threshold = 0.0` (the dense
default), the Duff-Reid growth bound `(|a22|*rmax + |amax|*tmax)*u ≤ |det|`
collapses to `0 ≤ |det|` and is trivially satisfied. L21 = A21·inv(D)
then scales by ≈10²⁹ and destroys the trailing submatrix. Residual 2.7e-2.

Verified via `grep` that `factor()` is only used by `numeric/factorize.rs`
in a test; the production sparse path uses `factor_frontal()`, so fixing
```

## Git Status
```
555b579 Add dense ACOPP30 triage example for task #19
c55bacf Shared bench-failure triage tooling + research note
ce09aa6 Phase 2.3 Step 9: delayed pivoting validation report
8f3fce0 Session 2026-04-13-04 addendum: refinement-termination fix
ed07ee3 Phase 2.3 refinement-termination fix: max_steps 3->10 + residual stop
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
spd_10             10           31            0     (10, 0, 0)
spd_50             50           21            2     (50, 0, 0)
spd_100           100           77            6    (100, 0, 0)
spd_200           200          399           16    (200, 0, 0)
kkt_10_3           13            2            0     (10, 3, 0)
kkt_30_10          40           19            1    (30, 10, 0)
kkt_50_15          65           47            2    (50, 15, 0)
kkt_100_30        130          203            7   (100, 30, 0)

8 matrices benchmarked

Loading KKT matrices from data/matrices/kkt ... 154588 matrices loaded

KKT summary: 154588 matrices (154481 dense-eligible n <= 1000, 107 skipped n > 1000)
  Inertia match: 152979/154481 (99.0%)
  Residual pass: 154141/154481 (99.8%)
  Worst residual: 2.80e-2 (ACOPP30_0026)

--- Sparse solver validation ---
Sparse solver: 154588/154588 total
  Inertia match vs MUMPS: 153009/154588 (99.0%)
  Residual pass: 154329/154588 (99.8%)
  Worst residual: 2.50e-4 (ERRINBAR_0824)

--- Dense failure analysis (1838 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.71e-13
QPNBLEND                    362        362          0       2.78e-15
MSS1                        240        240          0       2.78e-15
CORE1                       141        141          0       1.07e-15
CRESC50                      97         97          0       3.50e-15
ACOPP30                      68          0         68        2.80e-2
FBRAIN3LS                    50          6         48        2.82e-7
CERI651DLS                   42          3         39        7.06e-8
PFIT4                        38         38          0       9.22e-15
CERI651A                     37         37          0       7.69e-14
HS46                         27          0         27        7.51e-8
PFIT2                        23          0         23        5.39e-6
CERI651CLS                   21          1         20        2.06e-7
CRESC100                     19         19          0       4.65e-15
PALMER1ENE                   17          0         17        1.22e-8
CERI651ALS                   17          2         15        4.31e-8
DEVGLA2                      15          0         15        1.50e-7
KIRBY2                       12         12          0       1.28e-13
MISTAKE                      10          0         10        1.33e-6
ALLINITA                      9          2          7        5.43e-7
DISCS                         8          8          0       1.35e-15
BENNETT5                      8          8          0       1.29e-13
DJTL                          7          0          7        5.33e-7
SNAKE                         6          0          6        1.83e-9
LSC2LS                        5          0          5        1.95e-8
  ... and 45 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
ACOPP30_0026                   209      2.80e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0018                   209      2.76e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0000                   209      2.74e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0037                   209      2.69e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0012                   209      2.69e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0046                   209      2.64e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0036                   209      2.63e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0051                   209      2.58e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0002                   209      2.55e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0024                   209      2.54e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0057                   209      2.53e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0055                   209      2.49e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0013                   209      2.46e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0045                   209      2.45e-2   (72, 137, 0)   (72, 137, 0)
ACOPP30_0010                   209      2.43e-2   (72, 137, 0)   (72, 137, 0)

--- Sparse failure analysis (1837 failures) ---

family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.22e-13
QPNBLEND                    362        362          0       2.78e-15
MSS1                        240        240          0       1.68e-15
CORE1                       141        141          0       8.83e-16
CRESC50                      97         97          0       5.12e-16
ACOPP30                      67         67          0       1.63e-14
FBRAIN3LS                    52          3         50        2.79e-7
CERI651DLS                   39          3         36        1.93e-7
PFIT4                        38         38          0       1.69e-14
CERI651A                     37         37          0       7.97e-14
HS46                         29          0         29        3.56e-8
PFIT2                        23          0         23        2.42e-6
CERI651CLS                   21          1         20        2.53e-7
CRESC100                     19         19          0       2.40e-15
PALMER1ENE                   16          0         16        1.22e-8
DEVGLA2                      15          0         15        7.78e-7
CERI651ALS                   15          2         13        1.28e-7
KIRBY2                       12         12          0       1.52e-13
VESUVIO                      10         10          0       1.40e-13
MISTAKE                      10          0         10        1.17e-6
ALLINITA                      9          2          7        4.84e-7
BENNETT5                      8          8          0       8.69e-14
DISCS                         8          8          0       2.09e-15
DJTL                          7          0          7        5.33e-7
SNAKE                         5          0          5        2.42e-9
  ... and 44 more families

Top 15 worst residuals:
name                             n     residual       expected         actual
ERRINBAR_0824                   27      2.50e-4     (18, 9, 0)     (18, 9, 0)
PRICE4_0002                      2      7.74e-6      (2, 0, 0)      (2, 0, 0)
FLETCHER_0002                   16      3.63e-6     (12, 4, 0)     (12, 4, 0)
PFIT2_0390                       6      2.42e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0591                       6      1.70e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0329                       6      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0327                       6      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0328                       6      1.36e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0547                       6      1.35e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0545                       6      1.35e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0546                       6      1.35e-6      (3, 3, 0)      (3, 3, 0)
PFIT2_0248                       6      1.22e-6      (3, 3, 0)      (3, 3, 0)
MISTAKE_0100                    22      1.17e-6     (9, 13, 0)     (9, 13, 0)
TRO3X3_0637                     43      9.18e-7    (30, 13, 0)    (30, 13, 0)
DEVGLA2_0417                     5      7.78e-7      (5, 0, 0)      (5, 0, 0)

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  1809
Failed in dense only:             29
Failed in sparse only:            28

Shared failure mode breakdown:
  Inertia mismatch on BOTH paths:          1499
  Residual-only fail on BOTH paths:         240
  Mixed (one inertia, other residual):       70

Shared failure size class breakdown:
  n <=  100:     315
  n <= 1000:    1494
  n >  1000:       0

Top 25 families in shared failures:
family                    total    inertia   residual      worst_res
HAHN1                       498        498          0       2.71e-13
QPNBLEND                    362        362          0       2.78e-15
MSS1                        240        240          0       2.78e-15
CORE1                       141        141          0       1.07e-15
CRESC50                      97         97          0       3.50e-15
ACOPP30                      67          0          0        2.80e-2
FBRAIN3LS                    48          3         42        2.82e-7
CERI651DLS                   38          3         35        1.93e-7
PFIT4                        38         38          0       1.69e-14
CERI651A                     37         37          0       7.97e-14
HS46                         24          0         24        7.51e-8
PFIT2                        22          0         22        5.39e-6
CERI651CLS                   21          1         20        2.53e-7
CRESC100                     19         19          0       4.65e-15
PALMER1ENE                   16          0         16        1.22e-8
DEVGLA2                      15          0         15        7.78e-7
CERI651ALS                   14          2         12        1.28e-7
KIRBY2                       12         12          0       1.52e-13
MISTAKE                      10          0         10        1.33e-6
ALLINITA                      9          2          7        5.43e-7
BENNETT5                      8          8          0       1.29e-13
DISCS                         8          8          0       2.09e-15
DJTL                          7          0          7        5.33e-7
LSC2LS                        4          0          4        1.95e-8
HS118                         3          0          3        9.68e-8
  ... and 40 more families

Top 15 worst shared residuals:
name                             n    dense_res   sparse_res       expected     actual(sp)
ACOPP30_0026                   209      2.80e-2     8.64e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0018                   209      2.76e-2     6.75e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0000                   209      2.74e-2     4.27e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0037                   209      2.69e-2     6.74e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0012                   209      2.69e-2     8.34e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0046                   209      2.64e-2     7.82e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0036                   209      2.63e-2     1.02e-14   (72, 137, 0)   (71, 137, 1)
ACOPP30_0051                   209      2.58e-2     8.78e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0002                   209      2.55e-2     1.63e-14   (72, 137, 0)   (71, 137, 1)
ACOPP30_0024                   209      2.54e-2     7.73e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0057                   209      2.53e-2     8.20e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0055                   209      2.49e-2     1.07e-14   (72, 137, 0)   (71, 137, 1)
ACOPP30_0013                   209      2.46e-2     7.91e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0045                   209      2.45e-2     9.38e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0010                   209      2.43e-2     9.45e-15   (72, 137, 0)   (71, 137, 1)
```

## Recent Decisions
is counted as a zero pivot. The 2×2 fallback routes through the
same path.

**Why.** Converting a small-but-clearly-nonzero pivot into a
zero loses inertia information and produces residuals that
iterative refinement cannot recover, because the pivot is driven
to exactly 0 instead of being preserved with its noisy-but-
nonzero value. This is exactly the DEGENLPA_0065 failure mode:
the reference reports `(20, 15, 0)` and feral reported
`(20, 14, 1)` with a 7.06e2 residual. MUMPS always reports
`n0 = 0` in the default configuration (INFOG(28) is only
computed when ICNTL(24)=1), so the reference oracle never
reports zero pivots — the comparison is partly a measurement
artifact on top of the real sign-loss bug. SSIDS handles the
same case by breaking at the root and leaving the pivot
un-eliminated (the outer multifrontal driver reassembles it);
sign preservation is a strictly smaller change that captures
the correctness gain without touching the root-break logic.

**Evidence.** Parity 14/28 → 22/28 (flipped CERI651A×3,
DEGENLPA_0065, DEGENLPB_0045/0046/0047, PALMER2ANE_0000).
Sparse worst residual 7.06e2 → 3.22e-4 (six orders of
magnitude). Full measurements in
`dev/sessions/2026-04-13-04.md`. No test tolerances were
loosened. The `factor_frontal_root_force_accepts_without_delay`
unit test was updated to use `d = 0` exactly (matching the
absolute-zero branch), and
`factor_frontal_root_accepts_small_pivot_with_sign` was added to
cover the new sign-preserving branch with a clearly-negative
pivot.

## Recent Tried-and-Rejected
  sqrt(eps) threshold.

The triage harness (`examples/triage_dense_acopp30.rs` — committed
as 555b579) and bench cross-comparison metrics (committed as c55bacf)
remain valid infrastructure for the next attempt.

**Evidence.**
- `cargo run --release --example triage_dense_acopp30` after both
  fixes: ACOPP30_{0026,0018,0000} all produce residuals 1e-13..1e-14.
- `cargo run --release --bin bench` after both fixes:
  - Dense inertia match 146037/154481 (94.5%)
  - Dense residual pass 149390/154481 (96.7%)
  - Dense worst residual 1.85e0 on MISTAKE_0101 (expected `(9,13,0)`,
    got `(8,13,1)`)
  - 8836 total dense failures vs baseline 1838
- `cargo run --release --example triage_dense_acopp30` after revert:
  back to baseline 2.8e-2 with inertia `(72,137,0)`.

**Code state.** `src/dense/factor.rs` fully reverted to HEAD
(555b579). The attempted fix is not present in the tree.

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

(truncated from      384 lines to 350 line budget)
