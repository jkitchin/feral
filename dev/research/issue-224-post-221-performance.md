# Issue #224 — what is recoverable after #221 without giving up correctness

Date: 2026-10-22. Follow-up to #221/#222 (feral 0.21.0). Companion plan:
`dev/plans/issue-224-recovery.md`.

## Summary

- **The per-matrix factor cost did not regress.** On fixed matrices feral
  0.21.0 factors *faster* than 0.18.0: 0.93× on the 100 KKTs of the 0.18
  qcqp1000-2c trajectory and 0.83× on the 164 KKTs of the 0.21 trajectory,
  with identical inertia on all 264 and the expected count on every one.
- **qcqp1000-2c's 4× is two multiplicative effects, neither a kernel
  regression on a given matrix:** (a) the 0.21 solve's path visits ~70 "hard"
  late KKTs (0.6 s each, 20k delays, 6 cascade-break fronts) where the 0.18
  path visits ~10 — a trajectory effect that starts as a round-off
  difference at iter 76; (b) pounce's measured per-factorization time is
  1.8× the harness time for the same matrices under 0.21 (0.617 s vs 0.343 s
  per factor), against 1.24× under 0.18 — a pounce-side or
  integration-side cost, quantified in §3.4 below.
- **brainpc0/brainpc1 are a trajectory effect, and feral 0.21's inertia at
  the split is the defensible one.** The two solvers' KKT/solve sequences are
  identical up to one solve (brainpc1 iter 41 solve 1; brainpc0 iter 95 solve
  1). On those matrices 0.18 floors 113–670 pivots, its unrefined residual
  is 84–362, and its count moves with congruence seeds; 0.21 floors none,
  meets `max|L| = 1/u`, and counts the same on every seed. The matrices are
  numerically singular to working precision (eigvalsh band), so the count
  falls under the singular-matrix exemption. The 10–18× iteration blow-up
  is pounce's reaction (restoration: 811 of 1073 iterations on brainpc1,
  1306 of 2117 on brainpc0); factorization is 3.6 s of brainpc1's 74 s.
- What is **not** recoverable in feral without giving up correctness: 0.18's
  speed on the late qcqp KKTs came with floored pivots and an inertia count
  that is wrong on nearby matrices (tried-and-rejected 2026-10-20: lowering
  the backstop K, the 1e-10 floor); the delays themselves are the #215 exact
  factor. The levers that remain are bit-parity-safe kernel costs
  (§4).

## 1. Reproduction (this machine, macOS arm64, 14 cores, sequential runs)

pounce v0.12.0 (feral 0.18.0, `~/projects/pounce-v012/target/release/pounce`)
and pounce 0.13.0 @ 9958a24e (feral 0.21.0, `~/projects/pounce/target/release/pounce`),
`--no-sol print_timing_statistics=yes`, 0.21 runs add `solve_quality_audit=no`.
Script: scratchpad `i224/run_e2e.sh`.

| model | feral 0.18.0 | feral 0.21.0 | LinearSystemFactorization 0.18 → 0.21 |
|---|---|---|---|
| qcqp1000-2c | 26.4 s, 99 it | 103.8 s, 163 it | 25.1 s → 100.5 s (0.25 → 0.62 s/it) |
| brainpc0 | 6.6 s, 185 it (+24 re-solve) | 52.2 s, 2117 it (+24) | 3.2 s → 11.9 s (14.6 → 5.2 ms/it) |
| brainpc1 | 1.06 s, 58 it | 75.0 s, 1073 it (+52) | 0.66 s → 3.6 s (11 → 3.1 ms/it) |

brainpc1 0.21: `ComputeAcceptableTrialPoint` 69.8 s of 74.3 s (restoration
phase: 811 restoration iterations in 52 calls); brainpc0 0.21: 36.6 s of 51.9 s
(1306 restoration iterations). Per main-iteration factor cost is *lower* on
0.21 for both brainpc models.

KKT dumps: `--dump kkt:all` (qcqp) and `--dump kkt:-100` (brainpc), one
`kkt_solve_NNN.jsonl` per factorization including inertia-correction
retries, under scratchpad `i224/dump/{f018,f021}-<model>/`. The pounce
dump records carry `num_neg_evals_expected`, `num_neg_evals_actual` and
`status`.

## 2. Fixed-matrix A/B harness

`i224/ab` (scratchpad): one binary linking both crates.io releases
(`feral018 = "=0.18.0"`, `feral021 = "=0.21.0"`), pounce-feral's defaults
(`NumericParams::default()`, `bk.pivot_threshold = 1e-8`, Auto ordering and
scaling, `solve_refined_opts` with the default 10-step cap), `--reuse` keeps
one `Solver` across the sequence as pounce does.

### 2.1 qcqp1000-2c, 0.18's own trajectory (100 KKTs)

| | feral 0.18.0 | feral 0.21.0 |
|---|---|---|
| factor total | 20.27 s | 18.92 s (0.93×) |
| refined-solve total | 0.46 s | 0.53 s |
| inertia | (6007,5107,0) on all 100 | identical |
| floored pivots (sum) | 20 (8 matrices) | 4313 (11 matrices, iters 78–88) |
| nnz(L) total | 164.3 M | 179.2 M |
| Σ n_delayed_in | 94 957 | 335 654 |

### 2.2 qcqp1000-2c, 0.21's own trajectory (164 KKTs)

| | feral 0.18.0 | feral 0.21.0 |
|---|---|---|
| factor total | 67.53 s | 56.32 s (0.83×) |
| refined-solve total | 1.21 s | 3.69 s (3.05×) |
| inertia | (6007,5107,0) on all 164, = expected | identical |
| floored pivots (sum) | 311 (66 matrices) | 39 386 (77 matrices) |
| nnz(L) total | 564.3 M | 661.2 M |
| Σ n_delayed_in | 479 778 | 1 606 093 |
| matrices with t_fac > 0.5 s | 71 | 70 |

Typical hard matrix (iter 120): 0.21 0.626 s, nnz(L) 7.86 M, 642 floored,
4235 threshold-fail accepts, 6 cascade-break nodes, 20 280 delays, max|L|
3.9e11; 0.18 0.771 s, 6.50 M, 4 floored, 6239 delays. Both count 5107.

So on the matrices the 0.21 solve actually factors, 0.21 is the faster
kernel. The 0.21 pounce run's 100.5 s / 163 = 0.617 s per factorization
against the harness's 56.3 s / 164 = 0.343 s is the remaining gap (§3.4).

### 2.3 Where the two trajectories part

Iteration logs (`i224/itdiff.py`) are identical through iter 75 at printed
precision and split at iter 76 by a round-off-level step difference
(objective 7.3810516e5 vs 7.3810506e5, same inertia, no inertia correction on
either path: all 100 + 164 solves `Success` with the expected 5107). The
dumped KKTs already differ by 9e-13 at iter 0 (pounce 0.12 vs 0.13 assemble
them slightly differently), so the split is not attributable to one
factorization. After the split the 0.21 path spends iters ~78–150 in the hard
regime; the 0.18 path reaches it at iter 81 and converges at 99.

## 3. brainpc0 / brainpc1

### 3.1 The sequences are identical up to one solve

`i224/seqdiff.py` over the per-solve (expected, actual, status) triples:

- brainpc1: identical through iter 40. Iter 41, solve 1: 0.18 reports
  `Singular` with 6903 negatives, 0.21 reports `WrongInertia` with 6902.
  pounce then takes a δ_c step on 0.18's path (4 solves) and a δ_w step on
  0.21's (3 solves). The iteration lines split at iter 42 (α_pr 2.3e-2 vs
  1.0). Both matrices (f018 vs f021 dump) agree to 4.5e-6 absolute on
  entries of size 2e14.
- brainpc0: iteration lines split at iter 72 by round-off (4.0019482e-01
  vs 4.0019492e-01) with identical solve sequences through iter 94; the
  sequences split at iter 95 solve 1: 0.18 `Singular` 6901 then `Singular`
  6902; 0.21 `WrongInertia` 6901 then `Success`.

### 3.2 Fixed-matrix comparison at the split

brainpc1 iter 41 solve 1 (n = 13805), harness, cold solver:

| | feral 0.18.0 | feral 0.21.0 |
|---|---|---|
| inertia | (6903, 6902, 0) | (6903, 6902, 0) |
| floored pivots | 650 | 0 (tfa 0, cb 0) |
| nnz(L) | 381 609 | 258 146 |
| unrefined residual max|Ax−b| | 3.6e2 | 1.8e-6 |
| max|L| | — | 9.97e7 (= 1/u) |
| congruence seeds 1/4/8 | 6902/6902/6902, tiny 567/641/607, residual 2.5e6/2.2e-2/2.3e-4 | 6902/6902/6902, tiny 0/0/0 |

(pounce v0.12's own run counted 6903 on this matrix; a factor with 650
floored pivots does not give a repeatable count.)

brainpc0 iter 95 solve 1:

| | feral 0.18.0 | feral 0.21.0 |
|---|---|---|
| inertia | (6903, 6902, 0) | (6904, 6901, 0) |
| floored pivots | 113 | 0 |
| unrefined residual | 8.4e1 | 6.5e-10 |
| seeds 1/4/8 | 6902/6901/6902 (tiny 485/278/563) | 6901/6901/6901 (tiny 0/0/0; seed 4: cb 1, tfa 23) |

### 3.3 Oracle

Dense `numpy.linalg.eigvalsh` on the dumped matrices (`i224/eig224.py`,
128 s each; both matrices have ‖A‖₂ = 2.0e14, so the eigensolver's own
error is about n·eps·‖A‖ ≈ 6e2 in absolute terms):

| matrix | eigvalsh neg (strict) | smallest \|w\| | feral 0.18 | feral 0.21 |
|---|---|---|---|---|
| brainpc1 i41/s1 | 6903 | 1.8e-15, 5.0e-15, 1.4e-14, 4.5e-11, 3.7e-9 | 6902 (harness; 6903 in pounce's run) | 6902, all seeds |
| brainpc0 i95/s1 | 6901 | 8.2e-6, 2.7e-5, 6.7e-5, 1.0e-4 | 6902 (pounce run and seeds 1, 8); 6901 (seed 4) | 6901, all seeds |

brainpc1 i41 has three eigenvalues at 1e-15–1e-14 on a 2e14-norm matrix:
the count is undecidable in working precision (singular-matrix
exemption) and the two solvers differ only by one. brainpc0 i95 has no
eigenvalue closer to zero than 8e-6 but the unscaled eigvalsh cannot
resolve below ~6e2 either; a Sylvester-scaled rerun
(`eig224s.py`: D·A·D with D = 1/√(row inf-norm), same inertia) was
queued (`i224/run_eig2.sh`, output `i224/res/eig_scaled.txt`) and must be
read by the implementer before the brainpc0 verdict is final. What is
already decisive without it: 0.21's count is the same on every
congruence seed with no floored pivot and a 6.5e-10 unrefined residual,
while 0.18's moves with the seed, floors 113–563 pivots, and has an
unrefined residual of 84. 0.18's `Singular`/6902 is what pushed pounce
onto the δ_c path. Nothing here is a feral 0.21 defect.

### 3.4 pounce's per-factorization time vs the harness (qcqp)

pounce 0.13 reports 100.5 s of `LinearSystemFactorization` for 163
factorizations (0.617 s each); the harness factors the same 164 dumped
matrices with the same crates.io feral 0.21.0, same parameters, one
persistent `Solver`, in 56.3 s (0.343 s each). Under 0.18 the same ratio
is 25.1 s / 99 = 0.253 s vs 20.3 s / 100 = 0.203 s (1.24×). So about 44 s of
the 100 s is not in the factorization kernel as the harness exercises it.
Candidates, untested (the time budget ran out before the timing-sensitive
runs): (a) the thread pool pounce runs feral under (the harness's
`--serial` flag was added for this A/B and is built, not yet run); (b)
pounce's `LinearSystemFactorization` timer covering work other than
`Solver::factor` (CSC assembly, the second-opinion machinery, refinement
diagnostics); (c) the quality-ladder state (the 0.21 qcqp log reports no
escalation, so unlikely). pounce's `--json-output` report carries
`LinearSolverSummary { n_factors, total_factor_secs, total_delayed_cols,
total_n_tiny }`, which separates (a)/(b) in one run. This gap is the
single largest recoverable item for qcqp and it is on the integration
side, not in the kernel.

## 4. Where the time goes on a hard qcqp KKT (feral 0.21)

Not yet phase-timed in this session. `diag_221_rook_cost` (built,
`target/release/diag_221_rook_cost`, `FERAL_221_PHASES=1 FERAL_221_PANEL_DIAG=1`)
on `i224/dump/f021-qcqp1000-2c/iter_120/kkt_solve_001.jsonl` gives the
panel/Schur/scalar-tail split; the 2026-10-20 journal entry for the same
regime (nql180 at u = 0.01) put 70 % of dense time in the panel-bail →
flush → scalar-step cycle, and that is where the two bit-parity-safe
levers below act. On these matrices the 0.21 factor already carries fewer
multiplier-growth problems than 0.18 (max|L| ≤ 1.7e12 against 0.18's
unbounded floored columns) at 0.8× its time.

## 5. What is feral's, what is pounce's, what is not recoverable

**feral's (bit-parity-safe, inertia-neutral):**
1. The integration gap of §3.4 if it turns out to be feral's parallel
   dispatch under pounce's pool (then: a `min_parallel_flops` or
   intrafront gate change, measured with the harness `--serial` A/B).
2. Row-max caching in `rook_rescue` (`src/dense/rook.rs::sym_row_argmax`)
   — every rook iteration rescans a full symmetric row of length `nrow`;
   on the hard qcqp fronts (2048 rows, ~100–140 rook rescues per matrix,
   up to 1152 on brainpc0 i95 under 0.18) the scans repeat rows already
   scanned by `scalar_pivot_step`'s `symmetric_row_offdiag_max`. Caching
   `(gamma_r, argmax)` per row for the duration of one pivot decision
   changes no arithmetic (max and argmax are exact), so bit-parity holds
   by construction.
3. A cheap flush when a panel bails with `n_elim ≤ 1`:
   `apply_blocked_schur` packs tiles for a rank-1 update; a strided
   rank-1 loop in the same evaluation order (`d·L_i` then `·L_j`, same FMA
   setting) is bit-identical and avoids the packing. The 2026-10-20 lazy-BK
   experiment measured flushes of 1–2 pivots through the packing kernel as
   the cost that ate its gain; the committed kernel has the same flushes
   (`PANEL_PARTIAL` with `PIVOTS_INLINE` small) but the flush-only change
   was never measured on its own.
4. The 3× refined-solve cost on 0.21's hard matrices (3.69 s vs 1.21 s per
   sequence) tracks the floored-pivot count (`tiny` 150–700 per matrix)
   and refinement steps; worth measuring the step count per solve before
   touching anything.

**pounce's:** the brainpc0/1 trajectories (restoration-phase cycling after a
δ_c/δ_w choice made on a count that 0.18 got wrong or could not know) and
most of the qcqp wall time (trajectory after a round-off split at iter 76,
plus the §3.4 integration gap unless it proves to be feral's).

**Not recoverable without giving up correctness:**
- 0.18's 0.77 s vs 0.21's 0.61 s on the hard qcqp matrices is already in
  0.21's favour; there is no 0.18 speed to recover on fixed matrices.
- The ~20k delays per hard qcqp matrix are the exact factor (#215). The
  two ways to shed them — a lower delay-chain backstop K or the 1e-10
  cascade-break floor — were rejected on 2026-10-20 because they are
  inertia-wrong on column25 (27/28 and 28/28 seeds) and on qc85/q15c76.
- 0.18's brainpc counts came from 113–670 floored pivots and vary with the
  congruence seed; reproducing them would mean reintroducing #213/#215.
- An a-posteriori-pivoting panel would cut the bail cost but breaks the
  blocked/scalar bit-parity contract (`tests/blocked_ldlt.rs`,
  `tests/golden_bits.rs`), which the maintainer chose to keep.

## Files

- Harness: scratchpad `i224/ab/` (`cargo build --release`; `ab224 <018|021>
  [--reuse] [--serial] [--seed S] [--u U] files…`), results `i224/res/`,
  join script `i224/join.py`, trajectory diff `i224/itdiff.py`, solve-sequence
  diff `i224/seqdiff.py`, dump summary `i224/dumpstats.py`.
- Dumps (3.9 GB): `i224/dump/{f018,f021}-{qcqp1000-2c,brainpc0,brainpc1}/`.
- Scratchpad root:
  `/private/tmp/claude-501/-Users-jkitchin-projects-feral/a1458847-2377-4832-a0c1-096910fb944e/scratchpad/`.
