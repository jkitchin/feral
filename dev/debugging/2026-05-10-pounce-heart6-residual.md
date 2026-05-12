# Pounce HEART6: large residuals + silent wrong-inertia on ill-conditioned KKT

**Date:** 2026-05-10
**FERAL HEAD:** (whatever HEAD is — pull as of 2026-05-10)
**Reporter:** pounce IPM port (jkitchin@andrew.cmu.edu)
**Severity:** correctness regression on ill-conditioned indefinite systems
**Status:** filed; pounce-side workaround landing in `pounce-feral` (post-solve residual check + `IncreaseQuality` escalation)

## TL;DR

While running the pounce IPM on the CUTEst HEART6 problem (n=6, m=6
equality NLP), several augmented KKT systems FERAL produces have
solution residuals 8–14 orders of magnitude larger than MA57 produces
on the **same matrix and RHS**. In some cases FERAL also reports
`Success` with the **wrong** negative-eigenvalue count, which causes
the IPM's perturbation handler to stop escalating regularization and
charge ahead with a Newton direction whose residual is `~1e13`.

This is observed across many of the 41 problems where pounce+FERAL
regresses against pounce+MA57 on the CUTEst sweep. HEART6 is the
clearest small reproducer.

## Reproduction

Three captured systems are checked in at
`data/matrices/kkt/HEART6_pounce_diag/`:

| File | n | nnz | cond(A) | LAPACK true neg | Notes |
|------|---|-----|---------|-----------------|-------|
| `heart6_iter_a.mtx` (+ `_rhs.mtx`) | 12 | 64 | ~1.8e21 | 6 | FERAL `Success`, residual ~1e11; MA57 ~4e-4 |
| `heart6_iter_b.mtx` (+ `_rhs.mtx`) | 12 | 64 | ~4.3e24 | 8 | FERAL reports `Success` **with neg=6** (wrong!); MA57 reports `WrongInertia, neg=8` |
| `heart6_iter_c.mtx` (+ `_rhs.mtx`) | 12 | 70 | ~1.7e3  | 6 | FERAL residual 1.2e4; MA57 1e-4 — **at cond=1.7e3** |

Files use Matrix Market `symmetric` format (1-based, lower triangle,
duplicates summed). RHS is a column vector.

A minimal reproduction example is straightforward — use `read_mtx`
to load each matrix + RHS, then `factor` (pass `Some(Inertia { positive, negative, zero })`
matching LAPACK's true count), then `solve_refined`, then compute
`‖A x − b‖∞`. Compare to the values in the table.

The pounce-side capture pipeline that produced the JSONL these MTX
files were extracted from:

1. Build pounce with both backends:
   `cargo build --release -p pounce-cutest --features ma57 --bin cutest_suite --bin kkt_compare`
2. Capture all 326 augmented systems from a HEART6 solve:
   `POUNCE_DUMP_KKT=/tmp/heart6_kkt.jsonl cutest_suite --single HEART6 --solver pounce`
3. Compare per-row across FERAL, MA57, dense LAPACK `dsyev`:
   `kkt_compare /tmp/heart6_kkt.jsonl`

The `kkt_compare` binary reports `(LAPACK_neg|min|λ|, cond)`,
`(FERAL_status, FERAL_neg, FERAL_residual)`, `(MA57_status, MA57_neg,
MA57_residual)` for each system. Selected rows captured below.

## Observed (from `kkt_compare`)

| row | cond(A)  | LAPACK\_neg | FERAL                | MA57                 |
|----:|---------:|------------:|----------------------|----------------------|
|  15 | 1.8e+21  | 6           | `Success`/6/**9.4e+10** | `Success`/6/4.4e-04  |
|  27 | 1.8e+21  | 6           | `Success`/6/**1.0e+11** | `Success`/6/5.6e-04  |
| 280 | 9.8e+23  | 6           | `Success`/6/**8.2e+12** | `Success`/6/2.0e-03  |
| 290 | 1.5e+24  | 6           | `Success`/6/**2.6e+12** | `Success`/6/1.1e-03  |
| 304 | 4.3e+24  | **8**       | **`Success`/6/1.3e+14** | `WrongInertia`/8/4.6e+25 |
| 312 | 2.9e+24  | **8**       | **`Success`/6/9.3e+14** | `WrongInertia`/8/4.4e+25 |
| 315 | 1.2e+25  | 6           | `Success`/6/**4.5e+13** | `Success`/6/3.0e-02  |
| 320 | 1.7e+03  | 6           | `Success`/6/**1.2e+04** | `Success`/6/1.0e-04  |

(Format `Status / num_neg_evals / ‖Ax−b‖∞`.)

Bolded values are the discrepancies. Rows 304 and 312 are the
correctness bug — FERAL declares `Success` and a 6-negative inertia
on a matrix that LAPACK's `dsyev` shows has 8 negative eigenvalues.
Rows 15, 27, 280, 290, 315 are the residual-quality issue:
agreement on inertia, agreement on `Success`, but the solution is
8–14 orders of magnitude off. Row 320 shows the residual gap
appears even at modest condition (cond=1.7e3, residual gap of 8
orders of magnitude).

## Hypothesis (best guess from the outside)

Two distinct issues are in play.

1. **Residual quality on ill-conditioned indefinite matrices.** MA57
   does iterative refinement internally on its solve path, which
   buys back several digits when the condition number is high.
   FERAL's `solve` (vs. `solve_refined`) does not, and even
   `solve_refined`'s FERAL fix-up may bottom out at the factorization's
   raw quality. The pounce wrapper currently calls FERAL's
   non-refined solve path.

2. **Silent wrong-inertia at extreme conditioning (rows 304, 312).**
   FERAL reports `Success` with `num_negative_eigenvalues() = 6` on
   matrices whose true negative count is 8. This is a real
   correctness bug: a downstream IPM consumer (pounce, IPOPT, etc.)
   that uses inertia to drive perturbation correction will
   under-regularize and produce broken Newton steps. MA57 catches
   the wrong-inertia case on the same matrix.

## What pounce does in the meantime

`pounce-feral` (the wrapper exposing FERAL through pounce's
`SparseSymLinearSolverInterface`) is being patched to:

- post-compute `‖A x − b‖∞ / max(‖b‖∞, 1)` after every successful
  FERAL solve;
- if it exceeds a threshold, internally call
  `feral::Solver::increase_quality()` and return `CallAgain` so
  pounce's `TSymLinearSolver` re-fills values and re-factors;
- if quality is already at `Exhausted`, return `Singular` so
  pounce's perturbation handler escalates regularization and tries
  again.

This is a pounce-side workaround. The underlying FERAL behavior is
the bug — pounce can paper over it for now, but a downstream user
who calls FERAL directly will get the same broken solutions.

## Suggested fix on the FERAL side

1. **Inertia robustness at extreme cond:** when the smallest
   accepted pivot is below some `eps × ‖A‖∞` threshold, return
   `WrongInertia`/`Singular` rather than `Success` even if the raw
   factor completed. Compare against MA57's behavior on rows 304
   and 312 — it gets this right.
2. **Default residual check:** make `solve_refined` (or the
   `factor`+`solve` chain) include at least a one-step iterative
   refinement, and refuse to return `Success` if the post-refinement
   relative residual is above (say) `sqrt(eps)`.

## Cross-links

- pounce side: `crates/pounce-feral/src/lib.rs` (the wrapper getting
  the residual-check workaround)
- pounce diagnostic tool: `benchmarks/cutest/src/bin/kkt_compare.rs`
  (the comparison harness; reusable on any captured KKT JSONL)
- pounce dump hook: `crates/pounce-algorithm/src/kkt/std_aug_system_solver.rs`
  (the `POUNCE_DUMP_KKT` env-var-controlled dump)

## Aggregate impact

On the 727-problem CUTEst sweep, switching pounce from MA57 to
FERAL costs ~5 percentage points solve rate (562 → 523 successes).
41 problems regress. Bucketed by pounce status:

- 15× `Error_In_Step_Computation`
- 12× `Maximum_Iterations_Exceeded`
- 10× `Restoration_Failed`
- 2× `Infeasible_Problem_Detected`
- 2× `Search_Direction_Becomes_Too_Small`
- 1× `Timeout`

The HEART6 trace bucketed as `Restoration_Failed`. The diagnosis
above is consistent with the other restoration failures (resto-NLP
KKT systems push conditioning into 1e15+ territory), and is likely
the dominant cause of the regression as a whole, but only HEART6
has been instrumented end-to-end so far.
