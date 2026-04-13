# Phase 2.2.2 — Scaling-Aware Pivot Rejection Validation Sweep

**Date:** 2026-04-12
**Head commit under test:** `09955c2` (Phase 2.2.2 Steps 5-6 complete)
**Status:** OUTCOME C — ACOPP30 regression recovered (47 orders),
remaining 3 regression targets unchanged. MC64 no longer blocks
Phase 2.2; full gap closure is deferred to Phase 2.3 (delayed
pivoting).

## Summary

Phase 2.2.2 added `BunchKaufmanParams::pivot_threshold` and a
column-relative 1×1 rejection clause plus the Duff-Reid 2×2 growth
bound, opted in at `u = 0.01` by the MC64 callers (`ldlt_params`
in `tests/mc64_regression.rs` and `params_kkt` in
`src/bin/bench.rs`).

**The big win:** the ACOPP30_0000 `2.27e+46` regression introduced
in Phase 2.2.1 is gone. Residual drops to `1.076e-1` — ~17 orders
*better* than the pre-MC64 Identity baseline (`2.84e+16`) and 47
orders better than Post-2.2.1.

**The honest miss:** the other three regression matrices
(CRESC132, CHWIRUT1, CRESC100) show essentially no change in
residual. Their inertia is already exact (CHWIRUT1, CRESC100) or
nearly exact (CRESC132 is ±2), so the column-relative rejection
has no tiny pivots to reject. The plateau at `1e+02 – 1e+05` is
not a pivot-acceptance issue; it is either iterative refinement
stagnation on large KKT systems or the need for delayed pivoting
to produce a better first iterate.

Every `tests/mc64_regression.rs` target remains `#[ignore]`'d with
an updated Post-2.2.2 status comment. No test tolerances were
loosened. The sanity panel shows no regressions vs Post-2.2.1.

## Regression test results

Temporarily un-ignored all 4 tests and ran
`cargo test --release --test mc64_regression -- --test-threads=1
--nocapture` (log at `/tmp/phase-2.2.2-step7.log`). The
`#[ignore]` attributes have been **restored** with updated status
comments.

| Matrix        |    n | Pre-fix (Identity) | Post-2.2.1 (MC64) | Post-2.2.2 (MC64 + u=0.01) |  Target | Canonical MUMPS |                              Verdict |
| ------------- | ---: | -----------------: | ----------------: | -------------------------: | ------: | --------------: | -----------------------------------: |
| ACOPP30_0000  |  209 |           2.84e+16 |      **2.27e+46** |               **1.076e-1** | < 1e-08 |        5.01e-14 | RECOVERED 47 ORDERS, still above tgt |
| CHWIRUT1_0000 |  645 |           1.41e+09 |          8.50e+02 |                   8.497e+2 | < 1e-08 |        9.51e-13 |                   UNCHANGED vs 2.2.1 |
| CRESC100_0000 |  806 |           2.54e+04 |          1.43e+02 |                   1.426e+2 | < 1e-08 |        6.15e-15 |                   UNCHANGED vs 2.2.1 |
| CRESC132_0000 | 5314 |           2.39e+08 |          1.37e+05 |                   1.370e+5 | < 1e-06 |        2.48e-11 |                   UNCHANGED vs 2.2.1 |

ACOPP30 is the only matrix where the column-relative rejection
fires on a meaningful number of pivots. This is consistent with
the Phase 2.2.1 diagnosis in
`dev/debugging/2026-04-12-acopp30-regression.md` that the
regression was caused by 5 forced-zero pivots in the
`ForceAccept` branch interacting with the unscaled residual
recompose. Rejecting those pivots via `pivot_threshold = 0.01`
before they reach `ForceAccept` removes the catastrophic
amplification.

The other three matrices have exact or near-exact inertia with
no `ForceAccept` fires to reject, so Step 3/4's `u = 0.01` clause
is simply a no-op on them.

## Sanity panel comparison

Run: `cargo run --release --example triage_large_cresc132`.
(Example was updated to set `pivot_threshold = 0.01` to match the
new KKT defaults.) Log at `/tmp/phase-2.2.2-step8.log`.

| Matrix        |    n | Post-2.2.1 (MC64) | Post-2.2.2 (MC64 + u=0.01) | Δ orders |     MUMPS residual | Inertia (feral / MUMPS) |
| ------------- | ---: | ----------------: | -------------------------: | -------: | -----------------: | ----------------------- |
| CHWIRUT1_0000 |  645 |          8.50e+02 |                   8.497e+2 |     0.00 |           9.51e-13 | (431,214,0) MATCH       |
| HAHN1_0000    |  715 |          3.76e+04 |                   3.758e+4 |     0.00 | 3.19e-14 (SSIDS)   | (478,237,0) ±1          |
| GAUSS2_0000   |  758 |          1.84e+03 |                   1.843e+3 |     0.00 |           3.32e-14 | (507,251,0) ±1          |
| CRESC100_0000 |  806 |          1.43e+02 |                   1.426e+2 |     0.00 |           6.15e-15 | (606,200,0) MATCH       |
| MUONSINE_0000 | 1537 |          2.35e-02 |                   2.349e-2 |     0.00 |           2.25e-15 | (1026,511,0) ±1         |
| VESUVIO_0000  | 3083 |          1.82e+05 |                   1.815e+5 |     0.00 |           1.10e-12 | (2057,1026,0) ±1        |
| CRESC132_0000 | 5314 |          1.37e+05 |                   1.370e+5 |     0.00 |           2.48e-11 | (2658,2656,0) ±2        |

**No matrix improved, no matrix regressed.** The panel is
invariant under `u = 0.01` on these 7 matrices. This is expected
given the inertia match rates — if the pivot stream on each
column is already well-conditioned at absolute `zero_tol`, a
column-relative threshold has nothing to reject.

The implication: Phase 2.2.2's value on this panel is the
*non-regression* guarantee (no matrix got worse when `u = 0.01`
was flipped on) rather than a residual improvement. The panel
residuals still sit at `1e+02 – 1e+05`, requiring either delayed
pivoting or a solve-side fix (iterative refinement that actually
converges on these systems).

## ACOPP30 isolated story

| Stage              | Residual  | Delta |
| ------------------ | --------: | ----: |
| Identity (no MC64) |  2.84e+16 |    —  |
| Post-MC64 (2.2.1)  |  2.27e+46 | +30   |
| Post-u=0.01 (2.2.2)| 1.076e-01 | −47   |
| Target             |   < 1e-08 |       |
| Canonical MUMPS    |  5.01e-14 |       |

`1.076e-1` vs `5.01e-14` is ~13 orders above canonical MUMPS and
~7 orders above the target. The residual is still bad — but it is
now *bad in the ordinary way* (rank-deficient KKT needing delayed
pivoting and/or refinement that converges) rather than bad via a
scaling-exponent blowup. Before this commit the matrix reported
`1e+46`; now it reports `1e-1`, which the iterative refinement
path can at least reason about.

## Full test suite regression check

`cargo test --release` log at `/tmp/phase-2.2.2-full-suite.log`.

```
Total passed: 146
Total failed:   0
Total ignored:  7   (4 mc64_regression + 1 threshold_consistency
                     + 1 refinement + 1 misc)
```

Identical to the pre-Step-7 numbers. Every `#[ignore]` that was
lifted for measurement has been restored. No existing test
regressed as a result of the `pivot_threshold = 0.01` flip in
`ldlt_params` and `params_kkt`.

`cargo run --release --bin bench` sparse summary:

```
Sparse solver: 154588/154588 total
  Inertia match vs MUMPS: 152968/154588 (99.0%)
  Residual pass:          154202/154588 (99.8%)
```

Compared to Post-2.2.1 (153151 total; residual pass 152788/153151
= 99.76%): corpus grew by ~1.4k and the residual pass rate is
99.75% vs 99.76%, i.e. unchanged within rounding. No aggregate
regression.

Top 3 worst residuals in the bench (sparse side):

| Rank | Matrix         | Residual | Expected inertia | Actual inertia | Notes                                |
| ---: | -------------- | -------: | ---------------- | -------------- | ------------------------------------ |
|    1 | POLAK6_0021    |   1.20e9 | (5,4,0)          | (3,4,2)        | Rank-deficient, pre-existing ignore  |
|    2 | CERI651A_0076  |   1.82e0 | (129,61,0)       | (128,61,1)     | ±1 inertia, CERI family              |
|    3 | CERI651A_0136  |  1.09e-2 | (129,61,0)       | (127,62,1)     | ±2 inertia, CERI family              |

POLAK6_0021 (1.20e9) has always been the bench outlier; this is
unchanged by Phase 2.2.2.

## What was committed

- `examples/triage_large_cresc132.rs` — opt the triage sanity
  panel into `pivot_threshold = 0.01`, mirroring `params_kkt` in
  the bench harness.
- `tests/mc64_regression.rs` — appended Phase 2.2.2 Step 7 status
  comments to each of the 4 regression tests, preserving
  `#[ignore]`. Updated comments cite the Post-2.2.2 residuals and
  point at this document.
- `dev/validation/phase-2.2.2-pivot-rejection.md` — this report.
- `CHANGELOG.md` — Unreleased entry under `### Fixed` noting the
  ACOPP30 recovery.
- `dev/decisions.md` — Phase 2.2.2 entry documenting the
  `pivot_threshold = 0.01` default for MC64 callers and the
  explicit deferral of delayed pivoting to Phase 2.3.
- `dev/journal/2026-04-12-09.org` — real-time journal entry.

Nothing in `src/` was modified in this session. No tolerances
were loosened. No tests were un-ignored permanently.

## Open questions for Phase 2.3

1. CHWIRUT1 and CRESC100 have exact inertia and still plateau at
   `8.5e+02` and `1.4e+02`. If the factor is essentially
   structurally correct, why doesn't `solve_sparse_refined`
   converge? Next step: instrument the refinement loop to print
   per-iteration residuals on CHWIRUT1 and see whether the
   plateau is "iteration 1 stops improving" or "residual grows".
2. ACOPP30 at `1.076e-1` — is the remaining gap dominated by the
   2 `ForceAccept` zero pivots that now survive after `u = 0.01`,
   or by downstream sparse-solve rounding? The un-scaled residual
   recompose path is shared with the panel matrices; ACOPP30 may
   need delayed-pivoting specifically.
3. CRESC132's ±2 inertia mismatch. This is the Phase 2.2.2
   `count_2x2_inertia` trace-vs-a00 bug carried forward from
   Phase 2.2.1. Not a Phase 2.2.2 deliverable; still open.
4. MUONSINE at `2.349e-2` is the only panel matrix even in the
   neighborhood of the residual-pass bench threshold (~1e-7).
   Why is this one 7 orders closer than the others? If the
   answer is "denser supernodes amortize the rounding better",
   that points at a specific sparse-solve fix.
5. Does Phase 2.3 delayed pivoting (SPRAL SSIDS `ldlt_tpp.cxx`
   style) fix the remaining residual plateau in a single blow,
   or is delayed pivoting only the ACOPP30 fix and the
   CHWIRUT1/CRESC100 plateaus need an independent solve-side
   investigation?

## Verdict

**Phase 2.2.2: PARTIAL WIN, OUTCOME C.** The minimum acceptance
criterion ("ACOPP30 no longer a 30-order regression vs Identity")
is met with ~17 orders of margin. None of the 4 regression tests
pass their targets, and all 4 remain `#[ignore]`'d with updated
status comments. The sanity panel is flat (no improvement, no
regression) and the broader bench residual-pass rate is unchanged
within rounding. Phase 2.2.2 can be closed as a correctness fix
for ACOPP30; full closure of the MC64 gap requires Phase 2.3
delayed pivoting plus a separate investigation of the
CHWIRUT1/CRESC100 solve-side plateau.
