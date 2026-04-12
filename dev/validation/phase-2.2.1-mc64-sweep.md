# Phase 2.2.1 Step 8 — MC64 Validation Sweep

**Date:** 2026-04-12
**Head commit under test:** `0a13515` (Phase 2.2.1 Steps 6+7 complete)
**Status:** PARTIAL WIN — MC64 improved 6 of 7 sanity-panel matrices
by 2–10 orders of magnitude, but no regression test met its target.
One matrix (ACOPP30_0000) regressed catastrophically.

## Summary

MC64 matching-based scaling is demonstrably doing non-trivial work:
every `n > 500` sanity-panel matrix now solves to a residual many
orders of magnitude better than the pre-fix baseline. However, the
four explicit regression targets in `tests/mc64_regression.rs`
(`< 1e-8` for ACOPP30/CHWIRUT1/CRESC100, `< 1e-6` for CRESC132)
remain **unmet** — the residuals are in the 10² – 10⁵ range, not
near machine precision. Phase 2.2.1 cannot be closed on these
numbers; at least one follow-up fix is required in Phase 2.2.2.

Notably, ACOPP30_0000 (n=209) went from a pre-fix sparse-path
residual of `2.84e+16` to a post-MC64 residual of `2.27e+46` — a
30-order-of-magnitude *regression*. Root-cause hypothesis: on this
structurally-singular KKT matrix, MC64 scaling interacts
pathologically with the `ForceAccept` zero-pivot path. The
scaled factor produces inertia `(62, 142, 5)` (5 forced zero
pivots) while canonical MUMPS reports `(71, 137, 1)` and SSIDS
reports `(71, 138, 0)`. The huge exponent in `2.27e+46` strongly
suggests an exp-clamp or dual-variable overflow leaking through
the `s[i] = 1` fallback for unmatched rows.

## Regression test results

All 4 tests were temporarily un-ignored for this sweep and run via
`cargo test --release --test mc64_regression -- --test-threads=1`.
Log: `/tmp/mc64_regression_step8.log`. The 4 `#[ignore]` attributes
have been **restored** before committing; each now carries a Phase
2.2.1-Step-8 status comment pointing at this document.

| Matrix | n | Pre-fix residual | Post-MC64 residual | Target | Canonical MUMPS | Verdict |
|---|---:|---:|---:|---:|---:|---|
| ACOPP30_0000  |  209 | 2.84e+16 | **2.27e+46** | < 1e-8 | 5.01e-14 | **REGRESSED** |
| CHWIRUT1_0000 |  645 | 1.41e+09 | 8.50e+02 | < 1e-8 | 9.51e-13 | IMPROVED, FAIL |
| CRESC100_0000 |  806 | 2.54e+04 | 1.43e+02 | < 1e-8 | 6.15e-15 | IMPROVED, FAIL |
| CRESC132_0000 | 5314 | 2.39e+08 | 1.37e+05 | < 1e-6 | 2.48e-11 | IMPROVED, FAIL |

*ACOPP30 "pre-fix" number is the `solve_sparse_refined` result on
commit `c01235f`. The `3.15e-2` figure cited in the test docstring
and the MC64 plan comes from the `triage_acopp30` example, which
exercises a different solve path. This discrepancy itself should be
reconciled in Phase 2.2.2.*

## Sanity panel comparison

Run: `cargo run --release --example triage_large_cresc132`.
Pre-fix log reconstructed by checking out `c01235f` and re-running
the same command. Post-fix log: `/tmp/triage_step8.log`.

| Matrix | n | Pre-fix residual | Post-MC64 residual | Δ orders | MUMPS residual | Inertia (feral / MUMPS) |
|---|---:|---:|---:|---:|---:|---|
| CHWIRUT1_0000 |  645 | 1.41e+09 | 8.50e+02 | −6.2 | 9.51e-13 | (431,214,0) / (431,214,0) MATCH |
| HAHN1_0000    |  715 | 1.44e+14 | 3.76e+04 | −9.6 | 3.19e-14 (SSIDS) | (478,237,0) / (479,236,0) ±1 |
| GAUSS2_0000   |  758 | 1.34e+09 | 1.84e+03 | −5.9 | 3.32e-14 | (507,251,0) / (508,250,0) ±1 |
| CRESC100_0000 |  806 | 2.54e+04 | 1.43e+02 | −2.3 | 6.15e-15 | (606,200,0) / (606,200,0) MATCH |
| MUONSINE_0000 | 1537 | 3.52e+03 | 2.35e-02 | −5.2 | 2.25e-15 | (1026,511,0) / (1025,512,0) ±1 |
| VESUVIO_0000  | 3083 | 5.61e+14 | 1.82e+05 | −9.5 | 1.10e-12 | (2057,1026,0) / (2058,1025,0) ±1 |
| CRESC132_0000 | 5314 | 2.39e+08 | 1.37e+05 | −3.2 | 2.48e-11 | (2658,2656,0) / (2660,2654,0) ±2 |

Every matrix improved. Six improved by 2–10 orders of magnitude.
MUONSINE is the best of the panel at 2.35e-2 and the only one even
approaching the "residual pass" bench threshold of ~1e-7. No matrix
is within the Phase 2.2.1 plan's aspirational "within 3 orders of
magnitude of canonical" criterion; CHWIRUT1 and CRESC100 are the
closest at ~15 orders above canonical, while MUONSINE is ~13.

The ±1/±2 inertia disagreements on 5 of 7 matrices are the
Phase 2.2.2 `count_2x2_inertia` trace-vs-a00 bug, visible before
MC64 and still visible now (inertia is invariant under congruence,
so scaling does not fix it — this is expected).

## Full test suite regression check

`cargo test --release --no-fail-fast` (with all 4 regression tests
temporarily un-ignored):

```
Total passed:  141
Total failed:    4   (only the 4 un-ignored mc64_regression tests)
Total ignored:   3
```

After restoring `#[ignore]` on the 4 regression tests:

```
Total passed:  141
Total failed:    0
Total ignored:   7
```

No tests in `src/` or the other integration suites regressed. The
Phase 2.2.1 MC64 plumbing does not break any existing assertion.
This is consistent with MC64 being a congruence transformation that
shouldn't change computed inertia on matrices the existing suite
covers.

## Root-cause hypotheses

### ACOPP30_0000 catastrophic regression (highest priority)

Feral inertia `(62, 142, 5)` under MC64 differs from both the
pre-fix feral result (equivalent to the test docstring assumption)
and from MUMPS `(71, 137, 1)` / SSIDS `(71, 138, 0)`. Five
zero-eigenvalue pivots under `ForceAccept` strongly suggests MC64
scaling has pushed multiple near-singular 2×2 blocks below the
factor's `zero_tol` threshold, and the `ForceAccept` branch is
emitting zeros for the corresponding solve entries. When the
resulting `x` is plugged back into the **unscaled** `A`, the large
post-scaling `s[i]` multiplication on each returned entry produces
the `1e46` blow-up.

Candidate investigations for Phase 2.2.2:

1. Verify the MC64 symmetric dual clamp in
   `src/scaling/mc64.rs`. The research note specifies clamping any
   dual > `LOG_HUGE ≈ 709` and rewriting zero scales to 1. Check
   that the clamp preserves symmetry (clamping only row OR only
   column duals would break the symmetric average).
2. Instrument `compute_symmetric` to dump the ACOPP30 scaling
   vector. Look for entries near `f64::MAX` or near zero that
   survive the safety guards.
3. Check the interaction with `ForceAccept` explicitly. If
   `PartialSingular` is set, the identity-fallback rows should
   completely disable scaling for those indices — verify no
   post-scaling multiplication occurs at those positions.
4. Re-check the sign convention from SPRAL's `hungarian_wrapper`
   (plan Step 4 §4, research-note R2). A sign error here would
   produce exactly the kind of "two-sided over-scaling" we see.

### CHWIRUT1 / CRESC100 / CRESC132 / VESUVIO plateaus at 10² – 10⁵

All four show 2–10 orders of improvement but plateau well above
the target. These have near-correct inertia (exact match on
CHWIRUT1 and CRESC100; ±1 on CRESC132). The plateau pattern
suggests MC64 is working but not alone sufficient. Candidate
causes:

1. **No iterative refinement convergence.** `solve_sparse_refined`
   does up to 4 refinement sweeps with best-iterate tracking. On
   these large KKT matrices the first iterate may be so far from
   the solution that refinement never reduces the residual enough.
   Confirmation: collect the per-iteration residual trace and see
   whether the refinement plateau is 1 step or 4 steps.
2. **Sparse-solve rounding accumulation at scale** (research note
   R4). Per-supernode `Vec` allocations and the
   loop-over-supernodes structure in `solve_sparse` may accumulate
   rounding that MC64 cannot correct. The small 3×3 `mc64_end_to_end`
   tests do converge, so this is scale-dependent.
3. **Factor threshold too loose under scaling.** `zero_tol`
   defaults to `f64::EPSILON`. Under MC64 the diagonal magnitudes
   change, which may push marginal pivots across the threshold
   and into the `ForceAccept` branch. Compare `zero_tol` behavior
   with/without scaling on these matrices.

### HAHN1 / GAUSS2 / MUONSINE healthy improvement with ±1 inertia

These 3 matrices all improved 5–10 orders and show the Phase 2.2.2
trace-vs-a00 ±1 inertia signature. The residual plateau here
likely has the same cause as CHWIRUT1/CRESC100, but the ±1 inertia
adds one forced-zero eigenvector of error which could itself
account for the 10² – 10⁴ floor.

## What was committed

- `tests/mc64_regression.rs` — added Phase 2.2.1 Step 8 status
  comment to each of the 4 regression tests, preserving `#[ignore]`
  with a pointer to this document.
- `dev/validation/phase-2.2.1-mc64-sweep.md` — this report.
- `dev/journal/2026-04-12-05.org` — real-time journal entry.

Nothing in `src/` was modified. No tolerances were loosened.

## Open questions for Phase 2.2.2

1. Why did ACOPP30 regress 30 orders when the 6 sanity-panel
   matrices improved? Diagnose before any other fix.
2. Does removing scaling from the post-solve step (symmetric
   `D·A·D` factor + unscaled apply) fix ACOPP30?
3. Is the `ForceAccept` + MC64 combination a fundamental design
   conflict, or just a threshold-tuning issue?
4. Do we need to raise `zero_tol` under scaling, or is the factor
   correct and the solve applying scaling wrong?
5. Should we gate `solve_sparse_refined` to use the scaled matrix
   in its residual loop, or is the current unscaled-residual design
   correct? (Research note says unscaled is correct; verify
   empirically on ACOPP30.)

## Verdict

**Phase 2.2.1 Step 8: NOT CLOSED.** MC64 scaling is doing real work
but is not sufficient on its own. The plan's acceptance criterion
"residuals within 3 orders of magnitude of canonical" is not met
on any matrix, and the regression tests remain gated. Phase 2.2.1
should remain open pending Phase 2.2.2 triage.
