# F-03 — `bloweybl` rank-deficient rejection

**Finding:** `GHS_indef/bloweybl.mtx` (n=30003, CUTEr bound-constrained
QP) is rejected by feral with `NumericallyRankDeficient`. Both
reference oracles factor it cleanly with one zero pivot.

## Oracle (external)

| solver | inertia (pos, neg, zero) | factor (µs) | rel residual |
|--------|--------------------------|-------------|--------------|
| MUMPS 5.8.2 | (20001, 10001, 1) | 153 014 | 2.4e-14 |
| MA57 2023.11.17 | (20001, 10001, 1) | 6 430 | 2.7e-15 |
| feral (current) | — | — | (rejection) |
| sibling `bloweybq` n=10001 | (10001, 0, 0) | — | 3.7e-16 |

Both oracles agree the matrix has rank 30002 (one zero pivot). The
smaller sibling `bloweybq` (same generator, half the size) factors
cleanly with no zero pivots.

## Matrix structure

Diagnostic probe of `bloweybl.mtx`:

```
n               = 30003
explicit zero diagonal entries   = 10001
missing diagonal entries (treated as zero) = 10001
explicit nonzero diagonal entries = 10001, all value = 0.02
sign breakdown                   = pos: 10001, neg: 0, zero: 20002
```

Two thirds of the diagonals are zero. This is a saddle-point block:

```
A  ≈  [ H   A^T ]
      [ A    0  ]
```

with `H = 0.02 * I_{10001}` (the constant-positive third) and a zero
(n=20002) block. The off-diagonal coupling makes the matrix
non-singular (rank 30002), but the eliminator must use 2×2 pivots that
cross the zero block to expose this.

## Root cause

`NumericParams::default()` at `src/numeric/factorize.rs:382-417` uses
`BunchKaufmanParams::default()`, which sets
`on_zero_pivot = ZeroPivotAction::Fail` (`src/dense/factor.rs:343-356`).

At the root supernode (`may_delay == false`), any column whose
remaining pivot magnitude falls inside `|d| <= zero_tol` (≈ machine
epsilon) is short-circuited to `Err(NumericallyRankDeficient)`. The
trigger sites are:

- `src/dense/factor.rs:445` — last-pivot 1×1 path
- `src/dense/factor.rs:2519` — 2×2 rejection, `det_floor_fail`
- `src/dense/factor.rs:2648` — 1×1 frontal rejection, true zero

On `bloweybl`, the 2/3 structurally-zero diagonals produce enough
candidates that one of them hits a root-level Fail before BK can
absorb it into a 2×2 with the small-positive 0.02 diagonals.

## Why the default was wrong

The current `Fail` default was inherited from
`BunchKaufmanParams::default()` and is appropriate for dense factor
callers that have no delayed-pivoting infrastructure. For the *sparse*
multifrontal path it's strictly worse than the two competing options:

| action | what happens at a true zero pivot | matches |
|--------|-----------------------------------|---------|
| `Fail` | abort with `NumericallyRankDeficient`, discard factor | nothing — SSIDS at root breaks out leaving columns un-eliminated, MUMPS/MA57 don't fail |
| `ForceAccept` | count zero in inertia, zero L column, continue | MUMPS (`INFOG(28)`) |
| `PerturbToEps` | perturb to `sign(d) * max(\|d\|, abs_floor)` | MA57 (`cntl[4]`) |

`Fail` loses information: callers can't distinguish "exactly one zero
pivot" from "the whole matrix is structurally degenerate." Both
oracles (MUMPS, MA57) produce a usable factor in both cases. Users
who need to detect singularity can read `inertia.zero > 0` after a
`ForceAccept` factor.

## Fix

Change `NumericParams::default()` to set
`on_zero_pivot = ZeroPivotAction::ForceAccept`. Leave
`BunchKaufmanParams::default()` unchanged (dense entry point stays
`Fail`-by-default per the 2026-04-13 dense-vs-sparse split documented
in `dev/decisions.md:325-344`).

Behavior change for `Solver::new()`:

- Matrices that previously returned `FactorStatus::Singular` now
  return `FactorStatus::Success` with `inertia.zero > 0` (or
  `WrongInertia` if `check_inertia` is passed with a mismatched
  expectation). The factor is preserved; the caller can solve
  against it and inspect `inertia` to detect rank-deficiency.
- This matches MUMPS and MA57 behavior under their default settings.
- No matrix that previously factored successfully changes outcome.

Tests that opted into the `Fail` semantics explicitly via
`BunchKaufmanParams { on_zero_pivot: Fail, .. }` are unaffected.
Two pounce-interface tests that relied on the `Solver::new()` default
producing `Singular` on `diag(1, 0, 1)` need updating: either opt
into `Fail` explicitly via `Solver::with_params`, or assert the new
contract (`Success` with `inertia.zero == 1`).

## Acceptance

1. Regression test in `tests/` that constructs a small saddle
   `[I 0; 0 0]` and asserts feral produces correct inertia
   (without expecting `Singular`).
2. `bloweybl` flagged-status flips from `status=fail` to `ok` in
   `external_benchmarks/stress/report.py`, with `inertia.zero == 1`
   matching the MUMPS/MA57 oracle.
3. No regression on the existing test suite (the two tests that
   relied on the old default are updated to opt-in to `Fail`).

## References

- MUMPS 5.8 User's Guide §3.4 (INFOG(28) — number of null pivots)
- MA57 (HSL Specification) §2.7 (CNTL(4) — static pivot tolerance)
- Wächter-Biegler 2006 §3.1 (inertia correction in IPM)
- `dev/decisions.md:325-344` (dense vs sparse default split)
- `dev/research/issue-2-kkt-pivot-default.md` (pivot threshold history)
- `dev/research/cascade-break-l-perturbation-2026-05-15.md`
  (`PerturbToEps` analysis)
