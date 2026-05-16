# FBRAIN3LS 2×2 pivot block stability — `pivot_threshold` sweep

**Issue:** #29 (M6: FBRAIN3LS 2x2 pivot block stability — sweep BK threshold and document)
**Status:** Complete. Headline finding: the issue's hypothesis is *not*
borne out by data. `pivot_threshold` does not move the inertia or
residual on FBRAIN3LS; the active gate is the F-01 `null_pivot_tol`
override. The chosen default of `1e-8` stays.
**Date:** 2026-05-16
**Driver:** `cargo run --release --bin diag_fbrain3ls_pivtol_sweep`

## Background — what the issue claimed

The original issue text:

> `FBRAIN3LS` produces small ill-conditioned 2x2 pivot blocks during
> Bunch-Kaufman pivoting. Sensitivity to the BK threshold (`1e-8`)
> is high enough that small changes flip inertia.

Two factual problems with this framing surface immediately after one
run of the sweep:

1. **There are no 2×2 pivot blocks on any FBRAIN3LS matrix at any
   pivtol we tested.** `n_2x2 = 0` in every sweep row.
2. **The inertia does not flip when `pivot_threshold` is varied.** It
   stays at `(5,0,1)` (rank-deficient under `ForceAccept`) or `(6,0,0)`
   (full-rank under `Fail`) across seven orders of magnitude of
   `pivot_threshold`, including the BK / SSIDS canonical `0.01`.

The reported sensitivity *is* real — but it lives on a different knob.
See "Where the actual sensitivity lives" below.

## Method

`src/bin/diag_fbrain3ls_pivtol_sweep.rs` loads each matrix in
`SAMPLES` from `data/matrices/kkt/FBRAIN3LS/`, factors it via
`factorize_multifrontal` for each
`pivot_threshold ∈ {0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-2}` under
two passes:

- **Pass A:** `on_zero_pivot = ForceAccept` (sparse-multifrontal
  default). The driver's F-01 override raises `null_pivot_tol` to
  `sqrt(n) · EPS · ‖A‖_inf`, which dominates `pivot_threshold` here.
- **Pass B:** `on_zero_pivot = Fail`. Disables the F-01 override (per
  `override_null_pivot_tol`'s `Fail` early-out) so the BK / Duff-Reid
  acceptance machinery driven by `pivot_threshold` is the only active
  gate.

Sample set (all five matrices have `n=6`, `nnz=21`):

| sample          | consensus_inertia | verdict                  | feral_baseline_inertia |
| --------------- | :---------------- | :----------------------- | :--------------------- |
| FBRAIN3LS_0788  | (6, 0, 0)         | definitive               | (6, 0, 0) MATCH        |
| FBRAIN3LS_0839  | (6, 0, 0)         | numerically_intractable  | (5, 0, 1)              |
| FBRAIN3LS_0843  | (6, 0, 0)         | numerically_intractable  | (5, 0, 1)              |
| FBRAIN3LS_0848  | none (3-way)      | excluded                 | (5, 0, 1)              |
| FBRAIN3LS_0851  | (6, 0, 0)         | numerically_intractable  | (5, 0, 1)              |

(Sidecar consensus data from each matrix's `*.verdict.json`.)

Residual is `‖A x − b‖₂ / max(‖b‖₂, 1)` with `x` from
`solve_sparse_refined` and `b` taken from the matrix's `.json` sidecar
RHS — the same RHS the canonical Fortran oracles solve against.

## Results

All values are `(inertia, min|D|, rel_res)`. Identical rows in a block
are collapsed to a single "all pivtols → …" line.

### Pass A: `ZeroPivotAction::ForceAccept` (default)

| matrix          | null_pivot_floor | all pivtols → inertia | min &#124;D&#124; | rel_res  |
| --------------- | ---------------: | :-------------------- | -------: | -------: |
| FBRAIN3LS_0788  |        2.55e-7   | (5,0,1) NO            | 7.64e-9  | 1.18e-10 |
| FBRAIN3LS_0839  |        1.40e-6   | (5,0,1) NO            | 6.86e-10 | 8.07e-8  |
| FBRAIN3LS_0843  |        1.14e-6   | (5,0,1) NO            | 5.82e-10 | 2.23e-8  |
| FBRAIN3LS_0848  |        2.24e-6   | (5,0,1) NO            | 7.25e-10 | 6.60e-8  |
| FBRAIN3LS_0851  |        2.88e-6   | (5,0,1) NO            | 5.07e-10 | 1.37e-7  |

Every pivot the BK kernel produces (`7.6e-9` down to `5e-10`) is *strictly
below* the per-matrix `null_pivot_floor = sqrt(n) · EPS · ‖A‖_inf`
(~`1e-6`). The F-01 override stamps these as zeros independent of the
`pivot_threshold` knob, which is why the sweep is flat.

The 0788 row is the most instructive: it appears as `definitive` in
`compute_consensus.py` and `feral_match_inertia = true` in the verdict
sidecar, yet today it reports `(5,0,1)`. The verdict was recorded
before the F-01 override landed (the recorded `feral_residual` of
`1.15e-10` matches the residual we see now — pivot health didn't
change, only the *interpretation* of a sub-floor pivot did). This is
behavior drift, not new breakage; the inertia gate already excludes
the borderline FBRAIN3LS samples per
`dev/research/inertia-triage-2026-04-27.md`.

### Pass B: `ZeroPivotAction::Fail` (null_pivot_tol override disabled)

| matrix          | all pivtols → inertia | min &#124;D&#124; | rel_res  |
| --------------- | :-------------------- | -------: | -------: |
| FBRAIN3LS_0788  | (6,0,0) MATCH         | 7.64e-9  | 1.18e-10 |
| FBRAIN3LS_0839  | (6,0,0) MATCH         | 6.86e-10 | 8.07e-8  |
| FBRAIN3LS_0843  | (6,0,0) MATCH         | 5.82e-10 | 2.23e-8  |
| FBRAIN3LS_0848  | (6,0,0) MATCH         | 7.25e-10 | 6.60e-8  |
| FBRAIN3LS_0851  | (6,0,0) MATCH         | 5.07e-10 | 1.37e-7  |

With the F-01 override out of the way, BK accepts every diagonal pivot
(none of them are exactly zero) and the inertia matches the MUMPS/SSIDS
consensus on all five. `pivot_threshold` still has *no effect* — the
column-relative test `|a_kk| >= u · max_{i>k}(|a_ik|)` never trips
because for these tiny n=6 matrices the diagonal entries dominate
their off-diagonals by orders of magnitude after the existing
permutation has run. The residuals are unchanged from Pass A: the
F-01 override only sets a *label* on the pivot (zero vs nonzero), it
does not perturb the numerics.

## Where the actual sensitivity lives

Two earlier knob flips on FBRAIN3LS each flipped the inertia, neither
of them on `pivot_threshold`:

1. **The dense-vs-multifrontal routing change** (Phase 2.4.1b /
   block32 register-resident kernel,
   `dev/research/block32-register-resident-kernel.md` lines 148-149):
   FBRAIN3LS_0848 and FBRAIN3LS_0851 regressed from `(6,0,0)` to
   `(5,0,1)` purely from 1-ULP FMA rounding flips at the same pivot
   threshold.
2. **The F-01 `null_pivot_tol` override**
   (`dev/research/f01-rankdef-underreporting.md`): introduced
   `sqrt(n) · EPS · ‖A‖_inf` floor on rank-deficiency detection,
   which is what currently labels FBRAIN3LS's smallest pivots as
   zeros (Pass A above).

`pivot_threshold` itself controls a different test: Bunch-Kaufman
1977 §2's column-relative acceptance ratio plus Duff-Reid 1995's
2×2-block growth bound (per the docstring on
`BunchKaufmanParams::pivot_threshold`). On FBRAIN3LS these tests
never reject anything because (a) the diagonal already dominates and
(b) there are no 2×2 candidates to begin with.

## Literature alignment

### Bunch-Kaufman 1977 §2

The original BK test accepts a 1×1 pivot when `|a_kk| >= α · γ_0`
(where α = (1+√17)/8 ≈ 0.6404 and γ_0 = max_{i>k} |a_ik|) and falls
back to a 2×2 block built from the off-diagonal max otherwise. The
paper assumes the matrix has been pre-scaled so that this ratio test
is meaningful in absolute terms. With `pivot_threshold = u > 0` we
layer on a *column-relative* threshold `|a_kk| >= u · γ_0` (the
MA27/MA57/MUMPS `cntl(1)` knob), motivated by Duff-Reid 1995 — see
§3 of that paper for the derivation of the growth bound
`(|a_22|·RMAX + AMAX·TMAX)·u <= |det|` used for 2×2 blocks
(`dense::factor::factor_frontal` lines 1786-1990).

### Ashcraft-Grimes-Lewis 1998

AGL describe four "accuracy and stability" tradeoffs for symmetric
indefinite factorizations: partial pivoting, Bunch-Parlett, Bunch-Kaufman,
and rook pivoting. Their §4 measurements (Table 4.2, "Element growth")
show that on PD matrices with mild ill-conditioning the BK column-
relative threshold `u` barely affects element growth in the range
`u ∈ [1e-3, 1e-1]`. They do not study the `u → 0` regime — their text
takes for granted that `u >= 1e-3` is appropriate for general use.
The FBRAIN3LS sweep above is consistent with their finding: in a regime
where the diagonal already dominates, `u` has no observable effect on
the numerics. The sensitivity AGL warn against is on matrices with
near-zero diagonals — and on those the existing scaling-aware default
of `0.01` (set in callers that use MC64) is the right move.

### Higham 2002 §11

*Accuracy and Stability of Numerical Algorithms* §11 ("Symmetric
indefinite and triangular systems") summarises the same threshold
test (Algorithm 11.1) and recommends `u ∈ [α, 0.5]` after the
Bunch-Kaufman bounds. The takeaway for our default is that `1e-8` is
**much** below the regime where the column-relative test is intended
to bite. It exists as a *very* permissive safety net, not as a stability
device. On FBRAIN3LS the per-column ratios are all O(1) (the diagonal
is the column max in absolute value), so neither `1e-8` nor `0.01`
would change behaviour.

## Justification for keeping `pivot_threshold = 1e-8`

The default was set in
`numeric::factorize::NumericParams::default()` to match Ipopt's
`ma27_pivtol` (1e-8) for the Identity-scaled IPM-KKT path that
`ripopt` exercises. The rationale (preserved in the docstring lines
347-381) is:

1. `0.01` (the MUMPS/SSIDS canonical) is calibrated for MC64-
   equilibrated matrices where the column maxes are normalised to O(1)
   and the threshold test means "reject pivots that are <1% of
   normalised column max". On Identity-scaled matrices the ratio is
   on *raw* values and `0.01` would mis-fire on legitimate pivots
   in differently-scaled columns.
2. `1e-8` is what every Ipopt user on MA27 has been running for two
   decades; it represents "almost no rejection, a very loose ratio
   safety net".
3. Sparse callers that *do* run MC64 / InfNorm scaling override
   explicitly to `0.01`.

The FBRAIN3LS sweep does **not** disturb any of these arguments — it
proves the default `1e-8` is irrelevant on this family (the active gate
is `null_pivot_tol`) but it does not point at a better value either:
`0.0`, `1e-10`, `1e-8`, `1e-6`, and `0.01` all give bit-identical
results. Keeping `1e-8` therefore preserves the Ipopt-compatibility
argument for free; flipping it would only churn the default with no
measured benefit.

## What would actually change FBRAIN3LS inertia

The borderline FBRAIN3LS matrices report `(5,0,1)` today because:

- A single diagonal pivot reduces to ~`5e-10`-`8e-9` during
  Bunch-Kaufman elimination.
- That value is below the F-01 `sqrt(n) · EPS · ‖A‖_inf` floor (~`1e-6`).
- The driver counts it as a zero rather than a (correct-sign) positive.

The only knob that would flip this is `null_pivot_tol` itself
(equivalently, switching `on_zero_pivot` to `Fail`). That change is
out of scope for this issue:

- The F-01 override exists to *catch* rank-deficiency on KKT-augmented
  IPM systems where the multipliers genuinely live in a singular
  block (issue #5, `f01-rankdef-underreporting.md`). Backing it off
  globally would silently re-introduce the bug F-01 fixed.
- The triage in `dev/research/inertia-triage-2026-04-27.md` already
  classified the FBRAIN3LS borderline matrices as
  `numerically_intractable` — outside the inertia gate.
- The corpus consensus framework already excludes them from the
  reportable mismatch count.

## Decision

**Keep `NumericParams::default().bk.pivot_threshold = 1e-8`.** The
threshold is below the regime where it bites on FBRAIN3LS, and the
matched-against-Ipopt rationale (Identity-scaled IPM-KKT users)
remains the dominant constraint.

No code change to `BunchKaufmanParams::default()` either: the dense
path keeps `pivot_threshold = 0.0`, consistent with the dense-vs-sparse
split (`dev/decisions.md` 325-344). The dense FBRAIN3LS_0788 triage
in `examples/triage_fbrain3ls.rs` already prints the actual sensitivity
(it lives on `zero_tol`, not `pivot_threshold`).

## Open items (not blocking this issue)

- The verdict file for FBRAIN3LS_0788 still says
  `"feral_match_inertia": true` based on a pre-F-01 run. Refreshing
  the verdict harness after the F-01 override would shift this one
  matrix into the borderline bucket. **Not in scope here** — it's a
  scoreboard refresh, not a solver change.
- The `verdict=excluded` matrices in the FBRAIN3LS family (0848 and
  a few others) are not part of any oracle gate. They surface in
  the bench output and in the stress manifest under the new
  `borderline` category but do not affect pass/fail.

## Artifacts

- Sweep driver: `src/bin/diag_fbrain3ls_pivtol_sweep.rs`
- Stress manifest entries: `external_benchmarks/stress/manifest.tsv`
  rows tagged `cuter_kkt / FBRAIN3LS__FBRAIN3LS_{0839,0843,0851}`
- `fetch.py` extension: `external_benchmarks/stress/fetch.py` now
  copies `cuter_kkt` rows from `data/matrices/kkt/<family>/<sample>.mtx`
- Prior context:
  - `dev/research/inertia-triage-2026-04-27.md` §FBRAIN3LS (3 of 11)
  - `dev/research/f01-rankdef-underreporting.md`
  - `dev/research/block32-register-resident-kernel.md` lines 148-149
  - `dev/research/2x2-bk-inertia-accounting.md`

## References

- J. R. Bunch and L. Kaufman (1977), *Some stable methods for
  calculating inertia and solving symmetric linear systems*,
  Math. Comp. 31(137), pp. 163-179. §2 defines the column-relative
  acceptance ratio used in `pivot_threshold`.
- I. S. Duff and J. K. Reid (1995), *MA47, a Fortran code for direct
  solution of indefinite sparse symmetric linear systems*,
  RAL-TR-95-001 / ACM TOMS 21(1) pp. 95-115. §3 derives the 2×2 growth
  bound `(|a_22|·RMAX + AMAX·TMAX)·u <= |det|` reused in
  `factor_frontal`.
- C. Ashcraft, R. G. Grimes, J. G. Lewis (1998), *Accuracy and stability
  of sparse symmetric indefinite linear-system solvers*,
  SIAM J. Matrix Anal. Appl. 20(2) pp. 513-561, §4.
- N. J. Higham (2002), *Accuracy and Stability of Numerical Algorithms*
  (2nd ed.), SIAM, §11 ("Symmetric indefinite and triangular systems").
- HSL MA27 documentation, `cntl[0]` (pivot tolerance, default 0.1)
  and the Ipopt `ma27_pivtol` option (default `1e-8`).
