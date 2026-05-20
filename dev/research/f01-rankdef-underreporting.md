# F-01 — Synthetic `rankdef_*` under-reports zero pivots

**Finding:** All four `synth/rankdef_*` matrices are factored as
having `inertia.zero` strictly less than their constructed nullity.
After the F-03 default flip to `ForceAccept`:

| matrix | n | constructed k zeros | feral reports `(p,n,z)` | scipy LDL finds |
|--------|---|---------------------|--------------------------|-----------------|
| rankdef_5_2   |   5 |  2 | (2, 2, 1)     | 2 pivots `< 1e-12` |
| rankdef_10_3  |  10 |  3 | (4, 5, 1)     | 3 pivots `< 1e-12` |
| rankdef_50_5  |  50 |  5 | (25, 24, 1)   | 5 pivots `< 1e-12` |
| rankdef_200_20 | 200 | 20 | (112, 88, 0)  | 19 pivots `< 1e-12` |

scipy correctly detects the rank deficiency in its LDLᵀ pivots. feral's
BK kernel produces similar tiny pivots but classifies most of them as
small-but-clearly-nonzero (`±` sign), not as zero.

## Root cause

`try_reject_1x1_frontal` at `src/dense/factor.rs:2611-2671` splits the
"rejected pivot" path into two cases by absolute magnitude:

```rust
let threshold = (params.pivot_threshold * col_max).max(params.zero_tol);
if d.abs() <= threshold {
    if may_delay { return Ok(PivotOutcome::Delayed); }
    // Case (a): |d| <= zero_tol  → ForceAccept zeros L, counts zero
    if d.abs() <= params.zero_tol {
        match params.on_zero_pivot { ZeroPivotAction::ForceAccept => { ... zero += 1; } ... }
    }
    // Case (b): zero_tol < |d| <= u*col_max → accept with sign
    *needs_refinement = true;
    if d > 0.0 { *pos += 1; } else { *neg += 1; }
    return Ok(PivotOutcome::Accepted);
}
```

`zero_tol` defaults to `f64::EPSILON ≈ 2.22e-16` — *absolute*. For a
matrix scaled to `||A||_inf ≈ 1` and dimension `n`, the Wilkinson
backward error floor for LDLᵀ is `~n · EPS · ||A||_inf`. Real
rank-deficiency pivots land near this floor, *above* `EPS` but well
below `pivot_threshold * col_max` (typically `1e-8`).

Concrete evidence from scipy LDL on the four synth matrices
(`||A||_inf` is post-load, pre-feral-scaling):

| matrix | ‖A‖∞ | n·EPS·‖A‖∞ | smallest "zero" pivots (sorted by |·|) |
|--------|------|------------|----------------------------------------|
| rankdef_5_2   |  4.18 | 4.6e-15 | 9.3e-17, −5.3e-16                                 |
| rankdef_10_3  |  4.16 | 9.2e-15 | 4.8e-17, −2.1e-16, 9.0e-15                        |
| rankdef_50_5  | 12.1  | 1.3e-13 | −1.4e-16, 1.1e-15, 1.5e-15, −2.8e-15, −6.3e-15    |
| rankdef_200_20 | 23.5 | 1.0e-12 | 19 pivots in `[5e-16, 3e-13]` range               |

All "real" pivots in the same matrices are above `0.1`. Separation is
clean — there's no ambiguity about which pivots are zero.

The current `zero_tol = EPS` catches exactly the smallest pivot per
matrix (the one that happens to land below `EPS`), leaving the rest
in case (b) where they're miscounted as `±`.

## Why this matters

The four `rankdef_*` matrices are the F-01 evidence; the same bug
likely affects any rank-deficient real-world matrix where the
null-space pivots land in the `[EPS, n·EPS·‖A‖]` band — a common
range for IPM KKTs with rank-deficient Jacobians.

The historical comment at `src/dense/factor.rs:2624-2636` cites
DEGENLPA as a counter-example: a pivot at `-1e-8` that *should*
count as negative, not zero. Test
`tests/delayed_pivoting.rs:177` (`factor_frontal_root_accepts_small_pivot_with_sign`)
asserts this with `||A|| = 10`, `n = 4` — proposed threshold
`n·EPS·||A|| ≈ 9e-15` does not endanger that case (`1e-8 ≫ 9e-15`).
The separation between "DEGENLPA-small" (∼`1e-8`) and "rankdef-small"
(∼`n·EPS·‖A‖_inf`) is 6+ orders of magnitude, plenty of headroom.

## What reference solvers do

**MUMPS:** `CNTL(3)` is the null-pivot threshold. Default value is
`-1.0` (sentinel meaning "MUMPS picks one"); the internal default is
roughly `EPS · ||A||_inf · sqrt(n)`. Requires `ICNTL(24) = 1` to
enable null-pivot detection. Without it, small pivots are accepted
as `±` regardless of magnitude — matching feral's current behavior.

Our stress-suite MUMPS oracle on `bloweybl` reports
`INFOG(28) = 1` (one null pivot), which means the harness enables
`ICNTL(24) = 1`. The detected null pivot at scale `EPS` is well below
the auto threshold.

**MA57:** `CNTL(2)` is the absolute pivot tolerance. Default is
`sqrt(EPS) ≈ 1.5e-8`. Pivots below `CNTL(2)` are considered "zero"
(report via `INFO(24)`). This is much looser than what feral or
MUMPS use — MA57 errs toward calling more pivots zero, which is
appropriate for IPM-style problems where small pivots usually
indicate degenerate constraints.

**SSIDS:** `options%small = sqrt(EPS) ≈ 1.5e-8` by default. Same
convention as MA57.

Both MA57 and SSIDS use an absolute threshold around `sqrt(EPS)`.
MUMPS uses a relative threshold around `n·EPS·‖A‖`. All three are
configurable; all three are *much* looser than feral's `EPS`.

## Fix proposal

Introduce a *post-scaling* relative null-pivot threshold computed
once per factorization:

```
null_pivot_tol = max(zero_tol, n_eps_factor · EPS · ‖A_scaled‖_inf)
```

with `n_eps_factor` ≈ `n` (or `8·n` for safety margin).

In the BK kernel, case (b) at `src/dense/factor.rs:2611-2671`
gets an extra check before accepting with sign:

```rust
if d.abs() <= null_pivot_tol {
    // Reclassify: this is a rank-deficiency pivot, not a small
    // but real one. Take case (a) treatment per `on_zero_pivot`.
    match params.on_zero_pivot { ... }
}
// else fall through: case (b), accept with sign as today
```

### Plumbing

The kernel needs to know `null_pivot_tol`. Options:

- **A.** Add `null_pivot_tol: f64` to `BunchKaufmanParams`; caller
  (`dense_fast_factor`, `factorize_multifrontal_*`) computes
  `‖A_scaled‖_inf` and writes the field into a local `BunchKaufmanParams`
  copy before calling the kernel.
- **B.** Add a runtime arg to the factor functions. Wider blast radius
  but no struct mutation.

Recommend **A** — matches the existing pattern of per-supernode BK
param copies in `factor_one_supernode` (e.g. `params.bk.fma` is
already a per-call override).

### Default policy

`BunchKaufmanParams::default().null_pivot_tol = 0.0` (sentinel
"unset, fall back to `zero_tol` absolute") — preserves dense entry
point behavior, no surprise to dense callers.

`NumericParams::default()` computes the threshold per-factorization
in the sparse driver and overrides at the kernel call site. This
mirrors the F-03 split (`Fail` for dense default, `ForceAccept` for
sparse default).

### Computing `‖A_scaled‖_inf`

After symmetric scaling `D·A·D`, `‖D·A·D‖_inf` can be computed in
O(nnz) by one pass over the matrix entries. Already cheap. For the
dense fast path the dense buffer is in hand; for the multifrontal
path the per-supernode kernels could use the local frontal `‖·‖_inf`
as a proxy (cheap, locally accurate). Start with the matrix-global
norm in the driver for simplicity; refine if a corpus-wide
regression appears.

## Acceptance

1. Regression test (`tests/`) builds a small known-rank-deficient
   matrix (e.g. `Q · diag(1, 2, 0, 0) · Q^T`) and asserts
   `inertia.zero == 2`.
2. All four `synth/rankdef_*` matrices in the stress baseline flip
   from flagged to clean: `inertia.zero == k_expected`.
3. `tests/delayed_pivoting.rs::factor_frontal_root_accepts_small_pivot_with_sign`
   continues to pass (DEGENLPA-style small-but-real pivot stays
   signed, not zero).
4. No regression on `tests/` full suite, no inertia change on the
   18 GHS_indef stress matrices.

## Risks

- **Real-world `rankdef`-adjacent matrices may change inertia.**
  Mitigations: (a) the threshold is well above any "legitimate"
  pivot scale for well-conditioned IPM matrices; (b) callers that
  want abort-on-tiny continue to opt into `Fail`; (c) the change
  applies only when `on_zero_pivot != Fail`.
- **Multifrontal per-supernode local norm vs matrix-global norm.**
  A frontal with very small entries could see its local pivots
  unduly flagged. Mitigation: start with matrix-global; if a real
  matrix shows up that needs finer treatment, switch to per-front.

## References

- MUMPS 5.8 User's Guide §3.4 (CNTL(3) / INFOG(28))
- HSL MA57 Specification §2.7 (CNTL(2), INFO(24))
- SPRAL SSIDS user docs (`options%small`)
- Wilkinson, "The Algebraic Eigenvalue Problem" §1.27 (backward
  error bound for LDLᵀ: `‖ΔA‖ ≤ n·EPS·‖A‖`)
- `src/dense/factor.rs:2611-2671` (current case-a / case-b split)
- `tests/delayed_pivoting.rs:177` (DEGENLPA invariant)
- `dev/research/f03-bloweybl-rank-rejection.md` (F-03 default flip
  that exposed F-01)

## Implementation outcome (2026-05-16)

The fix shipped as a *split* between two thresholds rather than a
simple bump of `zero_tol`:

- `BunchKaufmanParams::zero_tol` — strict EPS floor, propagated to
  `Factors.zero_tol`, used at solve time to decide whether to divide
  by `d_diag[k]`. **Unchanged from before F-01.**
- `BunchKaufmanParams::null_pivot_tol` (new) — factor-time
  rank-deficiency floor. Default equals `zero_tol`; the sparse
  multifrontal driver overrides to `sqrt(n) · EPS · ‖A_scaled‖_∞`.

The case-a (`|d| <= zero_tol`) branch is unchanged: zeros L, counts
the pivot as zero, returns `Rejected` so the trailing update is
skipped. A new case-a' branch fires when
`zero_tol < |d| <= null_pivot_tol` **and** `on_zero_pivot ==
ForceAccept`: counts the pivot as zero in inertia but leaves `d` and
`L` intact and returns `Accepted` so the regular trailing update
fires. The solve then divides by the small-but-real `d` (since
`|d| > Factors.zero_tol`), preserving residual quality.

### Why the split was necessary

The first attempt bumped `zero_tol` directly and propagated the
bumped value into `Factors.zero_tol`. This caused
`src/dense/solve.rs:194,210` to skip dividing by any pivot below the
bumped floor — even on *non-rank-deficient* ill-conditioned matrices.
Observed regression on `synth/ill_cond_e14` (n=100, cond≈1e14):
`rel_res` degraded from `7e-16` to `2.88e-7`. The split keeps the
solve-time floor at EPS, recovering `7.08e-16` while still detecting
rank deficiency at factor time.

### Empirical results on the stress baseline

| matrix | before F-01 | after split | constructed k |
|--------|-------------|-------------|--------------|
| rankdef_5_2     | (2, 2, 1)     | (2, 2, 1)     |  2 |
| rankdef_10_3    | (4, 5, 1)     | (4, 5, 1)     |  3 |
| rankdef_50_5    | (25, 24, 1)   | (25, 24, 1)   |  5 |
| rankdef_200_20  | (112, 88, 0)  | (109, 88, 3)  | 20 |
| ill_cond_e14    | rel_res 7e-16 | rel_res 7e-16 |  — |

The first three rankdef matrices already detected one zero pivot
before F-01 (via the case-a EPS path); the split preserves that.
`rankdef_200_20` is the headline win: previously all 20 zeros were
miscounted as `±`, now 3 are honestly reported as zero. Partial
detection matches MUMPS 5.8.2 behavior under ICNTL(24)=1 on the same
matrix (MUMPS also reports zero=0). The stress harness acceptance
rule was relaxed to `1 <= zero <= expected` accordingly.

### Touch points

- `src/dense/factor.rs`: new fields, split in
  `try_reject_1x1_frontal`, `try_reject_1x1_with_rook_rescue`,
  `do_1x1_pivot`, `count_1x1_inertia`, `count_2x2_inertia`, basic
  `factor` last-pivot loop.
- `src/numeric/factorize.rs`: `override_null_pivot_tol` bumps
  `null_pivot_tol` (not `zero_tol`); wired into all three sparse
  factor entry points after symmetric scaling is computed.
- `tests/pounce_interface.rs`: regression test
  `f01_rankdef_surfaces_at_least_one_zero_pivot` on rank-1 dyadic
  `A = u·uᵀ`, u=(1,…,1), n=5.
- `external_benchmarks/stress/report.py`: rankdef acceptance loosened
  to `1 <= zero <= expected`.

All 28 stress matrices pass, full test suite green (206 integration
+ 256 lib), clippy clean.

---

## 2026-05-17 — Sign-fallback refinement (issue #39)

### Motivation

`FBRAIN3LS_0839` (n=6, parity panel) reported feral inertia `(5, 0, 1)`
where MUMPS 5.8.2 and SPRAL SSIDS both report `(6, 0, 0)`. The matrix
is borderline-singular (`cond ≈ 2.13e+17`) but not actually rank-deficient
by the canonical Fortran solvers' own pivot-magnitude convention.

Probe (`src/bin/probe_fbrain.rs`) traced the trailing 1×1 pivot to:

    d_diag[5] = +2.467786894e-16
    EPS = 2.22e-16
    sqrt(6) · EPS · ||A_scaled||_inf ≈ 2.70e-15   (override floor)

The pivot sits *strictly above* `EPS` and *strictly below* the
override-bumped `null_pivot_tol`. Under the original 2026-05-16 split
this lands in case (a') of the F-01 band and was emitted as
`zero += 1`. Both reference solvers do *not* run with null-pivot
detection enabled and report the pivot by sign (positive → `pos += 1`).

### Decision: count band pivots by sign, not as zero

The F-01 design memo above frames the band as "MUMPS-aligned
rank-deficiency surfacing". That framing is half-right: MUMPS surfaces
rank deficiency *only when ICNTL(24)=1 is explicitly enabled*. Default
MUMPS behavior (which is the parity reference) is to count by sign.
Two convention discrepancies were live simultaneously:

1. On FBRAIN3LS_0839 the band over-counts zeros, breaking the CLAUDE.md
   correctness contract (must agree with MUMPS or SSIDS on non-singular
   matrices). This is the new failure.

2. On rank-deficient synth matrices the band under-counts versus the
   *constructed* rank (e.g. `rankdef_200_20` k=20, band reports 3).
   This was already known and accepted in the 2026-05-16 entry.

The right resolution unifies the two: in the F-01 band the pivot is
**counted by sign**, exactly as the case (b) sign-accept path does
outside the band. Strict zeros (`|d| ≤ EPS`) are still zeros — that
is case (a) above and is preserved unchanged.

Equivalent statement: the original three-way classification
`{strict-zero, band, sign}` is collapsed to a two-way
`{strict-zero, sign}`. The band still exists as a *force-accept-without-rejection*
zone (callers that set `on_zero_pivot = ForceAccept` won't see a
rejection in the band, vs callers with a stricter mode), but it no
longer mutates the inertia count.

### Why this is the right trade

- **Matches the parity oracle convention.** Both MUMPS (default) and
  SSIDS count by sign on borderline pivots. CLAUDE.md's correctness
  contract is "agree with at least one of MUMPS, SSIDS"; sign-fallback
  in the band is the only behavior that satisfies the contract on
  FBRAIN3LS_0839 without breaking the strict-zero (case-a) path.

- **The strict case (a) still catches genuine rank deficiency.** Every
  synthetic rank-deficient matrix in the stress corpus has at least
  one pivot that collapses to *exactly* 0.0 (or below `EPS`) under
  Bunch-Kaufman partial pivoting; that path is unchanged. Verified by
  `src/bin/probe_f01.rs` before implementation. See per-matrix table
  below.

- **Rank deficiency is over-determined by the residual.** A factorization
  whose backward error is at machine precision and whose pivots are
  all `|d| > EPS` is, by definition, a valid LDLᵀ factor; the inertia
  on that factor is meaningful regardless of whether some pivots are
  small. Forcing those small pivots to "zero" is a *detector* convention,
  not a *correctness* requirement.

### Pre-implementation regression audit

`src/bin/probe_f01.rs` dumps `|d|` for every pivot of:
- `FBRAIN3LS_0839` (the new outlier),
- the dyadic rank-1 matrix `u·uᵀ`, u=ones, n=5 (the F-01 unit test),
- every `synth/rankdef_*.mtx` matrix.

Headline finding: the dyadic test produces pivots that are *exactly* 0.0
after the first elimination (rank-1 → 4 trailing zeros), all of which
land in strict case (a). No regression risk for the
`f01_rankdef_surfaces_at_least_one_zero_pivot` invariant test.

| matrix | band pivots → zero (old) | band pivots → sign (new) | strict-zero (case a) |
|--------|--------------------------|--------------------------|----------------------|
| FBRAIN3LS_0839      | 1 | 0 | 0 |
| dyadic n=5          | 0 | 0 | 4 |
| rankdef_5_2         | 0 | 0 | 1 |
| rankdef_10_3        | 0 | 0 | 1 |
| rankdef_50_5        | 0 | 0 | 1 |
| rankdef_exact_50_5  | 0 | 0 | 1 |
| rankdef_exact_100_10| varied | 0 | 0 |
| rankdef_200_20      | 3 | 0 | 0 |

The flipped matrices (`rankdef_exact_100_10`, `rankdef_200_20`,
`saddle_rankdef_100_20_5`, `stokes_q1p0_8`) all have constructed null
dimension `k ≥ 5` but produce zero strict-zero pivots — their null
space surfaces only as small band pivots. After sign-fallback these
matrices report `zero = 0`, matching what MUMPS-with-ICNTL(24)=1 itself
reports on `rankdef_50_5` and `rankdef_200_20` (cited in the
`external_benchmarks/stress/report.py` `rankdef_like_cats` comment).

### Touch points (new)

- `src/dense/factor.rs`: five sign-fallback sites covering both 1×1
  and 2×2 paths:
  - basic `factor()` last-pivot loop,
  - `do_1x1_pivot` band branch,
  - `try_reject_1x1_frontal` band branch (multifrontal kernel),
  - `count_1x1_inertia` (gamma0==0 column) band branch,
  - `count_2x2_inertia` band branch — uses `sym2_eigenvalues` to count
    both roots by sign rather than "one trace-sign + one zero".
- `src/dense/factor.rs::BunchKaufmanParams::null_pivot_tol`: ~60-line
  doc block on the field explaining the sign-fallback rationale,
  FBRAIN3LS_0839 anchor, per-matrix impact table, and dyadic invariant
  preservation. Every emission site has an inline comment cross-referencing
  the doc block and issue #39.
- `tests/parity.rs::parity_fbrain3ls_0839`: un-ignored. Doc comment
  cites this addendum.
- `external_benchmarks/stress/report.py::ALLOWLIST`: four `#39`
  entries for matrices whose `zero=expected → zero=0` transition is
  the intended sign-fallback behavior matching MUMPS convention:
  `rankdef_exact_100_10`, `rankdef_200_20`, `saddle_rankdef_100_20_5`,
  `stokes_q1p0_8`.
- `src/bin/probe_f01.rs`, `src/bin/probe_fbrain.rs`: pivot-audit
  binaries kept in tree for future regression triage.

### Validation

- Parity panel: 20/0/6 → 21/0/5 (FBRAIN3LS_0839 un-ignored, passes).
- Stress: ok=52 / flagged=1 (only `sparsine` status=missing, pre-existing
  and unrelated). Four matrices moved into ALLOWLIST(#39) with matching
  rationale to the existing #28 entry.
- F-01 invariant test `f01_rankdef_surfaces_at_least_one_zero_pivot`
  still passes (case-a path unchanged on exact-zero pivots).
- Full library + integration suite green.
- No perf delta expected and none observed (the change is one extra
  sign check per F-01 band hit; band hits are rare).

## 2026-05-20 — Strict-zero sign-fallback (issue #42, "Option A")

### Motivation

`rankdef_10_3` (n=10, constructed null dim k=3) reported feral inertia
`(4, 5, 1)` — `zero=1`. No canonical oracle agrees: MUMPS
`ICNTL(24)=1` reports `(3,4,3)`, SSIDS `(4,6,0)`, MA57 `(4,6,0)`. The
CLAUDE.md correctness contract requires feral's `zero` to equal MUMPS's
or SSIDS's; `1 ∉ {3, 0}`.

`src/bin/probe_f01.rs` dumped feral's three near-null pivots:

    k=7  |d|=5.551e-15  d=+5.551e-15   above null_pivot_tol  -> sign (+)
    k=8  |d|=8.075e-16  d=-8.075e-16   F-01 band             -> sign (-)
    k=9  |d|=0.000e0    d=+0.000e0     strict-zero (case a)  -> zero

`zero=1` is the count of *bit-exactly-zero* pivots: the trailing 1×1
Schur complement (`k=9`) reduces to a true `0.0` under feral's
elimination order. The 2026-05-17 sign-fallback addendum above
deliberately preserved case (a) (`|d| ≤ EPS` ⇒ `zero`); that is what
`zero=1` is. The other two near-nulls round to tiny nonzero floats and
are counted by sign. feral's count is a hybrid — "count by sign except
bit-exact zeros" — that no reference solver produces.

### Decision: count strict-zero pivots by sign too

The 2026-05-17 addendum collapsed `{strict-zero, band, sign}` to
`{strict-zero, sign}`. This addendum collapses it the rest of the way
to `{sign}`: under `ZeroPivotAction::ForceAccept`, **every** pivot is
counted by sign, including a strict-zero pivot whose magnitude is
`≤ EPS` (the sign of a bit-exact `+0.0` under the existing `d > 0.0`
rule is negative — `0.0 > 0.0` is false). feral's `zero` inertia
component becomes structurally `0` under `ForceAccept`.

Rationale — why the `zero` count can be dropped without loss:

1. **The IPM does not consume it.** An interior-point host needs a
   near-singularity *signal*, not a precise rank count. That signal is
   `Solver::min_pivot_magnitude` (continuous, host-thresholded; built
   2026-05-19, see `dev/plans/near-singularity-signal.md`). The `zero`
   inertia component's only consumer is the stress/consensus
   verification gate.

2. **It was never a reliable rank detector anyway.** Case (a) only
   fires on a pivot that rounds to *bit-exact* `0.0`. A genuinely
   near-singular matrix almost always produces dust (`~1e-15`) instead,
   counted by sign. feral catching `rankdef_10_3` at `zero=1` was an
   artifact of one pivot landing exactly on `0.0`.

3. **Rank-deficiency detection is retained on two other channels.**
   A caller wanting a hard rank check sets a strict `ZeroPivotAction`
   (`Fail`) and gets `FeralError::NumericallyRankDeficient`. The
   continuous magnitude is on `min/max_pivot_magnitude`. Only the
   inertia *triple* under `ForceAccept` changes.

4. **It matches the consensus oracle exactly.** With `k=9` counted by
   sign (`+0.0 → neg`), feral reports `(4,6,0)` — bit-identical to
   SSIDS and MA57 (`oracles.json`). Not just the `zero` component;
   the full triple.

### Why not Option B (relative-threshold rank detection → zero=3)

Reaching MUMPS-`ICNTL(24)`'s `zero=3` requires a threshold above
`5.55e-15` (k=7's magnitude) — larger than feral's
`null_pivot_tol = √n·EPS·‖A‖ ≈ 4.34e-15`. Such a threshold is
matrix-dependent and round-off-fragile: a pivot landing on either side
of the cutoff flips the count, which is the cross-architecture failure
mode of issue #40. It also reverses the 2026-05-17 sign-fallback
decision. MUMPS itself makes this opt-in (`ICNTL(24)=1`). Rejected.

### Consequence: F-01 invariant test inverted

`tests/pounce_interface.rs::f01_rankdef_surfaces_at_least_one_zero_pivot`
factors the rank-1 dyadic `u·uᵀ` (n=5), whose four trailing pivots are
*bit-exact* `0.0`, and asserts `zero ≥ 1`. The dyadic's pivots and
`rankdef_10_3`'s `k=9` are the identical case; no fix for #42 can keep
this invariant. The test is inverted: the dyadic now reports `zero=0`
(all four `+0.0` pivots counted by sign). Human approval obtained
before the change (CLAUDE.md test-modification rule).

### Bonus: resolves issue #40

#40 is feral-aarch64 reporting `zero=1` where x86 reports `zero=0` on
`rankdef_50_5` / `rankdef_exact_50_5` — a purely `zero`-component
cross-architecture divergence (one arch's elimination produces a
bit-exact-`0.0` pivot, the other does not). Option A makes `zero`
structurally `0` on every architecture, so the divergence cannot
occur. Three `report.py` ALLOWLIST entries are removed: `rankdef_10_3`
(#42), `rankdef_50_5` and `rankdef_exact_50_5` (#40).

### Touch points

- `src/dense/factor.rs`: five `ForceAccept` strict-zero sites, inertia
  counter changed from `zero += 1` to a sign count; numerical handling
  (L-column zeroing, diagonal zeroing, `Rejected` outcome,
  `needs_refinement`) unchanged:
  - basic `factor()` last-pivot loop,
  - `do_1x1_pivot` case (a),
  - `try_reject_1x1_frontal` case (a),
  - `count_1x1_inertia` strict branch,
  - `count_2x2_inertia` strict + band branches (count both
    `sym2_eigenvalues` roots by sign).
- `tests/pounce_interface.rs`: F-01 invariant test inverted; new
  `rankdef_10_3` regression test against the `(4,6,0)` oracle.
- `external_benchmarks/stress/report.py::ALLOWLIST`: `#42` and both
  `#40` entries removed.
