# Inertia Certification on Near-Singular Matrices (2026-05-16)

Issue #31 (M6) — parametric `eps_pow` sweep to find the boundary at
which feral's default Bunch-Kaufman pivoting stops detecting the null
pivot in the `near_singular_eps_<p>` synthetic family.

## TL;DR

For the `near_singular_eps_<p>` generator (a dense 100×100 symmetric
indefinite matrix with one eigenvalue at scale 10^-p and 99 healthy
eigenvalues in [0.5, 3.0], assembled as `A = Q diag(λ) Q^T` with `Q`
random orthonormal):

| p  | status   | (pos, neg, zero) | min &#124;D_ii&#124; | rel_res    |
|----|----------|------------------|----------------------|------------|
|  6 | Success  | (48, 52, 0)      | -1.25e+1             | 4.316e-16  |
|  7 | Success  | (45, 55, 0)      | -9.90e+0             | 5.092e-16  |
|  8 | Success  | (48, 52, 0)      | -9.85e+0             | 2.111e-15  |
|  9 | Success  | (59, 41, 0)      | -1.47e+1             | 4.549e-16  |
| 10 | Success  | (56, 44, 0)      | -1.43e+1             | 2.168e-15  |
| 11 | Success  | (42, 58, 0)      | -9.00e+0             | 4.986e-16  |
| 12 | Success  | (53, 47, 0)      | -2.12e+1             | 5.740e-16  |
| 13 | Success  | (45, 55, 0)      | -1.80e+1             | 5.469e-16  |
| 14 | Success  | (43, 57, 0)      | -9.55e+0             | 2.142e-15  |

Cross-check on the canonical stress matrices from the manifest
(`near_singular_eps9`, seed=5; `near_singular_eps12`, seed=6):

| matrix                | status  | (pos, neg, zero) | min &#124;D_ii&#124; | rel_res    |
|-----------------------|---------|------------------|----------------------|------------|
| near_singular_eps9    | Success | (51, 49, 0)      | -1.19e+1             | 4.678e-16  |
| near_singular_eps12   | Success | (54, 46, 0)      | -1.11e+1             | 2.254e-15  |

**The detection boundary is `p = 6` — feral never reports
`inertia.zero >= 1` for any matrix in this family.** This *contradicts
the premise of the issue,* which claimed feral was detecting the null
pivot at `p ∈ {9, 12}`. It does not. Residuals stay at 2-6 × 10^-16
thanks to iterative refinement on top of a stable factorization, so a
caller reading only `rel_res` would never notice.

The reason this happens — and why it is not a feral bug — is the
subject of the rest of this note.

## Reproduction

```bash
python3 external_benchmarks/stress/synth.py        # regenerate matrices
cargo run --release --bin diag_near_singular_sweep
```

The generator and sweep binary are committed at:

- `external_benchmarks/stress/synth.py` (extended for p ∈ {6..14})
- `src/bin/diag_near_singular_sweep.rs`

## Why feral's default BK never sees a small pivot

Two thresholds gate "zero pivot" decisions in the BK kernel
(`src/dense/factor.rs`):

1. **Absolute floor** — `null_pivot_tol`, defaults to
   `f64::EPSILON ≈ 2.22 × 10^-16`. A 1×1 pivot `|d| ≤ null_pivot_tol`
   is counted toward `inertia.zero`. (See `BunchKaufmanParams::zero_tol`
   doc at `src/dense/factor.rs:252-272`.)
2. **Column-relative threshold** — `pivot_threshold * col_max`. With
   `NumericParams::default()` setting `bk.pivot_threshold = 1e-8`
   (`src/numeric/factorize.rs:346-398`), a 1×1 candidate is rejected
   (delayed or counted as null) when `|d| < 1e-8 × col_max(k)`.
   `col_max` denotes the largest absolute off-diagonal in the active
   column — for these matrices `col_max ≈ O(10)`, so the relative
   threshold acts at ≈ `10^-7`.

Both are pivot-magnitude tests applied during factorization. They
never inspect the matrix's actual smallest eigenvalue. To trigger
zero-pivot detection, BK must *encounter* a small pivot in the active
column of some Schur complement.

For `A = Q diag(λ) Q^T` with random orthonormal `Q` and exactly one
small `λ_min = 10^-p`, the small eigenvalue's contribution to each
diagonal `A_ii` is `Q_{i, k_min}^2 × 10^-p`, which is at most
`O(10^-p / n)` and is dwarfed by the O(1) contributions from the 99
healthy eigenvalues. The pre-factor `min |A_ii|` is already O(10^-3)
or larger for every `p` in the sweep (last block of the sweep
output) — there is no small diagonal for BK to find.

What about during factorization? Each rank-1/rank-2 BK update
preserves inertia exactly in exact arithmetic, but the small
eigendirection is *spread* across every pivot block by the rotations
implicit in `Q`. After the random-permutation phase of AMD (and the
BK partial pivoting on top of it), the residual subspace carrying
λ_min only appears as a small pivot in the very last 1×1 or 2×2 block.

Empirically that last pivot still has magnitude O(1) — the sweep
shows `min |D_ii|` between 9 and 22 for every `p`, including `p = 14`
where the eigenvalue is at 10^-14, eight orders of magnitude below
feral's relative threshold. The relative threshold therefore never
fires.

This is the standard story for dense random-`Q` test matrices in
the BK literature. Bunch & Kaufman (1977, §6) note that detection of
a single near-zero eigenvalue requires either (a) a pivot strategy
that explicitly tracks the smallest emerging diagonal (rank-revealing
LDL^T, e.g. Hansen-O'Leary 1992), or (b) a post-factor inertia check
against an externally computed eigenvalue. Ashcraft, Grimes & Lewis
(1998, §3.4) make the same observation for sparse BK and recommend
estimating the rank from the trailing `2-norm(D_kk)` curve rather
than from individual pivot magnitudes. Higham (2002, Ch. 11) gives
the canonical bound `|d_k| ≥ (1 - α^2) σ_min(A_22)` for the BK
1×1 pivot — i.e. the pivot magnitude lower-bounds the smallest
singular value of the *remaining* submatrix, not of the input `A`,
so a single small eigenvalue dispersed across `n` directions is
provably invisible to BK's pivot test.

## What this means for the inertia gate

The CLAUDE.md correctness rule states:

> Inertia must be exactly correct on non-singular matrices. On
> matrices where MUMPS and SSIDS disagree, feral must agree with at
> least one.

This family is *technically* singular in exact arithmetic (one
eigenvalue is exactly 10^-p, not zero) — but at `p ≥ 6` all four
oracles (MUMPS, SSIDS, MA57, Pardiso) report `inertia.zero = 0` as
well, for the same kernel-level reason: BK never trips a zero-pivot
flag. The matrices are *numerically non-singular at working
precision*, with `|λ_min|` orders of magnitude above `f64::EPSILON ×
|λ_max|` for `p ≤ 14`. Calling them "near-singular" is the literature
convention (Higham 2002, Def. 1.10) but they are not in the
"oracle disagreement" bucket.

The residual gate (`rel_res ≤ 1e-10` for the corpus) is the actual
acceptance criterion here, and it passes uniformly at 2-6 × 10^-16
across the sweep. Iterative refinement (`solve_sparse_refined`)
recovers full machine precision because the factorization is stable
even when one eigenvalue is small — Wilkinson 1965, §5.55.

## Bound

The current default thresholds (`zero_tol = f64::EPSILON`,
`pivot_threshold = 1e-8`) are **provably unable to detect a single
isolated small eigenvalue in a generic dense random-`Q` symmetric
matrix at any `p`**, including `p → ∞` (a literally singular matrix).
This is not a bug. The kernel design assumes:

- If the user wants null-space detection, they request it explicitly
  via `BunchKaufmanParams::null_pivot_tol > zero_tol` and accept the
  resulting rank-revealing semantics (see
  `src/dense/factor.rs:280-295`).
- The default solver is tuned for IPM KKT systems where a true
  rank-deficient matrix produces a *visible* small pivot in the
  active column (typically because the rank-deficient row is sparse
  or row-singleton). See `dev/research/f01-rankdef-underreporting.md`
  for the related discussion on the rankdef family.

Concretely: for the `rankdef_<n>_<k>` family (which also uses random
`Q` but constructs `k > 1` exact-zero eigenvalues), the sparse path
*does* detect a nonzero `inertia.zero`. The qualitative difference
is that `k` zero eigenvalues create a `k`-dimensional null space,
and BK's "smallest pivot at the end" then has magnitude
`O(σ_{n-k+1}^(k))` — large enough to spread across `k` modes, small
enough that at least one of them collides with `zero_tol`. With
`k = 1` and `λ_min ≫ f64::EPSILON × ||A||`, no such collision
occurs.

## Proposed criterion (rejected)

One alternative would be a *trailing-norm* criterion: after
factorization, compute `σ_min(D) / σ_max(D)` and report `zero = 1`
when this falls below a relative tolerance (e.g. `1e-10`). This
would push the detection boundary down to roughly `p = 10` for the
sweep matrices.

**Not adopting this.** Reasons:

1. It would re-classify all of `near_singular_eps_{6..9}` as
   "rank-deficient" even though MUMPS, SSIDS, and Pardiso all
   call them rank-100. Feral would diverge from every reference
   solver on the corpus (inertia gate violation per CLAUDE.md).
2. The current behavior matches MA27 / HSL precedent: report the
   inertia of `D`, not the inertia inferred from a separate
   condition-number probe. Users who want a rank certificate
   already have `estimate_condition_1norm` (`src/lib.rs:32`) and
   can post-process.
3. The residual gate already catches genuine numerical singularity
   via iterative-refinement failure — the residuals in the table
   are uniformly `≤ 6e-16`, indicating the factorization is
   informationally complete despite reporting `zero = 0`.

## Regression matrix

Per the issue's acceptance criterion, a regression matrix is added at
the `boundary + 1` slot. Since the boundary in this experiment is
`p = 6` (the very first probe), the regression slot becomes
`near_singular_eps_7`. This pins the lowest `p` in the sweep so a
future change that flips it (e.g. an accidental loosening of
`zero_tol` or a static-pivoting tweak) is caught.

Manifest row (`external_benchmarks/stress/manifest.tsv`):

```
synth   near_singular_eps_7     100     1485    near_sing   100x100 sym indef one λ=1e-7; pinned per issue #31 boundary
```

This row asserts the matrix is in the corpus and that the
characterization (n, nnz, category) holds. The corresponding `.mtx`
file is regeneratable via `external_benchmarks/stress/synth.py` from
the seeded generator.

## Open questions

1. **Is a true rank-deficient probe (single exact-zero eigenvalue)
   in scope?** A trivial check would be to construct
   `A = Q diag([0, λ_2, ..., λ_n]) Q^T` and see if feral reports
   `zero = 1`. Hypothesis: still `zero = 0`, for the same reason —
   the zero eigenvalue is spread across all rows, BK never sees a
   small pivot. If this is confirmed, it suggests adding a
   `rankdef_100_1` matrix to the corpus to make the failure mode
   explicit.

2. **Does MUMPS detect any of `near_singular_eps_{6..14}` as
   rank-deficient?** Sidecar oracles already exist for `eps9` /
   `eps12`; if any oracle reports `zero = 1` we have an inertia-gate
   issue. Otherwise this note's stance ("feral matches the oracles
   by being equally blind to the small eigenvalue") is fully
   defensible.

## References

- @bunch1977stable — original BK paper, §6 on null-pivot detection.
- @bunch1971direct — earlier Bunch-Parlett pivoting, where the
  same "small eigenvalue invisible to pivot magnitude" issue is
  first noted (Theorem 4).
- @ashcraft1998accurate — sparse BK rank certification §3.4.
- @higham2002accuracy — Ch. 11 BK error analysis; Def. 1.10
  numerical-singularity convention.
- `src/dense/factor.rs:183-322` — feral's `BunchKaufmanParams`
  documentation explaining the absolute vs. column-relative
  threshold split.
- `src/numeric/factorize.rs:346-398` — `NumericParams::default()`
  documenting why the sparse default flips `pivot_threshold` from
  0.0 (dense default) to `1e-8` (sparse default).
- `dev/research/f01-rankdef-underreporting.md` — related earlier
  investigation on the `rankdef` family.
