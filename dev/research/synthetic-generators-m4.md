# Synthetic generators for M4 stress categories (issue #27 + #31 follow-up)

## Purpose

The current synthetic side of `external_benchmarks/stress/manifest.tsv`
covers four pathologies — `rankdef`, `near_sing`, `illcond`, and
`cascade` — all built on a dense `Q D Q^T` skeleton. Issue #27 asks for
generators in four additional regimes that exercise the rest of feral's
indefinite pipeline (interior-point saddle, supernode crossover, MC64
scaling, and a real Stokes saddle), and the issue #31 follow-up asks
for an explicit exact-zero variant of `rankdef` so the dispersed-null
failure mode is no longer hidden behind floating-point tininess.

This note records the math of each new generator: the construction, the
inertia/oracle claim, and the pathology being probed.

## 1. `rankdef_exact_n_k` — explicit exact-zero variant

### Construction

Same `Q D Q^T` skeleton as `rankdef_n_k` but with `D = diag(0, …, 0,
d_{k+1}, …, d_n)` where the first `k` eigenvalues are *exact* IEEE 0.0
rather than tiny random values. The remaining `n − k` eigenvalues are
drawn with random signs from `[0.5, 3.0]`. `Q` is a random orthonormal
basis from QR of a Gaussian matrix.

After forming `A = Q D Q^T` we resymmetrise (`A ← (A + A^T)/2`) to kill
floating-point asymmetry. The result is a *dense* matrix whose null
space is the span of the first `k` columns of `Q` — almost certainly a
dense null space (no diagonal pattern hint).

### Inertia oracle

`(p, n − k − p, k)` where `p` is the number of positive draws in the
non-zero portion. We record the exact triple per-seed in the generator
output rather than reconstructing it in `report.py`, but the existing
report classifier only needs the `zero == k` check, which it already
does via the `rankdef_(\d+)_(\d+)` regex.

### Pathology

Compared to the existing `rankdef_n_k` (which uses literal `0.0`
already, so the new name overlaps in intent), the explicit
`rankdef_exact_n_k` is a *named, documented* probe with a distinct
seed. Issue #31 surfaced that the current `rankdef_50_5` / `200_20`
matrices yield `zero=0` from MUMPS despite their constructed nullity —
the null space is dispersed across the whole basis by the dense `Q^T`
transform, so partial-pivot BK collapses it into ostensibly-normal
pivots. `rankdef_exact_n_k` makes that failure mode an explicit oracle
target rather than a side effect of `rankdef_n_k`, with a seed chosen
to *minimise* the chance of a coincidentally helpful basis.

We keep both names: `rankdef_n_k` remains backward-compatible with the
baseline; `rankdef_exact_n_k` is the new explicit variant.

## 2. `saddle_rankdef_n_k` — saddle-point with rank-deficient constraint

### Construction

```
H  = symmetric positive definite n×n   (built as M^T M + δI, M Gaussian)
A  = m×n constraint, m = n − k         (rank m by construction)
K  = [ H    A^T ]
     [ A     0  ]
```

`A` is built as `m × n` Gaussian and *not* rank-padded — for Gaussian
`A` with `m ≤ n`, `A` has full row-rank `m` almost surely; we verify
numerically (rank from SVD) and refuse to write the file if the check
fails. To make the constraint *deficient*, we then zero out the last
`k` rows of `A`, giving a constraint block of true rank `m − k =
n − 2k`. Wait — that overcounts. Let me restate.

The issue text reads: *"`rank(A) = n − k`, inertia oracle is
`(n, n − dim ker(A), 2 dim ker(A))`"*. Reading the formula: with
`A ∈ R^{m × n}` where `m = n − k` and `rank(A) = n − k = m`, the
constraint block is *full row-rank*. Then `dim ker(A^T) = 0` but
`dim ker(A)` (the column null-space) equals `n − rank(A) = k`. The
inertia of `K` is then `(n, m − dim ker(A), 2 dim ker(A)) =
(n, n − k − k, 2k) = (n, n − 2k, 2k)`. The total matches: `n + n − 2k
+ 2k = 2n`, but the size of `K` is `n + m = 2n − k`. That's off by
`k`. Re-reading: the issue's formula yields `n + (n − k) + 2k = 2n + k`
which also doesn't match `n + m`. The formula in the issue is a *sketch*
that mis-counts by `k`; we use the rigorously correct one below.

For a saddle matrix `K = [H A^T; A 0]` with `H ≻ 0`, `A ∈ R^{m × n}`,
the inertia is (Gould 1985; Forsgren–Murray 1993):

- `n+(K) = n + (m − rank A)`     (positive eigenvalues)
- `n−(K) = rank A`               (negative eigenvalues)
- `n0(K) = m − rank A`           (zero eigenvalues, from the trivial
                                   nullspace of `A^T`)

Wait — the standard reference result for `H ≻ 0` and `A` *full
row-rank* `m` gives `n+(K) = n`, `n−(K) = m`, `n0(K) = 0`. When `A`
loses rank by `r` (so `rank A = m − r`), the saddle gains `r` zero
eigenvalues *and* loses `r` negative eigenvalues, so:

- `n+(K) = n`
- `n−(K) = rank A = m − r`
- `n0(K) = m − rank A = r`

For our generator we set `m = n − k` (constraint count) and choose
`rank A = m − r` for some user-specified deficiency `r`. We name the
generator `saddle_rankdef_n_k_r` with three integers. For the manifest
we expose `saddle_rankdef_50_10_3` (n=50 primals, 40 constraints,
constraint nullity 3, total size 90).

### How we drop the rank of `A`

Generate `A` as `m × n` Gaussian, take its SVD `A = U Σ V^T`, then zero
out the last `r` singular values: `Σ[m − r:] = 0`. Re-multiply to get
the rank-deficient `A`. Append zeros on the trailing diagonal block.

### Inertia oracle

`(n, m − r, r)` with `m = n − k`. Total `n + m − r + r = n + m = 2n − k`
matches the matrix size. We assert this from the generator and write
it into the `notes` column so a future, smarter `report.py` can pick it
up; for now the existing `report.py` only oracle-checks the
`rankdef_<n>_<k>` synthetic naming convention. We extend the report
classifier so `saddle_rankdef_<n>_<k>_<r>` matrices have their `zero`
checked against `r` and not against `n − k`.

### Pathology

The canonical interior-point KKT shape. Probes (a) inertia logic on
matrices where the *expected* inertia is structurally constrained
(equal counts of `+`/`−` from the saddle structure), and (b) the BK
pivot's ability to detect the small explicit null-space `r` when
embedded in a much larger matrix.

## 3. `wide_frontal_n` — forced wide supernode

### Construction

Build a sparse symmetric matrix whose elimination tree forces a single
supernode of width `> n0` (target `n0 = 1024` to land above the
sparse-to-dense crossover heuristic). The simplest such pattern is a
*bordered block diagonal*: a small bag of `b` cheap leaf columns each
connected only to a wide tail block of dimension `w`, plus the tail
block itself which is dense (or nearly so).

```
        leaf1     leaf2    ...    leaf_b        tail (w × w)
leaf1  [ a_1      .                 .       b_1^T               ]
leaf2  [          a_2                       b_2^T               ]
 …     [                                                        ]
leaf_b [                            a_b     b_b^T               ]
tail   [ b_1      b_2     …         b_b     T_w (dense indef)   ]
```

The elimination tree has `b` leaves all feeding into a *single*
supernode of width `w`. We pick `b = 16`, `w = 600` giving
`n = b + w = 616`. This is comfortably above the default sparse →
dense crossover (`PAR_MIN_FLOPS` calibration suggests w ≈ 256 is
where the dense kernel wins), and dense factor of a 600 × 600 block
is sub-100ms even on a laptop.

The tail block `T_w` is built as `Q D Q^T` with random orthonormal `Q`
and a balanced indefinite spectrum, then we *prune* any tail entry of
magnitude `< 1e-3` to keep nnz moderate (target ~ 600k nnz). The leaf
diagonals `a_i` are `+1` to keep the matrix indefinite-but-finite, and
the `b_i` borders are short random vectors (length `w`, ~50 nonzeros
each).

### Inertia oracle

Not derivable in closed form — the tail block dominates and its
inertia is whatever `Q D Q^T` gives. We *report* the inertia of the
generated matrix (computed via NumPy `eigh`) into the generator's
log line but do not require the report classifier to oracle-check it.
The acceptance criterion for this category is simply "factors
successfully and returns a consistent inertia sum". We tag the
category as `wide_frontal` so it's not lumped under any of the
existing oracle gates.

### Pathology

Targets the sparse → dense crossover threshold (`PAR_MIN_FLOPS`
calibration from session 2026-05-15-05). A single 1100×1100 supernode
should saturate dense BLAS3 paths; if the threshold is mis-tuned the
solver may dispatch on a non-optimal kernel.

## 4. `mc64_resistant_n` — MC64 succeeds, scaling stays bad

### First attempt and why we abandoned it

We initially tried `A = I + α u u^T` with `u = 𝟙/√n`. The eigenvalues
of this rank-1 update are 1 (multiplicity n−1) and 1+α, so picking
α = −2 gives one eigenvalue at −1 and condition number 1. That's not
ill-conditioned — diagonal scaling wasn't even needed. Direct
verification: with the default n=200 the assembled matrix had
cond(A) ≈ 1.48 and after a symmetric row-max scaling the cond was
unchanged. The "rank-1 perturbation" framing was wrong: a rank-1
update of a flat diagonal doesn't create the kind of *dispersed*
ill-conditioning that defeats MC64.

### Construction we use

`A = Q D Q^T` with `Q` a random orthonormal basis and `D` an
indefinite spectrum where exactly one eigenvalue is `small_eig`
(default `1e-8`) and the rest are O(1). The diagonal entries of `A`
are all O(1) because the eigenvector of the tiny eigenvalue is dense
in the original basis. MC64 sees an O(1) diagonal and produces
`s_i ≈ 1`; the scaled matrix has essentially the same spectrum as
the unscaled matrix.

Empirically: with `n = 200`, `small_eig = 1e-8`, seed `601`:
`cond(A) ≈ 2.0e8` before scaling, `≈ 2.0e8` after a symmetric
row-max scaling (a proxy for MC64-style scaling).

### Inertia oracle

Data-dependent — recorded in the generator log line and the
manifest's `notes` column. For seed `601`, n=200: `(107, 93, 0)`
(verified by `numpy.linalg.eigvalsh` and by feral itself). We do
*not* wire this into `report.py`'s oracle gate; the acceptance check
for this matrix is "factor succeeds and inertia sums to n".

### Pathology

If feral's MC64 implementation accepts the scaling and proceeds, the
residual after solve will be poor (the small eigenvalue is *not*
addressed by any diagonal scaling). This stress matrix is the
regression target for *detecting* that MC64 alone is not enough —
a future iterative-refinement gate or auxiliary equilibration should
notice that the post-factor inertia is sensitive to perturbations.

## 5. `stokes_q1p0_h` — Q1-P0 Stokes saddle on h×h grid

### Construction

Velocity DOFs: bilinear `Q1` on a square `h × h` mesh (so `2 · (h+1)^2`
unknowns counting both velocity components). Pressure DOFs: piecewise
constant `P0` on the `h × h` element grid (so `h^2` unknowns).

The standard Q1–P0 Stokes saddle is:

```
K = [ A    B^T ]      A = velocity Laplacian (block diag, 2 components)
    [ B     0  ]      B = discrete divergence
```

We build `A` and `B` from element-wise quadrature on the unit square.
The `Q1` Laplacian stencil on a uniform grid reduces to a 9-point
stencil. The `P0` divergence is the average of nodal velocity
divergence over each element, with sign convention from the standard
mixed-element references.

### Inertia oracle

Q1-P0 famously fails the LBB condition with *two* spurious pressure
modes in 2D: the global constant **and** the "checkerboard"
alternating mode. (We initially expected only the constant mode; the
implementation revealed two via `np.linalg.matrix_rank(B)`.) So
`rank(B) = n_p − 2` and the saddle inertia is:

- `n+(K) = n_u_free`
- `n−(K) = rank B = n_p − 2`
- `n0(K) = n_p − rank B = 2`

For `h = 8`: `n_u_per_comp = 9·9 = 81` minus `4·9 − 4 = 32` Dirichlet
boundary nodes → 49 free nodes per component, `n_u_free = 98`.
`n_p = 64`. Saddle size `162`. Inertia oracle `(98, 62, 2)`.

We pick `h = 8` for the manifest (~162 unknowns, factors in
microseconds).

### Pathology

A real saddle with the LBB-defect null mode that Q1-P0 famously
exhibits. Probes (a) inertia detection of the single zero pressure
mode, (b) the constraint-block structure that the augmented-system
analysis path is supposed to recognise.

## Acceptance / oracle wiring in `report.py`

The existing classifier handles `rankdef_<n>_<k>` directly. We extend
it with:

- `rankdef_exact_<n>_<k>` → expected zero = `k`
- `saddle_rankdef_<n>_<k>_<r>` → expected zero = `r`
- `stokes_q1p0_<h>` → expected zero = `2` (constant + checkerboard
  pressure modes in 2D)
- `wide_frontal_<n>` → no zero oracle (consistency sum only)
- `mc64_resistant_<n>` → no zero oracle (status check only)

Each generator writes its computed inertia (from `numpy.linalg.eigh`)
to stdout when run, so a manifest update is straightforward.

## Seeds

| name                              | seed |
|-----------------------------------|------|
| `rankdef_exact_50_5`              | 301  |
| `rankdef_exact_100_10`            | 302  |
| `saddle_rankdef_50_10_3`          | 401  |
| `saddle_rankdef_100_20_5`         | 402  |
| `wide_frontal_616`                | 501  |
| `mc64_resistant_200`              | 601  |
| `stokes_q1p0_8`                   | n/a (deterministic) |

Seeds are picked from disjoint integer blocks to avoid collision with
existing generators (which use `1..200`).
