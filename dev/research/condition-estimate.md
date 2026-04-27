# F2.0 — Condition-Number Estimate: Research Note

**Date:** 2026-04-27
**Phase:** F2.0 of `dev/plans/kkt-feature-gaps.md`
**Goal:** Decide the algorithm, API, and validation strategy for a
1-norm condition estimator before writing F2.1 code.

## Why we want this

Ipopt's δ-ladder for inertia correction (`δ_w` Hessian
perturbation, `δ_c` constraint relaxation) is currently driven by
the inertia signal only — feral can detect "wrong inertia" but
cannot distinguish a *structurally* wrong factor (genuine
indefiniteness in the wrong direction) from an *ill-conditioned*
factor (bad scaling that more regularization will not fix).

ripopt's downstream choice of when to escalate δ_w vs when to
trigger restoration depends on knowing κ̂(A). MUMPS exposes this
via `INFOG(40)` when `ICNTL(11)=2`; SSIDS via `solve_inquiry` —
both compute Hager-Higham 1-norm estimates. feral has no
equivalent today.

Other consumers:
- Iterative-refinement termination (Skeel 1980): refine until
  forward-error bound `κ̂ · ε_machine` is reached, not just until
  residual is tiny.
- ripopt diagnostic logging (per `dev/plans/ripopt-integration.md`
  §F2): per-iteration `kappa_1_est` field for observability.
- Quality-of-factor heuristics for adaptive pivot threshold.

## Reference solver behavior

### MUMPS (Fortran 5.8.2)

`ICNTL(11)` — error-analysis control:
- 0 (default): no error analysis.
- 1: full statistics including condition number estimates,
  componentwise and normwise backward errors. Cost: ~5 extra
  solves.
- 2: condition number only (cheaper than 1). Cost: ~3-5 solves.

When enabled, MUMPS populates:
- `RINFOG(7)` — componentwise backward error.
- `RINFOG(8)` — normwise backward error.
- `RINFOG(10)` — `||A||₁`.
- `RINFOG(11)` — `||A⁻¹||₁` estimate (the κ piece we want).
- `INFOG(40)` is *not* a condition number — that was an error in
  the F2 motivation in `kkt-feature-gaps.md`. The actual MUMPS
  field is `RINFOG(11)`. (See MUMPS user manual §6.2 table for
  `RINFOG`.) Note this when updating the plan.

Internal algorithm: MUMPS calls Higham's `DLACON`-style 1-norm
estimator (LAPACK auxiliary), which is an implementation of
Hager 1984 with the Higham 1988 termination refinement.

### SSIDS (C++/Fortran)

`solve_inquiry` exposes per-RHS forward and backward errors. The
condition estimate path is in `core_solve.f90::condition_est_1`
and uses the same Hager-Higham power-iteration scheme. SSIDS
documents the cost as 4-5 solves with the stored factor.

### LAPACK reference: `DGECON` / `DLACON`

- `DGECON(NORM, N, A, LDA, ANORM, RCOND, ...)` — given `||A||`
  and a factor of A, returns `RCOND = 1 / (||A||·||A⁻¹||)`.
- `DLACON(N, V, X, ISGN, EST, KASE)` — reverse-communication
  driver implementing Hager's 1984 algorithm. `KASE` returns the
  operation type the caller has to perform (`Ax` or `Aᵀx`); the
  driver iterates internally until `EST` (the running estimate
  of `||A⁻¹||₁`) stops increasing.

DLACON is the canonical reference and a clean-room Rust port is
straightforward (~80 lines).

## Algorithm: Hager-Higham 1-norm power iteration

Given a factor of `A` (so we can compute `A⁻¹b` cheaply for any
RHS `b`), estimate `||A⁻¹||₁`:

1. **Initialize** `x = (1/n, 1/n, ..., 1/n)`. Set
   `est_old = 0`, `kase = 1`.
2. **Repeat:**
   - Compute `y = A⁻¹ x` (one solve).
   - Compute `est = ||y||₁`.
   - If `est ≤ est_old`, terminate (return `est_old`).
   - Compute `ξ = sign(y)` componentwise.
   - Compute `z = A⁻ᵀ ξ` (one solve).
   - If `||z||_∞ ≤ z·x`, terminate (return `est`).
   - Else set `x = e_j` where `j = argmax|z_i|`, `est_old = est`.
3. **Higham 1988 refinement**: after termination, do one final
   solve with the alternating-sign vector `b_i = (-1)^{i+1}·(1 +
   (i-1)/(n-1))` and replace `est` with `2·||A⁻¹b||₁ / (3n)` if
   that is larger. This catches the ~5% of cases where Hager
   1984 underestimates by a factor of 2 or more.

Termination: 3-5 iterations in practice; LAPACK caps at 5.

### Symmetric specialization

For symmetric `A` factored as `P A Pᵀ = L D Lᵀ`:

- `A⁻¹ = Pᵀ L⁻ᵀ D⁻¹ L⁻¹ P`
- `A⁻ᵀ = A⁻¹` (symmetry)

So **the inner loop uses one solve per iteration, not two**.
This halves the cost vs the general case. Total: 3-5 solves
total instead of 6-10.

For an indefinite `D` with 2×2 blocks (Bunch-Kaufman pivots), the
solve still works unchanged — `D⁻¹` is applied block-wise. No
algorithmic change in the estimator.

### Computing `||A||₁`

`||A||₁ = max over columns j of sum_i |A[i,j]|`. For a CSC
matrix this is a single linear pass. For symmetric storage
(upper-triangle only), a separate pass forms the column sum
including the symmetric mirror entries.

The estimator returns `κ̂_1 = ||A||₁ · ||A⁻¹||₁`. Both pieces
are computed; the user-facing API can return them separately or
combined.

## Decisions

### D1. Algorithm: Hager 1984 + Higham 1988 refinement

Implement DLACON-style reverse communication is overkill for a
clean-room Rust port; we have direct access to the factor and
can write the loop inline. Use the **forward** (non-reverse-
communication) form. The refinement step is small (~10 lines)
and catches the documented underestimation cases — include it.

### D2. Symmetric-specific solve path

Use the symmetric identity `A⁻ᵀ = A⁻¹`. This is correct for the
symmetric indefinite factor we produce. Code path:

```rust
loop {
    let y = solver.solve(&x)?;       // A⁻¹ x
    let est = l1_norm(&y);
    if est <= est_old { break est_old; }
    let xi = signs(&y);
    let z = solver.solve(&xi)?;      // A⁻ᵀ ξ = A⁻¹ ξ
    if linf(&z) <= dot(&z, &x) { break est; }
    x = unit_vector(argmax_abs(&z));
    est_old = est;
}
```

### D3. API shape

```rust
impl Solver {
    /// Hager-Higham 1-norm condition estimate.
    /// Returns ||A||_1 * ||A^{-1}||_1.
    /// Cost: 3-5 solves with the stored factor.
    pub fn estimate_condition_1norm(&mut self, matrix: &CscMatrix)
        -> Result<f64, FeralError>;

    /// Lower-level: returns the components separately.
    pub fn estimate_inverse_norm_1(&mut self)
        -> Result<f64, FeralError>;
}
```

`estimate_condition_1norm` is the user-facing single-call form.
The lower-level `estimate_inverse_norm_1` lets a caller who has
already computed `||A||_1` (e.g., from a separate analysis pass)
skip recomputing it.

`&mut self`: the estimator allocates a small workspace
(`SolveWorkspace` reuse if available, two `Vec<f64>` of length
`n` otherwise). `&mut` makes the workspace lifecycle explicit.

### D4. Free-function form

Mirror the multi-RHS surface — also expose:

```rust
pub fn estimate_condition_1norm_free(
    matrix: &CscMatrix,
    factors: &SparseFactors,
) -> Result<f64, FeralError>;
```

For callers who hold `SparseFactors` directly without going
through `Solver`. Internally allocates a `SolveWorkspace`.

### D5. Failure modes

The estimator returns `Err` only on:
- Solve failure during the inner loop (factor is singular —
  return `FeralError::SingularFactor`).
- Dimension mismatch between `matrix` and the factor.

It does *not* return `Err` for "estimator did not converge in 5
iterations" — Hager-Higham's contract is that the running
estimate is monotone-non-decreasing and 5 iterations is the
LAPACK-blessed cap. Return whatever the iteration produced.

Special case `n == 0`: return `Ok(0.0)`. (Convention: empty
matrix has condition 0, matching the convention for an empty
sum.)

### D6. Workspace reuse

The estimator does ~5 solves. Allocating fresh `SolveWorkspace`
per solve is wasteful. Add a `ConditionWorkspace` analogous to
`SolveWorkspace`:

```rust
pub struct ConditionWorkspace {
    solve_ws: SolveWorkspace,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    xi: Vec<f64>,
}
```

Allocated once, reused across iterations. The free-function form
`estimate_condition_1norm_free` allocates internally; an
`_into_ws` form lets callers in a hot loop reuse.

## Validation strategy

### Calibration set (F2.1 acceptance gate)

Three matrix families with known κ:

1. **Hilbert matrices** `H_n[i,j] = 1/(i+j-1)` for n=4..10. κ
   grows as `O((1+√2)^{4n} / sqrt(n))`. Documented exact values
   in Higham 2002 §28.1 — our estimate must be within 10× (Hager
   is a lower bound; 10× factor is the textbook conservative
   tolerance).

2. **Diagonal-of-known-spectrum**: `A = diag(σ_1, ..., σ_n)`
   with σ chosen so `κ_1 = max/min` is exactly a target value.
   Test points: 1.0, 1e3, 1e6, 1e10, 1e15. Estimator must be
   within 2× of true value (diagonal case is trivial — Hager
   converges in 1 iteration).

3. **Random KKT panels with known SVD**: build `A = QΛQᵀ` with Q
   random orthogonal and Λ a chosen indefinite spectrum. Test
   that κ̂ tracks `max|λ|/min|λ|` within 10×.

### Cross-validation set (F2.2 acceptance gate)

Run feral's estimator on N ≥ 1000 corpus matrices for which the
MUMPS oracle has produced `RINFOG(11)` data. Compare:

- **Geomean ratio** `feral_kappa / mumps_kappa` should be in
  [0.5, 5.0]. Both estimators are statistical lower bounds; they
  use the same algorithm, so disagreement above 5× indicates a
  bug in one of them.
- **Per-matrix ratio** can vary more widely (Hager is non-
  deterministic in the unit-vector tiebreak); the 5× geomean
  bound captures systematic-error-only.

This requires the MUMPS oracle to be re-run with `ICNTL(11)=2`
enabled. Add this to the F2.2 task: extend
`external_benchmarks/mumps_oracle/mumps_bench.F` to write
`RINFOG(11)` to verdict.json.

### Negative-control set

The estimator should *not* report κ̂ < 1. Run on a few
deliberately well-conditioned matrices (orthogonal matrices,
identity scalings) and verify κ̂ ≥ 1 - sqrt(eps).

## Code-touch map (anticipating F2.1 implementation)

New file `src/numeric/condition.rs`:

```rust
pub struct ConditionWorkspace { ... }
impl ConditionWorkspace {
    pub fn for_factors(factors: &SparseFactors) -> Self { ... }
}

pub fn matrix_norm_1(matrix: &CscMatrix) -> f64 { ... }
pub fn estimate_inverse_norm_1(
    factors: &SparseFactors,
    n: usize,
    ws: &mut ConditionWorkspace,
) -> Result<f64, FeralError> { ... }

pub fn estimate_condition_1norm_free(
    matrix: &CscMatrix,
    factors: &SparseFactors,
) -> Result<f64, FeralError> {
    let anorm = matrix_norm_1(matrix);
    let mut ws = ConditionWorkspace::for_factors(factors);
    let inv_norm = estimate_inverse_norm_1(factors, matrix.n, &mut ws)?;
    Ok(anorm * inv_norm)
}
```

Modifications to `src/numeric/solver.rs`:
- Add `Solver::estimate_condition_1norm(&mut self, matrix)`.
- Add `Solver::estimate_inverse_norm_1(&mut self)` if useful.

Modifications to `src/lib.rs`:
- Re-export `estimate_condition_1norm_free` and the workspace
  type.

Tests in `tests/condition.rs`:
- Hilbert κ matches Higham table within 10×.
- Diagonal-spectrum κ matches max/min within 2×.
- κ ≥ 1 on identity-scaled orthogonal matrices.
- Cross-validation harness skeleton (MUMPS comparison wired in
  F2.2).

## Cost analysis

Per call:
- 1 pass to compute `||A||_1`: O(nnz).
- 3-5 solves with the stored factor: O(k · nnz(L)) where k is
  iterations.
- Negligible vector ops (signs, l1, dot, argmax).

For the small-frontal panel (n ≈ 100, nnz(L) ≈ 1000):
- Estimator cost ≈ 5 × solve cost ≈ 50 µs.
- F2 acceptance gate per `kkt-feature-gaps.md`: estimator should
  add ≤ 5% to the *invoked-on-demand* solve cost. Since the
  estimator is opt-in (not invoked per IPM iteration unless the
  caller asks), the "5%" is really "5% when called", which is
  trivially met — the alternative is "do the 5 solves yourself
  with no helper API".

## Open questions (close before F2.1)

1. **Should the estimator be on `Solver` or only on
   `SparseFactors`?** Decision: both. `Solver::estimate_*` is
   the convenience entry; the free fn for callers who hold
   `SparseFactors` directly. Mirrors the `solve_sparse` pattern.

2. **Should we expose `||A||_1` and `||A⁻¹||_1` separately?**
   Yes, via `estimate_inverse_norm_1`. The combined form is the
   default but the components are useful for diagnostic logging
   (a high κ̂ with small `||A||_1` means the inverse is the
   problem; with small `||A⁻¹||_1` and big `||A||_1`, the
   problem is scaling).

3. **Should the estimator try to enforce a sign convention on
   `ξ` ties?** LAPACK's DLACON uses `sign(0) = +1`. We adopt the
   same convention. The estimator is statistically correct for
   either tiebreak and matching LAPACK simplifies cross-
   validation.

4. **F2.3 wiring into iterative refinement: change behavior or
   diagnostic-only?** Per the F2 plan: diagnostic only in F2.3.
   Adaptive refinement termination is a research topic; the
   first version exposes the estimate without using it.

## References

- Hager, W. W. (1984). "Condition Estimates." *SIAM J. Sci.
  Stat. Comput.* 5(2), 311-316.
- Higham, N. J. (1988). "FORTRAN codes for estimating the
  one-norm of a real or complex matrix, with applications to
  condition estimation." *ACM Trans. Math. Softw.* 14(4),
  381-396.
- Higham, N. J. (2002). "Accuracy and Stability of Numerical
  Algorithms" 2nd ed. SIAM. §15 (condition number estimation),
  §28 (Hilbert matrix κ tables).
- LAPACK source: `DLACON` (auxiliary 1-norm estimator),
  `DGECON` (driver).
- MUMPS user manual 5.8.2 §6.2 — `RINFOG(11)`, `ICNTL(11)`.
- SSIDS source: `core_solve.f90::condition_est_1`.

## Errata for `kkt-feature-gaps.md`

The F2 motivation references "MUMPS's `INFOG(40)`". The correct
field is `RINFOG(11)` (real-valued, indexed at 11 in the
RINFOG array). Fix the plan when this note is committed.
