# LU-path condition estimate (issue #94)

## Problem

`src/numeric/condition.rs` has a Hager–Higham 1-norm condition estimator
(`κ₁ = ‖A‖₁·‖A⁻¹‖₁`, Hager 1984 + Higham 1988, DLACON-style, ≤5 iters). Its
`estimate_inverse_norm_1` is bound to the **symmetric** `SparseFactors` and drives
each Hager iteration through `numeric::solve` (the symmetric solver), exploiting
the specialization `A⁻ᵀ = A⁻¹` so one iteration = one solve. It cannot see an LU
basis: there is no `rcond()`/`condition_estimate()` on `SparseLu`/`DenseLu`.

Downstream (discopt#364) wants a trustworthy κ estimate on the *unsymmetric LU
basis* the simplex factors, so a B&B node can decide "ill-conditioned → refine /
re-solve with perturbation" inside one engine instead of swapping solvers.

## Algorithm (unchanged math, new solve callback)

Hager–Higham estimates `‖B⁻¹‖₁` by a power iteration on the cube
`{x : ‖x‖₁ ≤ 1}`:

```
x₀ = (1/n,…,1/n)
repeat (≤ MAX_ITER):
    y  = B⁻¹ x           ;  est = ‖y‖₁
    if est stopped growing → break
    ξ  = sign(y)         (sign(0)=+1, LAPACK)
    z  = B⁻ᵀ ξ
    if ‖z‖∞ ≤ zᵀx → break         (local max on the cube)
    x  = e_j, j = argmax|z|;  cycle-guard on repeated j
refine (Higham): b_i = (-1)^i (1 + i/(n-1)); est = max(est, 2‖B⁻¹ b‖₁/(3n))
```

For a **symmetric** factor `B⁻ᵀ = B⁻¹`, so the `z`-solve reuses the same solver
(1 solve/iter). For the **unsymmetric LU** the two differ: `y = B⁻¹x` is `ftran`
and `z = B⁻ᵀξ` is `btran` (~2 solves/iter). Both are O(1) factorizations; the
factor is reused.

`‖B‖₁ = max_j Σ_i |B_ij|` is the max absolute **column** sum — for a general
(unsymmetric) matrix this is a straight column-sum, no symmetry doubling (unlike
`matrix_norm_1`, which reflects the stored lower triangle).

## Design: factor out the shared driver

Introduce in `condition.rs`:

- `trait HagerHighamOperator { dim(); apply_inverse(&mut [f64]); apply_inverse_transpose(&mut [f64]); }`
  — the two in-place solves `B⁻¹·rhs` / `B⁻ᵀ·rhs`.
- `hager_higham_inverse_norm_1<O: HagerHighamOperator>(op) -> Result<f64>` — the
  driver loop above, verbatim from the current `estimate_inverse_norm_1`.

Then:
- `estimate_inverse_norm_1(&SparseFactors)` builds a `SymmetricHager` adapter
  (holds the pooled `SolveWorkspace` + one in-buffer, `apply_inverse_transpose`
  delegates to `apply_inverse`) and calls the driver. **Math is bit-identical**:
  same operation order, same pooled `ws` (the N5 one-build test still holds).
- `SparseLu::condition_estimate_1(&mut self, b: &SparseColMatrix)` and
  `DenseLu::condition_estimate_1(&mut self, b: &GeneralMatrix)` build an adapter
  over `ftran`/`btran` and multiply the driver result by `b.one_norm()`.

Signature note: issue #94 sketches `-> f64`, but the LU solves and the dimension
check are fallible and the crate forbids `unwrap`/`expect` in `src/`, so the
methods return `Result<f64, FeralError>`, matching `estimate_condition_1norm`.

## Oracles (external, hand-computed)

- Identity `n×n`: κ₁ = 1 (both paths).
- Diagonal `diag(1, 1e3, 1e6)`: κ₁ = 1e6 exactly (Hager on a diagonal is exact).
- `B = [[1,2],[3,4]]`: `B⁻¹ = [[-2,1],[1.5,-0.5]]`, `‖B‖₁ = 6`, `‖B⁻¹‖₁ = 3.5`,
  κ₁ = 21 (hand). Estimate is a lower bound on `‖B⁻¹‖₁`, so assert
  `κ_est ∈ [κ_true/2, κ_true·(1+ε)]`.
- Dense/sparse parity on the same basis (estimates agree bit-for-bit).
- Dimension-mismatch (`b.m ≠ self.m`) → `Err`.

## Scope

Additive: shared driver + trait in `condition.rs`, two `condition_estimate_1`
methods, two `one_norm` matrix helpers. Reuses the existing, tested Hager loop.
