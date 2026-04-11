# Dense LDLᵀ Factorization with Bunch-Kaufman Pivoting

**Status:** Pre-implementation research note (mandatory per Section 13.3)
**Date:** 2026-04-11
**Related spec sections:** 2.4 (Pivot selection), 2.5 (Inertia counting), 2.9 (Solve phase), 3.3 (Implementation order)
**Key references:** citep:bunch1977stable, citep:bunch1971direct, citep:hogg2013pivoting

## 1. Overview

This note covers the algorithmic foundation for FERAL's first implementation
target: a scalar (unblocked) dense LDLᵀ factorization with Bunch-Kaufman
pivoting. The factorization computes

    P A Pᵀ = L D Lᵀ

where P is a permutation, L is unit lower triangular, and D is block diagonal
with 1×1 and 2×2 blocks. The inertia of A equals the inertia of D by
Sylvester's Law of Inertia (Section 2).

This note resolves three design questions raised during expert review before
implementation begins:

1. In-place storage vs. separate `Factors` struct (Section 7)
2. Fused update+argmax as a day-one requirement (Section 6)
3. Full symmetric row search as the first test target (Section 8)


## 2. Why Block Diagonal D is Necessary

For symmetric positive definite matrices, Cholesky factorization (A = LLᵀ)
exists and is numerically stable without pivoting. For symmetric indefinite
matrices, this fails: the diagonal may contain zeros or near-zeros even when
A is nonsingular.

**Existence theorem** (citep:bunch1971direct): For any nonsingular symmetric
matrix A, there exists a permutation P and a unit lower triangular L such that
P A Pᵀ = L D Lᵀ, where D is block diagonal with blocks of size 1×1 or 2×2.
The 2×2 blocks absorb the cases where no single diagonal element is a safe
pivot but a 2×2 submatrix is nonsingular.

For KKT matrices arising in interior-point methods, the expected inertia is
(n, m, 0) — n positive eigenvalues from the Hessian block and m negative
eigenvalues from the constraint block (citep:wachter2006ipopt). The block
diagonal D must capture this structure exactly; naive diagonal pivoting
(1×1 only) would fail on the indefinite structure.

**Inertia via Sylvester's Law:** The congruence transformation P A Pᵀ = L D Lᵀ
preserves inertia. Since L is nonsingular (unit diagonal), the inertia of A
equals the inertia of D. This means we can read off the inertia from D's blocks
without computing eigenvalues of A. This is the mechanism that makes LDLᵀ
factorization useful for the IPM application: factorize once, get both the
solution and the inertia.


## 3. Bunch-Kaufman Pivot Selection (Theorem 3 of BK77)

The Bunch-Kaufman algorithm (citep:bunch1977stable, Theorem 3) selects either a
1×1 or 2×2 pivot at each step by examining a single column of the trailing
submatrix. The key parameter is

    α = (1 + √17) / 8 ≈ 0.6404

which balances stability (bounded element growth) against the frequency of
2×2 blocks.

### 3.1 The Selection Procedure

At step k, with trailing submatrix A(k:n, k:n):

**Step 1.** Compute γ₀ = max_{i≠0} |A[i,0]|, the largest off-diagonal magnitude
in column 0 (relative to the trailing submatrix). Let r be the row index
achieving this maximum.

**Step 2.** If γ₀ = 0, or the trailing submatrix is 1×1, accept a 1×1 pivot.
The matrix is (numerically) reducible at this point.

**Step 3.** Test whether A[0,0] is acceptable as a 1×1 pivot:

    |A[0,0]| ≥ α · γ₀

If yes, use A[0,0] as a 1×1 pivot (no permutation needed).

**Step 4.** If Test 3 fails, compute γᵣ = max_{i≠r} |A[i,r]|, the largest
off-diagonal magnitude in the full symmetric row/column r. This requires
searching both below and to the left of position r (since only the lower
triangle is stored).

**Step 5.** Test whether A[r,r] is acceptable as a 1×1 pivot:

    |A[r,r]| ≥ α · γᵣ

If yes, swap rows/columns 0↔r and use A[r,r] as a 1×1 pivot.

**Step 6.** Test whether A[0,0] is still usable with the new information:

    |A[0,0]| · γᵣ ≥ α · γ₀²

If yes, use A[0,0] as a 1×1 pivot (no permutation needed). This test is
the "LAPACK 3-way extension" — LAPACK's `dsytf2` adds this test which is
not in the original BK77 Theorem 3 but reduces the frequency of 2×2 pivots.

**Step 7.** Otherwise, use the 2×2 block formed by rows/columns {0, r}.
Swap row/column 1↔r so the 2×2 block occupies positions {0, 1}.

### 3.2 Guarantees

The element growth factor is bounded by (1 + 1/α)^(n-1) ≈ 2.56^(n-1). This
is exponential in theory but the bound is rarely approached in practice.
The algorithm requires O(n) comparisons per step (one column search plus
potentially one row search), giving O(n²) comparison work over the full
factorization. The factorization itself is O(n³/3).

### 3.3 α vs. Threshold Pivoting Parameter u

The BK threshold α is fundamentally different from the threshold pivoting
parameter u used by SSIDS (u=0.01) and MUMPS (CNTL(1)=0.01) in their
multifrontal factorizations:

- **α (BK):** Controls the tradeoff between 1×1 and 2×2 pivots in a full
  search of each column/row. Larger α → stricter acceptance → more 2×2 blocks
  but better stability. Used in the dense kernel only.

- **u (TPP/APP):** Controls how aggressively pivots are accepted in the
  multifrontal context, where rejected pivots are *delayed* to the parent
  node rather than forcing a 2×2 block. Smaller u → more permissive → fewer
  delays but potentially worse stability. Used in sparse factorization.

FERAL's roadmap: start with BK (α=0.6404) for the dense kernel, then
transition to threshold partial pivoting (u=0.01) when the multifrontal
engine is implemented (citep:hogg2013pivoting).


## 4. The Elimination Step

After selecting a pivot (1×1 or 2×2), the trailing submatrix is updated via
a Schur complement.

### 4.1 Rank-1 Update (1×1 Pivot)

When d = A[k,k] is the 1×1 pivot:

    L[i,k] = A[i,k] / d           for i = k+1, ..., n
    A[i,j] -= L[i,k] · d · L[j,k]  for i ≥ j > k

This is a symmetric rank-1 update of the trailing submatrix.

### 4.2 Rank-2 Update (2×2 Pivot)

When the 2×2 block is:

    B = [ a₀₀   a₁₀ᴴ ]
        [ a₁₀   a₁₁  ]

the L columns below the block are computed by solving B × [L₀ L₁]ᵀ = [col₀ col₁]ᵀ,
then the trailing submatrix is updated by a symmetric rank-2 operation.

The naive approach computes B⁻¹ via Cramer's rule:

    det = a₀₀ · a₁₁ − |a₁₀|²
    L₀[j] = (a₁₁ · A[j,0] − a₁₀ · A[j,1]) / det
    L₁[j] = (a₀₀ · A[j,1] − a₁₀ᴴ · A[j,0]) / det

This is numerically problematic when |a₁₀| is large relative to the diagonal,
because det = a₀₀·a₁₁ − |a₁₀|² suffers catastrophic cancellation.

### 4.3 Normalized 2×2 Computation (from faer)

faer uses a normalization-by-|a₁₀| technique that avoids this cancellation.
The key insight: divide through by |a₁₀| before computing the determinant.

    d₁₀_abs = |a₁₀|
    d₀₀ = a₀₀ / d₁₀_abs
    d₁₁ = a₁₁ / d₁₀_abs
    t = 1 / (d₀₀ · d₁₁ − 1)       // this is |a₁₀|² / det(B)
    d₁₀ = a₁₀ / d₁₀_abs           // unit-magnitude, phase only
    d = t / d₁₀_abs                // = |a₁₀| / det(B)

Then the L columns and rank-2 update are computed as:

    For each row j below the 2×2 block:
        x₀ = A[j, 0],  x₁ = A[j, 1]
        w₀ = (x₀ · d₁₁ − x₁ · d₁₀) · d    // this is L₀[j]
        w₁ = (x₁ · d₀₀ − x₀ · d₁₀ᴴ) · d   // this is L₁[j]
        For each column i ≥ j below the block:
            A[i,j] -= A[i,0] · w₀ᴴ + A[i,1] · w₁ᴴ

The determinant test becomes d₀₀·d₁₁ − 1 instead of a₀₀·a₁₁ − |a₁₀|².
When |a₁₀| is large relative to the diagonal (the typical case for 2×2 BK
pivots), the normalized version avoids the large-minus-large cancellation.

**Implementation note:** Store d₀₀, d₁₁, d₁₀, and d for the rank-2 update.
These same quantities are needed in the solve phase (Section 5.2).


## 5. Inertia Counting from D Blocks

This is the most critical correctness requirement. FERAL must report exact
inertia counts — any error means the IPM layer cannot detect indefinite
Hessians or degenerate constraints.

### 5.1 Counting from 1×1 Blocks

Trivial:
- d > 0 → +1 positive
- d < 0 → +1 negative
- d = 0 → +1 zero

### 5.2 Counting from 2×2 Blocks

Given a 2×2 D block:

    B = [ a   b ]
        [ b   c ]

**DO NOT count the signs of a and c independently.** This gives wrong results
when the block is indefinite. For example, a = 1, c = 2, b = 3 has both
diagonal elements positive, but det = 1·2 − 9 = −7 < 0, so the block has
one positive and one negative eigenvalue.

The correct procedure uses the determinant:

    det = a·c − b²

- det > 0 and a > 0 → inertia (2, 0, 0) — both eigenvalues positive
- det > 0 and a < 0 → inertia (0, 2, 0) — both eigenvalues negative
- det < 0 → inertia (1, 1, 0) — one positive, one negative
- det = 0 → inertia (1, 0, 1) if a > 0, else (0, 1, 1)

**Why this works:** The eigenvalues of B are (tr ± √(tr² − 4·det)) / 2 where
tr = a + c. The sign of det determines whether the eigenvalues have the same
sign (det > 0) or opposite signs (det < 0). When det > 0, the sign of tr
(equivalently, the sign of a or c, since both have the same sign as tr when
det > 0) determines which sign.

**BK pivot selection guarantees:** For a properly-selected 2×2 BK pivot, the
off-diagonal |b| is large relative to the diagonal, which means det < 0 in
most cases (one positive, one negative eigenvalue). However, the code must
handle all cases correctly because near-zero pivots and degenerate matrices
can produce any inertia pattern.

### 5.3 Near-Zero Pivot Handling

When a 1×1 pivot |d| < zero_tol or a 2×2 block |det| < zero_tol_2x2:

- **ForceAccept mode:** Accept the pivot, report zero in the inertia count,
  flag the factorization for iterative refinement. The IPM layer (POUNCE)
  decides whether to perturb and refactor.

- **Fail mode:** Return `FactorStatus::NumericallyRankDeficient` immediately.
  The IPM layer adds regularization and calls again.

FERAL never decides perturbation amounts. This architectural boundary is
critical — it matches Ipopt's separation between `PDPerturbationHandler`
(in the IPM layer) and `SymLinearSolver` (the factorization interface).


## 6. Fused Update + Argmax

The expert review identified this as a day-one requirement, not a future
optimization. The reasoning:

The Schur complement update (Section 4) touches every element of the trailing
submatrix: O(n²) memory accesses. The pivot selection for the *next* step
(Section 3) requires scanning a column/row of the updated submatrix: another
O(n²) access in the worst case (when a row search is needed).

If these are separate passes, the trailing submatrix is read from memory twice
per pivot step. With the fused approach, the argmax tracking is embedded in
the update loop, and the trailing submatrix is read once.

### 6.1 Rank-1 Fused Update+Argmax

    max_abs = 0
    max_idx = k+1
    For j = k+1 to n:
        For i = j+1 to n:
            A[i,j] -= L[i,k] · d · L[j,k]
            if |A[i,j]| > max_abs:
                max_abs = |A[i,j]|
                max_idx = (i, j)

This gives the global off-diagonal maximum of the updated trailing submatrix,
which is the starting point for the next pivot selection. If |A[max_idx]|
happens to be in column 0 of the trailing submatrix, Step 1 of the pivot
selection (Section 3.1) is already done.

### 6.2 Rank-2 Fused Update+Argmax

Same structure but with the rank-2 update formula from Section 4.3.

### 6.3 What to Fuse in the Scalar Kernel

For the initial scalar (unblocked) kernel, fuse the update with the search
for γ₀ (the column-0 off-diagonal maximum). This eliminates the most common
redundant pass. The full row search for γᵣ (Step 4 of pivot selection) is
less frequent (only when Test 3 fails) and can remain a separate scan.

When moving to the blocked kernel later, the fused update+argmax is used in
the full-pivoting path within each block, while the blocked Schur complement
update across blocks uses BLAS-3 operations (no fusion needed there).


## 7. Storage Layout Decision

The spec describes two approaches that are in tension:

1. **In-place factorization** (like LAPACK's `dsytf2`/`dsytrf`): Overwrite
   the lower triangle of A with L (unit diagonal implicit), store D's diagonal
   on A's diagonal, store D's subdiagonal in a separate vector.

2. **Separate `Factors` struct**: Allocate separate `l: Vec<f64>`,
   `d_diag: Vec<f64>`, `d_subdiag: Vec<f64>` vectors.

### 7.1 Decision: In-Place with Auxiliary Vectors

Use in-place factorization. The lower triangle of the input matrix is
overwritten with L (unit diagonal not stored). D's diagonal occupies the
main diagonal. D's subdiagonal is stored in a separate `subdiag: Vec<f64>`
of length n (zero for 1×1 blocks, the off-diagonal value for the first
row of each 2×2 block). The permutation is stored as a `perm: Vec<usize>`.

**Rationale:**

- Forward-compatible with the blocked kernel: the blocked algorithm
  (W-panel technique) also works in-place, accumulating updates in a
  separate W workspace. A copy-out `Factors` struct would require
  rethinking the data flow for the blocked case.

- Memory-efficient: no redundant copy of the matrix. For dense matrices
  this is n² doubles saved.

- Matches faer's approach: faer's `cholesky_in_place` overwrites A with
  L on the lower triangle and D's diagonal on the main diagonal, with
  subdiagonal stored separately. The high-level `Lblt` struct that
  separates these is a post-processing convenience layer, not the
  computational representation.

### 7.2 The `DenseFactorization` Return Type

The factorization function signature:

    pub fn factor_in_place(
        a: &mut [f64],       // n×n column-major, lower triangle overwritten
        n: usize,
        subdiag: &mut [f64], // length n, 2×2 block subdiagonals
        perm: &mut [usize],  // length n, pivot permutation
        config: &BkConfig,
    ) -> Result<Inertia, FactorError>

The matrix `a` is column-major with stride n. After factorization:
- `a[i + j*n]` for i > j contains L[i,j]
- `a[j + j*n]` contains D's diagonal element at position j
- `a[i + j*n]` for i < j is undefined (not zeroed, to save work)
- `subdiag[k]` is the off-diagonal of the 2×2 block starting at row k
  (zero if k is a 1×1 block or the second row of a 2×2 block)

This layout is identical to LAPACK's `dsytf2` (lower triangle variant).


## 8. Solve Phase

Given P L D Lᵀ Pᵀ = A and a right-hand side b, the solve computes x = A⁻¹b
in five steps (without equilibration — equilibration adds pre/post scaling):

1. **Apply permutation:** y = Pᵀ b
2. **Forward substitution:** z = L⁻¹ y (unit lower triangular)
3. **D-block solve:** w = D⁻¹ z
4. **Backward substitution:** v = L⁻ᵀ w
5. **Undo permutation:** x = P v

### 8.1 D-Block Solve Details

**1×1 block:** w[k] = z[k] / d[k] — scalar division.

**2×2 block** (normalized, following faer):

Given D block [[a, b], [b, c]] where b = subdiag[k]:

    b_inv = 1 / b
    ak = a · b_inv         // = a/b
    ck = c · b_inv         // = c/b
    denom = 1 / (ak · ck − 1)
    z₀k = z[k] · b_inv    // = z[k]/b
    z₁k = z[k+1] · b_inv  // = z[k+1]/b
    w[k]   = (ck · z₀k − z₁k) · denom
    w[k+1] = (ak · z₁k − z₀k) · denom

This is algebraically equivalent to Cramer's rule but normalizes by b,
avoiding overflow when b is large and improving conditioning of the
denominator (ak·ck − 1 instead of a·c − b²).

### 8.2 With Equilibration

When equilibration scaling D_eq is applied (D_eq A D_eq is factored instead
of A), the full solve sequence is:

1. b̂ = D_eq · b
2. y = Pᵀ b̂
3. z = L⁻¹ y
4. w = D_bk⁻¹ z
5. v = L⁻ᵀ w
6. x̂ = P v
7. x = D_eq · x̂

**Critical distinction:** D_eq (equilibration, always invertible, does not
contribute to inertia) and D_bk (the block diagonal from factorization,
may contain near-zero blocks, all inertia comes from here) are completely
different objects despite both being called "D" in different contexts.


## 9. First Test Strategy

The expert review identified the full symmetric row search as the #1 BK
implementation bug. The first test matrix should exercise the case where:

- The column-only maximum (γ₀) points to row r
- The symmetric row search for row r (computing γᵣ) reveals a larger
  element in the *row* part (stored in a different column of the lower
  triangle) than in the column part
- This changes the pivot decision compared to a column-only search

### 9.1 Concrete Test Matrices

From citep:bunch1977stable (the BK paper), use the small worked examples
that exercise each branch of the 3-way pivot selection:

1. **Test 3 passes (1×1 pivot, no swap):** A matrix where |A[0,0]| ≥ α·γ₀
2. **Test 5 passes (1×1 pivot with swap):** A matrix where Test 3 fails but
   |A[r,r]| ≥ α·γᵣ
3. **Test 6 passes (1×1 pivot, LAPACK extension):** A matrix where Tests 3
   and 5 fail but |A[0,0]|·γᵣ ≥ α·γ₀²
4. **2×2 pivot:** A matrix where all 1×1 tests fail, forcing a 2×2 block

### 9.2 Inertia-Critical Tests

5. **2×2 block with positive diagonals but negative determinant:** e.g.,
   D block = [[1, 3], [3, 2]], det = −7. Inertia must be (1, 1, 0),
   NOT (2, 0, 0).
6. **Near-zero eigenvalue:** A matrix with one eigenvalue at ~1e-14,
   testing the zero_tol boundary.
7. **Full KKT structure:** A small [[H, Jᵀ], [J, −δI]] matrix with
   known inertia (n, m, 0).


## 10. Path to Blocked Factorization

The scalar kernel is the foundation. The blocked kernel wraps it with a
W-panel accumulation technique:

1. Process `block_size` columns using the scalar kernel, but *defer* the
   Schur complement update to columns outside the current block.
2. Accumulate the deferred updates in a workspace matrix W of size
   n × block_size.
3. At block boundaries, apply the accumulated rank-k update to the
   trailing submatrix as a single BLAS-3 operation (symmetric rank-k
   update or equivalent).

This converts O(n) rank-1/rank-2 updates (BLAS-2, memory-bound) into
one rank-k update (BLAS-3, compute-bound), dramatically improving cache
utilization for large matrices.

### 10.1 Design Implications for the Scalar Kernel

The scalar kernel must be written so that it can operate on a *panel*
(a subset of columns) rather than the full trailing submatrix. Specifically:

- The update loop should accept column-range parameters (start, end)
  rather than always updating the full trailing submatrix.
- The L columns and D values must be stored in a layout compatible with
  the W-panel accumulation.
- The in-place storage layout (Section 7) naturally supports this: the
  blocked kernel simply calls the scalar kernel on the diagonal block,
  then reads the L columns to form the W panel.

### 10.2 SIMD Considerations

The scalar kernel does not need SIMD. The blocked kernel's rank-k update
is where SIMD pays off — it is a matrix-matrix multiply (or symmetric
rank-k update), which has high arithmetic intensity and benefits from
vectorized inner loops. This is a future concern; the scalar kernel's
memory access patterns matter more than its instruction-level parallelism.


## 11. Summary of Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Storage layout | In-place (LAPACK-style) | Forward-compatible with blocked kernel; matches faer |
| 2×2 normalization | Divide by \|a₁₀\| (faer approach) | Avoids catastrophic cancellation in det computation |
| Fused update+argmax | Day-one requirement | Halves memory traffic per pivot step |
| First test case | Full symmetric row search exercise | Most common BK implementation bug |
| Inertia counting | Determinant-based, not diagonal signs | Only correct method for 2×2 blocks |
| Zero pivot handling | ForceAccept or Fail, never perturb | Perturbation is IPM layer's responsibility |
| Pivot threshold | α = (1 + √17) / 8 ≈ 0.6404 | Standard BK value; TPP with u=0.01 comes in Stage 2 |
| Blocked kernel prep | Column-range parameters in scalar kernel | Enables W-panel technique without rewrite |
