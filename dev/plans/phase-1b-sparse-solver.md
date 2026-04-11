# Phase 1b Implementation Plan — Sparse Multifrontal Solver

## Overview

Build a single-threaded multifrontal sparse symmetric indefinite solver
in 9 incremental steps, each independently testable and committable.
The dense BK kernel from Phase 1a is embedded directly (no trait extraction).

## Step 1: CSC Sparse Matrix (`src/sparse/`)

**Files:** `src/sparse/mod.rs`, `src/sparse/csc.rs`

```rust
pub struct CscMatrix {
    pub n: usize,
    pub col_ptr: Vec<usize>,   // length n+1
    pub row_idx: Vec<usize>,   // length nnz
    pub values: Vec<f64>,      // length nnz
}
```

**Methods:**
- `from_triplets(n, rows, cols, vals)` — deduplicate, sort, build CSC
- `nnz()` — number of nonzeros
- `validate()` — check sorted, in-bounds, consistent lengths
- `symmetric_pattern()` → `CscPattern` — expand lower triangle to full
  symmetric pattern (needed for AMD and elimination tree)
- `symv(x, y)` — sparse symmetric matrix-vector product (for residuals)

**Also:** `MtxMatrix::to_csc()` conversion in `src/io/mtx.rs`.

**Tests:** Round-trip triplet→CSC→dense, symv against dense symv, validation
rejects bad input.

---

## Step 2: AMD Ordering (`src/ordering/amd.rs`)

**Files:** `src/ordering/mod.rs`, `src/ordering/amd.rs`

```rust
pub fn amd_order(pattern: &CscPattern) -> Vec<usize>
```

**Algorithm:** Implement from Amestoy/Davis/Duff 1996. Core loop:
1. Maintain approximate degree for each non-eliminated variable.
2. Select minimum-degree variable, eliminate it.
3. Update degrees of affected variables using element absorption.
4. Use aggressive absorption and mass elimination optimizations.

**Output:** Permutation vector `perm` of length n.

**Tests:**
- Identity ordering for already-optimal matrices (diagonal, banded)
- Compare fill count of AMD-ordered vs natural-ordered factorization
  on small matrices (< 20×20) with hand-verified results
- Verify `perm` is a valid permutation (unique values 0..n)

---

## Step 3: Elimination Tree (`src/ordering/elimination_tree.rs`)

```rust
pub struct EliminationTree {
    pub parent: Vec<Option<usize>>,  // parent[j] = min { i > j : fill(i,j) }
    pub n: usize,
}
```

**Algorithm:** Union-find with path compression on the permuted pattern.
For each column j of P·A·Pᵀ, walk up from each row index i > j in that
column, attaching subtrees.

**Methods:**
- `from_pattern(pattern: &CscPattern) -> Self`
- `children(&self) -> Vec<Vec<usize>>` — compute children lists
- `roots(&self) -> Vec<usize>` — nodes with no parent (forest roots)

**Tests:** Hand-verified trees for small matrices (5×5, 10×10).

---

## Step 4: Postordering (`src/ordering/postorder.rs`)

```rust
pub fn postorder(etree: &EliminationTree) -> (Vec<usize>, Vec<usize>)
// Returns (postorder_perm, inverse_perm)
```

**Algorithm:** DFS on the elimination tree, visiting children in order of
ascending subtree weight (smallest subtree first → minimizes peak memory).

**Properties to verify:**
- Every child appears before its parent in postorder
- postorder_perm is a valid permutation
- Composing AMD perm with postorder perm gives the final ordering

**Tests:** Verify topological ordering property, verify weight-based child
ordering.

---

## Step 5: Column Counts (`src/symbolic/column_counts.rs`)

```rust
pub fn column_counts(
    pattern: &CscPattern,
    etree: &EliminationTree,
    postorder: &[usize],
) -> Vec<usize>
```

**Algorithm:** Liu's row subtree algorithm. For each column j (in postorder),
the nonzero count in L[:,j] is determined by the row subtrees rooted at j's
row indices, compressed using the elimination tree.

**Tests:** Compare predicted counts against actual factorization NNZ for
small matrices where we can compute the dense factorization.

---

## Step 6: Supernode Detection (`src/symbolic/supernode.rs`)

```rust
pub struct Supernode {
    pub cols: Range<usize>,     // columns eliminated in this node
    pub nrow: usize,            // total rows (nrow >= ncol)
    pub row_indices: Vec<usize>, // row indices of the frontal
}

pub fn find_supernodes(
    pattern: &CscPattern,
    etree: &EliminationTree,
    col_counts: &[usize],
    params: &SupernodeParams,
) -> Vec<Supernode>
```

**Two passes:**
1. Fundamental supernodes: consecutive columns with identical structure
   (same row indices minus the column itself).
2. Amalgamation: merge small nodes using SSIDS merge rule (trivial chain
   + size-based with nemin threshold).

**Tests:** Verify supernodes for banded matrices (should merge into large
supernodes), arrow matrices (single large root supernode), and diagonal
matrices (all singletons with nemin=1).

---

## Step 7: Symbolic Factorization / MemoryPlan (`src/symbolic/mod.rs`)

```rust
pub struct MemoryPlan {
    pub supernodes: Vec<Supernode>,
    pub factor_nnz_estimate: usize,
    pub factor_slack: f64,
    pub contrib_sizes: Vec<usize>,
    pub peak_contrib_bytes: usize,
    pub amaps: Vec<Vec<(usize, usize)>>,
}

pub fn symbolic_factorize(
    matrix: &CscMatrix,
    perm: &[usize],
    params: &SupernodeParams,
) -> Result<MemoryPlan, FeralError>
```

Combines steps 3–6 into a single analysis phase. Precomputes assembly maps.

**Tests:** Verify factor_nnz_estimate >= actual NNZ, contrib_sizes correct,
amaps point to valid positions.

---

## Step 8: Numeric Factorization (`src/numeric/`)

**Files:** `src/numeric/mod.rs`, `src/numeric/factorize.rs`,
`src/numeric/assembly.rs`, `src/numeric/frontal.rs`

```rust
pub fn factorize_multifrontal(
    matrix: &CscMatrix,
    plan: &MemoryPlan,
    params: &BunchKaufmanParams,
) -> Result<(SparseFactors, Inertia), FeralError>
```

**Postorder traversal loop:**
1. Allocate frontal matrix (nrow × nrow) zeroed
2. Assembly-Pre: scatter original entries via amap, extend-add child contribs
3. Call dense BK kernel on the fully-summed region
4. Extract contribution block (Schur complement)
5. Assembly-Post: free child contrib blocks
6. Store factor data in FactorBump, accumulate inertia

**Tests:**
- Factor small sparse matrices, compare inertia against dense factorization
- Factor sparse KKT matrices from data/matrices/kkt/, compare inertia
  against MUMPS sidecar
- Verify sparse factorization matches dense factorization exactly for
  matrices small enough to compare

---

## Step 9: Sparse Solve (`src/numeric/solve.rs`)

```rust
pub fn solve_sparse(
    factors: &SparseFactors,
    rhs: &[f64],
) -> Result<Vec<f64>, FeralError>
```

**Supernodal forward/backward substitution:**
1. Apply equilibration: b̂ = D_eq · b
2. Permute: ŷ = Pᵀ · b̂
3. Forward solve through supernodes (postorder)
4. D-block solve (1×1 and 2×2)
5. Backward solve through supernodes (reverse postorder)
6. Unpermute: x̂ = P · v
7. Undo equilibration: x = D_eq · x̂

**Tests:**
- Solve small sparse systems, compare against dense solve
- Residual check on all collected KKT matrices
- solve_refined() wrapper for Phase 1b convention

---

## Session Sequencing

This is too large for one session. Suggested breakdown:

- **Session 1:** Steps 1–3 (CSC, AMD, elimination tree)
- **Session 2:** Steps 4–6 (postorder, column counts, supernodes)
- **Session 3:** Step 7 (symbolic factorization, MemoryPlan)
- **Session 4:** Step 8 (numeric factorization)
- **Session 5:** Step 9 (sparse solve) + full KKT validation

Each session produces working, tested code with a checkpoint.

## Exit Criterion

Sparse solver produces identical inertia and equivalent solutions (within
tolerance) to the dense solver on all 153k collected KKT matrices that
are parseable. The benchmark harness is extended to run both dense and
sparse paths and compare.
