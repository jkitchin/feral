# FERAL: Frontal Elimination for Robust Algebraic Linear Systems

## Project Initialization & Comprehensive Design Specification

**Version:** 1.0 — March 29, 2026
**Author:** John Kitchin (jkitchin@andrew.cmu.edu), Carnegie Mellon University
**License:** MIT

---

## 1. Project Overview

FERAL is a standalone, self-contained sparse symmetric indefinite direct solver implemented in pure Rust. It is the linear algebra foundation for POUNCE (Path-following Optimizer for Unconstrained and Nonlinear Constrained Equations), an interior point NLP solver. Together, FERAL and POUNCE form a modern, open-source, MIT-licensed solver stack targeting chemical engineering nonlinear programming, where Ipopt is the dominant solver.

### 1.1 Motivation

The existing landscape of sparse symmetric indefinite solvers — MA57, MUMPS, PARDISO — is dominated by legacy Fortran/C code with restrictive licenses (HSL), heavyweight MPI infrastructure (MUMPS), or vendor lock-in (MKL PARDISO). There is no modern, permissively licensed, self-contained implementation suitable for embedding in an open-source NLP solver.

FERAL fills this gap. The project is de-risked by the successful 6-week development of ripopt (a Rust reimplementation of IPOPT using Claude Code), which achieves competitive results against IPOPT on CUTEst benchmarks and 100% solve rate on electrolyte problems from the target domain, including ~40 unique solves that IPOPT cannot achieve.

### 1.2 Design Philosophy

- **One engine, not multiple solvers.** Dense, small sparse, and large sparse cases are configurations of the same multifrontal engine, not separate codepaths.
- **Zero non-Rust dependencies for the core solver.** No BLAS, no LAPACK, no Fortran. Dense kernels are implemented from the literature with research-grade quality, including SIMD-aware micro-kernels where performance demands it.
- **Pluggable strategies, not pluggable solvers.** Pivot strategy, ordering, supernode policy, and kernel selection are variation points within a single architecture.
- **Co-designed with POUNCE.** The outer NLP loop can adapt FERAL's behavior across interior point iterations — a capability no existing standalone solver offers.
- **Clean-room implementation.** All code is original, derived from published papers and BSD-licensed references (primarily SPRAL/SSIDS for sparse architecture, BLIS for dense kernel design). No code is copied.

### 1.3 The Solver Stack

```
POUNCE  ← NLP solver (interior point, future)
  └── FERAL   ← sparse symmetric indefinite solver (this project)
        └── own dense kernels (research-grade, from literature)
        └── own sparse infrastructure (CSC storage, ordering, etc.)
```

---

## 2. Architecture

### 2.1 Build Order: Dense First, Sparse Second

FERAL is built in two explicit stages. This is not a compromise — it is the correct engineering sequence.

**Stage 1 — Standalone dense solver (Phase 1):** A self-contained `factor(matrix) -> (Factors, Inertia)` and `solve(factors, rhs) -> solution` with full Bunch-Kaufman pivoting. No frontal matrices, no assembly tree, no ordering. The entire matrix is treated as a single dense system. This stage is complete and fully tested before Stage 2 begins.

**Stage 2 — Multifrontal sparse solver (Phase 2+):** The dense kernel from Stage 1 is embedded inside a multifrontal engine. The `DenseKernel` trait and `PivotStrategy` trait are designed at this boundary, informed by what the multifrontal engine actually requires. AMD ordering, elimination tree, symbolic factorization, supernode amalgamation, and assembly (extend-add) are all Stage 2 concerns.

The size regimes the final solver targets:
- **Dense (n < ~200):** Single frontal = whole matrix. Stage 1 result, unchanged.
- **Small sparse (n ~ 200–2000):** AMD + elimination tree + small frontals. Supernode amalgamation optional.
- **Medium sparse (n ~ 2k–50k):** Core target. Supernodes, cache blocking, aggressive amalgamation.
- **Large sparse (n ~ 50k–500k):** Shared-memory parallelism via Rayon on the assembly tree.
- **Very large sparse (n > 500k):** Distributed MPI multifrontal. Phase 4.

### 2.2 Engine Components

```
FERAL Engine
├── Ordering              ← AMD | METIS | custom
│   └── Elimination tree construction
│   └── Postordering
│   └── Column counts / fill estimation
│
├── Symbolic Factorization
│   └── Supernode detection and amalgamation (nemin-based, Section 2.6)
│   └── Assembly tree construction
│   └── Assembly map (amap) precomputation: original-to-frontal index mapping
│   └── MemoryPlan production (factor NNZ estimate, contribution block sizes)
│
├── Memory Allocation     ← driven by MemoryPlan
│   └── FactorBump    (append-only bump allocator, grows as needed)
│   └── ContribPool   (serial: LIFO stack; parallel: buddy allocator)
│
├── Numeric Factorization
│   ├── Assembly-Pre  (scatter original entries via amap + assemble child contribs into factor region)
│   ├── Dense Kernel  (factor_frontal → factors + contribution block)
│   ├── Assembly-Post (assemble child contribs into contribution block region, free child contribs)
│   └── Inertia accumulation across frontals
│
├── Solve Phase              ← full sequence: see Section 2.9
│   └── 1. Apply equilibration to RHS:   b̂ = D_eq · b
│   └── 2. Permute:                      ŷ = Pᵀ · b̂
│   └── 3. Forward substitution:         solve L · z = ŷ
│   └── 4. BK D-block solve:             solve D_bk · w = z  (1×1 or 2×2 systems)
│   └── 5. Backward substitution:        solve Lᵀ · v = w
│   └── 6. Permute back:                 x̂ = P · v
│   └── 7. Undo equilibration:           x = D_eq · x̂
│   └── 8. Iterative refinement (via solve_refined(); activated when factors.needs_refinement; see Section 2.10)
│
├── Quality Escalation       ← IncreaseQuality interface (Section 2.12)
│   └── Stage 1: activate iterative equilibration (if not already on)
│   └── Stage 2: raise pivot threshold parameter u
│   └── Returns false when all escalation options are exhausted
│
└── Configuration
    └── SupernodeParams (nemin, merge criteria — Section 2.6)
    └── BunchKaufmanParams (alpha, ZeroPivotAction)
    └── Kernel selection (CPU scalar → blocked → SIMD → GPU)
```

**Assembly split rationale (from SSIDS code review).** Assembly is split into pre-factorization and post-factorization phases because child contribution blocks can only be assembled into the parent's contribution block AFTER the dense kernel populates that region. The pre-phase assembles original matrix entries (via the precomputed `amap`) and child contributions into the factor (fully-summed) region. The post-phase assembles child contributions into the generated element (contribution block) region and frees child contribution blocks. SSIDS's `assemble.hxx` implements exactly this split.

**Assembly map (amap) precomputation.** The mapping from original matrix entries to positions in each frontal matrix is precomputed during symbolic factorization and stored per-supernode. This avoids searching during numeric factorization. Additionally, a dense lookup vector of size `n` (the full matrix dimension) is used at runtime for the extend-add child-to-parent index mapping, providing O(1) lookup instead of binary search.

### 2.3 Memory Model

FERAL uses a **two-allocator model** for dynamic memory during numeric factorization. This model is designed from the start for the single-thread case and extended cleanly to parallel in Phase 2.

> **Design note (from SSIDS code review).** SSIDS uses `AppendAlloc` (page-based bump allocator) for factor storage and `BuddyAllocator` (buddy-system pool with deallocation) for contribution blocks. MUMPS uses a single contiguous workspace with factor storage growing from the bottom and contributions from the top, with memory compresses as a fallback. FERAL's two-allocator model is a sound simplification for the serial case that separates permanent (factor) from transient (contribution) storage.

#### MemoryPlan (output of symbolic factorization)

Symbolic factorization produces a `MemoryPlan` before any numeric work begins:

```rust
pub struct MemoryPlan {
    /// Estimated total NNZ in the L factor across all supernodes.
    /// This is a lower bound: delayed pivots increase a node's factor size
    /// by ndelay_in rows and columns (only known at factorization time).
    /// Inflated by `factor_slack` to accommodate typical delayed pivot overhead.
    pub factor_nnz_estimate: usize,

    /// Slack factor applied to factor_nnz_estimate. Default 1.2 (20% overhead,
    /// matching MUMPS's ICNTL(14)=20 default). SSIDS uses 1.1.
    pub factor_slack: f64,

    /// For each node in postorder: the size (in f64s) of its contribution block.
    /// These are determined entirely by the symbolic structure:
    ///   contrib_size[k] = (snode.nrow - snode.ncol)²
    /// Contribution block dimensions are INVARIANT under delayed pivoting —
    /// delays increase the factor region, not the contribution block.
    /// (Confirmed by SSIDS code review: NumericNode.hxx allocates contrib
    /// from symbolic nrow/ncol, not delay-augmented dimensions.)
    pub contrib_sizes: Vec<usize>,

    /// Peak contribution pool depth (sum of all live contribution blocks at
    /// the deepest point of the tree). Used to preallocate ContribPool.
    pub peak_contrib_bytes: usize,

    /// Precomputed assembly maps (amap) per supernode: for each original matrix
    /// entry belonging to this node, stores (source_index_in_values, dest_index_in_frontal).
    /// Computed during symbolic factorization to avoid searching during numeric phase.
    pub amaps: Vec<Vec<(usize, usize)>>,
}
```

#### FactorBump (append-only bump allocator)

A page-based append-only allocator, preallocated with an initial page of `factor_nnz_estimate` entries. Each supernode's factor storage is allocated at assembly time — **not at symbolic time** — because delayed pivots from children change the node's actual dimensions. When a node is assembled, `ndelay_in` (the sum of `ndelay_out` from all children) is known, and the node allocates `(nrow + ndelay_in) × (ncol + ndelay_in)` entries from the bump allocator plus `2 × (ncol + ndelay_in)` entries for the D block diagonal.

If the current page fills, a new page is allocated (like SSIDS's `AppendAlloc`). Deallocation is not supported — factor storage is permanent for the lifetime of the `Factors` struct.

> **Why not pre-assigned offsets?** Symbolic factorization cannot predict the exact factor size per node because `ndelay_in` is only known during numeric factorization. Pre-assigned offsets would require either (a) worst-case allocation assuming maximum delays, wasting memory, or (b) overflow handling that defeats the purpose of pre-assignment. The bump allocator handles this naturally: allocate the right amount when the delay count is known. This is the approach SSIDS uses (`AppendAlloc` in `assemble.hxx:189`).

#### ContribPool

A pool allocator for transient contribution blocks. The elimination tree is traversed in **postorder** (leaves first). At each node:

1. **Assembly-Pre:** Allocate the contribution block from ContribPool. Scatter original matrix entries (via precomputed `amap`) + assemble child contribution blocks into the factor (fully-summed) region of the frontal.
2. **Factor** the frontal with the dense kernel → accepted pivots written to `FactorBump`, contribution block populated with the Schur complement.
3. **Assembly-Post:** Assemble child contribution blocks into the parent's contribution block region. Free child contribution blocks (return to pool).

**Phase 1 (serial):** ContribPool is a simple LIFO stack arena. In serial postorder traversal, the LIFO property holds: a child's contribution is always consumed by the parent before any unrelated node. Preallocate to `peak_contrib_bytes`. No overflow handling is needed because contribution block sizes are symbolic invariants (determined by `snode.nrow - snode.ncol`, not affected by delayed pivots).

**Phase 2 (parallel):** Independent subtrees of the assembly tree can be factored concurrently. The LIFO stack breaks here — threads working different subtrees interleave allocations. Phase 2 replaces the LIFO stack with a **buddy allocator** (SSIDS model: `BuddyAllocator` with 16 levels of subdivision, explicit deallocation, and dynamic page growth). `FactorBump` remains shared — bump allocation is thread-safe with a simple atomic pointer. The `MemoryPlan` is unchanged — it is the interface between the two phases.

### 2.4 Pivot Configuration (Stage 1)

The Bunch-Kaufman algorithm is one algorithm controlled by one parameter — the threshold `α`. The 1×1 vs 2×2 pivot decision is a single joint choice made by the BK selection procedure (Theorem 3, Bunch & Kaufman 1977); it cannot be decomposed into independent `accept_1x1` / `accept_2x2` calls. What varies between "strategies" is `α` and what to do when a pivot is numerically zero after selection.

> **Stage 2 pivot transition (from SSIDS/MUMPS code review).** Classic Bunch-Kaufman is used for Stage 1 (dense). For the multifrontal Stage 2, FERAL should transition to **threshold partial pivoting** (TPP) following SSIDS and MUMPS. SSIDS uses TPP with `u=0.01` (default) as the threshold parameter, and MUMPS uses `CNTL(1)=0.01` for SYM=2. Both test `|pivot| >= u * max_column_entry`, which is fundamentally different from BK's multi-step column search. The TPP approach integrates naturally with delayed pivoting (reject columns that don't meet the threshold) and is cheaper per step. The BK scalar kernel remains as a reference implementation and for small dense systems. See Section 3.3 for the implementation order.

> **LAPACK-style 3-way pivot selection (from faer code review).** The original BK77 Theorem 3 has a 2-way decision. LAPACK (and faer's `PartialDiag` mode) extends this to a 3-way decision: after finding the column-maximum entry at row `r`, if `|A[r,r]| >= α * γ_r` (where `γ_r` is the max off-diagonal in row `r`), accept row `r` as a 1×1 pivot instead of using a 2×2 block. This reduces unnecessary 2×2 blocks. FERAL should implement this LAPACK extension, not the raw BK77 version.

> **Full symmetric row search (from faer code review).** The BK column search must examine the full symmetric row/column of the active submatrix — both the column below the diagonal AND the row to the left of the diagonal (which are the same values by symmetry but stored in different locations in the lower triangle). faer's `offdiag_argmax` does this explicitly. A search of only the lower column would miss the maximum off-diagonal entry in some cases.

**Stage 1 implementation:** `BunchKaufmanParams` — a plain struct, no trait.

```rust
/// Parameters controlling Bunch-Kaufman factorization behavior.
pub struct BunchKaufmanParams {
    /// Pivot threshold α. BK standard: (1 + sqrt(17)) / 8 ≈ 0.6404.
    /// Smaller values (e.g. 0.01, MA57-like) produce more 2×2 pivot blocks
    /// and better numerical stability at the cost of more fill.
    pub alpha: f64,

    /// A 1×1 pivot |d| <= zero_tol is considered numerically zero and triggers
    /// `on_zero_pivot`. Default: 100.0 * f64::EPSILON ≈ 2.2e-14.
    /// For equilibrated matrices (Section 2.8), diagonal entries are O(1) after
    /// scaling, so this threshold is appropriate. Do not set below machine epsilon.
    pub zero_tol: f64,

    /// A 2×2 pivot block [[a,b],[b,c]] is considered near-singular when
    /// |det| = |a·c − b²| <= zero_tol_2x2. Default: zero_tol².
    /// When triggered, `on_zero_pivot` fires (same action as for 1×1 near-zero pivots).
    /// Rationale: BK selects a 2×2 block when neither diagonal is a good 1×1 pivot —
    /// the block can have tiny det even when a and c are individually O(1). Without
    /// this check, the solve (Step 4, Section 2.9) divides by a near-zero det.
    pub zero_tol_2x2: f64,

    /// What to do when the selected pivot is numerically zero.
    /// Applies to both 1×1 pivots (|d| <= zero_tol) and 2×2 pivot blocks
    /// (|det| <= zero_tol_2x2). See variant doc comments for semantics.
    pub on_zero_pivot: ZeroPivotAction,
}

#[derive(Debug, Clone)]
pub enum ZeroPivotAction {
    /// Accept the tiny pivot; flag the factorization for iterative refinement.
    /// Reports inertia correctly including zero counts. Use this when the caller
    /// wants the true inertia of a potentially rank-deficient matrix.
    ForceAccept,
    /// Return FeralError::NumericallyRankDeficient to the caller.
    /// The caller (e.g. POUNCE) decides what to do — typically adds diagonal
    /// perturbation to the matrix and refactorizes.
    Fail,
}
```

> **Architectural note on perturbation (from Ipopt code review).** In Ipopt, the linear solver NEVER perturbs the matrix internally. It reports `SYMSOLVER_SUCCESS`, `SYMSOLVER_SINGULAR`, or `SYMSOLVER_WRONG_INERTIA`, and `PDPerturbationHandler` (in the IPM layer, outside the solver) decides perturbation amounts, modifies the diagonal, and requests refactorization. FERAL follows this architecture: `ZeroPivotAction::Fail` is the solver reporting "I found a near-zero pivot" and letting the caller handle it. The previous `Perturb(f64)` variant has been removed because it conflated the linear solver layer with the IPM layer. POUNCE will manage all perturbation decisions externally, matching Ipopt's proven architecture. See Section 2.12 for POUNCE integration requirements.

```rust
impl Default for BunchKaufmanParams {
    fn default() -> Self {
        let zero_tol = 100.0 * f64::EPSILON;
        Self {
            alpha: (1.0 + 17f64.sqrt()) / 8.0,  // ≈ 0.6404
            zero_tol,
            zero_tol_2x2: zero_tol * zero_tol,
            on_zero_pivot: ZeroPivotAction::Fail,
        }
    }
}
```

| Configuration  | `alpha`               | `zero_tol_2x2`   | `on_zero_pivot`    | Use case                              |
|----------------|-----------------------|------------------|--------------------|---------------------------------------|
| BK standard    | `(1+√17)/8 ≈ 0.6404` | `zero_tol²`      | `Fail`             | Dense solver default                  |
| MA57-like      | `0.01`                | `zero_tol²`      | `Fail`             | More stability, more 2×2 blocks       |
| Phase 1b       | `(1+√17)/8 ≈ 0.6404` | `zero_tol²`      | `ForceAccept`      | Multifrontal default (correct inertia)|

Static pivoting (PARDISO-like) is a different algorithm — no BK selection, just perturb tiny diagonals in place — and is implemented separately when needed, not through this struct.

**Inertia counting for 1×1 and 2×2 D blocks.** Inertia is derived from the D blocks after factorization. The rule differs by block size:

- **1×1 block** `[d]`: contributes `+1` positive if d > 0; `+1` negative if d < 0; `+1` zero if d = 0.

- **2×2 block** `[[a, b], [b, c]]`: do NOT count diagonal signs. Compute eigenvalue signs from the characteristic equation:
  - `det = a·c − b²`
  - If `det > 0` and `a > 0` → contributes `(+2, 0, 0)` (both eigenvalues positive)
  - If `det > 0` and `a < 0` → contributes `(0, +2, 0)` (both eigenvalues negative)
  - If `det < 0` → contributes `(+1, +1, 0)` (one positive, one negative eigenvalue)
  - If `det = 0` → contributes `(+1, 0, +1)` if `a > 0`, else `(0, +1, +1)` (one zero eigenvalue)

This is justified by Sylvester's Law of Inertia: the inertia of A equals the inertia of D in the factorization P·L·D·Lᵀ·Pᵀ = A.

**The naive approach of counting signs of diagonal entries of 2×2 blocks is wrong.** A 2×2 block with `a > 0`, `c > 0` but `det < 0` is indefinite and must contribute `(1, 1, 0)`, not `(2, 0, 0)`. This case appears regularly in KKT matrices. Any test suite for the dense BK solver must include matrices that exercise this case, verified against the eigenvalues of the 2×2 blocks.

**Inertia policy for `ZeroPivotAction` variants:**

- **`ForceAccept` on a near-zero pivot reports the zero in inertia correctly** (it does not alter the pivot). The factorization may be numerically unstable for the solve, which is why `ForceAccept` triggers iterative refinement. This is the correct action when the caller wants the true inertia of a rank-deficient matrix.

- **`Fail` returns `FeralError::NumericallyRankDeficient`** — the caller decides what to do. POUNCE interprets this as either linearly dependent constraints (triggering constraint reduction) or an ill-conditioned Hessian (triggering diagonal perturbation and refactorization).

**Perturbation is external to FERAL.** When the caller (POUNCE) needs to perturb the diagonal to correct inertia, it adds the perturbation to the matrix entries BEFORE passing the matrix to `factor()`, then calls `factor()` again on the modified matrix. This matches Ipopt's architecture where `PDPerturbationHandler` modifies the KKT diagonal and the linear solver sees only the already-perturbed matrix. FERAL never decides how much to perturb — that is IPM-layer logic. See Section 2.12 for the full POUNCE integration requirements including the escalation heuristic.

**Stage 2 / POUNCE integration:** When POUNCE exists and needs to adapt solver behavior across IPM iterations, the integration surface is:

1. **Quality escalation:** FERAL exposes `increase_quality() -> bool` (Section 2.12). POUNCE calls this when iterative refinement fails or inertia is wrong. First call activates enhanced scaling; subsequent calls raise the pivot threshold. Returns false when all escalation options are exhausted.

2. **Inertia reporting:** FERAL reports exact `(positive, negative, zero)` counts. POUNCE compares against the expected `(n, m, 0)` for KKT systems and decides whether to perturb and refactorize.

3. **Factorization status:** FERAL returns `Ok((Factors, Inertia))` or `Err(FeralError::NumericallyRankDeficient)`. POUNCE maps these to its own perturbation logic.

The adaptive pivot strategy — exploiting outer loop structure to adjust the solver's behavior across IPM iterations — remains a target novel contribution. The `IncreaseQuality` interface provides the mechanism. Premature trait design is deferred until the interface can be grounded in real POUNCE usage.

### 2.5 The Dense Kernel (Stage 1)

#### Core types (Stage 1)

> **Storage format rationale (from faer code review).** faer uses full n×n column-major storage (not packed lower-triangular) because packed storage defeats SIMD vectorization and BLAS-3 blocking. The formula `j*n - j*(j+1)/2 + i` at every access is slower than column-major `j*n + i`, and varying column lengths in packed format prevent contiguous SIMD reads. The extra n(n-1)/2 f64s for a full matrix (~4 KB for n=32, ~400 KB for n=316) is negligible. FERAL uses full storage from the start to avoid a painful format transition when blocked algorithms arrive in Stage 2.

```rust
/// Symmetric matrix stored as full n×n column-major. Only the lower triangle
/// is meaningful; the strict upper triangle is ignored on input.
/// Entry (i, j) is at index j*n + i. Size: n*n f64 values.
/// The factorization overwrites the lower triangle in-place (like LAPACK's dsytrf).
pub struct SymmetricMatrix {
    pub n: usize,
    pub data: Vec<f64>,  // full n×n column-major, lower triangle is authoritative
}

/// Inertia of a symmetric matrix: counts of positive, negative, zero eigenvalues.
/// Invariant: positive + negative + zero == n.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

/// Factorization result from Stage 1 dense BK: P·L·D_bk·Lᵀ·Pᵀ = D_eq·A·D_eq.
/// Stage 2 (multifrontal) uses a different type: SparseFactors stored in FactorBump.
pub struct Factors {
    pub n: usize,
    /// Unit lower triangular L in full n×n column-major storage.
    /// Entry (i,j) at index j*n + i. Diagonal entries are always 1.0 (unit lower
    /// triangular) and ARE stored explicitly. The strict upper triangle is unused.
    pub l: Vec<f64>,
    /// D_bk diagonal entries in pivot order. Length n.
    /// For a 1×1 pivot at position k: d_diag[k] is the pivot value, d_subdiag[k] == 0.0.
    /// For a 2×2 pivot at positions k, k+1: d_diag[k] = a, d_diag[k+1] = c,
    ///   d_subdiag[k] = b (the off-diagonal), d_subdiag[k+1] = 0.0.
    /// The 2×2 block is [[a, b], [b, c]].
    /// Discriminant: d_subdiag[k] != 0.0 means positions k and k+1 form a 2×2 block.
    pub d_diag: Vec<f64>,
    /// D_bk sub-diagonal entries. Length n. Zero for 1×1 pivots.
    /// For a 2×2 block at (k, k+1): d_subdiag[k] = b (the off-diagonal element).
    pub d_subdiag: Vec<f64>,
    /// BK pivot permutation (forward). Length n.
    /// Convention: perm[i] = j means original row j was moved to pivot position i.
    /// Step 2 (Pᵀ·b̂): y[i] = b_hat[perm[i]]  for i in 0..n
    /// Step 6 (P·v):   x_hat[perm[i]] = v[i]   for i in 0..n
    pub perm: Vec<usize>,
    /// Inverse permutation. Length n. perm_inv[perm[i]] == i for all i.
    /// Stored explicitly for O(1) inverse lookups during the solve (following faer).
    pub perm_inv: Vec<usize>,
    /// Equilibration scaling diagonal D_eq (from Section 2.8). Length n.
    pub d_eq: Vec<f64>,
    /// True when ZeroPivotAction::ForceAccept fired during factorization.
    /// Signals that solve_refined() should be used instead of solve() — see Section 2.10.
    pub needs_refinement: bool,
}
```

> **D block storage rationale (from faer code review).** faer stores the D factor as separate diagonal and sub-diagonal vectors with `subdiag[k] == 0.0` as the discriminant for 1×1 vs 2×2 blocks. This enables O(1) random access to any pivot position (vs. iterating over a `Vec<DBlock>` enum to find position k). The two-vector representation also maps directly to LAPACK's `dsytrf` output format and simplifies the D-block solve loop.

#### Stage 1 public API

```rust
/// Factor a symmetric indefinite matrix using Bunch-Kaufman pivoting.
/// Applies equilibration (Section 2.8) transparently before factoring.
pub fn factor(matrix: &SymmetricMatrix, params: &BunchKaufmanParams)
    -> Result<(Factors, Inertia), FeralError>;

/// Solve A·x = rhs using previously computed factors.
/// Full 7-step sequence per Section 2.9. No iterative refinement.
pub fn solve(factors: &Factors, rhs: &[f64]) -> Result<Vec<f64>, FeralError>;

/// Solve A·x = rhs with iterative refinement (Section 2.10).
/// Activate when factors.needs_refinement is true (ForceAccept fired).
/// Requires the original matrix to compute residuals.
pub fn solve_refined(matrix: &SymmetricMatrix, factors: &Factors, rhs: &[f64])
    -> Result<Vec<f64>, FeralError>;
```

**Stage 2 — DenseKernel trait (deferred):** When the multifrontal engine is built, the `DenseKernel` trait will be designed at the dense/sparse boundary. The key constraint it must satisfy — which the original spec violated — is that `factor_frontal` and the Schur complement update cannot be separated: with delayed pivoting, the contribution block (what flows to the parent frontal) is produced *during* the pivot loop, not after it. The trait must expose a single `factor_frontal(frontal) -> FrontalResult` that returns factors and contribution block together.

Performance phases for the dense kernel:
- **Stage 1:** Scalar, correct, well-tested.
- **Stage 2:** Blocked with cache-aware panel factorization.
- **Stage 2+:** SIMD micro-kernel for Schur complement update (Goto/BLIS architecture).
- **Phase 4:** GPU kernel for large frontals (threshold ~128×128).

### 2.6 Supernode Amalgamation

> **Design rationale (from SSIDS and MUMPS code review).** SSIDS uses a single integer parameter `nemin` (default 32) with a two-condition merge rule (`core_analyse.f90:806-822`). MUMPS uses `NEMIN=5` with additional fill-ratio bounds. Both approaches outperform a binary on/off policy because even small sparse matrices benefit from amalgamation — without it, many tree nodes have a single elimination variable, making BLAS calls pointless. The binary `None / Aggressive` enum previously here was too coarse.

```rust
/// Parameters controlling supernode amalgamation.
pub struct SupernodeParams {
    /// Minimum number of eliminated columns in a supernode. Nodes with
    /// fewer eliminations are candidates for merging with their parent.
    /// Default: 32 (matching SSIDS). MUMPS uses 5; both work well.
    /// Setting nemin=1 effectively disables amalgamation.
    pub nemin: usize,
}

impl Default for SupernodeParams {
    fn default() -> Self {
        Self { nemin: 32 }
    }
}
```

**Merge rule (from SSIDS `core_analyse.f90:806-822`).** A child node is merged with its parent if EITHER:

1. **Trivial chain extension:** The parent has exactly one eliminated column AND the parent's column count equals the child's minus one. This merges perfect chain structures where parent and child share identical column patterns.

2. **Size-based amalgamation:** BOTH the parent and child have fewer than `nemin` eliminated columns. This introduces explicit zeros (fill-in) but ensures supernodes are large enough for BLAS-3 efficiency.

The explicit zeros introduced by amalgamation are tracked for memory accounting. For parallel execution (Phase 2), MUMPS doubles `nemin` to create larger supernodes that amortize thread scheduling overhead.

### 2.7 Future: Distributed MPI (Phase 4)

The multifrontal assembly tree is naturally parallel — independent subtrees can be factored on different MPI ranks. Key challenges:

- **Tree mapping:** Assigning subtrees to ranks balancing load while minimizing communication at merge points
- **Assembly communication:** Schur complement contributions between ranks at merge nodes
- **Root problem:** Largest frontal matrices with least parallelism at tree top — where dense kernel and GPU quality matters most

The engine architecture from Phase 1 must not preclude this. Assembly tree traversal, frontal matrix management, and the kernel trait boundary are designed with distributed extension in mind.

### 2.8 Equilibration (Diagonal Scaling)

KKT matrices from interior point solvers have entries spanning many orders of magnitude: the Hessian block is O(1), the Jacobian block is O(1), but the constraint block diagonal grows as O(1/μ) at convergence — up to 10¹⁰ for small barrier parameters. Without scaling, BK pivot selection degrades: the algorithm sees tiny pivots next to large off-diagonals and either misclassifies 2×2 blocks or delays too many pivots.

**Algorithm (Phase 1):** Iterative infinity-norm equilibration (Knight-Ruiz-Ucar, 2012). Compute a diagonal scaling matrix D such that each row i of D·A·D has ||row i||∞ ≈ 1, using an iterative procedure:

```
d[i] = 1.0   for i = 1..n
repeat (max 10 iterations):
    for i = 1..n:
        maxentry[i] = max_j |d[i] * A[i,j] * d[j]|   (using symmetry: only lower triangle)
        d[i] = d[i] / sqrt(maxentry[i])
    if max_i |1 - maxentry[i]| < 1e-8:
        break   (converged)
```

Apply symmetrically: factor Â = D·A·D, then for a solve with RHS b, solve Â·ŷ = D·b and return x = D·ŷ. The caller sees only `factor(A)` and `solve(b)` — scaling is transparent.

> **Why iterative, not one-shot (from SPRAL code review).** The one-shot formula `d[i] = 1/sqrt(max_j |A[i,j]|)` is just the first iteration of this procedure. SPRAL's `equilib_scale_sym` (`scaling.f90:480-520`) implements the iterative version with 10 iterations. MUMPS uses Ruiz iterative equilibration (ICNTL(8)=7). For well-scaled problems, one iteration suffices. For badly-scaled KKT matrices (which FERAL targets), the iterative version with 5-10 iterations converges significantly better — the remaining row-norm imbalance after one iteration can still be 100× or more on KKT matrices with O(1/μ) entries. The iterative version is trivially more code (a loop around the one-shot computation) and is the standard in modern solvers.

**When scaling is active:**
- Stage 1 (dense): always on in production; can be toggled off in tests to isolate BK behavior from scaling effects.
- Stage 2 (sparse): always on; applied to the assembled matrix before factorization.

**Degenerate case:** If any row has all zeros (singular row), d[i] = 1 (no scaling). This cannot happen for a non-singular matrix but must be guarded.

**Quality escalation (from Ipopt code review).** Ipopt activates MC19 scaling on-demand as the first `IncreaseQuality()` escalation step, before raising the pivot tolerance. FERAL follows a similar pattern: the default iterative equilibration is always active, but `increase_quality()` can activate more expensive scaling methods when available.

**Phase 2 improvement:** MC64 (optimal matching-based scaling and ordering) produces better pivot quality on structurally difficult systems. This requires matching-algorithm infrastructure and is deferred to Phase 2. The MC64 reference is: Duff & Koster (1999/2001), HSL_MC64, available in SuiteSparse as `MATLAB/CSparse/cs_dmperm.m`. FERAL will implement a clean-room version from the algorithm description, not from HSL source. For KKT matrices specifically, MC64 provides both scaling and matching/reordering that together significantly improve factorization quality.

The engine diagram in Section 2.2 is updated to name each step explicitly. Section 2.9 gives the full sequence.

### 2.9 Solve Sequence (Complete)

The solve sequence for `A·x = b` using FERAL's stored factorization has **two distinct diagonal operations** that must not be confused. This section exists to make that explicit, because both appear in the Solve Phase and conflating them produces wrong answers.

Given:
- `D_eq`: the equilibration scaling diagonal (computed in Factor Phase from Section 2.8)
- `P`: the BK permutation (from pivot selection)
- `L`: unit lower triangular factor
- `D_bk`: the BK D factor — a block diagonal with 1×1 and 2×2 blocks

The factorization is: `P · L · D_bk · Lᵀ · Pᵀ = D_eq · A · D_eq`

**Solve steps in order:**

```
Step 1:  b̂  = D_eq · b                  (apply equilibration to RHS)
Step 2:  ŷ  = Pᵀ · b̂                   (apply BK permutation)
Step 3:  z   = L⁻¹ · ŷ                  (forward substitution through unit lower L)
Step 4:  w   = D_bk⁻¹ · z               (solve each D block: trivial for 1×1,
                                          2×2 system solve for 2×2 blocks)
Step 5:  v   = L⁻ᵀ · w                  (backward substitution through Lᵀ)
Step 6:  x̂  = P · v                     (undo BK permutation)
Step 7:  x   = D_eq · x̂                 (undo equilibration)
```

**`D_eq` and `D_bk` are different objects with different roles:**

| | `D_eq` | `D_bk` |
|--|--------|---------|
| What | Equilibration diagonal (Section 2.8) | BK factorization D block (Section 2.4) |
| Shape | Diagonal n×n | Block diagonal: 1×1 and 2×2 blocks |
| When computed | Factor Phase, before BK runs | During BK pivot loop |
| Invertible? | Always (guarded in Section 2.8) | Not always — near-zero blocks trigger `ZeroPivotAction` |
| Inertia contribution | None (applied symmetrically, preserves eigenvalue signs) | Entirely — all inertia comes from signs of `D_bk` blocks |

**Step 4 in detail for 2×2 blocks:** For a 2×2 D block `[[a, b], [b, c]]`, solve `[[a,b],[b,c]] · [w1, w2]ᵀ = [z1, z2]ᵀ`. Two equivalent formulations:

**Direct (Cramer's rule):** `det = a·c − b²`, `w1 = (c·z1 − b·z2) / det`, `w2 = (a·z2 − b·z1) / det`.

**Normalized (faer's approach, preferred):** Scale by `1/b` first: `ak = a/b`, `ck = c/b`, `denom = 1/(ak·ck − 1)`, `z1k = z1/b`, `z2k = z2/b`, then `w1 = (ck·z1k − z2k) · denom`, `w2 = (ak·z2k − z1k) · denom`. This is better conditioned when `b` is large — which it typically is for BK 2×2 blocks, since BK selects 2×2 precisely when the off-diagonal dominates.

> **Why the normalized version is better (from faer code review).** BK selects a 2×2 block when neither diagonal alone is a good pivot, meaning `|a|` and `|c|` are small relative to `|b|`. In this regime, `det = a·c − b² ≈ −b²` and the Cramer's rule formulation divides by a quantity near `−b²`, which is large and well-conditioned. But the intermediate products `c·z1` and `b·z2` can have large magnitudes that cancel, causing precision loss. The normalized formulation avoids this by dividing through by `b` first, keeping intermediate values of order 1. faer implements exactly this approach (`bunch_kaufman/solve.rs:73-95`).

If `|det|` is near zero (or `|b|` is near zero for the normalized version), the block was near-singular — this should have been caught by `ZeroPivotAction` during factorization.

### 2.10 Iterative Refinement

Iterative refinement is activated when `ZeroPivotAction::ForceAccept` fires during factorization (the solve step may be inaccurate due to near-zero pivots). The `solve()` function performs at most **3 steps** of fixed-point iterative refinement after the initial solve.

**Algorithm (one step):**

```
Given: A, factors, current solution x̂ᵢ
1. Compute residual in higher precision:  r = b - A·x̂ᵢ   (compute A·x̂ᵢ in f64 accumulation)
2. Solve the correction:                  δx = factors.solve(r)    (standard 7-step sequence)
3. Update:                                x̂ᵢ₊₁ = x̂ᵢ + δx
```

**Stopping criterion:** Stop when `||δx||₂ / ||x̂ᵢ||₂ < macheps * sqrt(n)`, or after 3 steps, whichever is first. If `||x̂ᵢ||₂ = 0`, use `||δx||₂ < macheps * sqrt(n)`.

**Note:** When `ForceAccept` fires for a genuinely rank-deficient matrix, the solve is not expected to converge — the iterative refinement will not improve `δx` significantly and will exit after 3 steps. The caller (POUNCE) interprets this as a degenerate system. For near-zero pivots that are numerical noise (full-rank matrix), iterative refinement typically converges in 1–2 steps.

**Reference model:** LAPACK's `dgerfs` (double-precision iterative refinement for general systems). FERAL's version is simpler: no norm estimation, no condition number estimate, just 3 steps. MUMPS recommends 2-3 fixed steps without convergence test (ICNTL(10) = -2 or -3) for best results.

> **POUNCE outer refinement requirement (from Ipopt code review).** FERAL's iterative refinement operates on the reduced augmented system (the matrix FERAL factored). Ipopt performs additional refinement on the **full 8-component primal-dual system** (x, s, y_c, y_d, z_L, z_U, v_L, v_U) in `PDFullSpaceSolver::Solve` (`IpPDFullSpaceSolver.cpp:253-346`), including the perturbation terms (delta_x, delta_c) in the residual computation. Ipopt does at least `min_refinement_steps=1` step per solve regardless of pivot quality, and up to `max_refinement_steps=10` with a convergence test based on `residual_ratio = ||resid||_inf / (min(||res||_inf, 1e6*||rhs||_inf) + ||rhs||_inf)`. POUNCE must implement this outer refinement loop in addition to whatever FERAL does internally. FERAL's internal refinement is a useful quality measure for the standalone solver, but it is not sufficient for the IPM's needs. See Section 2.12 for the full POUNCE integration specification.

**Implementation note:** Step 1 must recompute `A·x̂ᵢ` from the original matrix (not the factors), otherwise the refinement corrects nothing. This requires keeping a reference to the original `SymmetricMatrix` in `solve()`, which takes it as a separate argument:

```rust
pub fn solve_refined(
    matrix: &SymmetricMatrix,
    factors: &Factors,
    rhs: &[f64],
) -> Result<Vec<f64>, FeralError>;
```

The plain `solve(factors, rhs)` always skips refinement. `solve_refined(matrix, factors, rhs)` always runs the refinement loop — it does not check `needs_refinement` internally. The `needs_refinement` flag is guidance for the caller: call `solve_refined` when the flag is true; call plain `solve` when you know the factorization is well-conditioned. An agent should not add a `needs_refinement` conditional inside `solve_refined` — the caller is responsible for choosing the right function.

```rust
pub struct Factors {
    // ... existing fields ...
    /// True when ZeroPivotAction::ForceAccept fired — solution may be inaccurate;
    /// use solve_refined() instead of solve() to activate iterative refinement.
    pub needs_refinement: bool,
}
```

### 2.11 Test Tolerance Formula

Each test tolerance is set before implementation based on the matrix's estimated condition number. The standard backward error bound for LDLᵀ factorization is:

```
tolerance = n · κ(A) · macheps
```

where `n` is the matrix dimension, `κ(A)` is the 2-norm condition number, and `macheps = f64::EPSILON ≈ 2.2e-16`.

For test matrices where the condition number is known analytically or easily estimated:
- Well-conditioned (κ ≈ 1–100): `tolerance = 1e-12`
- Moderately ill-conditioned (κ ≈ 10³–10⁶): `tolerance = 1e-10 · n`
- KKT matrices with regularization `delta_c = 1e-8`: `κ ≈ 1/delta_c = 10⁸`; `tolerance = n · 1e-8`
- KKT matrices at convergence (μ → 0, no regularization): condition number can reach 10¹²; iterative refinement required; `tolerance = 1e-6 · n`

For the benchmark harness (Section 4.2), the condition estimate is computed from the equilibration scaling: `κ_est = (max d_eq[i]) / (min d_eq[i])`, which tracks the matrix's range of scale. This is a lower bound on the true condition number — it is conservative (may flag some correct solves as failures) rather than permissive (would miss incorrect solves). This is intentional.

**Rule (from Section 5.2):** Tolerance is set at plan time. If a correct implementation exceeds the tolerance, the tolerance formula or the condition estimate is wrong — investigate before adjusting.

### 2.12 POUNCE Integration Requirements and Quality Escalation

> This section documents requirements for POUNCE (the IPM solver) that are informed by Ipopt's architecture. These are NOT implemented in FERAL — they are recorded here so FERAL's API is designed to support them. Based on expert code review of Ipopt's `PDFullSpaceSolver.cpp`, `PDPerturbationHandler.cpp`, `TSymLinearSolver.cpp`, and `SparseSymLinearSolverInterface.hpp`.

#### 2.12.1 FERAL's Linear Solver Interface

FERAL exposes a minimal interface to POUNCE, modeled on Ipopt's `SymLinearSolver`:

```rust
/// Information flowing from FERAL to POUNCE after factorization.
pub enum FactorStatus {
    /// Factorization succeeded. Inertia is available.
    Success,
    /// Matrix is numerically singular (near-zero pivot with ZeroPivotAction::Fail).
    Singular,
}

/// Quality escalation: ask the solver to try harder on the next factorization.
/// Returns false when all escalation options are exhausted.
/// Stage 1: activate enhanced scaling (if not already on).
/// Stage 2: raise pivot threshold parameter u.
/// Modeled on Ipopt's IncreaseQuality() — see IpTSymLinearSolver.cpp:432-441.
pub fn increase_quality(&mut self) -> bool;

/// Number of negative eigenvalues from last factorization.
pub fn num_negative_eigenvalues(&self) -> usize;

/// Whether the solver can report inertia (always true for FERAL).
pub fn provides_inertia(&self) -> bool { true }
```

**What flows INTO FERAL from POUNCE:**
- The assembled (possibly perturbed) KKT matrix
- The RHS vector(s)
- Whether to check inertia against an expected count

**What flows OUT of FERAL to POUNCE:**
- `FactorStatus` (Success or Singular)
- Exact inertia counts `(positive, negative, zero)`
- Solution vector(s)

**What does NOT flow through the interface:**
- No perturbation amounts (POUNCE adds delta_x, delta_c to the matrix before passing it to FERAL)
- No barrier parameter mu
- No zero pivot threshold (FERAL manages this internally via `increase_quality()`)
- No information about KKT structure (FERAL sees a generic symmetric indefinite matrix)

#### 2.12.2 POUNCE Perturbation Handler (PDPerturbationHandler equivalent)

POUNCE must implement the following perturbation escalation logic, based on Ipopt's `IpPDPerturbationHandler.cpp`:

**Inertia correction loop** (Ipopt's `PDFullSpaceSolver::SolveOnce`, lines 500-640):
```
while factorization fails:
    if FERAL reports Singular AND constraints exist:
        PerturbForSingularity: add delta_c to constraint block
    if FERAL reports wrong inertia (too few negative eigenvalues):
        first try increase_quality(); if that fails, treat as singular
    if FERAL reports wrong inertia (too many negative eigenvalues) OR Singular without constraints:
        PerturbForWrongInertia: add delta_x to primal block
    add perturbation to matrix diagonal, call FERAL.factor() again
```

**delta_x escalation heuristic** (Ipopt's `get_deltas_for_wrong_inertia`, lines 366-416):
```
if delta_x_curr == 0:
    if delta_x_last == 0:  delta_x = 1e-4                          (initial)
    else:                   delta_x = max(1e-20, delta_x_last / 3)  (restart from memory)
else:
    if delta_x_last == 0 or 1e5 * delta_x_last < delta_x_curr:
        delta_x = 100 * delta_x_curr                                (aggressive escalation)
    else:
        delta_x = 8 * delta_x_curr                                  (normal escalation)
if delta_x > 1e20: return false  // triggers restoration phase
delta_s = delta_x  // always equal
```

**Constraint regularization** (mu-dependent, Ipopt's `delta_cd`, line 465-468):
```
delta_c = delta_d = 1e-8 * mu^0.25
```
This shrinks toward zero as the barrier parameter decreases, so constraint regularization vanishes at convergence.

**Structural degeneracy detection** (Ipopt's `finalize_test`, lines 470-538):
Over the first 3 matrices, a state machine probes whether the Jacobian or Hessian is structurally degenerate by trying different combinations of perturbation. Once degenerate status is determined, the appropriate perturbation is applied automatically, avoiding unnecessary trial factorizations.

#### 2.12.3 POUNCE Outer Iterative Refinement

POUNCE must implement iterative refinement on the full primal-dual system (not the reduced augmented system FERAL sees). Based on Ipopt's `PDFullSpaceSolver::Solve` (lines 253-346):

- Compute residuals for all 8 components (x, s, y_c, y_d, z_L, z_U, v_L, v_U) including perturbation terms
- `residual_ratio = ||resid||_inf / (min(||res||_inf, 1e6 * ||rhs||_inf) + ||rhs||_inf)`
- At least `min_refinement_steps=1` step per solve (Ipopt default)
- At most `max_refinement_steps=10` steps
- Stop early if ratio did not improve by factor `1 - 1e-9`
- On failure (ratio > `residual_ratio_max=1e-10`): first `increase_quality()`, then `pretend_singular` if ratio > `residual_ratio_singular=1e-5`

#### 2.12.4 Ordering Considerations for KKT Matrices

MUMPS's ICNTL(12) provides LDLT-aware ordering preprocessing that computes the ordering on a compressed/constrained graph accounting for the symmetric indefinite structure. This can significantly reduce fill-in for KKT matrices. FERAL should investigate a similar capability in Phase 2 — computing the ordering on the quotient graph after maximum transversal, which makes the ordering aware of the 2×2 block structure.

For KKT systems, AMD typically produces more fill-in than METIS or SCOTCH. METIS should be prioritized in Phase 2, with AMD as the fallback for users who don't have METIS available.

---

## 3. Dense Kernel Implementation Strategy

> Full BibTeX entries for all references below: [`dev/references.bib`](dev/references.bib)

### 3.1 Literature Foundation

Each dense operation is implemented from research-grade understanding of the relevant literature, not from naive textbook algorithms.

**Required reading before implementation:**

- **Goto & van de Geijn (2008):** "Anatomy of High-Performance Matrix Multiplication." *ACM TOMS* 34(3):1–25. DOI: [10.1145/1356052.1356053](https://doi.org/10.1145/1356052.1356053). Explains why BLAS is fast — the micro-kernel architecture, cache blocking at L1/L2/L3 levels. This is the foundation for understanding dense kernel performance.
- **Van Zee & van de Geijn (2015):** "BLIS: A Framework for Rapidly Instantiating BLAS Functionality." *ACM TOMS* 41(3):1–33. DOI: [10.1145/2764454](https://doi.org/10.1145/2764454). Modernizes and clarifies Goto's approach. Explicitly designed to be reimplemented, with clean layering: micro-kernel → macro-kernel → full operation.
- **Bunch & Kaufman (1977):** "Some Stable Methods for Calculating Inertia and Solving Symmetric Linear Systems." *Mathematics of Computation* 31(137):163–179. DOI: [10.1090/S0025-5718-1977-0428694-0](https://doi.org/10.1090/S0025-5718-1977-0428694-0). The pivot theory — why 1×1 and 2×2 blocks are necessary and sufficient.
- **Bunch & Parlett (1971):** "Direct Methods for Solving Symmetric Indefinite Systems of Linear Equations." *SIAM J. Numer. Anal.* 8(4):639–655. DOI: [10.1137/0708060](https://doi.org/10.1137/0708060). Complete pivoting for symmetric indefinite systems. More expensive than BK but provides the theoretical ceiling.
- **Duff & Reid (1983):** "The Multifrontal Solution of Indefinite Sparse Symmetric Linear Equations." *ACM TOMS* 9(3):302–325. DOI: [10.1145/356044.356047](https://doi.org/10.1145/356044.356047). The foundational multifrontal paper.
- **Amestoy, Davis & Duff (1996):** "An Approximate Minimum Degree Ordering Algorithm." *SIAM J. Matrix Anal. Appl.* 17(4):886–905. DOI: [10.1137/S0895479894278952](https://doi.org/10.1137/S0895479894278952). Fill-reducing ordering; see also Algorithm 837 (2004) DOI: [10.1145/1024074.1024081](https://doi.org/10.1145/1024074.1024081) for the C/Fortran implementation.
- **Hogg & Scott (2013):** "Pivoting Strategies for Tough Sparse Indefinite Systems." *ACM TOMS* 40(1):1–19. DOI: [10.1145/2513109.2513113](https://doi.org/10.1145/2513109.2513113). Threshold pivoting and delayed pivot strategies in multifrontal factorization.
- **Hogg, Ovtchinnikov & Scott (2016):** "A Sparse Symmetric Indefinite Direct Solver for GPU Architectures." *ACM TOMS* 42(1):1–25. DOI: [10.1145/2756548](https://doi.org/10.1145/2756548). The SSIDS paper — GPU offload, delayed pivoting architecture. Primary BSD-licensed reference for FERAL's sparse architecture.
- **Davis & Hager (2009):** "Dynamic Supernodes in Sparse Cholesky Update/Downdate and Triangular Solves." *ACM TOMS* 35(4):1–23. DOI: [10.1145/1462173.1462176](https://doi.org/10.1145/1462173.1462176). Dynamic supernode amalgamation strategy.
- **George & Liu (1981):** *Computer Solution of Large Sparse Positive Definite Systems.* Prentice-Hall. ISBN: 0-13-165274-5. Elimination trees, fill-reducing orderings, and the graph-theoretic foundations of sparse factorization.

### 3.2 Code Inspection References

Read for algorithmic insight (clean-room discipline — understand the problem, then implement independently):

#### SPRAL/SSIDS (BSD-3-Clause) — https://github.com/ralna/spral

The primary reference for FERAL's sparse multifrontal architecture.

| Topic                                    | File                                                                                                                     |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| Multifrontal numeric driver              | [src/ssids/cpu/NumericSubtree.hxx](https://github.com/ralna/spral/blob/80bc843/src/ssids/cpu/NumericSubtree.hxx)         |
| Multifrontal numeric implementation      | [src/ssids/cpu/NumericSubtree.cxx](https://github.com/ralna/spral/blob/80bc843/src/ssids/cpu/NumericSubtree.cxx)         |
| Symbolic factorization & supernodes      | [src/ssids/cpu/SymbolicSubtree.hxx](https://github.com/ralna/spral/blob/80bc843/src/ssids/cpu/SymbolicSubtree.hxx)       |
| Symbolic node definition                 | [src/ssids/cpu/SymbolicNode.hxx](https://github.com/ralna/spral/blob/80bc843/src/ssids/cpu/SymbolicNode.hxx)             |
| Symbolic analysis driver                 | [src/ssids/anal.F90](https://github.com/ralna/spral/blob/80bc843/src/ssids/anal.F90)                                     |
| Factor routine with pivot selection      | [src/ssids/cpu/factor.hxx](https://github.com/ralna/spral/blob/80bc843/src/ssids/cpu/factor.hxx)                         |
| Threshold pivoting (TPP) kernel          | [src/ssids/cpu/kernels/ldlt_tpp.cxx](https://github.com/ralna/spral/blob/80bc843/src/ssids/cpu/kernels/ldlt_tpp.cxx)     |
| Approximate pivoting (APP) kernel        | [src/ssids/cpu/kernels/ldlt_app.cxx](https://github.com/ralna/spral/blob/80bc843/src/ssids/cpu/kernels/ldlt_app.cxx)     |
| APP kernel header (delayed pivoting)     | [src/ssids/cpu/kernels/ldlt_app.hxx](https://github.com/ralna/spral/blob/80bc843/src/ssids/cpu/kernels/ldlt_app.hxx)     |
| Block LDLT with delayed pivoting         | [src/ssids/cpu/kernels/block_ldlt.hxx](https://github.com/ralna/spral/blob/80bc843/src/ssids/cpu/kernels/block_ldlt.hxx) |
| Assembly (extend-add / scatter-gather)   | [src/ssids/cpu/kernels/assemble.hxx](https://github.com/ralna/spral/blob/80bc843/src/ssids/cpu/kernels/assemble.hxx)     |
| Pivot strategy configuration (datatypes) | [src/ssids/datatypes.f90](https://github.com/ralna/spral/blob/80bc843/src/ssids/datatypes.f90)                           |
| High-level Fortran driver                | [src/ssids/ssids.f90](https://github.com/ralna/spral/blob/80bc843/src/ssids/ssids.f90)                                   |

**Focus areas:** `ldlt_tpp.cxx` for threshold pivot acceptance criteria and numerical guards. `ldlt_app.cxx` for the blocked delayed-pivot algorithm. `SymbolicSubtree.hxx` for supernode amalgamation decisions.

#### BLIS (BSD-3-Clause) — https://github.com/flame/blis

The reference for the micro-kernel architecture underlying high-performance dense operations.

| Topic | File |
|-------|------|
| GEMM blocking loop kernel | [frame/3/gemm/bli_gemm_ker_var2.c](https://github.com/flame/blis/blob/b5d5783/frame/3/gemm/bli_gemm_ker_var2.c) |
| GEMM control structure | [frame/3/gemm/bli_gemm_cntl.c](https://github.com/flame/blis/blob/b5d5783/frame/3/gemm/bli_gemm_cntl.c) |
| SUP (small matrix) integration | [frame/3/bli_l3_sup_int.c](https://github.com/flame/blis/blob/b5d5783/frame/3/bli_l3_sup_int.c) |
| Panel packing for cache blocking | [frame/3/bli_l3_sup_packm_var.c](https://github.com/flame/blis/blob/b5d5783/frame/3/bli_l3_sup_packm_var.c) |
| Block size / cache tuning | [frame/base/bli_blksz.c](https://github.com/flame/blis/blob/b5d5783/frame/base/bli_blksz.c) |
| Block size header | [frame/base/bli_blksz.h](https://github.com/flame/blis/blob/b5d5783/frame/base/bli_blksz.h) |
| Auxiliary micro-kernel info | [frame/base/bli_auxinfo.h](https://github.com/flame/blis/blob/b5d5783/frame/base/bli_auxinfo.h) |
| Context / arch dispatch | [frame/base/bli_cntx.h](https://github.com/flame/blis/blob/b5d5783/frame/base/bli_cntx.h) |

**Focus areas:** `bli_gemm_ker_var2.c` for the 5-loop GEMM structure and how the micro-kernel fits in. `bli_blksz.c` for cache-level block size selection. The packing files for understanding why panel packing matters for the Schur complement update.

#### faer-rs (MIT) — https://github.com/sarah-ek/faer-rs

The reference for Rust-specific implementation patterns: SIMD via `std::arch`, memory layout, and the Bunch-Kaufman algorithm in idiomatic Rust.

| Topic                       | File                                                                                                                                                  |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| Bunch-Kaufman factorization | [faer/src/linalg/cholesky/bunch_kaufman/factor.rs](https://github.com/sarah-ek/faer-rs/blob/8dfccee/faer/src/linalg/cholesky/bunch_kaufman/factor.rs) |
| Bunch-Kaufman solve         | [faer/src/linalg/cholesky/bunch_kaufman/solve.rs](https://github.com/sarah-ek/faer-rs/blob/8dfccee/faer/src/linalg/cholesky/bunch_kaufman/solve.rs)   |
| LDLT factorization          | [faer/src/linalg/cholesky/ldlt/factor.rs](https://github.com/sarah-ek/faer-rs/blob/8dfccee/faer/src/linalg/cholesky/ldlt/factor.rs)                   |
| LDLT solve                  | [faer/src/linalg/cholesky/ldlt/solve.rs](https://github.com/sarah-ek/faer-rs/blob/8dfccee/faer/src/linalg/cholesky/ldlt/solve.rs)                     |
| Triangular solve framework  | [faer/src/linalg/triangular_solve.rs](https://github.com/sarah-ek/faer-rs/blob/8dfccee/faer/src/linalg/triangular_solve.rs)                           |
| Dense matrix data structure | [faer/src/mat.rs](https://github.com/sarah-ek/faer-rs/blob/8dfccee/faer/src/mat.rs)                                                                   |

**Focus areas:** `bunch_kaufman/factor.rs` for how BK pivot selection translates to safe Rust (edge cases, index arithmetic). The LDLT files for the basic unblocked factorization structure before adding pivoting.

#### SuiteSparse AMD (BSD-3-Clause) — https://github.com/DrTimothyAldenDavis/SuiteSparse

Reference for AMD fill-reducing ordering implementation.

| Topic                                  | File                                                                                                                       |
|----------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Main ordering driver                   | [AMD/Source/amd_order.c](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/16178b1/AMD/Source/amd_order.c)           |
| Core AMD algorithm (graph compression) | [AMD/Source/amd_1.c](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/16178b1/AMD/Source/amd_1.c)                   |
| Core AMD algorithm (elimination)       | [AMD/Source/amd_2.c](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/16178b1/AMD/Source/amd_2.c)                   |
| Symmetrize (A + A^T pattern)           | [AMD/Source/amd_aat.c](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/16178b1/AMD/Source/amd_aat.c)               |
| Postordering of elimination tree       | [AMD/Source/amd_postorder.c](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/16178b1/AMD/Source/amd_postorder.c)   |
| Graph preprocessing                    | [AMD/Source/amd_preprocess.c](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/16178b1/AMD/Source/amd_preprocess.c) |

**Focus areas:** `amd_2.c` contains the full AMD elimination loop — the core of the algorithm. `amd_postorder.c` for postordering the elimination tree, which is needed for cache-friendly frontal matrix traversal.

#### SuiteSparse CHOLMOD (Apache 2.0 since SuiteSparse v7.0, 2023; earlier versions LGPL/BSD) — https://github.com/DrTimothyAldenDavis/SuiteSparse

Reference for supernodal techniques (CHOLMOD is positive definite but the supernode infrastructure applies).

| Topic | File |
|-------|------|
| Supernodal symbolic analysis | [CHOLMOD/Supernodal/cholmod_super_symbolic.c](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/16178b1/CHOLMOD/Supernodal/cholmod_super_symbolic.c) |
| Supernodal numeric factorization | [CHOLMOD/Supernodal/cholmod_super_numeric.c](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/16178b1/CHOLMOD/Supernodal/cholmod_super_numeric.c) |
| Supernodal triangular solve | [CHOLMOD/Supernodal/cholmod_super_solve.c](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/16178b1/CHOLMOD/Supernodal/cholmod_super_solve.c) |
| Symbolic analysis (etree, column counts) | [CHOLMOD/Cholesky/cholmod_analyze.c](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/16178b1/CHOLMOD/Cholesky/cholmod_analyze.c) |
| Row-column counts (fill prediction) | [CHOLMOD/Cholesky/cholmod_rowcolcounts.c](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/16178b1/CHOLMOD/Cholesky/cholmod_rowcolcounts.c) |

**Focus areas:** `cholmod_super_symbolic.c` for the supernode amalgamation algorithm — how it merges columns into supernodes to enable BLAS-3 operations. `cholmod_rowcolcounts.c` for fill estimation used to preallocate factor storage.

### 3.3 Implementation Order

1. Scalar LDLᵀ with full Bunch-Kaufman pivoting (correct first) — LAPACK-style 3-way pivot selection, full symmetric row search
2. Blocked BK LDLᵀ with W-panel and deferred Schur complement update (faer's approach: block_size=64, accumulate W = A·L·D⁻¹ during block, apply BLAS-3 update at block boundaries)
3. Threshold partial pivoting (TPP) kernel for multifrontal integration (u=0.01 default, matching SSIDS/MUMPS)
4. A posteriori pivoting (APP) blocked kernel (SSIDS model: factor each diagonal block without pivoting, check threshold after the fact, rollback on failure, BLAS-3 inter-block updates)
5. SIMD micro-kernel for the inner loop (Rust `std::arch` or `pulp` crate)
6. Full micro-kernel architecture per BLIS (if profiling justifies)

> **BK → TPP → APP transition rationale (from SSIDS/MUMPS code review).** Classic BK is ideal for Stage 1 (scalar dense) because it has exact literature references (BK77 Theorem 3), test oracles from the paper, and strong growth-factor guarantees. For the blocked multifrontal engine (Stage 2), threshold partial pivoting (TPP) is more natural: delayed pivots are just columns that don't meet the threshold. SSIDS and MUMPS both use TPP with u=0.01 (not BK) for their multifrontal engines. The APP (a posteriori pivoting) approach is the state of the art: factor each block without pivoting for maximum BLAS-3 efficiency, then check the threshold test after the fact and roll back if needed. SSIDS uses APP as its primary kernel with TPP as fallback. This staged approach — BK → TPP → APP — lets each kernel be thoroughly tested before the next is attempted, and earlier kernels remain as reference implementations and fallback paths.

> **Fused update+argmax optimization (from faer code review).** faer's `rank_1_update_and_argmax` / `rank_2_update_and_argmax` functions compute the off-diagonal maximum during the Schur complement rank update, avoiding a separate O(n²) scan pass per pivot. This is a meaningful optimization for the scalar kernel and should be adopted when performance matters.

### 3.4 SIMD Portability Policy

FERAL targets the **stable Rust toolchain** with no nightly features. SIMD is introduced in Phase 2+ only, after the scalar implementation is correct and benchmarked.

**ISA baseline:** x86-64 with AVX2 (Haswell, 2013+). This covers all modern workstations and university HPC clusters. SSE4.2 fallback is not required.

**Implementation approach:** Runtime CPU feature detection via `is_x86_feature_detected!("avx2")`, with an `#[target_feature(enable = "avx2")]` hot path and a scalar cold path. This keeps the binary portable: it runs on pre-AVX2 hardware (slower, scalar path) without a compile-time flag.

**Non-x86 architectures (Apple Silicon, ARM HPC):** The scalar path runs correctly on all targets. NEON/SVE SIMD paths are a Phase 4 concern. Do not introduce `#[cfg(target_arch = "x86_64")]`-only paths in the scalar code — keep scalar logic architecture-agnostic.

**Constraint:** No `std::arch` intrinsics in Phase 1. All `unsafe` SIMD code lives in `src/dense/micro_kernel.rs` (a single, isolated file). Phase 2 may add it there without touching the rest of the codebase.

---

## 4. Benchmark Framework

### 4.1 Benchmark Matrix Sources

**Tier 1: KKT systems from ripopt.** Dump symmetric indefinite KKT matrices during CUTEst and electrolyte problem solves. Each matrix includes: the matrix itself (Matrix Market symmetric format, lower triangle), the RHS vector, expected inertia `(n, m, 0)`, and regularization metadata (`delta_w`, `delta_c`). Correctness is validated by residual check — no separate MUMPS reference solve is needed. Provides hundreds of matrices from the exact application domain. The problems ripopt currently fails on are especially valuable — those are the hardest KKT systems.

ripopt has built-in KKT dump support (no modification required). Set in `SolverOptions` before calling `ripopt::solve()`:

```rust
opts.kkt_dump_dir  = Some(PathBuf::from("../feral/data/matrices/kkt/hs071"));
opts.kkt_dump_name = "hs071".to_string();
```

Each successful IPM iteration writes two files:
- `<name>_<iter:04>.mtx` — Matrix Market `coordinate real symmetric`, lower triangle, 1-indexed, full 17-digit precision
- `<name>_<iter:04>.json` — Sidecar with the following fields:

```json
{
  "problem_name": "hs071",
  "iteration": 4,
  "n": 4,
  "m": 2,
  "rhs": [...],
  "inertia": { "positive": 4, "negative": 2, "zero": 0 },
  "delta_w": 0.0,
  "delta_c": 0.0,
  "status": "ongoing"
}
```

**Field notes:**

- `delta_w` — amount added to the first `n` diagonal entries (primal block) during inertia correction. `0.0` for well-conditioned matrices; positive when the Hessian was indefinite or degenerate. The `.mtx` file already contains this perturbation — the inertia recorded is for the matrix *as written*.
- `delta_c` — amount subtracted from the last `m` diagonal entries (constraint block). Typically a small constant (e.g. `1e-8`) or `0.0`.
- `rhs` — the right-hand side for the scaled NLP problem. ripopt applies gradient-based NLP scaling internally; the matrix and RHS are both in scaled space. This is irrelevant to FERAL's correctness — it just factors a symmetric indefinite matrix and solves a linear system.

The dump fires only in the main IPM loop (not restoration), only when factorization succeeds, and is a no-op when `kkt_dump_dir` is `None`.

**Collecting matrices:** Run the ripopt CUTEst suite with `kkt_dump_dir` set. Each of the 727 problems contributes one matrix per IPM iteration — typically 10–50 matrices per problem, covering a full trajectory from initial point to convergence or failure. The collection harness is `collect_kkt` in the ripopt repository (see Section 13.4).

**Trajectory-aware testing.** Don't just test individual matrices. Test sequences of matrices from a single IPM solve (iteration 1 through convergence). This tests the realistic usage pattern where FERAL sees a sequence of similar but progressively more ill-conditioned matrices. The sidecar JSON includes the iteration number, enabling the benchmark harness to group matrices by problem and report trajectory-level statistics (e.g., "all iterations of hs071 solve correctly" vs "fails at iteration 23 of hs071"). Matrices from failed solves are the most valuable — problems where ripopt currently fails produce the hardest KKT systems and should be priority test cases.

**Cross-solver harvesting (Phase 2+).** Consider instrumenting Ipopt itself (via its debug output or by writing a custom `SparseSymLinearSolverInterface` wrapper that dumps the assembled KKT system) to harvest matrices from a mature IPM with different regularization strategies. This provides diversity beyond ripopt's approach — Ipopt may produce different perturbation patterns, different ordering of regularization attempts, and matrices from its restoration phase that ripopt does not yet have.

**Tier 2: SuiteSparse Matrix Collection.** Filter for symmetric indefinite matrices. Coverage outside NLP/KKT structure — structural mechanics, electromagnetics, saddle point systems from CFD. Sorted by size with representatives at each order of magnitude.

Tier 2 sidecars (expected inertia) are generated by running rmumps on each matrix before FERAL exists. The script `scripts/fetch-suitesparse.sh` downloads matrices AND generates sidecars in one pass using rmumps as a one-time oracle. These sidecars are committed to the repo under `data/benchmark-config.toml` alongside the matrix metadata. rmumps is used here as a bootstrap oracle only — it is not called during FERAL's benchmark runs. Once sidecars exist, FERAL validates against them directly (residual + inertia match). This is the only sanctioned use of rmumps as a reference solver; it does not create an architectural dependency.

**Tier 3: Constructed stress tests.** Targeted at specific failure modes:

- Nearly singular matrices (tiny pivots — tests pivot threshold handling)
- Matrices where AMD gives poor fill vs METIS (tests ordering sensitivity)
- Arrow / bordered diagonal structure (common in decomposition-based NLP)
- Matrices requiring 2×2 pivot blocks (tests Bunch-Kaufman correctness)
- Highly indefinite with balanced positive/negative inertia (worst case for delayed pivoting)

**Tier 4: Scaling frontier.** SuiteSparse matrices at n > 100k. Not solved in Phase 1; defined from the start to track when they become feasible.

### 4.2 Benchmark Harness

A cargo subcommand or binary (`cargo run --bin bench --release` or `cargo run --bin bench`) that:

1. Loads all benchmark matrices from the data directory
2. Runs FERAL on each: symbolic analysis → numeric factorization → solve
3. Checks correctness using the **self-contained validation contract** (see below)
4. Records: factorization time, solve time, peak memory, fill ratio (actual vs predicted NNZ in factor)
5. Compares timings against MUMPS via rmumps (reference timings stored from previous runs)
6. Writes machine-readable results to `dev/benchmarks/run-YYYY-MM-DD-NN.json`
7. Diffs against previous run and produces summary

**Validation contract.** FERAL's correctness is validated without a separate MUMPS solve. For each matrix:

1. **Inertia:** FERAL's reported `(positive, negative, zero)` must exactly match the sidecar. For KKT matrices this is always `(n, m, 0)`.
2. **Residual:** Given the RHS `b` from the sidecar and FERAL's solution `x`, compute `‖Ax − b‖₂ / ‖b‖₂`. Must be below `tolerance`, where tolerance is set at plan time based on the estimated condition number of the matrix (not tuned post-hoc).
3. **Regularization flag:** Matrices with `delta_w > 1e-4` are tagged as *regularized* in the benchmark report. These are numerically degenerate KKT systems. FERAL is expected to pass them (the inertia is well-defined for the matrix as written) but they are reported separately from clean matrices.

No MUMPS reference solution is needed or used for correctness checking. MUMPS is used only for timing comparison.

### 4.3 Metrics

| Metric | How measured | Target |
|--------|-------------|--------|
| Solve rate | Correct inertia + solution within tolerance | 100% on KKT set |
| Median time ratio vs MUMPS (small frontals, max_front < 64) | FERAL time / MUMPS time | < 2× |
| Median time ratio vs MUMPS (medium frontals, max_front 64–256) | FERAL time / MUMPS time | < 3× (blocked kernel needed) |
| Median time ratio vs MUMPS (large frontals, max_front > 256) | FERAL time / MUMPS time | < 5× (SIMD kernel needed) |
| Worst-case time ratio | Max across benchmark set | < 10× |
| Peak memory ratio vs MUMPS | FERAL peak / MUMPS peak | < 2× |
| Fill ratio | Actual NNZ / predicted NNZ | < 1.5 |
| CUTEst integration solve rate | Problems solved when FERAL replaces MUMPS in ripopt | ≥ ripopt's current rate |

> **Performance target stratification (from MUMPS code review).** MUMPS's speed advantage comes primarily from optimized BLAS-3 calls (DGEMM/DSYRK) for the Schur complement update. For small frontals (<64), BLAS call overhead makes scalar code competitive and the <2× target is achievable. For medium frontals (64–256), a blocked panel factorization with a hand-written GEMM-like kernel is needed. For large frontals (>256), SIMD micro-kernels are essential to match optimized BLAS throughput. KKT matrices from chemical engineering NLP typically have small-to-medium frontals (20–200), so the primary target domain is well-served by the scalar and blocked kernels.

### 4.4 Benchmark Matrices in Git

Large binary matrices are stored outside the main repo (separate data repo or downloaded by setup script). The benchmark harness knows their location via a config file (`data/benchmark-config.toml`). The format:

```toml
# data/benchmark-config.toml
# One [[matrix]] entry per benchmark matrix. Generated by fetch-suitesparse.sh
# and by the KKT collection harness. Do not edit by hand.

[[matrix]]
path = "data/matrices/kkt/hs071/hs071_0001.mtx"
tier = "kkt"
n = 10          # total KKT dimension (primal + dual)
m = 2           # number of constraints (= expected negative inertia)
rhs_path = "data/matrices/kkt/hs071/hs071_0001.json"  # JSON sidecar has rhs field
inertia = { positive = 8, negative = 2, zero = 0 }
condition_estimate = 1e4  # used by bench harness to set residual tolerance

[[matrix]]
path = "data/matrices/suitesparse/bcsstk14.mtx"
tier = "suitesparse"
n = 1806
m = 0           # not a KKT matrix; m unused
rhs_path = ""   # no sidecar; bench harness generates random RHS
inertia = { positive = 1806, negative = 0, zero = 0 }
condition_estimate = 1e6
```

The `tier` field controls how the harness loads the RHS (`kkt` reads from the JSON sidecar; `suitesparse` generates a random RHS or uses a stored one). The `condition_estimate` field drives the residual tolerance via Section 2.11's formula. The main repo contains:

- Small exact test matrices (inline in test code or small CSV/MTX files)
- Benchmark metadata (matrix names, sizes, expected inertia, reference timings)
- Benchmark result history (JSON files, committed after each run)

---

## 5. Development Process

### 5.1 Feature Development Lifecycle

Every new feature follows this sequence. No step is optional.

#### Step 1: Literature Research

Search the foundational and recent literature for the specific algorithmic details needed. Use litdb, PubMed, web search, and the canonical references listed in Section 3.1. Identify:

- The canonical algorithm and its variants
- Known failure modes and edge cases
- What modern solvers actually do vs what the textbook says
- Whether anyone has solved this sub-problem better since the canonical paper

Document findings in `dev/research/feature-name.md` containing:

- Feature goal
- Canonical references read (with specific sections/theorems cited)
- What each reference says about the approach
- Disagreements or alternatives in the literature
- Chosen approach with rationale

#### Step 2: Code Inspection

Inspect existing implementations (SPRAL/SSIDS, BLIS, faer, SuiteSparse) for algorithmic choices not fully described in papers. Focus on:

- Edge cases and numerical guards
- Actual threshold values used
- Fallback behavior when the primary algorithm fails
- Things the authors learned the hard way that aren't in the paper

Document in the research note, separately from literature findings. Maintain clean-room discipline: read to understand the problem, then close the code and implement from algorithmic understanding.

#### Step 3: Implementation Plan

Before touching `src/`, write a plan in `dev/plans/feature-name.md`:

- Exact function signatures to be added or changed
- Invariants the implementation must maintain (inertia correctness, symmetry preservation, backward error bounds)
- Test cases derived from research (known matrices with known factorizations, edge cases from code inspection)
- Performance target referencing benchmark suite
- Failure modes identified from literature and code inspection

#### Step 4: Write Tests First

Tests are written before the implementation, derived from the plan:

- **Exact tests:** Small matrices with hand-computed or reference-solver factorizations. Verify correct pivots, correct inertia, factors multiply back to original within tolerance. Source: literature examples, constructed cases.
- **Reference tests:** Medium matrices from benchmark suite with MUMPS ground truth. Solution must match to tolerance. Inertia must match exactly.
- **Property tests:** Random symmetric indefinite matrices verifying structural invariants: inertia sums to n, factors have correct shape, pivot blocks are 1×1 or 2×2 only, memory within predicted bound.
- **Regression tests:** Every bug gets a permanent test case with a note about the original failure.

#### Step 5: Implement

Write the code. Commit frequently and atomically.

#### Step 6: Benchmark and Record

Run the full benchmark suite. Record results. Compare against previous run.

### 5.2 Test-Driven Development Rules

1. **Tests before implementation.** Test oracles (expected answers) come from independent sources: hand calculation, reference solvers, or tests written before the implementation exists. An agent must never write both the implementation and the test oracle in the same session without the oracle being derived from an external source.

2. **Tolerances are justified, not tuned.** Each test tolerance is set during the plan phase based on the condition number of the test matrix and the expected error bound. If the implementation exceeds the tolerance, the implementation is wrong — not the tolerance. **Rule: never loosen a test tolerance without recording the justification in the session checkpoint and getting human approval.**

3. **Benchmark numbers are mechanical.** The harness writes results to files. Agents read and summarize but do not generate numbers in prose. The context assembly script diffs automatically.

4. **Coverage is a signal, not a target.** Low coverage on a new module means tests weren't written first or code paths are unexercised. Coverage drops are flagged but chasing 100% is not the goal.

### 5.3 Code Quality

- `cargo clippy` with strict settings (deny warnings in CI)
- `cargo fmt` enforced
- No `unsafe` without a documented safety argument in a comment block explaining the invariants
- No `unwrap()` or `expect()` in library code — proper error handling with `Result`
- No panics in library code
- Errors are structured types, not strings
- All public APIs have doc comments with examples
- Internal modules have module-level doc comments explaining purpose and invariants

### 5.4 Preventing Agent Rationalization

Structural safeguards to ensure honest reporting:

| Problem | Safeguard |
|---------|-----------|
| Agent encodes bug in both implementation and test | Test oracles from independent sources only |
| Agent loosens tolerance to pass | Requires human approval + justification in checkpoint |
| Agent selectively reports good benchmark results | Harness writes raw files; context script diffs mechanically |
| Agent frames failure as design choice | CLAUDE.md instruction: state symptoms, incorrect outputs, failing cases — not summaries |
| Agent makes irreversible architectural decision | The following changes require human review and explicit approval before being committed — agent must stop, present the proposed change, and wait for a reply: (1) any change to the `BunchKaufmanParams` struct fields or `ZeroPivotAction` enum variants; (2) any change to the `MemoryPlan` struct fields or the two-allocator model (`FactorBump` / `ContribPool`); (3) any change to the public `FeralError` type (adding, removing, or renaming variants); (4) any change to the benchmark matrix file format (`.mtx` schema or sidecar `.json` field names/types); (5) any change to inertia counting semantics (what counts as positive/negative/zero); (6) any change to assembly tree traversal order or extend-add operation contract; (7) adding any non-Rust dependency to the core solver crate (no BLAS, LAPACK, C, or Fortran); (8) changing the Stage 1 / Stage 2 build order or promoting Stage 2 work before Stage 1 correctness is complete; (9) the initial design of the `DenseKernel` trait or `PivotStrategy` trait (Section 2.5 defers these to Stage 2 — when the time comes, their interface is a major decision that cascades into every frontal matrix operation and must not be chosen unilaterally). |
| Slow quality degradation | CI gates: clippy, fmt, no unwrap, commit message body required |
| Agent retries abandoned approach | tried-and-rejected.md consulted at session start |

---

## 6. Session Protocol & Agent ELN

### 6.1 Session Lifecycle

Every Claude Code session follows three phases:

**Orient.** Run `dev/assemble-context.sh`. Read `dev/context.md`. Understand: where the project stands, what was decided, what was tried and failed, current benchmark numbers, what the next priority is.

**Work.** Normal development following the feature lifecycle (Section 5.1).

**Checkpoint.** Before the session ends:

1. Run `cargo run --bin bench --release` and record results
2. Write session summary to `dev/sessions/YYYY-MM-DD-NN.md`
3. Append to `dev/tried-and-rejected.md` if anything was abandoned
4. Append to `dev/decisions.md` if any architectural decisions were made
5. Append to `CHANGELOG.md` Unreleased section if changes are user-visible
6. Commit everything including the session file as the final commit

### 6.2 Persistent State Files

All persistent state lives in the repo under `dev/`:

```
dev/
├── sessions/                    # One file per session, append-only
│   └── YYYY-MM-DD-NN.md
├── research/                    # Literature research notes per feature
│   └── feature-name.md
├── plans/                       # Implementation plans per feature
│   └── feature-name.md
├── benchmarks/                  # Raw benchmark results per run
│   ├── run-YYYY-MM-DD-NN.json
│   └── summary.md              # Current headline numbers + trend
├── decisions.md                 # Architectural decision log (append-only)
├── tried-and-rejected.md        # Abandoned approaches with detail (append-only)
├── constraints.md               # Hard constraints (rarely changes)
├── context.md                   # Auto-assembled orientation file
├── assemble-context.sh          # Script to rebuild context.md
└── templates/
    └── session.md               # Template for session checkpoint
    └── research.md              # Template for research notes
    └── plan.md                  # Template for implementation plans
```

### 6.3 Session Checkpoint Template

```markdown
# Session YYYY-MM-DD-NN

**Branch:** feature/name
**Commits:** abc1234, def5678, ...

## Goal
What this session set out to accomplish.

## What Was Done
Concrete accomplishments. Reference commits.

## Tried and Abandoned
What was attempted and didn't work. Include:
- What the approach was
- What symptoms appeared (failing tests, wrong output, performance)
- Why it was abandoned
- Under what conditions it might be worth revisiting

## Decisions Made
Any architectural decisions, with rationale and alternatives considered.
(Also appended to dev/decisions.md)

## Benchmark Results
Paste or reference the benchmark diff from this session.
Solve rate: X/N (change from previous)
Median time ratio: Y× (change from previous)

## Open Questions
Unresolved questions for future sessions.

## Next Session Should
Specific, actionable starting point for the next session.
```

### 6.4 Context Assembly

`dev/assemble-context.sh` produces `dev/context.md` with a **hard 350-line budget**. Items are written in priority order; lower-priority items are truncated or omitted if the budget is exhausted. The budget exists because `context.md` is loaded into every session — exceeding it silently degrades agent quality.

**Priority order (highest to lowest):**

1. **Next session goal** — the "Next Session Should" line from the most recent session (1–3 lines). Always included.
2. **Hard constraints** — from `dev/constraints.md` (~20 lines). Always included.
3. **Benchmark headline** — from `dev/benchmarks/summary.md` (~10 lines). Always included.
4. **Last 3 session summaries** — most recent first (~150 lines total). Always included.
5. **Open questions** — unresolved questions from the last 3 sessions only (~20 lines). Older open questions are closed or moved to a research note.
6. **Recent decisions** — from `dev/decisions.md`, last 14 days only (~30 lines).
7. **Recent tried-and-rejected** — entries dated within the last 30 days from `dev/tried-and-rejected.md`. Truncated to fit remaining budget. Always ends with:

```
[X of Y total entries shown. Full list: dev/tried-and-rejected.md]
```

**Accessing older tried-and-rejected entries:** Read `dev/tried-and-rejected.md` directly. The file is append-only and contains the full history. When starting work on a feature, search it for entries tagged with that feature name before beginning.

### 6.5 Tried-and-Rejected Entry Format

```markdown
## [YYYY-MM-DD] Greedy 2×2 pivot detection

**What:** Detect 2×2 pivot blocks greedily by examining adjacent diagonal
entries without lookahead.

**Why it seemed good:** Simpler than full Bunch-Kaufman scan, O(1) per pivot
instead of O(n) column search.

**What happened:** Fails on arrow matrices. The greedy approach misses cases
where the off-diagonal element is not adjacent to the diagonal entry that
needs the 2×2 block. Test `arrow_5x5_indefinite` produces wrong inertia
(3,0,2) instead of expected (2,1,2).

**Why abandoned:** The column search in Bunch-Kaufman is necessary — the
greedy shortcut is incorrect, not just suboptimal.

**Revisit if:** Someone finds a paper proving a restricted greedy approach
works for specific matrix structures (e.g., banded).
```

### 6.6 Decision Log Entry Format

```markdown
## [YYYY-MM-DD] Dense kernels implemented in-house, no faer dependency

**Decision:** Implement all dense linear algebra operations (LDLᵀ, Schur
update, triangular solve) from the literature rather than depending on faer.

**Rationale:** Full control over numerical behavior, co-design with pivot
strategy, zero external dependencies, publishable as self-contained.

**Alternatives considered:**
- Use faer behind DenseKernel trait: simpler initially, but adds dependency
  and limits co-design of pivot/kernel interaction
- Use raw BLAS/LAPACK via C FFI: defeats Rust-only goal

**Reversibility:** The DenseKernel trait boundary means faer could be
re-introduced as a backend later if self-implemented kernels prove
insufficient. Low-risk decision.
```

---

## 7. Git Practices

### 7.1 Branch Strategy

- `main` is always correct: all tests pass, clippy clean, fmt clean. No exceptions.
- Feature branches: `feature/bunch-kaufman-pivot`, `fix/arrow-matrix-inertia`, `bench/add-cutest-kkt-matrices`, `research/schur-update-blocking`
- Short-lived branches (2-3 sessions max). If longer, split and land partial work.

### 7.2 Commit Discipline

Commits are small and atomic. One commit does one thing. Never mix implementation and unrelated test changes. This enables `git bisect` when regressions appear.

### 7.3 Commit Message Format

```
Short imperative summary, under 72 characters

Body answers: what changed, why, and what evidence supports it.

For implementations: reference the research note and literature.
For bug fixes: state symptom, root cause, and regression test added.
For benchmarks: state what matrices were added/changed and why.

Tests: list test names that verify this change.
Ref: dev/research/relevant-note.md
Ref: Author, Journal, Year (for literature references)
```

**Rules:**
- Every commit has a body (not just the summary line) — enforced by CI
- No vague messages: "fix bug," "improve performance," "clean up" are rejected
- Bug fix commits include symptom, root cause, and regression test
- `Ref:` lines link to research notes and literature

### 7.4 Tags and Releases

Tag at phase milestones with annotated messages including benchmark state:

- `v0.1.0` — FERAL solves 100% of small KKT set correctly
- `v0.2.0` — Within 5× of MUMPS on medium set
- `v0.3.0` — Shared-memory parallel, within 2× of MUMPS
- `v0.4.0` — Replaces MUMPS in ripopt with no regression
- `v1.0.0` — Full solver: competitive performance, MPI, GPU

Intermediate tags for significant points: `bench-baseline`, `first-cutest-integration`.

### 7.5 CI Enforcement

Pre-commit or CI checks (non-negotiable):

- `cargo test` passes
- `cargo clippy -- -D warnings` passes
- `cargo fmt --check` passes
- No `unwrap()` in `src/` (grep check)
- Commit message has body (not just summary)

### 7.6 Benchmark Results in Git

- Raw benchmark output files (`dev/benchmarks/run-*.json`) are committed after each run
- `git diff` on benchmark summary shows exactly what changed
- Large binary benchmark matrices are NOT in the main repo — separate data repo or download script
- Benchmark metadata (names, sizes, expected inertia) is in the main repo

---

## 8. Changelog

Maintain `CHANGELOG.md` in the repo root following Keep a Changelog format.

### 8.1 Format

```markdown
# Changelog

## [Unreleased]

### Added
- (accumulated during development sessions)

### Changed

### Fixed

### Performance
- Solve rate: X/N
- Median time ratio vs MUMPS: Y×

## [0.1.0] - YYYY-MM-DD

### Added
- ...

### Performance
- Solve rate: X/N
- Median time ratio vs MUMPS: Y×
```

### 8.2 Rules

- Agent appends to Unreleased section during session checkpoint
- Only user-visible changes: would a user of FERAL or POUNCE notice this? If not, it belongs in the session log only
- Every versioned release includes a Performance section with headline benchmark numbers
- Moving Unreleased to a version is a human decision
- The Performance section is non-standard but essential for a solver project — makes trajectory visible to anyone reading the changelog

---

## 9. Success Criteria

### 9.1 Phase Milestones

| Phase | Milestone | Key Metric |
|-------|-----------|------------|
| 1 | Correct single-thread solver | 100% solve rate on small/medium KKT set, within 5× of MUMPS |
| 2 | Optimized + parallel | Shared-memory parallel, within 2× of MUMPS on medium set |
| 3 | Integration | Replaces MUMPS in ripopt/POUNCE, no regression in CUTEst solve rate |
| 4 | Full scale | Distributed MPI, GPU offload, competitive on SuiteSparse large matrices |

### 9.2 Research Success

The publishable claim: a modern, self-contained Rust implementation of a multifrontal symmetric indefinite solver with adaptive pivot strategies (tuned by the outer NLP loop) is competitive with legacy Fortran solvers and offers capabilities they lack.

Evidence required:
- CUTEst benchmark results showing competitive solve rates
- Demonstration of adaptive pivot strategy improving convergence on hard problems
- Electrolyte problem performance matching or exceeding ripopt's current results with MUMPS
- Any unique solves (problems FERAL handles that MUMPS/MA57 cannot) are the strongest publishable result

### 9.3 Dashboard

Every benchmark run produces:

```
Solve rate:              X / N matrices correct
Median time ratio:       Y× vs MUMPS
Worst-case time ratio:   Z× vs MUMPS
Peak memory ratio:       W× vs MUMPS
Fill ratio:              actual NNZ / predicted NNZ
CUTEst integration:      A / B problems (when available)
```

These numbers appear at the top of every session's context file. The trend over the last several sessions is included. An agent sees immediately whether things are improving and where the frontier is.

---

## 10. Constraints

These are hard constraints. They do not change without explicit human decision and a recorded rationale in `dev/decisions.md`.

1. **License:** MIT — applies to all code in this repository (`src/`, `benches/`, `tests/`, `scripts/`). No exceptions.
2. **License isolation:** Code that depends on ripopt (EPL-2.0) or rmumps (CeCILL-C) does not live in this repo. The KKT matrix collection script lives in the ripopt repository. FERAL only consumes the output data files, which are not copyrightable.
3. **Language:** Rust, stable toolchain
4. **Core dependencies:** Zero non-Rust dependencies for the solver itself. No BLAS, LAPACK, or Fortran.
5. **Clean-room:** All code original, derived from published papers and BSD-licensed references. No code copied from any source.
6. **rmumps role:** Testing reference for validation only, not an architectural dependency. FERAL's design assumes it will eventually cover the full problem size range.
7. **Correctness over performance:** A correct solver that is slow is better than a fast solver that produces wrong inertia. Performance optimization never compromises correctness.
8. **Inertia is exact:** There is no tolerance on inertia. The reported inertia (positive, negative, zero eigenvalue counts) must be exactly correct. This is not negotiable — POUNCE's convergence depends on it.
9. **No unsafe without justification:** Every `unsafe` block requires a comment documenting the safety invariants being upheld.
10. **Target community:** Chemical engineering nonlinear programming, where Ipopt is the dominant solver. POUNCE is designed to be an Ipopt-like solver in capability and interface. FERAL is its linear algebra foundation.

---

## 11. Repository Structure

**The layout below shows the Stage 1 state** (end of Phase 1a). Stage 2 directories (`ordering/`, `symbolic/`, `numeric/`) are created in Phase 1b. Trait files (`pivot/trait.rs`, `dense/trait.rs`) do NOT exist in Stage 1 — they are created when the `DenseKernel` and `PivotStrategy` traits are designed in Stage 2 (with human review, per Section 5.4 trigger #9). Do not create them in Phase 1.

```a
feral/
├── Cargo.toml                   # [lib] + [[bin]] bench; see Section 11.1 for contents
├── CHANGELOG.md
├── CLAUDE.md                    # Agent protocol (points to dev/ state)
├── LICENSE                      # MIT
├── README.md
├── src/
│   ├── lib.rs                   # Public API: factor(), solve(), SymmetricMatrix, Factors, Inertia, FeralError
│   ├── dense/                   # Dense kernel — Stage 1
│   │   ├── mod.rs
│   │   ├── matrix.rs            # SymmetricMatrix type (full n×n column-major, lower triangle authoritative)
│   │   ├── factor.rs            # Bunch-Kaufman LDLᵀ factorization
│   │   ├── solve.rs             # Full solve sequence per Section 2.9
│   │   └── equilibrate.rs       # Iterative Knight-Ruiz equilibration (Section 2.8)
│   ├── inertia.rs               # Inertia type and counting (Section 2.4)
│   └── error.rs                 # FeralError type (Section 11.2)
│
│   # Stage 2 additions (Phase 1b and beyond — do not create in Phase 1a):
│   # ├── ordering/              # AMD ordering, elimination tree, postordering
│   # ├── symbolic/              # Symbolic factorization, supernodes, MemoryPlan
│   # ├── numeric/               # Multifrontal engine, assembly, factorize driver
│   # │   ├── assembly.rs        # Extend-add, scatter-gather
│   # │   ├── frontal.rs         # Frontal matrix management, FactorBump, ContribPool
│   # │   └── factorize.rs       # Main multifrontal factorization driver
│   # ├── sparse/                # CSC storage, sparse matrix utilities
│   # ├── pivot/                 # PivotStrategy trait (Stage 2, human review required)
│   # │   ├── trait.rs           # PivotStrategy trait — do not create until Stage 2
│   # │   └── bunch_kaufman.rs
│   # └── dense/trait.rs         # DenseKernel trait — do not create until Stage 2
│
├── tests/
│   ├── exact/                   # Small matrices, hand-verified (from Bunch & Kaufman 1977)
│   ├── property/                # Randomized invariant checks (inertia sums to n, etc.)
│   └── regression/              # Bug reproduction tests (added as bugs are found)
├── benches/                     # Criterion microbenchmarks
├── src/bin/
│   └── bench.rs                 # Full benchmark harness (cargo run --bin bench)
├── data/
│   ├── matrices/                # Small test matrices (MTX format)
│   └── benchmark-config.toml    # Matrix metadata, paths, expected values
├── dev/
│   ├── sessions/
│   ├── research/
│   ├── plans/
│   ├── benchmarks/
│   ├── decisions.md
│   ├── tried-and-rejected.md
│   ├── constraints.md
│   ├── context.md
│   ├── assemble-context.sh
│   └── templates/
│       ├── session.md
│       ├── research.md
│       └── plan.md
└── scripts/
    └── fetch-suitesparse.sh     # Download SuiteSparse symmetric indefinite matrices
                                 # collect_kkt lives in ../ripopt/cutest_suite/ — see Section 13.4
```

### 11.1 Stage 1 Cargo.toml

```toml
[package]
name = "feral"
version = "0.1.0"
edition = "2021"
license = "MIT"
description = "Sparse symmetric indefinite direct solver in pure Rust"

[lib]
name = "feral"

[[bin]]
name = "bench"
path = "src/bin/bench.rs"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
approx = "0.5"
criterion = { version = "0.5", features = ["html_reports"] }
# rmumps used only for reference timing comparisons in benchmarks:
# rmumps = { path = "../ripopt/rmumps" }  # uncomment when needed

[[bench]]
name = "dense_factor"
harness = false
```

Note: `rmumps` is kept commented out by default — it is a dev-only reference dependency and requires the ripopt workspace. Uncomment it when running timing comparisons in Phase 1b+.

### 11.2 Initial `FeralError` Variants

The initial `FeralError` type for Stage 1. Variants are locked (Section 5.4 trigger #3 — changes require human review):

```rust
/// Errors returned by FERAL's public API.
#[derive(Debug)]
pub enum FeralError {
    /// The matrix is numerically rank-deficient: a pivot was exactly or
    /// near-zero and `ZeroPivotAction::Fail` was specified. The factorization
    /// is incomplete. The `zero` count in the returned partial `Inertia` (if
    /// any) indicates how many zero pivots were encountered.
    /// POUNCE maps this to its perturbation handler (Section 2.12): add diagonal
    /// perturbation to the matrix and call factor() again.
    NumericallyRankDeficient,

    /// Input matrix dimensions are inconsistent or the matrix is not square.
    InvalidInput(String),

    /// The RHS vector length does not match the factored matrix dimension.
    DimensionMismatch { expected: usize, got: usize },
}
```

These three variants cover Stage 1. `NumericallyRankDeficient` is what POUNCE checks to detect degenerate constraints or trigger inertia correction. `PerturbationFailed` was removed because perturbation is now managed externally by the IPM layer (Section 2.4). Additional variants (e.g., `OutOfMemory`, `SymbolicAnalysisFailed`) will be added in Stage 2 with human review.

**Input validation checklist for `factor()`.** Perform these checks at the top of `factor()` before any computation, returning `FeralError::InvalidInput` with a descriptive message on failure:

1. `n == 0` → error: `"matrix dimension is zero"`
2. `matrix.data.len() != n*n` → error: `"matrix data length {got} != expected {n*n} for n={n}"`
3. Any `f64::is_nan()` or `f64::is_infinite()` in the lower triangle → error: `"matrix contains NaN or Inf at index ({i},{j})"`

For `solve()` and `solve_refined()`:

4. `rhs.len() != factors.n` → `FeralError::DimensionMismatch { expected: factors.n, got: rhs.len() }`

No other input validation is required. The caller is responsible for providing a symmetric matrix in n×n column-major format with the lower triangle populated. FERAL only reads the lower triangle; the strict upper triangle is ignored.

---

## 12. CLAUDE.md Contents

This file goes in the repo root. It is the agent's entry point.

```markdown
# FERAL — Agent Protocol

## At Session Start

1. Run `./dev/assemble-context.sh`
2. Read `dev/context.md` — this is your orientation
3. Identify your goal for this session from the "next session should" section
4. If starting a new feature, read the relevant research note in `dev/research/`
   and implementation plan in `dev/plans/` before writing any code

## During Work

- Follow the feature development lifecycle in the project spec
- Commit frequently and atomically — one commit per logical change
- Every commit message must have a body explaining what, why, and evidence
- Run `cargo test` before every commit
- If you try something and abandon it, note it immediately (don't wait for checkpoint)

## At Session End (MANDATORY)

1. Run `cargo run --bin bench --release` (or `cargo run --bin bench`)
2. Write session checkpoint to `dev/sessions/YYYY-MM-DD-NN.md` using the
   template in `dev/templates/session.md`
3. If anything was abandoned: append to `dev/tried-and-rejected.md`
4. If any decisions were made: append to `dev/decisions.md`
5. If changes are user-visible: append to CHANGELOG.md Unreleased section
6. Commit the session file and all dev/ changes as the final commit:
   `git commit -m "Session YYYY-MM-DD-NN checkpoint\n\n[summary of session]"`

## Hard Rules

- NEVER loosen a test tolerance without human approval. Record justification.
- NEVER skip the checkpoint. A session without a checkpoint is lost work.
- NEVER modify existing entries in decisions.md or tried-and-rejected.md.
  These are append-only logs.
- NEVER use `unwrap()` or `expect()` in src/. Use proper error handling.
- NEVER introduce `unsafe` without a safety comment.
- NEVER commit without running `cargo test` and `cargo clippy`.
- When recording abandoned approaches: state what actually happened —
  symptoms, incorrect outputs, failing test cases. Do not summarize failure
  as a design choice.
- When benchmark numbers appear worse than previous session: report this
  explicitly. Do not omit unfavorable comparisons.

## Constraints

- MIT license
- Pure Rust, stable toolchain, no non-Rust dependencies in core solver
- Clean-room implementation from published papers and BSD-licensed references
- Inertia must be exactly correct (no tolerance)
- Correctness before performance, always
- rmumps is a testing reference, not a dependency
```

---

## 13. Getting Started

### 13.1 Initial Setup

```bash
# Create the repository
cargo init --lib feral
cd feral

# Create the directory structure
mkdir -p src/{ordering,symbolic,numeric,pivot,dense,sparse}
mkdir -p tests/{exact,reference,property,regression}
mkdir -p benches bin data/matrices
mkdir -p dev/{sessions,research,plans,benchmarks,templates}
mkdir -p scripts

# Initialize dev files
touch dev/decisions.md
touch dev/tried-and-rejected.md
touch dev/constraints.md
touch CHANGELOG.md

# Copy this spec to the repo
cp FERAL-PROJECT-SPEC.md dev/

# Create CLAUDE.md from Section 12 above

# Initialize git
git init
git add .
git commit -m "Initialize FERAL project structure

Frontal Elimination for Robust Algebraic Linear Systems.
Sparse symmetric indefinite direct solver in pure Rust.

Project spec: dev/FERAL-PROJECT-SPEC.md
Agent protocol: CLAUDE.md
License: MIT"
```

### 13.2 First Session Goals

The first Claude Code session should:

1. Set up CI (GitHub Actions: test, clippy, fmt, grep for unwrap). Use this workflow template at `.github/workflows/ci.yml`:

```yaml
name: CI
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - name: fmt
        run: cargo fmt --check
      - name: clippy
        run: cargo clippy -- -D warnings
      - name: test
        run: cargo test
      - name: no-unwrap
        run: |
          if grep -rn '\.unwrap()' src/; then
            echo "ERROR: unwrap() found in src/ — use proper Result handling"
            exit 1
          fi
      - name: commit-body
        # Reject bare summary-only commits on PRs (advisory — does not block CI)
        run: |
          git log --format="%B" -1 | awk 'NR==2{found=1} END{if(!found){print "WARNING: commit has no body";}}'
```

The `no-unwrap` grep checks for `.unwrap()` only. `unwrap_or`, `unwrap_or_else`, and `unwrap_or_default` are permitted. The commit-body check is advisory (prints a warning, does not fail the build — enforcement is by convention).
2. Implement the Stage 1 data structures: `SymmetricMatrix` (full n×n column-major dense matrix, lower triangle authoritative), `Inertia`, `Factors` (with `d_diag`/`d_subdiag` vectors for D blocks, `perm`/`perm_inv` for both permutation directions) (CSC sparse matrix is Phase 1b — do not implement in Session 1)
3. Implement a scalar (unblocked) dense LDLᵀ with Bunch-Kaufman pivoting
4. Write exact tests for the dense LDLᵀ using small matrices from the Bunch-Kaufman paper
5. Write the benchmark harness skeleton (`src/bin/bench.rs`). With no matrices present, it must print the header, attempt to load from `data/benchmark-config.toml` (absent on first run), print `0 matrices benchmarked`, and exit 0. Exact expected output:
   ```
   FERAL benchmark harness
   Loading matrices from data/benchmark-config.toml ... not found
   0 matrices benchmarked
   ```
   This confirms the binary compiles and the harness control flow works before any data exists.

6. Wire `src/lib.rs` reexports. The lib.rs file should contain only `pub use` declarations — no implementation. Exact structure:
   ```rust
   pub mod dense;
   pub mod inertia;
   pub mod error;

   // Flat public API re-exported at crate root:
   pub use dense::factor::{factor, Factors, BunchKaufmanParams, ZeroPivotAction};
   pub use dense::solve::{solve, solve_refined};
   pub use dense::matrix::SymmetricMatrix;
   pub use inertia::Inertia;
   pub use error::FeralError;
   ```
   This matches the file layout: `factor()`, `Factors`, params types live in `dense/factor.rs`; `solve()` and `solve_refined()` live in `dense/solve.rs` (the "full solve sequence" file per Section 11). This lets callers write `use feral::{factor, solve, SymmetricMatrix, ...}` without knowing the internal module structure.

7. Write the context assembly script
8. Write the first session checkpoint

This establishes the foundation: you can factorize dense matrices correctly, the testing and benchmark infrastructure exists, and the session protocol is operational.

### 13.3 First Research Note

Before implementing the dense LDLᵀ, the first research note (`dev/research/dense-ldlt.md`) should cover:

- Bunch-Kaufman 1977 Theorem 3 (the pivot selection algorithm)
- The LAPACK 3-way extension: the third branch that allows accepting the column-max row `r` as a 1×1 pivot when `|A[r,r]| >= α * γ_r` (from faer's `PartialDiag` mode)
- The threshold parameter α = (1 + √17) / 8 and how it differs from the threshold pivoting parameter u used by SSIDS (u=0.01) and MUMPS (CNTL(1)=0.01)
- Why 1×1 and 2×2 blocks are necessary and sufficient (existence theorem)
- Inertia preservation via Sylvester's Law of Inertia
- 2×2 block inertia counting: eigenvalue analysis via determinant sign + diagonal sign (not naive diagonal sign counting)
- What faer does for Bunch-Kaufman (code inspection: `PartialDiag` mode, full symmetric row search via `offdiag_argmax`, fused update+argmax)
- How SSIDS/MUMPS use threshold pivoting instead of BK, and the planned transition (BK → TPP → APP)
- Numerical guard for exact zero pivots vs near-zero pivots, and the architectural decision that perturbation lives in the IPM layer (not the solver)
- faer's 2×2 D-block solve normalization (scale by 1/b for better conditioning)
- How the scalar version extends to blocked factorization (W-panel technique from faer, then APP from SSIDS)

### 13.4 Benchmark Matrix Bootstrap

**Step 1: Collect KKT matrices from ripopt CUTEst runs.**

ripopt has built-in dump support via `SolverOptions::kkt_dump_dir`. The collection script lives in the **ripopt repository** (`cutest_suite/collect_kkt.rs`) — it does not belong in FERAL because it depends on ripopt (EPL-2.0) and uses CUTEst infrastructure. FERAL only needs the output files.

To regenerate the dataset, run the collection script from the ripopt repo and copy results to `data/matrices/kkt/`:

```bash
# In ../ripopt:
cargo run --bin collect_kkt --release --features cutest -- \
    --output /path/to/feral/data/matrices/kkt/
```

Each problem produces one `.mtx` + `.json` pair per IPM iteration.

This produces ~5,000–15,000 matrices covering the full trajectory of each solve. The ripopt failures (currently ~158 problems) are the most valuable — those are the hard systems FERAL must handle.

**Step 2: Fetch SuiteSparse symmetric indefinite matrices.**

```bash
bash scripts/fetch-suitesparse.sh
```

This downloads a curated set of symmetric indefinite matrices from [sparse.tamu.edu](https://sparse.tamu.edu) in Matrix Market format to `data/matrices/suitesparse/`. The script filters for `symmetric` + `real` + `indefinite` with `n < 50000` for Phase 1.

**Step 3: Constructed stress tests** are added inline as each feature is implemented — see Section 5.1, Step 4.

**File format contract** (both tiers use the same format so the benchmark harness has one reader):
- Matrix: `%%MatrixMarket matrix coordinate real symmetric`, lower triangle, 1-indexed
- Sidecar JSON (KKT tier only): `n`, `m`, `rhs`, `inertia {positive, negative, zero}`
- SuiteSparse tier: sidecar contains `n`, `m`, and expected inertia from a reference MUMPS run

---

## 14. Phase Roadmap

### Phase 1a: Correct Dense Solver

- Dense LDLᵀ with full Bunch-Kaufman pivoting (LAPACK-style 3-way decision, full symmetric row search)
- `BunchKaufmanParams` struct (no trait); `ZeroPivotAction` with `ForceAccept` and `Fail` (no `Perturb` — perturbation is external)
- `SymmetricMatrix` in full n×n column-major storage (not packed)
- `Factors` with `d_diag`/`d_subdiag` vectors (not `DBlock` enum) and both `perm`/`perm_inv`
- `factor(matrix, params) -> (Factors, Inertia)` and `solve(factors, rhs)` functions
- Inertia tracking: exact counts via eigenvalue signs of each D block (1×1: sign of scalar; 2×2: eigenvalue signs from det and sign of a — see Section 2.4 for the complete algorithm)
- Iterative Knight-Ruiz equilibration (Section 2.8)
- 2×2 D-block solve using faer's normalized formulation (Section 2.9)
- Forward/backward substitution with permutation
- Benchmark harness wired to dense matrices only
- **Exit criterion:** 100% correct inertia and solution on (a) all exact test matrices hand-verified from Bunch & Kaufman 1977 — these exist before any data collection — and (b) all KKT matrices collected via `collect_kkt` from the ripopt repo. Part (b) requires running `collect_kkt` first; if the dataset is not yet available, Phase 1a exits on part (a) alone and part (b) is validated at the start of Phase 1b. No timing requirement.

### Phase 1b: Correct Sparse Solver (Multifrontal, Single Thread)

- CSC sparse matrix infrastructure
- AMD ordering — produces a **permutation vector** `perm: Vec<usize>` (not an elimination tree). Note: AMD is the weakest ordering for KKT systems; METIS is deferred to Phase 2 but should be prioritized
- Elimination tree construction from the permuted sparsity pattern of `P·A·Pᵀ` — separate step after AMD
- Postordering of the elimination tree (required for LIFO property of ContribPool)
- Column counts / fill estimation from the postordered tree (Liu's algorithm; reference: CHOLMOD `cholmod_rowcolcounts.c`)
- Symbolic factorization: supernode detection (nemin-based amalgamation, Section 2.6), `MemoryPlan` production including `amap` precomputation
- `FactorBump` (append-only bump allocator) + `ContribPool` (LIFO stack for serial) allocated from `MemoryPlan` before numeric work begins
- Numeric factorization: postorder traversal, split assembly (pre/post — Section 2.2), dense kernel embedded
- Frontal matrix layout: **column-major, full symmetric (not packed)** — required for efficient Schur complement update
- `DenseKernel` trait designed at this boundary (see Section 2.5) with human review (Section 5.4 trigger #9)
- No delayed pivoting initially — use `ZeroPivotAction::ForceAccept` as the Phase 1b default; this correctly reports inertia (including zeros for rank-deficient KKT matrices) and flags the factorization for iterative refinement. Delayed pivoting — which eliminates the need for this compromise — is added in Phase 2
- **Phase 1b solve convention:** Because `ForceAccept` is the default, `factors.needs_refinement` will be true for most KKT matrices in Phase 1b. Use `solve_refined()` for all solves in Phase 1b — the refinement loop exits immediately (0 or 1 steps) for well-conditioned matrices, so the overhead is negligible. Phase 2 (delayed pivoting) reduces the frequency of `ForceAccept`, making plain `solve()` the common path again
- `increase_quality() -> bool` interface (Section 2.12)
- Full benchmark harness with KKT and SuiteSparse matrices, including trajectory-aware testing
- **Exit criterion:** 100% correct inertia + solution on the collected KKT benchmark set. Preferred: run `collect_kkt` from the ripopt repo before this phase — those are the target matrices. Fallback (if CUTEst infrastructure is unavailable): 100% correct inertia + solution on all Tier 2 SuiteSparse matrices collected via `scripts/fetch-suitesparse.sh`. At least one of these two datasets must be available before Phase 1b can exit. No timing requirement.

### Phase 2: Optimized and Parallel

- Threshold partial pivoting (TPP) kernel for multifrontal (u=0.01, matching SSIDS/MUMPS)
- Delayed pivoting mechanism (reject columns below threshold, pass to parent — Section 2.3)
- A posteriori pivoting (APP) blocked kernel (SSIDS model — factor blocks without pivoting, check threshold after, rollback on failure)
- Blocked dense LDLᵀ with W-panel and cache-aware panel factorization (faer's approach, block_size=64)
- SIMD micro-kernel for Schur complement update (fused update+argmax from faer)
- Shared-memory parallelism on assembly tree (Rayon); ContribPool transitions from LIFO stack to buddy allocator
- MC64 matching-based scaling (Section 2.8)
- METIS ordering option (priority for KKT systems where AMD underperforms)
- LDLT-aware ordering preprocessing (MUMPS ICNTL(12)-like compressed graph ordering for KKT structure)
- Optimized memory allocation and fill prediction
- **Exit criterion:** Within 2× of MUMPS on small-frontal KKT set; within 3× on medium set

### Phase 3: POUNCE Integration

- Replace MUMPS in ripopt with FERAL
- Implement `PDPerturbationHandler` equivalent in POUNCE (Section 2.12): escalation heuristic, degeneracy detection, mu-dependent delta_c formula
- Implement outer iterative refinement on full 8-component primal-dual system (Section 2.10)
- Adaptive pivot strategy driven by barrier parameter via `increase_quality()` / quality reduction
- Cross-solver KKT matrix harvesting from Ipopt for benchmark diversity
- CUTEst full-suite benchmarking
- **Exit criterion:** No regression in CUTEst solve rate; ideally unique solves

### Phase 4: Full Scale

- Distributed MPI multifrontal
- GPU offload for large frontal matrices (threshold ~128×128)
- Scaling frontier benchmarks (SuiteSparse n > 100k)
- **Exit criterion:** Competitive on large-scale benchmarks

---

*This document is the authoritative reference for the FERAL project. It lives in `dev/FERAL-PROJECT-SPEC.md` and is updated only by explicit human decision.*
