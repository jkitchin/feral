# Sparse Multifrontal LDLᵀ — Research Note

## Feature Goal

Build a single-threaded multifrontal sparse symmetric indefinite solver
(Phase 1b) that wraps the proven dense BK kernel from Phase 1a. The sparse
solver must produce exact inertia and correct solutions on the 153k collected
KKT matrices. No timing requirement.

## Canonical References

### Multifrontal Method
- **Duff & Reid (1983)** "The Multifrontal Solution of Indefinite Sparse
  Symmetric Linear Equations." ACM TOMS 9(3):302–325.
  - Introduces frontal matrices, assembly (elimination) tree, extend-add.
  - Key idea: factor each frontal independently using dense methods, pass
    Schur complement (contribution block) to parent via extend-add.
  - Indefinite case requires pivoting within each frontal — our dense BK
    kernel handles this.

### AMD Ordering
- **Amestoy, Davis & Duff (1996)** "An Approximate Minimum Degree Ordering
  Algorithm." SIAM J. Matrix Anal. Appl. 17(4):886–905.
- **Amestoy, Davis & Duff (2004)** "Algorithm 837: AMD." ACM TOMS 30(3):381–388.
  - Approximate degree via element absorption and aggressive absorption.
  - Output: permutation vector P such that P·A·Pᵀ has reduced fill.
  - Complexity: O(nnz) in practice.
  - Reference implementation: SuiteSparse `amd_2.c` (BSD-3).
  - Note: AMD is weak for KKT saddle-point structure. METIS deferred to Phase 2.

### Elimination Trees
- **George & Liu (1981)** *Computer Solution of Large Sparse Positive Definite
  Systems.* Chapters 4–6.
  - Elimination tree: parent[j] = min { i > j : L(i,j) ≠ 0 }.
  - Computed from the sparsity pattern of P·A·Pᵀ (upper triangle) in O(nnz).
  - Algorithm: for each column j, walk up the tree from each row index i > j
    in column j using path compression (union-find).
  - Postordering: DFS traversal ordering children by subtree weight (smallest
    first) to minimize peak memory.

### Column Counts / Fill Estimation
- **Liu (1990)** "The Role of Elimination Trees in Sparse Factorization."
  SIAM J. Matrix Anal. Appl. 11(1):134–172.
- Reference implementation: CHOLMOD `cholmod_rowcolcounts.c` (Apache 2.0).
  - Computes the number of nonzeros in each column of L using the elimination
    tree and the row subtree concept.
  - Output feeds into MemoryPlan for preallocation.

### Supernode Detection & Amalgamation
- **Davis & Hager (2009)** "Dynamic Supernodes in Sparse Cholesky." ACM TOMS 35(4).
- SSIDS `core_analyse.f90:806–822` merge rule:
  1. Trivial chain: parent has 1 elimination column AND parent col count =
     child col count − 1.
  2. Size-based: both parent and child have < nemin eliminated columns.
  - nemin default: 32 (SSIDS), 5 (MUMPS).
  - Amalgamation introduces explicit zeros tracked for memory accounting.

### SSIDS Architecture
- **Hogg, Ovtchinnikov & Scott (2016)** "A Sparse Symmetric Indefinite Direct
  Solver for GPU Architectures." ACM TOMS 42(1).
  - Primary architecture reference for FERAL's sparse engine.
  - BSD-3 licensed SPRAL implementation.

## Code Inspection Findings

### Assembly Split (SSIDS `assemble.hxx`)
Assembly is split into pre-factorization and post-factorization phases:
- **Pre:** Scatter original matrix entries via precomputed amap into frontal.
  Assemble child contributions into the fully-summed (factor) region.
- **Post:** Assemble child contributions into the contribution block region.
  Free child contribution blocks from ContribPool.
- Rationale: contribution block is the Schur complement, which only exists
  after the dense kernel runs.

### FactorBump vs Pre-assigned Offsets
SSIDS uses `AppendAlloc` (page-based bump allocator) because delayed pivots
change node dimensions unpredictably. Pre-assigned offsets would waste memory
or need overflow handling. Phase 1b has no delayed pivots, but using a bump
allocator now avoids a rewrite in Phase 2.

### ContribPool LIFO Property
In serial postorder traversal, the LIFO property holds naturally: a child's
contribution is always consumed by its parent before any unrelated node needs
the pool space. Contribution block sizes are symbolic invariants — determined
by (nrow − ncol), unaffected by delayed pivots.

### Frontal Matrix Layout
Column-major, full symmetric (not packed). Required for efficient Schur
complement update via GEMM-like rank-k update. The frontal has nrow rows
and ncol eliminated columns; the contribution block is (nrow−ncol)².

### Assembly Map (amap)
Precomputed during symbolic factorization, one per supernode. Maps
(source_index_in_csc_values, dest_index_in_frontal). Eliminates search
during numeric phase. Additionally, a dense lookup vector of size n is
used at runtime for child-to-parent extend-add mapping.

## Phase 1b Specific Constraints

1. **No delayed pivoting.** Use ZeroPivotAction::ForceAccept. Correctly
   reports inertia (including zeros). Flags factorization for refinement.
2. **solve_refined() for all solves.** Refinement exits in 0–1 steps for
   well-conditioned matrices. Negligible overhead.
3. **No DenseKernel trait.** Embed the dense BK kernel directly. Trait
   extraction requires human review (Phase 2).
4. **No PivotStrategy trait.** Single BK strategy hardcoded. Phase 2 adds
   TPP/APP with threshold escalation.
5. **Serial only.** ContribPool is a LIFO stack. Rayon parallelism is Phase 2.

## Chosen Approach

### Pipeline
```
CSC input → AMD ordering → elimination tree → postorder → column counts
→ supernode detection → MemoryPlan → numeric factorization → solve
```

### Implementation Order (each step independently testable)

1. **CSC sparse matrix** — storage format, construction, validation,
   symmetric expansion (lower → full pattern), matrix-vector product.

2. **AMD ordering** — implement from Amestoy/Davis/Duff 1996 algorithm
   description. Input: symmetric CSC. Output: permutation vector.
   Test: compare fill-in against known orderings for small matrices.

3. **Elimination tree** — construct from permuted sparsity pattern.
   Algorithm: union-find with path compression.
   Test: verify parent pointers against hand-computed trees.

4. **Postordering** — DFS on elimination tree, children ordered by
   subtree weight (smallest first for peak memory minimization).
   Test: verify postorder is a valid topological ordering.

5. **Column counts** — Liu's algorithm using the elimination tree.
   Test: compare predicted fill against actual factorization fill on
   small matrices.

6. **Supernode detection** — fundamental supernodes from column count
   patterns, then nemin-based amalgamation.
   Test: verify supernode structure on known matrices.

7. **Symbolic factorization** — combine supernodes, column counts,
   elimination tree into MemoryPlan. Compute amaps.
   Test: verify MemoryPlan sizes are correct upper bounds.

8. **Numeric factorization** — postorder traversal, assembly, dense BK
   kernel per front, contribution block management.
   Test: factor sparse KKT matrices, compare inertia and solution
   against dense factorization result.

9. **Sparse solve** — supernodal forward/backward substitution with
   permutation and equilibration.
   Test: ||Ax − b||/||b|| on all collected KKT matrices.

### Key Data Structures

```
CscMatrix        { n, col_ptr, row_idx, values }
EliminationTree  { parent: Vec<Option<usize>>, children: Vec<Vec<usize>> }
Supernode        { ncol, nrow, col_indices, row_indices, children }
MemoryPlan       { factor_nnz_estimate, factor_slack, contrib_sizes, peak_contrib_bytes, amaps }
FactorBump       { pages: Vec<Vec<f64>>, current_offset }
ContribPool      { arena: Vec<f64>, stack_pointer }
FrontalMatrix    { data: &mut [f64], nrow, ncol }  // view into FactorBump
```

## Disagreements / Alternatives

- **AMD vs METIS:** AMD is simpler but produces worse orderings for KKT
  matrices (saddle-point structure). The spec mandates AMD first, METIS
  in Phase 2. This is correct — get the pipeline working with AMD, then
  swap in METIS as a drop-in ordering improvement.

- **nemin=32 vs nemin=5:** SSIDS uses 32, MUMPS uses 5. Larger nemin
  creates fewer, larger supernodes (better BLAS-3 efficiency) but more
  fill. For Phase 1b with no BLAS-3 optimization, nemin=32 is fine —
  the overhead is memory, not compute. Can tune later.

- **Full symmetric vs lower-triangle frontal storage:** Spec mandates
  full symmetric for efficient Schur complement update. This doubles
  frontal memory but simplifies the GEMM call and is standard practice
  (SSIDS, CHOLMOD both use full symmetric frontals).
