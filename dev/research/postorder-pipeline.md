# Postorder in the Symbolic Pipeline — Research Note

## Problem Statement

The current `symbolic_factorize` (src/symbolic/mod.rs:62) computes the
elimination tree of the AMD-permuted pattern and immediately runs supernode
detection on it, **without applying postorder to the etree first**. As a
result, supernode amalgamation merges columns whose indices are not
consecutive in the column numbering. Every consumer of `Supernode.first_col`
and `Supernode.ncol` then assumes a contiguous range `first_col..first_col+ncol`
of eliminated columns, which is wrong for the merged set.

This is the root cause of the sparse path's wrong inertia and catastrophic
residuals (e.g. (50,1,0) instead of (35,16,0) and a 2.61e21 residual on
MGH10S_0000) on a measurable fraction of the 153k KKT corpus.

The original Phase 1b plan (`dev/plans/phase-1b-sparse-solver.md` Step 4,
line 96) explicitly called for *"Composing AMD perm with postorder perm
gives the final ordering"*. The implementation regressed against the plan:
`src/ordering/postorder.rs` exists with a working `postorder()` function
and tests, but is **never called** from the symbolic pipeline. It is an
orphan module.

## Why Postorder Is Required (Intuition)

In a postordered elimination tree the columns of any subtree form a
contiguous range, and a parent's column index always immediately follows
the columns of its last-visited child. This single property is what makes
all supernode bookkeeping work with `(first_col, ncol)` integer ranges:

- Fundamental supernode detection relies on `parent[j-1] == j` — this is
  the postorder rule that says "j's only child in the subtree just visited
  is j-1".
- Amalgamation merges a child supernode with its parent, and the merged
  result is still a contiguous range exactly because the child's columns
  are immediately followed by the parent's.
- Frontal row indices `first_col..first_col+nrow` are consecutive only if
  the supernode columns are consecutive.

When the etree is *not* postordered, none of these hold. AMD orderings
sometimes coincidentally produce a postordered tree (e.g. tridiagonal,
banded), which is why the bug stayed hidden through the existing test
suite — but for KKT matrices with bordered or saddle-point structure
the AMD output and the postorder are typically different.

### Worked example: MGH10S_0000

51×51 KKT, AMD returns (near-)identity. The etree has roots
`{0, 1, 2, 35, 36, …, 50}` and the children of constraint row 35+i are
`{3+i, 19+i}` (each variable column couples to exactly one constraint
row). In the natural ordering, the children of constraint 35 are columns
{3, 19} — not contiguous, separated by 15 other columns.

Without postorder, fundamental supernode detection emits 51 singleton
supernodes (no `parent[j-1] == j` matches). Amalgamation then merges each
variable column into its parent constraint snode by size-based merging
(`ncol < nemin = 32`), updating `first_col = min(child, parent)`. The
result is 19 final supernodes whose `first_col` values are `0, 1, 2, 3,
4, …, 18` — but the merged column SETS are
`{0}, {1}, {2}, {3,19,35}, {4,20,36}, …, {18,34,50}`. The supernode
struct stores only `(first_col=3, ncol=3)`, which downstream code
interprets as the contiguous range `{3, 4, 5}`. So the frontal that was
meant to eliminate {3, 19, 35} ends up eliminating {3, 4, 5}, and the
negative constraint diagonals at cols 35..50 are mostly never reached.

A postorder permutation of this etree would relabel the columns so that
the subtree rooted at 35 occupies positions {3, 4, 5}, the subtree rooted
at 36 occupies {6, 7, 8}, and so on. After this relabeling, the merged
supernode for the first subtree literally is the column range
[3, 6) — `(first_col=3, ncol=3)` then correctly identifies the same set
of columns `{3, 4, 5}` in the new numbering, which corresponds to the
original {3, 19, 35} via the composed permutation.

## Canonical References

### CHOLMOD
- **Davis (2006)** *Direct Methods for Sparse Linear Systems.* Chapter 4
  ("Sparse Cholesky factorization"), §4.6 ("Postordering the elimination
  tree").
  - Establishes the standard pipeline: order → etree → **postorder** →
    column counts → supernodes.
  - The postorder is composed with the fill-reducing ordering before
    column counts are computed; col counts are then computed on the
    final ordering.
- **CHOLMOD source `cholmod_postorder.c`** (BSD-3, SuiteSparse).
  - Implements the postorder of an etree with subtree-size weighting
    for memory minimization. Returns `Post[k] = node visited at step k`.
- **CHOLMOD `cholmod_analyze.c`** function `cholmod_analyze_p` shows the
  composition:
  ```
  Perm = AMD(A)
  AP   = A(Perm, Perm)
  Parent = etree(AP)
  Post = postorder(Parent)
  Perm = Perm[Post]      // compose
  AP   = A(Perm, Perm)
  Parent = etree(AP)     // re-etree on the composed permutation
  ColCount = rowcolcounts(AP, Parent, Post)
  ```
  Note that the etree is rebuilt on the composed-permutation pattern.
  This is necessary because the postorder reorders the columns, and the
  parent pointers must reference indices in the new numbering.

### SSIDS
- **Hogg, Reid & Scott (2010)** "Design of a Multicore Sparse Cholesky
  Factorization Using DAGs." SIAM J. Sci. Comput. 32(6):3627–3649.
- **SSIDS source `src/ssids/anal.F90`** (BSD-3, SPRAL).
  - The analysis phase explicitly applies postorder after the AMD/MeTiS
    ordering. SSIDS calls its postorder routine `core_anal_postord`
    (in `src/core_analyse.f90`).
  - SSIDS confirms the same merge rules used in feral's
    `find_supernodes` (fundamental + nemin amalgamation) require a
    postordered tree as a precondition.

### George & Liu Foundation
- **George & Liu (1981)** *Computer Solution of Large Sparse Positive
  Definite Systems*, §5.4 ("The compressed elimination tree").
- **Liu (1990)** "The Role of Elimination Trees in Sparse
  Factorization." SIMAX 11(1):134–172.
  - Theorem 4.6 in Liu's paper: postorder is the canonical traversal
    that enables LIFO stack-based assembly. This is the original
    justification for the LIFO ContribPool design that feral's
    Phase 1b plan calls out.

## Algorithm (CHOLMOD-style composition)

Given input matrix `A` and AMD permutation `amd_perm` (new-to-old):

1. **Build etree on the AMD-permuted pattern.**
   `etree₁ = EliminationTree::from_pattern(permute_pattern(A, amd_perm))`

2. **Compute postorder of the etree.**
   `(post, post_inv) = postorder(&etree₁)`
   - `post[k]` = node visited at step k in the postorder
   - `post[k]` is in the *AMD-numbering* of columns

3. **Compose permutations.**
   The AMD perm maps new→old: `amd_perm[k]` = original column at AMD
   position k. The postorder maps new→AMD-numbering: `post[k]` = AMD
   position visited at postorder step k. The composed permutation
   `final_perm` maps postorder position → original column:

   ```
   final_perm[k] = amd_perm[post[k]]
   ```

   Equivalently, `final_perm = amd_perm ∘ post` as vector composition.
   Compute `final_perm_inv` from `final_perm`.

4. **Re-permute the matrix and rebuild the etree** in the final
   numbering. The new etree's parent pointers reference indices in the
   final (postordered) numbering, where children of any subtree are
   contiguous.

   ```
   permuted_pattern = permute_pattern(A.symmetric_pattern(), final_perm)
   etree            = EliminationTree::from_pattern(&permuted_pattern)
   ```

5. **Compute column counts** on the final-numbered pattern + etree.

6. **Supernode detection and amalgamation** on the postordered etree.
   Now the merged supernodes have contiguous columns by construction.

7. The rest of the pipeline (numeric factorization, solve) is unchanged
   because it already assumes contiguous-column supernodes.

## Correctness Properties

After the fix, the following invariants must hold:

- **Postorder topological property:** for every column j with parent
  `p`, `inv_post[j] < inv_post[p]` (children appear before parents in
  postorder). Already verified in
  `tests/postorder.rs::test_postorder_valid_topological_order`.

- **Subtree contiguity:** for any node `v` in the postordered etree,
  the set of descendants of `v` in postorder forms a contiguous range
  `[start_v..start_v + size_v)`. This is what makes
  `(first_col, ncol)` a sound representation of a supernode's
  eliminated columns.

- **Final perm validity:** `final_perm` is a permutation of `0..n`,
  and `final_perm_inv[final_perm[k]] == k`.

- **Composition correctness:** for any column index `j` in the
  original numbering, applying `final_perm_inv[j]` to find its
  position in the final ordering and then reading
  `final_perm[final_perm_inv[j]]` returns `j`.

- **Sparse inertia matches dense inertia** on every test matrix in
  `tests/dense_ldlt.rs`, `tests/property_tests.rs`,
  `tests/kkt_hardening.rs`, plus the MGH10S triage example. This is
  the smoke test that the fix actually closes the bug.

## Failure Modes Considered

- **Postorder of a forest** — the elimination tree may be a forest
  with multiple roots (e.g. block-diagonal matrices, MGH10S, any
  matrix with disconnected components). The existing
  `postorder()` function in `src/ordering/postorder.rs` already
  handles this: it iterates `etree.roots()` and visits each root's
  subtree. No change needed there.

- **AMD perm and postorder both produce the same result** — for
  tridiagonal, banded, and other "naturally ordered" matrices, AMD
  may already return a perm whose etree is postordered. In this case
  postorder is identity and the composition `amd_perm ∘ identity =
  amd_perm`. The existing tests still pass — this is why the bug was
  invisible until KKT matrices were exercised.

- **Postorder reorders within a fundamental supernode** — by
  definition, the columns of a fundamental supernode form a path in
  the etree (`parent[j-1] == j`), so postorder visits them in
  ascending order. The internal structure of fundamental supernodes
  is preserved.

- **Effect on AMD's fill prediction** — AMD minimizes fill on the
  pattern of `P·A·Pᵀ`. Postorder is a *symmetric* permutation
  (relabeling columns and rows by the same permutation), which does
  not change the fill count. So the fill-reducing benefit of AMD is
  preserved.

## Testing Strategy

1. **Unit test** — extend `tests` in `src/symbolic/mod.rs` to verify
   that for each matrix in the existing fixtures, every supernode
   `s` satisfies: `for j in s.first_col..s.first_col + s.ncol, the
   etree subtree containing j is contained in this range`.

2. **Regression test** — load MGH10S_0000 (or a smaller hand-built
   bordered KKT), factor sparse, assert inertia matches the sidecar.
   This must FAIL before the fix and PASS after. Add to
   `tests/sparse_kkt.rs` (new file).

3. **Cross-check against dense** — pick ~10 KKT matrices from `data/`
   covering different block structures (small bordered, large
   bordered, banded, dense block) and assert sparse inertia ==
   dense inertia. Add as a property test.

4. **Bench delta** — re-run the 153k bench. Sparse inertia and
   residual numbers should jump to (or very near) 100%. If a
   non-trivial gap remains it indicates a *second* bug worth
   investigating before claiming Phase 1b exit.

## Risks and Open Questions

- **Two etree builds** — the algorithm rebuilds the etree after
  composition (Step 4). For very large matrices this doubles the
  symbolic cost. Acceptable for Phase 1b (correctness > speed).
  Phase 2 can fold the second etree build into a single pass that
  reuses the first build.

- **`SymbolicFactorization.perm` semantics change** — currently
  `sym.perm` is the AMD perm. After the fix it becomes the composed
  perm. Any external consumer that compares `sym.perm` against AMD
  output needs to be updated. Searching the codebase: only
  `factorize.rs` and `solve.rs` use `sym.perm`, both as opaque
  new-to-old maps, so they should keep working.

- **`postorder()` API mismatch** — the current return type is
  `(Vec<usize>, Vec<usize>)` (postorder, inv). For the fix we
  need a permutation in the AMD-numbering applied to the etree.
  The existing function already does the right thing — the etree it
  takes is in AMD numbering, so the result is in AMD numbering. No
  API change required.

- **Children-of-supernode tracking** — `find_supernodes` builds
  `Supernode.children` from `snode_parent`. After postorder this
  should still be correct, since postorder preserves the tree
  structure (just relabels nodes). Verify in implementation.
