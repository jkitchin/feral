# Phase 2.5.1 — Liu's row-subtree column counts

**Scope:** replace the current O(n²) `column_counts` in
`src/symbolic/column_counts.rs` with the Gilbert–Ng–Peyton
`O(nnz(A) + n·α(n))` algorithm based on Liu's row-subtree
description. α is the inverse Ackermann function — effectively
linear for all practical n.

## Current implementation — what's there today

`src/symbolic/column_counts.rs:20` implements a direct elimination
simulation:

1. For each column j, build `col_rows[j] = {i > j : (i,j) ∈ A}`
   (sort + dedup).
2. Process columns j = 0..n−1. When j is eliminated, all rows in
   `col_rows[j]` become pairwise connected — concretely the code
   pushes every row r ∈ `col_rows[j][1:]` into
   `col_rows[min_row]` (then sort + dedup).
3. `counts[j] = 1 + |col_rows[j]|` (diagonal plus off-diagonals).

Correctness: this is the standard "eliminate, propagate to
parent" rule — the minimum remaining row is the parent in the
elimination tree, and fill from j is exactly `col_rows[j] \ {p}`
being added to column p. Matches Davis "Direct Methods" §4.1
(`cs_leaf` / `cs_ereach` model).

Cost: worst case `O(n·fill(L))` because every fill entry may
trigger a re-sort of `col_rows[parent]`. Contains
`col_rows[min_row].sort_unstable()` + `dedup()` inside the outer
loop, plus `col_rows[min_row].contains(&row)` — both linear in
`|col_rows[min_row]|`, summing to O(n²) worst case on dense
patterns. Confirmed by the module header comment in
`src/symbolic/column_counts.rs` and by the Phase 2 plan.

## Algorithm — Gilbert–Ng–Peyton via Liu's row subtrees

Canonical references:
- Gilbert, Ng, Peyton, "An efficient algorithm to compute row and
  column counts for sparse Cholesky factorization",
  *SIAM J. Matrix Anal. Appl.* 15(4):1075–1091, 1994.
- Liu, "The role of elimination trees in sparse factorization",
  *SIAM J. Matrix Anal. Appl.* 11(1):134–172, 1990.
- Davis, *Direct Methods for Sparse Linear Systems*, SIAM 2006,
  Chapter 4 (§4.4 "Row and column counts").
- The textbook reference C implementation is `cs_counts` in
  CSparse (BSD-licensed) — we will follow its control flow.

### Data structures (all length n, O(n) total memory)

- `parent[i]`: elimination-tree parent of node i (given). Already
  computed by `EliminationTree::from_pattern`.
- `first[i]`: index of the first descendant of i in a postorder
  traversal.  Equivalently, `first[i] = min(first[c] for c child
  of i)`, with `first[leaf] = leaf_postorder_index`.
- `level[i]`: depth of i in the etree (root = 0).
- `ancestor[i]`: disjoint-set-union ancestor pointer for path
  compression (initialized `ancestor[i] = i`).
- `prev_leaf[i]` (a.k.a. `maxfirst`): postorder index of the most
  recent leaf j in row i's subtree, or −1 if none seen yet.
- `delta[i]`: column-count delta; final `counts[i]` is the prefix
  sum of `delta` over the subtree rooted at i (specifically:
  `counts[i] = sum of delta[d] for d descendant of i, plus 1`).

### Algorithm sketch

```
for each i in 0..n:
    delta[i] = if is_leaf(i) then 1 else 0

# Walk A row-by-row (i = 0..n-1), for each row look at its
# entries (i, k) with k < i — these are the original pattern of
# row i. The row subtree is the union of tree paths from each
# k up to the highest common ancestor in the etree.
for i in 0..n:
    for each k in A[i,:] with k < i:
        if first[i] > prev_leaf[k]:    # k is a "leaf" of row i's subtree
            delta[i] += 1
            if prev_leaf[k] != -1:
                q = find(ancestor, prev_leaf[k])  # LCA with path compression
                delta[q] -= 1
            prev_leaf[k] = first[i]
    if parent[i] != -1:
        ancestor[i] = parent[i]       # union step for DSU

# Accumulate deltas up the etree to get column counts
for i in 0..n:
    if parent[i] != -1:
        delta[parent[i]] += delta[i]
counts[i] = delta[i] + 1   # +1 for diagonal
```

The `find` uses path compression for amortized α. Each `k` in
`A[i,:]` is visited once, so the outer work is O(nnz(A)); the
DSU `find`s over the etree amortize to `O(n·α)` by the standard
Tarjan analysis.

Final `counts[i]` is the number of nonzeros in column i of L
(including the diagonal), matching the current function's
return contract.

### First-descendant computation

`first` is a simple postorder-first traversal of the etree:

```
for i in postorder:
    if first[i] undefined:
        first[i] = i                     # leaf
    if parent[i] != -1 and first[parent[i]] undefined:
        first[parent[i]] = first[i]
```

O(n).

## Test plan

Phase 2.5.1 is a pure refactor at the API level — same inputs,
same output type, same semantics. Correctness proof obligation:
**bit-exact equality with the existing implementation across the
full KKT corpus.**

Test steps:

1. **Golden tests retained.** The existing 5 unit tests in
   `src/symbolic/column_counts.rs` (diagonal, tridiagonal, dense,
   arrow, block-diagonal) stay — they must pass bit-exact.
2. **Cross-check sweep.** Add a new test/example that, for every
   matrix in `data/matrices/kkt/`, computes `column_counts_fast`
   and `column_counts_slow` and asserts equality. This is the
   same pattern as `examples/triage_sparse_kernel_diff.rs`.
3. **Differential in symbolic_factorize.** `factor_nnz` from
   `total_factor_nnz(col_counts)` feeds into allocation sizing.
   Bit-exact equality at the count level guarantees identical
   downstream behaviour. Existing symbolic unit tests plus the
   full KKT bench at session close are the regression net.

Oracle: the current `column_counts` function itself, renamed
internally to `column_counts_reference` behind a `#[cfg(test)]`
module or a public API kept for bench-time comparison (TBD in
the plan).

## Integration points

Single production call site: `src/symbolic/mod.rs:334`

```rust
let col_counts = column_counts(&permuted_pattern, &etree);
let factor_nnz = total_factor_nnz(&col_counts);
```

Test call sites (keep using old name for minimum churn):
`src/symbolic/supernode.rs:299..418` (7 tests).

The cleanest drop-in is to replace the body of
`column_counts` with the Liu/GNP algorithm and add an
`#[cfg(test)] fn column_counts_reference(...)` behind the
existing body. No call-site changes.

## Risks and unknowns

1. **Is this actually the bottleneck?** The Phase 2 plan calls
   it "the highest-leverage Phase 2.5 item," but we haven't
   profiled `symbolic_factorize` explicitly. Sparse factor/MUMPS
   p90 = 1.67 (context.md) includes both symbolic + numeric;
   numeric dominates on large matrices. **Action:** before
   implementing, add a simple timing split in the bench harness
   (symbolic_us vs numeric_us) for one 10k-matrix subset. If
   symbolic is < 5% of factor time even on the largest corpus
   matrices, defer Phase 2.5.1 in favor of 2.5.2 (Rayon on
   assembly tree).
2. **Postorder assumption.** The GNP algorithm requires the
   elimination tree be walked in postorder, which our
   `EliminationTree` may or may not expose. If not, we add a
   `postorder()` method (O(n), straightforward).
3. **Off-by-one in `first`.** Implementations differ in whether
   `first` is indexed by node or by postorder number. CSparse
   uses postorder-number indexing; we'll follow that to avoid
   reinventing the control flow.
4. **No existing `find` / DSU.** We'll add a tiny inline
   path-compression `find` scoped to this module. No new
   dependency, no new module.

## Effort estimate

Consistent with the Phase 2 plan: **4–6 hours**, broken down:

- 30 min: add symbolic/numeric timing split to bench, confirm
  symbolic-time share on the large-n corpus subset. If < 5%,
  abort and move to 2.5.2.
- 60 min: implement `first` + postorder helpers in
  `EliminationTree` (or in the new module).
- 90 min: implement `column_counts_gnp` with inline DSU.
- 30 min: add corpus cross-check example.
- 60 min: full `cargo test` + bench run + validation report.

## Acceptance

**Hard:**
1. All 118 existing lib tests pass.
2. New corpus-sweep test/example reports **zero** per-matrix
   column-count differences across all 154588 KKT matrices.
3. `symbolic_factorize` throughput on the large-n subset
   (n > 500) improves by ≥ 3× (asymptotic gain from n²→n·α).

**Soft:**
4. Overall sparse factor/MUMPS p90 improves (any measurable
   improvement — may be tiny if symbolic isn't a bottleneck).
