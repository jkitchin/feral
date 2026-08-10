use super::elimination_tree::EliminationTree;

/// CSR-style child arena: node `j`'s children are `idx[ptr[j]..ptr[j + 1]]`.
///
/// Replaces [`EliminationTree::children`]'s `Vec<Vec<usize>>` inside the
/// postorder traversals (issue #128 item D). That allocated `n` separate
/// `Vec`s per call and each traversal then cloned-and-sorted a fresh `Vec`
/// per node on top — `2n+` allocations for a pass the pipeline can run twice
/// per factorization. Here the whole child structure is two `Vec`s, each
/// node's slice is ordered **once in place** before the walk, and the DFS
/// stack carries only `(node, cursor)`.
///
/// Ordering is identical to the old code by construction: the arena is built
/// by a counting sort over ascending `j`, so each parent's slice starts in
/// ascending child-index order — exactly what `children()` produced — and
/// the same sort routine is then applied to the same input sequence.
struct ChildArena {
    ptr: Vec<usize>,
    idx: Vec<usize>,
}

impl ChildArena {
    fn from_etree(etree: &EliminationTree) -> Self {
        let n = etree.n;
        // ptr[j + 1] accumulates the child count of j, then prefix-sums.
        let mut ptr = vec![0usize; n + 1];
        for j in 0..n {
            if let Some(p) = etree.parent[j] {
                ptr[p + 1] += 1;
            }
        }
        for j in 0..n {
            ptr[j + 1] += ptr[j];
        }
        let mut idx = vec![0usize; ptr[n]];
        // Per-parent write cursor. Scanning `j` upward makes each parent's
        // slice ascending, matching `EliminationTree::children`.
        let mut fill = ptr[..n].to_vec();
        for j in 0..n {
            if let Some(p) = etree.parent[j] {
                idx[fill[p]] = j;
                fill[p] += 1;
            }
        }
        ChildArena { ptr, idx }
    }

    #[inline]
    fn children(&self, j: usize) -> &[usize] {
        &self.idx[self.ptr[j]..self.ptr[j + 1]]
    }

    /// Apply `order_children` to every node's slice, once. The ordering
    /// rules are pure functions of `(slice, sizes, ...)` — they never depend
    /// on traversal state — so hoisting them out of the DFS is behavior
    /// preserving.
    fn order_all(&mut self, mut order_children: impl FnMut(&mut [usize])) {
        for j in 0..self.ptr.len() - 1 {
            let (lo, hi) = (self.ptr[j], self.ptr[j + 1]);
            if hi - lo > 1 {
                order_children(&mut self.idx[lo..hi]);
            }
        }
    }
}

/// Depth-first walk of the arena from `root`, calling `emit` on each node in
/// postorder. The stack holds `(node, cursor)` — no per-node allocation.
fn dfs_postorder(
    arena: &ChildArena,
    root: usize,
    stack: &mut Vec<(usize, usize)>,
    mut emit: impl FnMut(usize),
) {
    stack.clear();
    stack.push((root, 0));
    while let Some(&mut (node, ref mut cursor)) = stack.last_mut() {
        let kids = arena.children(node);
        if *cursor < kids.len() {
            let child = kids[*cursor];
            *cursor += 1;
            stack.push((child, 0));
        } else {
            emit(node);
            stack.pop();
        }
    }
}

#[cfg(test)]
thread_local! {
    /// S1 (dev/research/repo-review-2026-06-09.md) work counter: total
    /// number of child-list elements materialized+sorted across all
    /// per-node sorts in [`postorder`]. Linear in `n` for the fixed
    /// (sort-once-per-node) traversal; quadratic for the old
    /// sort-on-every-stack-visit version. Test-only; compiled out of
    /// production builds.
    static SORT_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Compute a postorder traversal of the elimination tree.
///
/// Returns `(postorder, inv_postorder)` where:
/// - `postorder[k]` = the node visited at position k (new-to-old)
/// - `inv_postorder[node]` = the position of node in the postorder (old-to-new)
///
/// Children are visited in order of ascending subtree size (smallest first)
/// to minimize peak memory usage in the ContribPool.
pub fn postorder(etree: &EliminationTree) -> (Vec<usize>, Vec<usize>) {
    let n = etree.n;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let sizes = etree.subtree_sizes();
    let roots = etree.roots();

    let mut order = Vec::with_capacity(n);

    // Each node's children are sorted exactly once, in place in the arena,
    // before the walk; the DFS stack then carries only `(node, cursor)`.
    //
    // Two earlier shapes are pinned against here. The original stored
    // `(node, child_idx)` and re-cloned and re-sorted `children[node]` on
    // every `stack.last_mut()` iteration; a node with `c` children sits on
    // top of the stack `c+1` times, so it paid `O(c²·log c)` — `O(n²·log n)`
    // on a star etree (the arrow/bordered-KKT shape AMD produces for a dense
    // trailing border). See S1, dev/research/repo-review-2026-06-09.md. The
    // fix for that carried a freshly sorted `Vec` per stack entry, which is
    // linear in work but still allocates once per node; issue #128 item D
    // removes those allocations without changing the emitted order.
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut arena = ChildArena::from_etree(etree);
    arena.order_all(|kids| sort_children_by_size(kids, &sizes));

    // Process roots in ascending subtree size order
    let mut sorted_roots = roots;
    sorted_roots.sort_unstable_by_key(|&r| sizes[r]);

    for &root in &sorted_roots {
        dfs_postorder(&arena, root, &mut stack, |node| order.push(node));
    }

    // Compute inverse
    let mut inv = vec![0usize; n];
    for (k, &node) in order.iter().enumerate() {
        inv[node] = k;
    }

    (order, inv)
}

/// Phase 2.12 merge-biased postorder.
///
/// Like [`postorder`], but when descending into a parent's children
/// it partitions them into `bias[child] == false` (emit *first*) and
/// `bias[child] == true` (emit *last*). Within each partition,
/// children are still ordered by ascending subtree size (peak-memory
/// minimization, same as [`postorder`]).
///
/// Effect: children whose `bias[child]` is `true` have their subtrees
/// emitted adjacent to (immediately before) the parent's column in
/// the resulting numbering. When the bias matches the SSIDS desired
/// merges (per [`crate::symbolic::supernode::predict_merges`]), the
/// returned ordering makes every desired merge adjacent in the
/// column numbering, so the standard adjacency check in
/// `find_supernodes` succeeds for it.
///
/// Invariant: `biased_postorder(etree, &vec![false; n]) ==
/// postorder(etree)`.
pub fn biased_postorder(etree: &EliminationTree, bias: &[bool]) -> (Vec<usize>, Vec<usize>) {
    let n = etree.n;
    debug_assert_eq!(
        bias.len(),
        n,
        "biased_postorder bias length must equal etree.n"
    );
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let sizes = etree.subtree_sizes();
    let roots = etree.roots();

    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut scratch: Vec<usize> = Vec::new();
    let mut arena = ChildArena::from_etree(etree);
    arena.order_all(|kids| merge_bias_partition(kids, &sizes, bias, &mut scratch));

    // Roots are not biased (no parent to be adjacent to). Use the
    // unbiased subtree-size order.
    let mut sorted_roots = roots;
    sorted_roots.sort_unstable_by_key(|&r| sizes[r]);

    for &root in &sorted_roots {
        dfs_postorder(&arena, root, &mut stack, |node| order.push(node));
    }

    let mut inv = vec![0usize; n];
    for (k, &node) in order.iter().enumerate() {
        inv[node] = k;
    }
    (order, inv)
}

/// Schur-constrained postorder of an elimination tree (F3.2a).
///
/// Given an `is_schur` indicator (length `etree.n`), produce a postorder
/// such that **every Schur node appears at its etree-index position** in
/// the output. That is, `post[j] == j` for every `j` where
/// `is_schur[j] == true`, provided the Schur subset is closed under the
/// `parent` relation in the etree (the "top-forest" invariant). When that
/// invariant holds, the constraint is satisfiable: non-Schur descendants
/// of Schur nodes are emitted first; Schur nodes are then emitted in
/// strict ascending etree-index order, which equals their input order
/// because [`super::schur::compute_schur_aware_perm`] places Schur
/// indices at positions `[n - n_schur, n)` in the supplied order.
///
/// **Caller's responsibility.** The function does not validate the
/// top-forest invariant (no Schur node has a non-Schur parent). Callers
/// inside `symbolic_factorize_with_schur` get this for free because
/// `compute_schur_aware_perm` puts Schur indices at the highest
/// positions, and `parent[j] > j` for every node in the etree of the
/// permuted pattern.
///
/// **Children ordering rule** (applied at every parent and at the root
/// list): non-Schur children first, sorted by ascending subtree size
/// (peak-memory minimization, identical to [`postorder`]). Schur
/// children second, sorted by ascending etree index (preserves the
/// user's input order across the Schur tail).
///
/// Invariant: `schur_constrained_postorder(etree, &vec![false; n]) ==
/// postorder(etree)`.
pub fn schur_constrained_postorder(
    etree: &EliminationTree,
    is_schur: &[bool],
) -> (Vec<usize>, Vec<usize>) {
    let n = etree.n;
    debug_assert_eq!(
        is_schur.len(),
        n,
        "schur_constrained_postorder is_schur length must equal etree.n"
    );
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let sizes = etree.subtree_sizes();
    let roots = etree.roots();

    let mut order = Vec::with_capacity(n);

    // Phase 1: emit non-Schur nodes only. Walk the entire etree in DFS
    // postorder but only push non-Schur nodes onto `order`. A Schur node
    // is "transparent" — we recurse through it (so its non-Schur
    // descendants are reached) but we skip emitting it. After phase 1,
    // every non-Schur node sits at some position in `[0, n_f)` in a
    // valid postorder of the non-Schur subgraph (where each non-Schur's
    // sub-parent is its nearest non-Schur ancestor, or None).
    let mut scratch: Vec<usize> = Vec::new();
    let mut sorted_roots = roots;
    schur_partition_children(&mut sorted_roots, &sizes, is_schur, &mut scratch);
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut arena = ChildArena::from_etree(etree);
    arena.order_all(|kids| schur_partition_children(kids, &sizes, is_schur, &mut scratch));

    for &root in sorted_roots.iter() {
        dfs_postorder(&arena, root, &mut stack, |node| {
            if !is_schur[node] {
                order.push(node);
            }
        });
    }

    // Phase 2: emit Schur nodes in ascending etree-index order. The
    // contract from `compute_schur_aware_perm` places Schur indices at
    // `[n - n_schur, n)`, so iterating `k` from `0..n` and pushing when
    // `is_schur[k]` yields exactly the identity tail: `post[n_f + i] ==
    // n_f + i` for every Schur position. A DFS over the Schur subtree
    // would emit them in tree-walk order — correct only when the Schur
    // etree is a single ascending chain. With a forest of Schur roots
    // (e.g. KKT matrices like ACOPP30 where Schur cols 158, 159, 160,
    // 161, 167, 168 are roots while 157 is parented under chain root
    // 208), DFS reorders Schur indices and breaks the tail identity
    // that `symbolic_factorize_with_schur` relies on. Direct ascending
    // emission preserves the postorder validity (every Schur node's
    // Schur children have smaller etree index, so they emit earlier;
    // non-Schur descendants emitted in phase 1 already sit at positions
    // `< n_f`).
    for (k, &flag) in is_schur.iter().enumerate() {
        if flag {
            order.push(k);
        }
    }

    let mut inv = vec![0usize; n];
    for (k, &node) in order.iter().enumerate() {
        inv[node] = k;
    }
    (order, inv)
}

/// Partition children for the Schur-constrained postorder.
///
/// Non-Schur children first, ascending by subtree size. Schur children
/// second, ascending by etree index (preserves input order across the
/// Schur tail).
/// In place; `scratch` is a reusable buffer so the partition costs no
/// allocation per node (issue #128 item D).
fn schur_partition_children(
    children: &mut [usize],
    sizes: &[usize],
    is_schur: &[bool],
    scratch: &mut Vec<usize>,
) {
    let split = stable_partition(children, scratch, |c| !is_schur[c]);
    children[..split].sort_unstable_by_key(|&c| sizes[c]);
    children[split..].sort_unstable();
}

/// Sort a node's children by ascending subtree size (smallest first), the
/// peak-memory-minimizing visit order used by [`postorder`]. Applied once
/// per node, in place in the [`ChildArena`] (see S1,
/// `dev/research/repo-review-2026-06-09.md`, and issue #128 item D).
fn sort_children_by_size(children: &mut [usize], sizes: &[usize]) {
    #[cfg(test)]
    SORT_WORK.with(|w| w.set(w.get() + children.len()));
    children.sort_unstable_by_key(|&c| sizes[c]);
}

/// Order a parent's children for the merge-biased postorder.
///
/// Partition: `bias[child] == false` first (emit early), then
/// `bias[child] == true` (emit late, adjacent to the parent). Within
/// each partition, ascending subtree size — the same heuristic as
/// the unbiased postorder, applied independently to each partition.
///
/// In place; `scratch` is reused across nodes (issue #128 item D).
fn merge_bias_partition(
    children: &mut [usize],
    sizes: &[usize],
    bias: &[bool],
    scratch: &mut Vec<usize>,
) {
    let split = stable_partition(children, scratch, |c| !bias[c]);
    children[..split].sort_unstable_by_key(|&c| sizes[c]);
    children[split..].sort_unstable_by_key(|&c| sizes[c]);
}

/// Move every element satisfying `pred` to the front of `v`, preserving the
/// relative order **within each** group, and return the length of the front
/// group. `scratch` is reused across calls.
///
/// Stability matters: the code this replaces built each group with
/// `iter().copied().filter(..).collect()`, which preserves input order, and
/// the subsequent `sort_unstable_by_key` is only deterministic given a fixed
/// input sequence. An unstable partition here could feed the sorts a
/// different permutation and silently change the emitted postorder — and
/// hence the fill-reducing ordering.
fn stable_partition(
    v: &mut [usize],
    scratch: &mut Vec<usize>,
    mut pred: impl FnMut(usize) -> bool,
) -> usize {
    scratch.clear();
    let mut split = 0usize;
    for i in 0..v.len() {
        if pred(v[i]) {
            v[split] = v[i];
            split += 1;
        } else {
            scratch.push(v[i]);
        }
    }
    v[split..].copy_from_slice(scratch);
    split
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    #[test]
    fn test_postorder_tridiagonal() {
        // Chain: 0→1→2→3. Postorder should be [0, 1, 2, 3].
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (order, inv) = postorder(&etree);

        assert_eq!(order.len(), 4);
        // In a chain, postorder visits from leaf to root
        assert_eq!(order, vec![0, 1, 2, 3]);

        // Verify inverse
        for (k, &node) in order.iter().enumerate() {
            assert_eq!(inv[node], k);
        }
    }

    #[test]
    fn test_postorder_valid_topological_order() {
        // For any matrix: every child appears before its parent in postorder
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (order, inv) = postorder(&etree);

        assert_eq!(order.len(), 5);

        // Verify topological property: parent appears after child
        for j in 0..5 {
            if let Some(p) = etree.parent[j] {
                assert!(
                    inv[j] < inv[p],
                    "child {} (pos {}) should appear before parent {} (pos {})",
                    j,
                    inv[j],
                    p,
                    inv[p]
                );
            }
        }
    }

    #[test]
    fn test_postorder_diagonal() {
        // Forest of singletons: any order is a valid postorder
        let m = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0; 3]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (order, _) = postorder(&etree);

        assert_eq!(order.len(), 3);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_postorder_inverse_roundtrip() {
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (order, inv) = postorder(&etree);

        // order[inv[j]] == j for all j
        for j in 0..4 {
            assert_eq!(order[inv[j]], j);
        }
        // inv[order[k]] == k for all k
        for k in 0..4 {
            assert_eq!(inv[order[k]], k);
        }
    }

    #[test]
    fn test_schur_postorder_no_schur_matches_postorder() {
        // is_schur all-false should reproduce standard postorder exactly.
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (a, _) = postorder(&etree);
        let (b, _) = schur_constrained_postorder(&etree, &[false; 5]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_schur_postorder_chain_tail_pinned() {
        // Chain 0→1→2→3→4, mark {3,4} as Schur. Standard postorder is
        // [0,1,2,3,4]; constrained must keep 3,4 at positions 3,4.
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 1, 2, 2, 3, 3, 4, 4],
            &[0, 0, 1, 1, 2, 2, 3, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let mut is_schur = vec![false; 5];
        is_schur[3] = true;
        is_schur[4] = true;
        let (post, inv) = schur_constrained_postorder(&etree, &is_schur);
        assert_eq!(post[3], 3);
        assert_eq!(post[4], 4);
        // Identity check on the tail; non-Schur prefix is some valid
        // topological order.
        assert_eq!(inv[3], 3);
        assert_eq!(inv[4], 4);
    }

    #[test]
    fn test_schur_postorder_topological_property() {
        // For arbitrary etree + is_schur, every child still precedes its
        // parent in the postorder (topological invariant).
        let m = CscMatrix::from_triplets(
            6,
            &[0, 1, 2, 3, 4, 5, 1, 4, 2, 4, 3, 4, 5],
            &[0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4],
            &[1.0; 13],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let mut is_schur = vec![false; 6];
        is_schur[4] = true;
        is_schur[5] = true;
        let (_post, inv) = schur_constrained_postorder(&etree, &is_schur);
        for j in 0..6 {
            if let Some(p) = etree.parent[j] {
                assert!(
                    inv[j] < inv[p],
                    "child {} (pos {}) must precede parent {} (pos {})",
                    j,
                    inv[j],
                    p,
                    inv[p]
                );
            }
        }
        // Schur nodes are at the tail in their etree-index order.
        // For this matrix the Schur subset {4,5} forms a top of the tree.
        assert!(inv[4] >= 4);
        assert!(inv[5] >= 4);
        assert!(inv[4] < inv[5] || inv[5] < inv[4]); // both valid positions
    }

    #[test]
    fn test_schur_postorder_forest_tail_identity() {
        // F3.3 regression: when the Schur subtree is a *forest* (multiple
        // Schur roots) with at least one internal Schur node whose parent
        // is also Schur, a DFS over the Schur subtree emits Schur nodes
        // in tree-walk order, not etree-index order. That breaks the
        // tail identity post[k] == k that
        // `symbolic_factorize_with_schur` relies on for the
        // schur_indices contract.
        //
        // ACOPP30_0000 hit this: Schur roots were {158, 159, 160, 161,
        // 167, 168, 195, 196, 197, 203, 204} plus a chain 157 → 162 →
        // ... → 208. Tail identity was violated from col 174 onward,
        // so the original A[174, 174] = -28.56 ended up at permuted
        // (184, 184) and the Schur block had max relative error 0.997
        // vs the dense oracle. The fix is to emit phase-2 Schur nodes
        // directly in ascending etree-index order, not via DFS.
        //
        // Construction: n=8, Schur = {4, 5, 6, 7}. Etree:
        //   non-Schur chain 0 → 1 → 2 → 3 → root 5 (Schur)
        //   internal Schur 4 → root 7 (Schur)
        //   Schur roots {5, 6, 7}; Schur 4 is a non-root Schur node.
        let etree = EliminationTree {
            parent: vec![
                Some(1),
                Some(2),
                Some(3),
                Some(5),
                Some(7),
                None,
                None,
                None,
            ],
            n: 8,
        };
        let is_schur = vec![false, false, false, false, true, true, true, true];
        let (post, inv) = schur_constrained_postorder(&etree, &is_schur);
        // Tail identity: post[k] == k for every Schur k.
        for k in 4..8 {
            assert_eq!(
                post[k], k,
                "tail identity violated: post[{}] = {} (expected {})",
                k, post[k], k
            );
            assert_eq!(inv[k], k);
        }
        // Topological invariant: every child precedes its parent.
        for j in 0..8 {
            if let Some(p) = etree.parent[j] {
                assert!(
                    inv[j] < inv[p],
                    "child {} (pos {}) must precede parent {} (pos {})",
                    j,
                    inv[j],
                    p,
                    inv[p]
                );
            }
        }
        // Non-Schur prefix: post[0..4] is a permutation of {0, 1, 2, 3}.
        let mut prefix: Vec<usize> = post[0..4].to_vec();
        prefix.sort();
        assert_eq!(prefix, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_schur_postorder_empty_etree() {
        let etree = EliminationTree {
            parent: Vec::new(),
            n: 0,
        };
        let (post, inv) = schur_constrained_postorder(&etree, &[]);
        assert!(post.is_empty());
        assert!(inv.is_empty());
    }

    #[test]
    fn test_postorder_empty() {
        let etree = EliminationTree {
            parent: Vec::new(),
            n: 0,
        };
        let (order, inv) = postorder(&etree);
        assert!(order.is_empty());
        assert!(inv.is_empty());
    }

    /// Build a star elimination tree: nodes `0..n-1` are leaves whose only
    /// parent is the last node `n-1` (the root). This is the etree of an
    /// arrow/bordered matrix whose dense border sits at the *trailing*
    /// index (`A[n-1, i] != 0` for every `i < n-1`) — exactly the shape
    /// AMD produces for the dense-border KKT rows in this codebase's tests.
    fn star_etree(n: usize) -> EliminationTree {
        // Lower-triangle: diagonal + a dense trailing column n-1.
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i); // diagonal
            if i < n - 1 {
                rows.push(n - 1);
                cols.push(i); // (row n-1, col i): border in the lower triangle
            }
        }
        let vals = vec![1.0; rows.len()];
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let pat = m.symmetric_pattern();
        EliminationTree::from_pattern(&pat)
    }

    /// S1 (dev/research/repo-review-2026-06-09.md): the previous `postorder`
    /// re-cloned and re-sorted `children[node]` on every stack visit, so a
    /// node with `c` children (on top of the stack `c+1` times) paid
    /// O(c²·log c). On a star etree (one root with `n-1` children) that is
    /// O(n²·log n) — quadratic — in the default symbolic pipeline.
    ///
    /// Reproduction is deterministic via the `SORT_WORK` counter (total
    /// child-list elements materialized across all per-node sorts), so no
    /// flaky wall-clock timing is needed. Pre-fix the root's `(n-1)`-element
    /// child list is materialized `n` times → `~n²` elements. Post-fix it is
    /// materialized exactly once → `~n` elements. The assertion `work ≤ 4·n`
    /// fails on the quadratic version and passes on the linear fix.
    #[test]
    fn test_postorder_star_sort_work_is_linear() {
        let n = 2000;
        let etree = star_etree(n);

        // Sanity: this really is a star (root n-1, all others its children).
        assert_eq!(etree.children()[n - 1].len(), n - 1);
        assert_eq!(etree.roots(), vec![n - 1]);

        SORT_WORK.with(|w| w.set(0));
        let (order, inv) = postorder(&etree);
        let work = SORT_WORK.with(|w| w.get());

        // Output correctness still holds (every child before its parent).
        assert_eq!(order.len(), n);
        for j in 0..n {
            if let Some(p) = etree.parent[j] {
                assert!(inv[j] < inv[p], "child {j} must precede parent {p}");
            }
        }

        // The fix: child-sorting work is linear, not quadratic. The old
        // sort-on-every-visit code materializes ~n² elements here.
        assert!(
            work <= 4 * n,
            "postorder sort work {work} exceeds the linear bound {} (n={n}); \
             the O(n²·log n) sort-on-every-stack-visit regression is back",
            4 * n
        );
    }
}
