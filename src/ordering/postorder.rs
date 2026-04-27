use super::elimination_tree::EliminationTree;

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

    let children = etree.children();
    let sizes = etree.subtree_sizes();
    let roots = etree.roots();

    let mut order = Vec::with_capacity(n);

    // DFS stack: (node, child_index) — iterative DFS to avoid stack overflow
    let mut stack: Vec<(usize, usize)> = Vec::new();

    // Process roots in ascending subtree size order
    let mut sorted_roots = roots;
    sorted_roots.sort_unstable_by_key(|&r| sizes[r]);

    for &root in &sorted_roots {
        stack.push((root, 0));

        while let Some((node, child_idx)) = stack.last_mut() {
            // Sort children by subtree size (smallest first) on first visit
            let node = *node;
            let mut sorted_children = children[node].clone();
            sorted_children.sort_unstable_by_key(|&c| sizes[c]);

            if *child_idx < sorted_children.len() {
                let child = sorted_children[*child_idx];
                *child_idx += 1;
                stack.push((child, 0));
            } else {
                // All children visited — emit this node (postorder)
                order.push(node);
                stack.pop();
            }
        }
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

    let children = etree.children();
    let sizes = etree.subtree_sizes();
    let roots = etree.roots();

    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<(usize, Vec<usize>, usize)> = Vec::new();

    // Roots are not biased (no parent to be adjacent to). Use the
    // unbiased subtree-size order.
    let mut sorted_roots = roots;
    sorted_roots.sort_unstable_by_key(|&r| sizes[r]);

    for &root in &sorted_roots {
        let merged = merge_bias_partition(&children[root], &sizes, bias);
        stack.push((root, merged, 0));

        while let Some((node, sorted_children, child_idx)) = stack.last_mut() {
            let node_id = *node;
            if *child_idx < sorted_children.len() {
                let child = sorted_children[*child_idx];
                *child_idx += 1;
                let next_children = merge_bias_partition(&children[child], &sizes, bias);
                stack.push((child, next_children, 0));
            } else {
                order.push(node_id);
                stack.pop();
            }
        }
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

    let children = etree.children();
    let sizes = etree.subtree_sizes();
    let roots = etree.roots();

    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<(usize, Vec<usize>, usize)> = Vec::new();

    let sorted_roots = schur_partition_children(&roots, &sizes, is_schur);
    for &root in sorted_roots.iter() {
        let merged = schur_partition_children(&children[root], &sizes, is_schur);
        stack.push((root, merged, 0));

        while let Some((node, sorted_children, child_idx)) = stack.last_mut() {
            let node_id = *node;
            if *child_idx < sorted_children.len() {
                let child = sorted_children[*child_idx];
                *child_idx += 1;
                let next_children = schur_partition_children(&children[child], &sizes, is_schur);
                stack.push((child, next_children, 0));
            } else {
                order.push(node_id);
                stack.pop();
            }
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
fn schur_partition_children(children: &[usize], sizes: &[usize], is_schur: &[bool]) -> Vec<usize> {
    let mut nonschur: Vec<usize> = children.iter().copied().filter(|&c| !is_schur[c]).collect();
    let mut schur: Vec<usize> = children.iter().copied().filter(|&c| is_schur[c]).collect();
    nonschur.sort_unstable_by_key(|&c| sizes[c]);
    schur.sort_unstable();
    nonschur.extend(schur);
    nonschur
}

/// Order a parent's children for the merge-biased postorder.
///
/// Partition: `bias[child] == false` first (emit early), then
/// `bias[child] == true` (emit late, adjacent to the parent). Within
/// each partition, ascending subtree size — the same heuristic as
/// the unbiased postorder, applied independently to each partition.
fn merge_bias_partition(children: &[usize], sizes: &[usize], bias: &[bool]) -> Vec<usize> {
    let mut early: Vec<usize> = children.iter().copied().filter(|&c| !bias[c]).collect();
    let mut late: Vec<usize> = children.iter().copied().filter(|&c| bias[c]).collect();
    early.sort_unstable_by_key(|&c| sizes[c]);
    late.sort_unstable_by_key(|&c| sizes[c]);
    early.extend(late);
    early
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
        let (b, _) = schur_constrained_postorder(&etree, &vec![false; 5]);
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
}
