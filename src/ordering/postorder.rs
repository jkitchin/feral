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
