use crate::sparse::csc::CscPattern;

/// Compute an approximate minimum degree (AMD) ordering.
///
/// Given a symmetric sparsity pattern, returns a permutation vector `perm`
/// such that factoring P·A·Pᵀ produces less fill than the natural ordering.
///
/// This is a simplified AMD implementation based on the quotient graph model
/// from Amestoy, Davis & Duff (1996). It uses exact external degree (not
/// approximate) which is correct but slower than full AMD with element
/// absorption. Sufficient for Phase 1b matrices (dim <= 500).
///
/// The permutation maps new indices to old: column `perm[k]` of the original
/// matrix becomes column `k` in the reordered matrix.
pub fn amd_order(pattern: &CscPattern) -> Vec<usize> {
    let n = pattern.n;
    if n == 0 {
        return Vec::new();
    }

    // Build adjacency lists (excluding self-loops)
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (j, adj_j) in adj.iter_mut().enumerate() {
        for k in pattern.col_ptr[j]..pattern.col_ptr[j + 1] {
            let i = pattern.row_idx[k];
            if i != j {
                adj_j.push(i);
            }
        }
    }

    // Track which nodes are eliminated
    let mut eliminated = vec![false; n];
    // Degree of each node (number of adjacent non-eliminated nodes)
    let mut degree = vec![0usize; n];
    for i in 0..n {
        degree[i] = adj[i].len();
    }

    let mut perm = Vec::with_capacity(n);

    for _ in 0..n {
        // Find non-eliminated node with minimum degree
        let mut min_deg = usize::MAX;
        let mut pivot = 0;
        for i in 0..n {
            if !eliminated[i] && degree[i] < min_deg {
                min_deg = degree[i];
                pivot = i;
            }
        }

        // Eliminate pivot
        eliminated[pivot] = true;
        perm.push(pivot);

        // Collect neighbors of pivot (reachable set)
        let neighbors: Vec<usize> = adj[pivot]
            .iter()
            .copied()
            .filter(|&i| !eliminated[i])
            .collect();

        // Add fill edges: all pairs of neighbors become adjacent
        // (this is the key insight — elimination creates a clique among neighbors)
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let (a, b) = (neighbors[i], neighbors[j]);
                if !adj[a].contains(&b) {
                    adj[a].push(b);
                    adj[b].push(a);
                }
            }
        }

        // Update degrees: neighbors' degrees change due to fill edges
        // and removal of pivot
        for &nb in &neighbors {
            degree[nb] = adj[nb].iter().filter(|&&x| !eliminated[x]).count();
        }
    }

    perm
}

/// Apply a permutation to row/column indices: compute P·A·Pᵀ pattern.
///
/// Given a symmetric CscPattern and permutation `perm` (new-to-old mapping),
/// returns the permuted pattern.
pub fn permute_pattern(pattern: &CscPattern, perm: &[usize]) -> CscPattern {
    let n = pattern.n;

    // Build inverse permutation: inv_perm[old] = new
    let mut inv_perm = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        inv_perm[old] = new;
    }

    // Collect permuted entries (lower triangle of P·A·Pᵀ)
    let mut entries: Vec<Vec<usize>> = vec![Vec::new(); n];
    for old_j in 0..n {
        let new_j = inv_perm[old_j];
        for k in pattern.col_ptr[old_j]..pattern.col_ptr[old_j + 1] {
            let old_i = pattern.row_idx[k];
            let new_i = inv_perm[old_i];
            // Store in full pattern (both triangles)
            entries[new_j].push(new_i);
            if new_i != new_j {
                entries[new_i].push(new_j);
            }
        }
    }

    // Build CSC pattern
    let mut col_ptr = vec![0usize; n + 1];
    for j in 0..n {
        entries[j].sort_unstable();
        entries[j].dedup();
        col_ptr[j + 1] = col_ptr[j] + entries[j].len();
    }
    let row_idx: Vec<usize> = entries.into_iter().flatten().collect();

    CscPattern { n, col_ptr, row_idx }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    fn arrow_matrix_5() -> CscPattern {
        // Arrow matrix: dense first row/column, diagonal elsewhere
        // This is the worst case for natural ordering, best case for
        // ordering that puts the dense row last
        //
        // [ 1 1 1 1 1 ]
        // [ 1 1 0 0 0 ]
        // [ 1 0 1 0 0 ]
        // [ 1 0 0 1 0 ]
        // [ 1 0 0 0 1 ]
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        m.symmetric_pattern()
    }

    #[test]
    fn test_amd_valid_permutation() {
        let pat = arrow_matrix_5();
        let perm = amd_order(&pat);
        assert_eq!(perm.len(), 5);

        // Check it's a valid permutation
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_amd_arrow_matrix() {
        // For an arrow matrix, AMD should eliminate leaf nodes before the hub.
        // Leaves 1,2,3,4 each start with degree 1; hub (node 0) starts at degree 4.
        // As leaves are eliminated, hub's degree drops. The exact final position
        // of the hub depends on tie-breaking, but leaves should go first.
        let pat = arrow_matrix_5();
        let perm = amd_order(&pat);

        // First 3 positions should be leaf nodes (degree 1)
        assert!(
            perm[..3].iter().all(|&p| p != 0),
            "leaf nodes should be eliminated before hub"
        );

        // AMD on arrow should produce zero fill (any ordering of leaves first works)
        let fill = estimate_fill(&pat, &perm);
        assert_eq!(fill, 0, "AMD on arrow matrix should produce zero fill");
    }

    #[test]
    fn test_amd_diagonal_matrix() {
        // Diagonal matrix: all degrees are 0, any ordering is optimal
        let m = CscMatrix::from_triplets(
            4,
            &[0, 1, 2, 3],
            &[0, 1, 2, 3],
            &[1.0; 4],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let perm = amd_order(&pat);
        assert_eq!(perm.len(), 4);

        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_amd_tridiagonal() {
        // Tridiagonal: natural ordering is already near-optimal
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 1, 2, 2, 3, 3, 4, 4],
            &[0, 0, 1, 1, 2, 2, 3, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let perm = amd_order(&pat);
        assert_eq!(perm.len(), 5);
    }

    #[test]
    fn test_amd_empty() {
        let pat = CscPattern {
            n: 0,
            col_ptr: vec![0],
            row_idx: vec![],
        };
        let perm = amd_order(&pat);
        assert_eq!(perm.len(), 0);
    }

    #[test]
    fn test_permute_pattern() {
        // Simple 3x3 tridiagonal: [[1,-1,0],[-1,2,-1],[0,-1,1]]
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[1.0, -1.0, 2.0, -1.0, 1.0],
        )
        .unwrap();
        let pat = m.symmetric_pattern();

        // Reverse permutation: [2, 1, 0]
        let perm = vec![2, 1, 0];
        let permuted = permute_pattern(&pat, &perm);

        // After reversing, the pattern should be the same (tridiagonal is symmetric)
        assert_eq!(permuted.n, 3);
        assert_eq!(permuted.col_ptr[3], pat.col_ptr[3]);
    }

    #[test]
    fn test_amd_reduces_fill_on_arrow() {
        // Count fill for natural vs AMD ordering on the arrow matrix
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();

        let amd_perm = amd_order(&pat);
        let natural_perm: Vec<usize> = (0..5).collect();

        let natural_fill = estimate_fill(&pat, &natural_perm);
        let amd_fill = estimate_fill(&pat, &amd_perm);

        // AMD should produce less or equal fill
        assert!(
            amd_fill <= natural_fill,
            "AMD fill {} > natural fill {}",
            amd_fill,
            natural_fill
        );
    }

    /// Estimate fill-in by simulating elimination on the permuted pattern.
    fn estimate_fill(pattern: &CscPattern, perm: &[usize]) -> usize {
        let n = pattern.n;
        let permuted = permute_pattern(pattern, perm);

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for j in 0..n {
            for k in permuted.col_ptr[j]..permuted.col_ptr[j + 1] {
                let i = permuted.row_idx[k];
                if i != j && !adj[j].contains(&i) {
                    adj[j].push(i);
                }
            }
        }

        let mut eliminated = vec![false; n];
        let mut fill = 0usize;

        for j in 0..n {
            eliminated[j] = true;
            let neighbors: Vec<usize> = adj[j]
                .iter()
                .copied()
                .filter(|&i| !eliminated[i])
                .collect();

            for a in 0..neighbors.len() {
                for b in (a + 1)..neighbors.len() {
                    let (u, v) = (neighbors[a], neighbors[b]);
                    if !adj[u].contains(&v) {
                        adj[u].push(v);
                        adj[v].push(u);
                        fill += 1;
                    }
                }
            }
        }
        fill
    }
}
