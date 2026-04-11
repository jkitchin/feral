use crate::sparse::csc::CscPattern;

/// Elimination tree of a symmetric matrix.
///
/// For a symmetric matrix A, the elimination tree has the property:
/// parent[j] = min { i > j : L(i,j) ≠ 0 } where L is the Cholesky factor.
/// For indefinite matrices, the same structure applies to the fill pattern.
///
/// Constructed from the symmetric sparsity pattern using union-find with
/// path compression (George & Liu 1981, Chapter 4).
#[derive(Debug, Clone)]
pub struct EliminationTree {
    /// parent[j] = Some(i) where i > j, or None if j is a root.
    pub parent: Vec<Option<usize>>,
    pub n: usize,
}

impl EliminationTree {
    /// Build the elimination tree from a symmetric sparsity pattern.
    ///
    /// Uses the column-by-column algorithm with path compression
    /// (Liu 1990, based on George & Liu 1981):
    ///
    /// For each column j (in order 0..n), examine all rows i < j in column j
    /// (the upper triangle entries). Walk from i up the partially built tree
    /// using path compression until finding a root or reaching j. Make j the
    /// parent of that root. This produces parent[j] = min { i > j : L(i,j) ≠ 0 }.
    pub fn from_pattern(pattern: &CscPattern) -> Self {
        let n = pattern.n;
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut ancestor = vec![0usize; n]; // union-find forest

        for j in 0..n {
            ancestor[j] = j; // j is its own root initially
            for k in pattern.col_ptr[j]..pattern.col_ptr[j + 1] {
                let i = pattern.row_idx[k];
                if i >= j {
                    continue; // only process entries with i < j
                }

                // Find the root of i's subtree (with path compression)
                let mut r = i;
                while ancestor[r] != r {
                    r = ancestor[r];
                }
                // Path compression: make all nodes on the path point to r
                let mut node = i;
                while node != r {
                    let next = ancestor[node];
                    ancestor[node] = r;
                    node = next;
                }

                // If r != j, make j the parent of r
                if r != j {
                    parent[r] = Some(j);
                    ancestor[r] = j; // union: attach r's tree under j
                }
            }
        }

        EliminationTree { parent, n }
    }

    /// Compute children lists from parent pointers.
    pub fn children(&self) -> Vec<Vec<usize>> {
        let mut ch = vec![Vec::new(); self.n];
        for j in 0..self.n {
            if let Some(p) = self.parent[j] {
                ch[p].push(j);
            }
        }
        ch
    }

    /// Return root nodes (nodes with no parent).
    pub fn roots(&self) -> Vec<usize> {
        (0..self.n)
            .filter(|&j| self.parent[j].is_none())
            .collect()
    }

    /// Compute subtree sizes (number of nodes in each subtree, including self).
    pub fn subtree_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![1usize; self.n];
        // Process in reverse topological order (children before parents)
        // Since parent[j] > j always, processing 0..n in order is fine
        // if we accumulate into parents.
        for j in 0..self.n {
            if let Some(p) = self.parent[j] {
                sizes[p] += sizes[j];
            }
        }
        sizes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    #[test]
    fn test_etree_tridiagonal() {
        // Tridiagonal 5x5: elimination tree is a path 0→1→2→3→4
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 1, 2, 2, 3, 3, 4, 4],
            &[0, 0, 1, 1, 2, 2, 3, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);

        assert_eq!(etree.parent[0], Some(1));
        assert_eq!(etree.parent[1], Some(2));
        assert_eq!(etree.parent[2], Some(3));
        assert_eq!(etree.parent[3], Some(4));
        assert_eq!(etree.parent[4], None); // root
    }

    #[test]
    fn test_etree_arrow() {
        // Arrow matrix: node 0 is connected to all others
        // After natural ordering, etree should have 0 as root
        // with nodes 1,2,3,4 filling through 0
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);

        // With natural ordering on arrow matrix:
        // All nodes 1-4 connect to 0, and eliminating 0 creates a clique
        // among 1-4. So the etree should be 0→1→2→3→4 (chain from fill).
        // Actually: parent[j] = min { i > j : L(i,j) != 0 }
        // For column 0: rows 1,2,3,4 all have entries → parent[0] = 1 (not a root!)
        // Wait — arrow has column 0 connected to rows 1,2,3,4
        // Column 0: entries at rows 1,2,3,4 → parent[0] = min(1,2,3,4) = 1
        // Column 1: entry at row 0 (but 0 < 1, skip). Fill from eliminating 0: rows 2,3,4
        //   → parent[1] = 2
        // etc. So etree is a chain 0→1→2→3→4, root = 4
        assert_eq!(etree.parent[4], None);
        assert_eq!(etree.roots(), vec![4]);
    }

    #[test]
    fn test_etree_diagonal() {
        // Diagonal: no off-diagonal entries → forest of singletons
        let m = CscMatrix::from_triplets(
            4,
            &[0, 1, 2, 3],
            &[0, 1, 2, 3],
            &[1.0; 4],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);

        for j in 0..4 {
            assert_eq!(etree.parent[j], None);
        }
        assert_eq!(etree.roots().len(), 4);
    }

    #[test]
    fn test_etree_children() {
        // Tridiagonal: children of node k = [k-1] (except 0)
        let m = CscMatrix::from_triplets(
            4,
            &[0, 1, 1, 2, 2, 3, 3],
            &[0, 0, 1, 1, 2, 2, 3],
            &[1.0; 7],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let ch = etree.children();

        assert_eq!(ch[0], Vec::<usize>::new());
        assert_eq!(ch[1], vec![0]);
        assert_eq!(ch[2], vec![1]);
        assert_eq!(ch[3], vec![2]);
    }

    #[test]
    fn test_subtree_sizes() {
        // Tridiagonal 4x4: chain 0→1→2→3
        let m = CscMatrix::from_triplets(
            4,
            &[0, 1, 1, 2, 2, 3, 3],
            &[0, 0, 1, 1, 2, 2, 3],
            &[1.0; 7],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let sizes = etree.subtree_sizes();

        assert_eq!(sizes[0], 1);
        assert_eq!(sizes[1], 2);
        assert_eq!(sizes[2], 3);
        assert_eq!(sizes[3], 4);
    }
}
