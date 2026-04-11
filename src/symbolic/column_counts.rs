use crate::ordering::elimination_tree::EliminationTree;
use crate::sparse::csc::CscPattern;

/// Compute the number of nonzeros in each column of the Cholesky factor L.
///
/// Uses elimination graph simulation: process columns left to right,
/// maintaining the fill pattern. For column j, L[:,j] contains:
/// - The diagonal entry (j,j)
/// - All original entries (i,j) with i > j
/// - All fill entries propagated from earlier columns
///
/// For indefinite factorization (LDL^T), the fill pattern is the same as
/// Cholesky — pivoting changes values but not structure (ignoring delayed
/// pivots, which are Phase 2).
///
/// Input `pattern` should be the full symmetric pattern (both triangles).
///
/// Returns a vector of length n where `counts[j]` is the number of nonzeros
/// in column j of L (including the diagonal).
pub fn column_counts(pattern: &CscPattern, _etree: &EliminationTree) -> Vec<usize> {
    let n = pattern.n;
    if n == 0 {
        return Vec::new();
    }

    // Simulate the elimination to compute the exact fill pattern.
    // For each column j, track the set of row indices i > j that will
    // have nonzeros in L[:,j].
    //
    // When column j is eliminated, for every pair of rows (i1, i2) in
    // L[:,j] with i1 > j and i2 > j, a fill entry is created at (max(i1,i2), min(i1,i2)).
    // These fill entries propagate to subsequent columns.

    // Build adjacency: for each column j, the set of rows > j
    let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (j, col_j) in col_rows.iter_mut().enumerate() {
        for k in pattern.col_ptr[j]..pattern.col_ptr[j + 1] {
            let i = pattern.row_idx[k];
            if i > j {
                col_j.push(i);
            }
        }
        col_j.sort_unstable();
        col_j.dedup();
    }

    let mut counts = vec![1usize; n]; // diagonal always present

    for j in 0..n {
        let rows = std::mem::take(&mut col_rows[j]);
        counts[j] += rows.len();

        // Propagate fill: all rows in this column become connected.
        // The minimum row index inherits all other row indices.
        if rows.len() > 1 {
            let min_row = rows[0]; // smallest row > j
            for &row in &rows[1..] {
                // Add row to column min_row's pattern (if not already present)
                if !col_rows[min_row].contains(&row) {
                    col_rows[min_row].push(row);
                }
            }
            col_rows[min_row].sort_unstable();
            col_rows[min_row].dedup();
        }
    }

    counts
}

/// Compute the total number of nonzeros in L from column counts.
pub fn total_factor_nnz(counts: &[usize]) -> usize {
    counts.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    #[test]
    fn test_column_counts_diagonal() {
        // Diagonal matrix: each column of L has exactly 1 nonzero (the diagonal)
        let m = CscMatrix::from_triplets(
            4,
            &[0, 1, 2, 3],
            &[0, 1, 2, 3],
            &[1.0; 4],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![1, 1, 1, 1]);
        assert_eq!(total_factor_nnz(&counts), 4);
    }

    #[test]
    fn test_column_counts_tridiagonal() {
        // Tridiagonal 4x4: L has entries on diagonal and one subdiagonal
        // Column 0: rows 0, 1 → count = 2
        // Column 1: rows 1, 2 → count = 2
        // Column 2: rows 2, 3 → count = 2
        // Column 3: row 3      → count = 1
        let m = CscMatrix::from_triplets(
            4,
            &[0, 1, 1, 2, 2, 3, 3],
            &[0, 0, 1, 1, 2, 2, 3],
            &[1.0; 7],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![2, 2, 2, 1]);
        assert_eq!(total_factor_nnz(&counts), 7);
    }

    #[test]
    fn test_column_counts_dense() {
        // Dense 3x3: L is full lower triangle
        // Column 0: rows 0, 1, 2 → count = 3
        // Column 1: rows 1, 2    → count = 2
        // Column 2: row 2        → count = 1
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 2, 1, 2, 2],
            &[0, 0, 0, 1, 1, 2],
            &[1.0; 6],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![3, 2, 1]);
        assert_eq!(total_factor_nnz(&counts), 6); // n*(n+1)/2
    }

    #[test]
    fn test_column_counts_arrow() {
        // Arrow 5x5: column 0 has entries at rows 0-4, others are diagonal
        // Eliminating column 0 creates fill among rows 1-4
        // Column 0: rows 0,1,2,3,4 → count = 5
        // Column 1: rows 1,2,3,4 (fill from col 0) → count = 4
        // Column 2: rows 2,3,4 → count = 3
        // Column 3: rows 3,4 → count = 2
        // Column 4: row 4 → count = 1
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![5, 4, 3, 2, 1]);
        assert_eq!(total_factor_nnz(&counts), 15); // fully dense: n*(n+1)/2
    }

    #[test]
    fn test_column_counts_block_diagonal() {
        // Two 2x2 dense blocks: no fill between blocks
        // [a b 0 0]
        // [b c 0 0]
        // [0 0 d e]
        // [0 0 e f]
        let m = CscMatrix::from_triplets(
            4,
            &[0, 1, 1, 2, 3, 3],
            &[0, 0, 1, 2, 2, 3],
            &[1.0; 6],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![2, 1, 2, 1]);
        assert_eq!(total_factor_nnz(&counts), 6);
    }
}
