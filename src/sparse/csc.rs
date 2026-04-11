use crate::error::FeralError;

/// Compressed Sparse Column (CSC) matrix storage for symmetric matrices.
///
/// Only the lower triangle is stored. `col_ptr[j]..col_ptr[j+1]` gives the
/// range of entries in column j. Row indices within each column are sorted
/// in ascending order.
#[derive(Debug, Clone)]
pub struct CscMatrix {
    pub n: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub values: Vec<f64>,
}

/// Symmetric sparsity pattern (full, not just lower triangle).
/// Used for AMD ordering and elimination tree construction.
#[derive(Debug, Clone)]
pub struct CscPattern {
    pub n: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
}

impl CscMatrix {
    /// Number of stored nonzeros (lower triangle only).
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Build a CSC matrix from coordinate (triplet) format.
    ///
    /// Entries must be lower-triangle (row >= col). Duplicate entries are summed.
    /// Row indices within each column are sorted.
    pub fn from_triplets(
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
    ) -> Result<Self, FeralError> {
        if rows.len() != cols.len() || cols.len() != vals.len() {
            return Err(FeralError::InvalidInput(
                "triplet arrays must have equal length".to_string(),
            ));
        }

        // Count entries per column
        let mut col_counts = vec![0usize; n];
        for &c in cols {
            if c >= n {
                return Err(FeralError::InvalidInput(format!(
                    "column index {} out of bounds for n={}",
                    c, n
                )));
            }
            col_counts[c] += 1;
        }

        // Build col_ptr
        let mut col_ptr = vec![0usize; n + 1];
        for j in 0..n {
            col_ptr[j + 1] = col_ptr[j] + col_counts[j];
        }
        let nnz = col_ptr[n];

        // Place entries
        let mut row_idx = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        let mut offsets = col_ptr[..n].to_vec();
        for k in 0..rows.len() {
            let (r, c) = (rows[k], cols[k]);
            if r >= n {
                return Err(FeralError::InvalidInput(format!(
                    "row index {} out of bounds for n={}",
                    r, n
                )));
            }
            let pos = offsets[c];
            row_idx[pos] = r;
            values[pos] = vals[k];
            offsets[c] += 1;
        }

        // Sort each column by row index, summing duplicates
        let mut result = CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        };
        result.sort_and_sum_duplicates();
        Ok(result)
    }

    /// Sort row indices within each column and sum duplicate entries.
    fn sort_and_sum_duplicates(&mut self) {
        // Two-pass approach: first sort and deduplicate into a compact representation,
        // then rebuild the arrays.
        let mut new_row_idx = Vec::with_capacity(self.row_idx.len());
        let mut new_values = Vec::with_capacity(self.values.len());
        let mut new_col_ptr = vec![0usize; self.n + 1];

        for j in 0..self.n {
            let start = self.col_ptr[j];
            let end = self.col_ptr[j + 1];
            let col_start = new_row_idx.len();

            if start == end {
                new_col_ptr[j + 1] = col_start;
                continue;
            }

            // Collect (row, val) pairs for this column and sort by row
            let mut pairs: Vec<(usize, f64)> = (start..end)
                .map(|k| (self.row_idx[k], self.values[k]))
                .collect();
            pairs.sort_unstable_by_key(|&(r, _)| r);

            // Deduplicate by summing
            let mut prev_row = pairs[0].0;
            let mut prev_val = pairs[0].1;
            for &(r, v) in &pairs[1..] {
                if r == prev_row {
                    prev_val += v;
                } else {
                    new_row_idx.push(prev_row);
                    new_values.push(prev_val);
                    prev_row = r;
                    prev_val = v;
                }
            }
            new_row_idx.push(prev_row);
            new_values.push(prev_val);

            new_col_ptr[j + 1] = new_row_idx.len();
        }

        self.col_ptr = new_col_ptr;
        self.row_idx = new_row_idx;
        self.values = new_values;
    }

    /// Validate the CSC structure.
    pub fn validate(&self) -> Result<(), FeralError> {
        if self.col_ptr.len() != self.n + 1 {
            return Err(FeralError::InvalidInput(format!(
                "col_ptr length {} != n+1={}",
                self.col_ptr.len(),
                self.n + 1
            )));
        }
        if self.row_idx.len() != self.values.len() {
            return Err(FeralError::InvalidInput(
                "row_idx and values length mismatch".to_string(),
            ));
        }
        if self.col_ptr[self.n] != self.row_idx.len() {
            return Err(FeralError::InvalidInput(
                "col_ptr[n] != nnz".to_string(),
            ));
        }
        for j in 0..self.n {
            let start = self.col_ptr[j];
            let end = self.col_ptr[j + 1];
            for k in start..end {
                if self.row_idx[k] >= self.n {
                    return Err(FeralError::InvalidInput(format!(
                        "row index {} out of bounds in column {}",
                        self.row_idx[k], j
                    )));
                }
            }
            // Check sorted
            for k in (start + 1)..end {
                if self.row_idx[k] <= self.row_idx[k - 1] {
                    return Err(FeralError::InvalidInput(format!(
                        "row indices not sorted in column {} ({}>={})",
                        j,
                        self.row_idx[k - 1],
                        self.row_idx[k]
                    )));
                }
            }
        }
        Ok(())
    }

    /// Expand the lower-triangle CSC to a full symmetric sparsity pattern.
    ///
    /// The result contains both (i,j) and (j,i) for every off-diagonal entry.
    /// Used for AMD ordering and elimination tree construction.
    pub fn symmetric_pattern(&self) -> CscPattern {
        // Count entries per column in the full pattern
        let mut col_counts = vec![0usize; self.n];
        for j in 0..self.n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                col_counts[j] += 1; // lower triangle entry in column j
                if i != j {
                    col_counts[i] += 1; // transpose entry in column i
                }
            }
        }

        // Build col_ptr
        let mut pat_col_ptr = vec![0usize; self.n + 1];
        for j in 0..self.n {
            pat_col_ptr[j + 1] = pat_col_ptr[j] + col_counts[j];
        }
        let pat_nnz = pat_col_ptr[self.n];
        let mut pat_row_idx = vec![0usize; pat_nnz];

        // Place entries
        let mut offsets = pat_col_ptr[..self.n].to_vec();
        for j in 0..self.n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                // (i, j) in lower triangle
                pat_row_idx[offsets[j]] = i;
                offsets[j] += 1;
                if i != j {
                    // (j, i) — transpose
                    pat_row_idx[offsets[i]] = j;
                    offsets[i] += 1;
                }
            }
        }

        // Sort row indices within each column
        for j in 0..self.n {
            let start = pat_col_ptr[j];
            let end = pat_col_ptr[j + 1];
            pat_row_idx[start..end].sort_unstable();
        }

        CscPattern {
            n: self.n,
            col_ptr: pat_col_ptr,
            row_idx: pat_row_idx,
        }
    }

    /// Symmetric matrix-vector product: y = A * x.
    ///
    /// Uses only the stored lower triangle; implicitly applies symmetry.
    pub fn symv(&self, x: &[f64], y: &mut [f64]) {
        for yi in y.iter_mut().take(self.n) {
            *yi = 0.0;
        }
        for j in 0..self.n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                let v = self.values[k];
                y[i] += v * x[j];
                if i != j {
                    y[j] += v * x[i];
                }
            }
        }
    }

    /// Convert to dense symmetric matrix.
    pub fn to_dense(&self) -> crate::dense::matrix::SymmetricMatrix {
        let entries: Vec<(usize, usize, f64)> = (0..self.n)
            .flat_map(|j| {
                (self.col_ptr[j]..self.col_ptr[j + 1])
                    .map(move |k| (self.row_idx[k], j, self.values[k]))
            })
            .collect();
        crate::dense::matrix::SymmetricMatrix::from_lower_triangle(self.n, &entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_3x3() -> CscMatrix {
        // [ 2 -1  0 ]
        // [-1  3 -1 ]
        // [ 0 -1  4 ]
        CscMatrix::from_triplets(
            3,
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[2.0, -1.0, 3.0, -1.0, 4.0],
        )
        .unwrap()
    }

    #[test]
    fn test_from_triplets_basic() {
        let m = sample_3x3();
        assert_eq!(m.n, 3);
        assert_eq!(m.nnz(), 5);
        m.validate().unwrap();
    }

    #[test]
    fn test_from_triplets_duplicate_summing() {
        let m = CscMatrix::from_triplets(
            2,
            &[0, 0, 1],
            &[0, 0, 1],
            &[1.0, 2.0, 3.0],
        )
        .unwrap();
        assert_eq!(m.nnz(), 2);
        assert_eq!(m.values[0], 3.0); // 1.0 + 2.0
        assert_eq!(m.values[1], 3.0);
    }

    #[test]
    fn test_symmetric_pattern() {
        let m = sample_3x3();
        let pat = m.symmetric_pattern();
        assert_eq!(pat.n, 3);
        // Full pattern: (0,0), (1,0), (0,1), (1,1), (2,1), (1,2), (2,2)
        // = 7 entries total
        assert_eq!(pat.col_ptr[3], 7);

        // Column 0: rows 0, 1
        assert_eq!(&pat.row_idx[pat.col_ptr[0]..pat.col_ptr[1]], &[0, 1]);
        // Column 1: rows 0, 1, 2
        assert_eq!(&pat.row_idx[pat.col_ptr[1]..pat.col_ptr[2]], &[0, 1, 2]);
        // Column 2: rows 1, 2
        assert_eq!(&pat.row_idx[pat.col_ptr[2]..pat.col_ptr[3]], &[1, 2]);
    }

    #[test]
    fn test_symv() {
        let m = sample_3x3();
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0; 3];
        m.symv(&x, &mut y);
        // A * x = [2-2, -1+6-3, -2+12] = [0, 2, 10]
        assert!((y[0] - 0.0).abs() < 1e-14);
        assert!((y[1] - 2.0).abs() < 1e-14);
        assert!((y[2] - 10.0).abs() < 1e-14);
    }

    #[test]
    fn test_to_dense_roundtrip() {
        let m = sample_3x3();
        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), 2.0);
        assert_eq!(dense.get(1, 0), -1.0);
        assert_eq!(dense.get(0, 1), -1.0);
        assert_eq!(dense.get(1, 1), 3.0);
        assert_eq!(dense.get(2, 1), -1.0);
        assert_eq!(dense.get(1, 2), -1.0);
        assert_eq!(dense.get(2, 2), 4.0);
        assert_eq!(dense.get(2, 0), 0.0);
    }

    #[test]
    fn test_validate_rejects_bad_input() {
        let mut m = sample_3x3();
        m.row_idx[0] = 5; // out of bounds
        assert!(m.validate().is_err());
    }

    #[test]
    fn test_diagonal_matrix() {
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 2],
            &[0, 1, 2],
            &[1.0, 2.0, 3.0],
        )
        .unwrap();
        assert_eq!(m.nnz(), 3);
        let pat = m.symmetric_pattern();
        assert_eq!(pat.col_ptr[3], 3); // no off-diagonal, so 3 entries total
    }

    #[test]
    fn test_empty_matrix() {
        let m = CscMatrix::from_triplets(3, &[], &[], &[]).unwrap();
        assert_eq!(m.nnz(), 0);
        m.validate().unwrap();
        let pat = m.symmetric_pattern();
        assert_eq!(pat.col_ptr[3], 0);
    }

    #[test]
    fn test_kkt_structure() {
        // Small KKT: [H  A^T; A  -delta*I]
        // H = [2 0; 0 3], A = [1 1], delta = 1e-8
        // Full matrix (3x3):
        // [ 2    0    1  ]
        // [ 0    3    1  ]
        // [ 1    1  -1e-8]
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 2, 2, 2],
            &[0, 1, 0, 1, 2],
            &[2.0, 3.0, 1.0, 1.0, -1e-8],
        )
        .unwrap();
        assert_eq!(m.nnz(), 5);
        m.validate().unwrap();

        // symv check
        let x = [1.0, 1.0, 1.0];
        let mut y = [0.0; 3];
        m.symv(&x, &mut y);
        assert!((y[0] - 3.0).abs() < 1e-14); // 2 + 0 + 1
        assert!((y[1] - 4.0).abs() < 1e-14); // 0 + 3 + 1
        assert!((y[2] - (2.0 - 1e-8)).abs() < 1e-14); // 1 + 1 - 1e-8
    }
}
