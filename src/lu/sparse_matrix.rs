//! General (unsymmetric) sparse square matrix in CSC form for the LU basis.
//!
//! Unlike [`crate::sparse::csc::CscMatrix`], which stores only the lower
//! triangle of a *symmetric* matrix, an LU basis is general and every entry is
//! stored. Columns are compressed; row indices within a column are sorted
//! ascending. This module also builds the patterns the column ordering needs:
//! the transpose and the `AᵀA` (column-intersection) graph.

use crate::error::FeralError;
use crate::sparse::csc::CscPattern;

/// A general `m`×`m` sparse matrix in compressed-sparse-column form.
#[derive(Debug, Clone)]
pub struct SparseColMatrix {
    /// Dimension (square).
    pub m: usize,
    /// Column pointers, length `m + 1`.
    pub col_ptr: Vec<usize>,
    /// Row indices, length `nnz`, sorted ascending within each column.
    pub row_idx: Vec<usize>,
    /// Stored values, length `nnz`.
    pub values: Vec<f64>,
}

impl SparseColMatrix {
    /// Number of stored entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Build from `m` dense columns (compressing out exact zeros). `cols[j]` is
    /// column `j`, length `m`. The simplex entry point for dense-ish columns.
    pub fn from_dense_columns(m: usize, cols: &[Vec<f64>]) -> Result<Self, FeralError> {
        if cols.len() != m {
            return Err(FeralError::DimensionMismatch {
                expected: m,
                got: cols.len(),
            });
        }
        let mut col_ptr = Vec::with_capacity(m + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        for col in cols.iter() {
            if col.len() != m {
                return Err(FeralError::DimensionMismatch {
                    expected: m,
                    got: col.len(),
                });
            }
            for (i, &v) in col.iter().enumerate() {
                if v != 0.0 {
                    if !v.is_finite() {
                        return Err(FeralError::InvalidInput(
                            "LU basis column contains non-finite entries".to_string(),
                        ));
                    }
                    row_idx.push(i);
                    values.push(v);
                }
            }
            col_ptr.push(row_idx.len());
        }
        let mat = SparseColMatrix {
            m,
            col_ptr,
            row_idx,
            values,
        };
        mat.validate()?;
        Ok(mat)
    }

    /// Build from explicit sparse columns. `cols[j]` is a list of
    /// `(row, value)` pairs for column `j` (need not be sorted; duplicates are
    /// summed).
    pub fn from_sparse_columns(m: usize, cols: &[Vec<(usize, f64)>]) -> Result<Self, FeralError> {
        if cols.len() != m {
            return Err(FeralError::DimensionMismatch {
                expected: m,
                got: cols.len(),
            });
        }
        let mut col_ptr = Vec::with_capacity(m + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        let mut acc = vec![0.0_f64; m];
        let mut touched = Vec::new();
        for col in cols.iter() {
            for &(r, v) in col.iter() {
                if r >= m {
                    return Err(FeralError::InvalidInput(format!(
                        "row index {} out of range for dimension {}",
                        r, m
                    )));
                }
                if acc[r] == 0.0 {
                    touched.push(r);
                }
                acc[r] += v;
            }
            touched.sort_unstable();
            for &r in touched.iter() {
                if acc[r] != 0.0 {
                    row_idx.push(r);
                    values.push(acc[r]);
                }
                acc[r] = 0.0;
            }
            touched.clear();
            col_ptr.push(row_idx.len());
        }
        let mat = SparseColMatrix {
            m,
            col_ptr,
            row_idx,
            values,
        };
        mat.validate()?;
        Ok(mat)
    }

    /// Validate dimensions, sortedness, and finiteness.
    pub fn validate(&self) -> Result<(), FeralError> {
        if self.col_ptr.len() != self.m + 1 {
            return Err(FeralError::InvalidInput(
                "col_ptr length != m+1".to_string(),
            ));
        }
        if self.row_idx.len() != self.values.len() {
            return Err(FeralError::InvalidInput(
                "row_idx and values length mismatch".to_string(),
            ));
        }
        for j in 0..self.m {
            let (s, e) = (self.col_ptr[j], self.col_ptr[j + 1]);
            if e < s || e > self.row_idx.len() {
                return Err(FeralError::InvalidInput("malformed col_ptr".to_string()));
            }
            for k in s..e {
                if self.row_idx[k] >= self.m {
                    return Err(FeralError::InvalidInput(
                        "row index out of range".to_string(),
                    ));
                }
                if k > s && self.row_idx[k] <= self.row_idx[k - 1] {
                    return Err(FeralError::InvalidInput(
                        "row indices not strictly ascending".to_string(),
                    ));
                }
            }
        }
        if self.values.iter().any(|x| !x.is_finite()) {
            return Err(FeralError::InvalidInput("non-finite value".to_string()));
        }
        Ok(())
    }

    /// Column `j` as `(row_indices, values)` slices.
    #[inline]
    pub fn column(&self, j: usize) -> (&[usize], &[f64]) {
        let (s, e) = (self.col_ptr[j], self.col_ptr[j + 1]);
        (&self.row_idx[s..e], &self.values[s..e])
    }

    /// `y = A · x`.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for yi in y.iter_mut() {
            *yi = 0.0;
        }
        for (j, &xj) in x.iter().enumerate().take(self.m) {
            if xj == 0.0 {
                continue;
            }
            let (rows, vals) = self.column(j);
            for (&i, &v) in rows.iter().zip(vals.iter()) {
                y[i] += v * xj;
            }
        }
    }

    /// `y = Aᵀ · x`.
    pub fn matvec_transpose(&self, x: &[f64], y: &mut [f64]) {
        for (j, yj) in y.iter_mut().enumerate().take(self.m) {
            let (rows, vals) = self.column(j);
            let mut acc = 0.0;
            for (&i, &v) in rows.iter().zip(vals.iter()) {
                acc += v * x[i];
            }
            *yj = acc;
        }
    }

    /// The pattern of `AᵀA` as a full-symmetric [`CscPattern`] — the
    /// column-intersection graph used for fill-reducing column ordering. Two
    /// columns are adjacent iff they share a nonzero row. The diagonal is
    /// included.
    pub fn ata_pattern(&self) -> CscPattern {
        let m = self.m;
        // Row-wise lists: for each row, the columns with an entry there.
        let mut row_cols: Vec<Vec<usize>> = vec![Vec::new(); m];
        for j in 0..m {
            let (rows, _) = self.column(j);
            for &i in rows {
                row_cols[i].push(j);
            }
        }
        // Adjacency sets (use a marker array to dedup per column).
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); m];
        let mut mark = vec![usize::MAX; m];
        for j in 0..m {
            mark[j] = j; // include diagonal
            adj[j].push(j);
            let (rows, _) = self.column(j);
            for &i in rows {
                for &c in row_cols[i].iter() {
                    if mark[c] != j {
                        mark[c] = j;
                        adj[j].push(c);
                    }
                }
            }
            adj[j].sort_unstable();
        }
        let mut col_ptr = Vec::with_capacity(m + 1);
        let mut row_idx = Vec::new();
        col_ptr.push(0);
        for a in adj.iter() {
            row_idx.extend_from_slice(a);
            col_ptr.push(row_idx.len());
        }
        CscPattern {
            n: m,
            col_ptr,
            row_idx,
        }
    }
}
