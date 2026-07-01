//! General (unsymmetric) dense square matrix for the LU basis engine.
//!
//! Unlike [`crate::dense::matrix::SymmetricMatrix`], which stores only the
//! lower triangle of a symmetric matrix, an LU basis is a general square
//! matrix and every entry is stored. Layout is column-major: entry `(i, j)`
//! lives at `data[j * m + i]`. The simplex hands us the basic columns, so
//! [`GeneralMatrix::from_columns`] is the primary constructor.

use crate::error::FeralError;

/// A general `m`×`m` dense matrix in column-major storage.
#[derive(Debug, Clone)]
pub struct GeneralMatrix {
    /// Dimension (the matrix is square).
    pub m: usize,
    /// Column-major entries, length `m * m`. Entry `(i, j)` at `data[j*m + i]`.
    pub data: Vec<f64>,
}

impl GeneralMatrix {
    /// All-zeros `m`×`m` matrix.
    pub fn zeros(m: usize) -> Self {
        GeneralMatrix {
            m,
            data: vec![0.0; m * m],
        }
    }

    /// Build from `m` basic columns. `cols[j]` is column `j`, each of length
    /// `m`. This is the simplex entry point.
    pub fn from_columns(m: usize, cols: &[Vec<f64>]) -> Result<Self, FeralError> {
        if cols.len() != m {
            return Err(FeralError::DimensionMismatch {
                expected: m,
                got: cols.len(),
            });
        }
        let mut data = vec![0.0; m * m];
        for (j, col) in cols.iter().enumerate() {
            if col.len() != m {
                return Err(FeralError::DimensionMismatch {
                    expected: m,
                    got: col.len(),
                });
            }
            data[j * m..j * m + m].copy_from_slice(col);
        }
        let mat = GeneralMatrix { m, data };
        mat.validate()?;
        Ok(mat)
    }

    /// Build from a flat column-major buffer of length `m * m`.
    pub fn from_column_major(m: usize, data: Vec<f64>) -> Result<Self, FeralError> {
        if data.len() != m * m {
            return Err(FeralError::DimensionMismatch {
                expected: m * m,
                got: data.len(),
            });
        }
        let mat = GeneralMatrix { m, data };
        mat.validate()?;
        Ok(mat)
    }

    /// Read entry `(i, j)`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[j * self.m + i]
    }

    /// Write entry `(i, j)`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[j * self.m + i] = v;
    }

    /// Column `j` as a slice of length `m`.
    #[inline]
    pub fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.m..j * self.m + self.m]
    }

    /// Column `j` as a mutable slice of length `m`.
    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        &mut self.data[j * self.m..j * self.m + self.m]
    }

    /// Validate dimensions and finiteness.
    pub fn validate(&self) -> Result<(), FeralError> {
        if self.data.len() != self.m * self.m {
            return Err(FeralError::InvalidInput(format!(
                "GeneralMatrix data length {} != m*m = {}",
                self.data.len(),
                self.m * self.m
            )));
        }
        if self.data.iter().any(|x| !x.is_finite()) {
            return Err(FeralError::InvalidInput(
                "GeneralMatrix contains non-finite entries".to_string(),
            ));
        }
        Ok(())
    }

    /// `‖A‖₁ = max_j Σ_i |A_ij|`, the maximum absolute column sum.
    pub fn one_norm(&self) -> f64 {
        let mut best = 0.0f64;
        for j in 0..self.m {
            let s: f64 = self.col(j).iter().map(|v| v.abs()).sum();
            if s > best {
                best = s;
            }
        }
        best
    }

    /// `y = A · x`. `x` and `y` must have length `m`.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for yi in y.iter_mut() {
            *yi = 0.0;
        }
        for (j, &xj) in x.iter().enumerate().take(self.m) {
            if xj == 0.0 {
                continue;
            }
            let col = self.col(j);
            for (yi, &cij) in y.iter_mut().zip(col.iter()) {
                *yi += cij * xj;
            }
        }
    }

    /// `y = Aᵀ · x`. `x` and `y` must have length `m`.
    pub fn matvec_transpose(&self, x: &[f64], y: &mut [f64]) {
        for (j, yj) in y.iter_mut().enumerate().take(self.m) {
            let col = self.col(j);
            let mut acc = 0.0;
            for (&cij, &xi) in col.iter().zip(x.iter()) {
                acc += cij * xi;
            }
            *yj = acc;
        }
    }
}
