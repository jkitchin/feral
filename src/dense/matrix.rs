use crate::error::FeralError;

/// Symmetric matrix stored as full n×n column-major. Only the lower triangle
/// is meaningful; the strict upper triangle is ignored on input.
/// Entry (i, j) is at index j*n + i. Size: n*n f64 values.
pub struct SymmetricMatrix {
    pub n: usize,
    pub data: Vec<f64>,
}

impl SymmetricMatrix {
    /// Create a new n×n symmetric matrix initialized to zero.
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * n],
        }
    }

    /// Create a symmetric matrix from a flat column-major vector.
    /// The lower triangle is authoritative; the upper triangle is ignored.
    pub fn from_column_major(n: usize, data: Vec<f64>) -> Result<Self, FeralError> {
        if data.len() != n * n {
            return Err(FeralError::InvalidInput(format!(
                "matrix data length {} != expected {} for n={}",
                data.len(),
                n * n,
                n
            )));
        }
        Ok(Self { n, data })
    }

    /// Create a symmetric matrix from a dense 2D lower-triangular representation.
    /// `entries` provides (i, j, value) triples where i >= j.
    pub fn from_lower_triangle(n: usize, entries: &[(usize, usize, f64)]) -> Self {
        let mut mat = Self::zeros(n);
        for &(i, j, v) in entries {
            mat.set(i, j, v);
        }
        mat
    }

    /// Get entry (i, j), reading from lower triangle.
    /// For i >= j, returns data[j*n + i].
    /// For i < j, returns data[i*n + j] (symmetric).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i >= j {
            self.data[j * self.n + i]
        } else {
            self.data[i * self.n + j]
        }
    }

    /// Set entry (i, j) in the lower triangle.
    /// Also sets (j, i) for symmetry in the stored data.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        if i >= j {
            self.data[j * self.n + i] = val;
        } else {
            self.data[i * self.n + j] = val;
        }
    }

    /// Validate the matrix for factorization input.
    /// Checks: n > 0, data length, no NaN/Inf in lower triangle.
    pub fn validate(&self) -> Result<(), FeralError> {
        if self.n == 0 {
            return Err(FeralError::InvalidInput(
                "matrix dimension is zero".to_string(),
            ));
        }
        if self.data.len() != self.n * self.n {
            return Err(FeralError::InvalidInput(format!(
                "matrix data length {} != expected {} for n={}",
                self.data.len(),
                self.n * self.n,
                self.n
            )));
        }
        // Check lower triangle for NaN/Inf
        for j in 0..self.n {
            for i in j..self.n {
                let val = self.data[j * self.n + i];
                if val.is_nan() || val.is_infinite() {
                    return Err(FeralError::InvalidInput(format!(
                        "matrix contains NaN or Inf at index ({},{})",
                        i, j
                    )));
                }
            }
        }
        Ok(())
    }

    /// Symmetric matrix-vector product: y = A * x.
    /// Uses only the lower triangle.
    pub fn symv(&self, x: &[f64], y: &mut [f64]) {
        let n = self.n;
        for yi in y.iter_mut().take(n) {
            *yi = 0.0;
        }
        for j in 0..n {
            // Diagonal
            y[j] += self.data[j * n + j] * x[j];
            // Off-diagonal (lower triangle)
            for i in (j + 1)..n {
                let a_ij = self.data[j * n + i];
                y[i] += a_ij * x[j];
                y[j] += a_ij * x[i];
            }
        }
    }
}
