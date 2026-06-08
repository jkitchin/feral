//! Symbolic analysis for the sparse LU: the fill-reducing column ordering `Q`.
//!
//! Reuses feral's in-tree AMD (`feral_amd`) on the `AᵀA` (column-intersection)
//! pattern — a stand-in for COLAMD that needs no new ordering algorithm. The
//! resulting permutation is the reusable symbolic handle: across numerically
//! different but structurally identical bases, only the numeric factor is
//! recomputed.

use super::sparse_matrix::SparseColMatrix;
use crate::error::FeralError;

/// Reusable symbolic factorization: the column permutation `Q`.
#[derive(Debug, Clone)]
pub struct SparseLuSymbolic {
    /// Dimension.
    pub m: usize,
    /// Column order: factorization column position `k` is original column
    /// `qcol[k]`.
    pub qcol: Vec<usize>,
    /// Inverse: `qcol_inv[original_col] = column_position`.
    pub qcol_inv: Vec<usize>,
}

impl SparseLuSymbolic {
    /// Compute the column ordering via AMD on the `AᵀA` pattern.
    pub fn analyze(a: &SparseColMatrix) -> Result<Self, FeralError> {
        let m = a.m;
        if m == 0 {
            return Ok(SparseLuSymbolic {
                m,
                qcol: Vec::new(),
                qcol_inv: Vec::new(),
            });
        }
        let pat = a.ata_pattern();
        let col_ptr: Vec<i32> = pat
            .col_ptr
            .iter()
            .map(|&x| i32::try_from(x))
            .collect::<Result<_, _>>()
            .map_err(|_| FeralError::InvalidInput("AᵀA pattern index overflow".to_string()))?;
        let row_idx: Vec<i32> = pat
            .row_idx
            .iter()
            .map(|&x| i32::try_from(x))
            .collect::<Result<_, _>>()
            .map_err(|_| FeralError::InvalidInput("AᵀA pattern index overflow".to_string()))?;
        let cpat = feral_ordering_core::CscPattern::new(m, &col_ptr, &row_idx)
            .ok_or_else(|| FeralError::InvalidInput("malformed AᵀA pattern".to_string()))?;
        let perm = feral_amd::amd_order(&cpat)
            .map_err(|e| FeralError::InvalidInput(format!("AMD ordering failed: {:?}", e)))?;
        let qcol: Vec<usize> = perm.iter().map(|&x| x as usize).collect();
        let mut qcol_inv = vec![0usize; m];
        for (k, &q) in qcol.iter().enumerate() {
            qcol_inv[q] = k;
        }
        Ok(SparseLuSymbolic { m, qcol, qcol_inv })
    }

    /// Identity column ordering (natural order) — for testing and as a fallback.
    pub fn natural(m: usize) -> Self {
        SparseLuSymbolic {
            m,
            qcol: (0..m).collect(),
            qcol_inv: (0..m).collect(),
        }
    }
}
