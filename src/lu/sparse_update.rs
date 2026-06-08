//! Sparse rank-1 column-replacement update (product-form update of `U`).
//!
//! Replacing basis slot `leaving_slot` (at column position `q`) by a new column
//! `aₙₑw` yields `U' = U·F` with `F = I + (τ − e_q)e_qᵀ`, where
//! `τ = F_{prev}⁻¹ U⁻¹ L⁻¹ P aₙₑw` is the transformed spike — exactly the
//! `ftran` solve of `aₙₑw` in column-position space. Then `U'⁻¹ = F⁻¹ U⁻¹`, so
//! the update is recorded as the eta `(q, τ)` and applied after the `U`-solve
//! in `ftran` (transposed in `btran`). `τ[q]` is the update pivot: if it
//! vanishes the new basis is singular.
//!
//! This is a correct, genuinely sparse first-class rank-1 update with a refactor
//! budget. The full Forrest–Tomlin row-eta refinement (which keeps the eta
//! sparser than the dense `τ` stored here) is a documented optimization.

use super::sparse_factor::{EtaU, SparseLu};
use super::sparse_symbolic::SparseLuSymbolic;
use crate::error::FeralError;
use crate::lu::sparse_matrix::SparseColMatrix;

impl SparseLu {
    /// Number of rank-1 updates applied since the last factor/refactor.
    #[inline]
    pub fn updates_since_refactor(&self) -> usize {
        self.etas.len()
    }

    /// Replace basis slot `leaving_slot` with `entering_col` (`aₙₑw`).
    ///
    /// Returns [`FeralError::NeedsRefactor`] (leaving `self` unchanged) when the
    /// update or growth budget is exceeded, and [`FeralError::SingularBasis`]
    /// when the update pivot `τ[q]` vanishes.
    pub fn update(&mut self, leaving_slot: usize, entering_col: &[f64]) -> Result<(), FeralError> {
        let m = self.m;
        if entering_col.len() != m {
            return Err(FeralError::DimensionMismatch {
                expected: m,
                got: entering_col.len(),
            });
        }
        if leaving_slot >= m {
            return Err(FeralError::InvalidInput(format!(
                "leaving_slot {} out of range for basis dimension {}",
                leaving_slot, m
            )));
        }
        if self.updates_since_refactor() + 1 > self.params.max_updates {
            return Err(FeralError::NeedsRefactor);
        }

        // Scale the entering column into the factored frame Ã, then solve into
        // column-position space: τ = F⁻¹ U⁻¹ L⁻¹ P ãₙₑw. Identity scaling leaves
        // `entering_col` unchanged.
        let mut scaled = vec![0.0_f64; m];
        for (i, si) in scaled.iter_mut().enumerate() {
            *si = self.scale.d_row[i]
                * entering_col[self.scale.rperm[i]]
                * self.scale.d_col[leaving_slot];
        }
        let mut tau = vec![0.0_f64; m];
        self.solve_colspace(&scaled, &mut tau);

        let q = self.qcol_inv[leaving_slot];
        let tq = tau[q];
        if tq.abs() <= self.params.zero_pivot_tol {
            return Err(FeralError::SingularBasis { column: q });
        }
        let new_growth = self.growth.max(1.0 / tq.abs());
        if new_growth > self.params.max_growth {
            return Err(FeralError::NeedsRefactor);
        }

        self.etas.push(EtaU { q, tau });
        self.growth = new_growth;
        Ok(())
    }

    /// Discard all pending updates and re-factor from scratch on `a`, reusing
    /// the column ordering in `symbolic`.
    pub fn refactor(
        &mut self,
        a: &SparseColMatrix,
        symbolic: &SparseLuSymbolic,
    ) -> Result<(), FeralError> {
        *self = SparseLu::factor(a, symbolic, self.params.clone())?;
        Ok(())
    }
}
