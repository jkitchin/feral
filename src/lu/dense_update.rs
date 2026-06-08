//! Dense rank-1 column-replacement update (Bartels–Golub).
//!
//! Replacing basis slot `leaving_slot` by a new column whose spike is
//! `s = L⁻¹ P aₙₑw` (from [`DenseLu::ftran_partial`]) maintains the invariant
//! `P B Q = L U` in three steps.
//!
//! Step one overwrites column `q = qcol_inv[leaving_slot]` of `U` with the
//! spike `s`; `U` is then upper triangular except column `q`, which has a spike
//! below the diagonal. Step two cyclically shifts columns `q..m-1` of `U` left
//! by one and moves the spike column to position `m-1` (updating `Q`), turning
//! `U` into upper-Hessenberg with a single subdiagonal on positions `q..m-2`.
//! Step three eliminates that subdiagonal with a Gauss sweep (no in-bump
//! pivoting; the growth monitor and `NeedsRefactor` handle instability): each
//! elimination is a row operation on `U` and the corresponding column operation
//! on `L`, so `L` stays unit lower triangular and `U` upper triangular.
//!
//! The work is done on clones of `L`, `U`, and `Q`, committed only on success,
//! so a `NeedsRefactor` return leaves `self` unchanged and recoverable.

use super::dense_factor::DenseLu;
use crate::error::FeralError;

impl DenseLu {
    /// Replace basis slot `leaving_slot` with `entering_col` (the new basis
    /// column `aₙₑw`). The spike `L⁻¹ P aₙₑw` is computed internally, then folded
    /// into the factorization. On success the factors reflect the new basis.
    ///
    /// Returns [`FeralError::NeedsRefactor`] (leaving `self` unchanged) when the
    /// update budget (`max_updates`) or growth budget (`max_growth`) is
    /// exceeded, and [`FeralError::SingularBasis`] when a bump pivot vanishes.
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
        // Update-count budget (checked before doing any work).
        if self.updates_since_refactor + 1 > self.params.max_updates {
            return Err(FeralError::NeedsRefactor);
        }

        // Scale the entering column into the factored frame Ã, then form the
        // spike L⁻¹ P ãₙₑw (no factor mutation). With identity scaling this is
        // just `entering_col`.
        let mut spike = vec![0.0; m];
        for (i, si) in spike.iter_mut().enumerate() {
            *si = self.scale.d_row[i]
                * entering_col[self.scale.rperm[i]]
                * self.scale.d_col[leaving_slot];
        }
        self.ftran_partial(&mut spike)?;

        let q = self.qcol_inv[leaving_slot];
        let ztol = self.params.zero_pivot_tol;
        let max_growth = self.params.max_growth;

        // Work on clones; commit only on success.
        let mut u = self.u.clone();
        let mut l = self.l.clone();
        let mut qcol = self.qcol.clone();

        // 1. Overwrite column q of U with the spike.
        for i in 0..m {
            u[i + q * m] = spike[i];
        }

        // 2. Cyclic column shift q..m-1: spike column moves to position m-1.
        cyclic_shift_columns(&mut u, m, q);
        let leaving = qcol[q];
        for j in q..m - 1 {
            qcol[j] = qcol[j + 1];
        }
        qcol[m - 1] = leaving;

        // 3. Eliminate the Hessenberg subdiagonal on positions q..m-2.
        let mut growth = self.growth;
        for k in q..m.saturating_sub(1) {
            let piv = u[k + k * m];
            if piv.abs() <= ztol {
                return Err(FeralError::NeedsRefactor);
            }
            let sub = u[k + 1 + k * m];
            if sub == 0.0 {
                continue;
            }
            let mult = sub / piv;
            growth = growth.max(mult.abs());
            if growth > max_growth {
                return Err(FeralError::NeedsRefactor);
            }
            // Row op on U: row_{k+1} -= mult · row_k, columns k..m-1.
            for j in k..m {
                u[k + 1 + j * m] -= mult * u[k + j * m];
            }
            u[k + 1 + k * m] = 0.0; // enforce exact zero
                                    // Column op on L: col_k += mult · col_{k+1} (keeps L unit lower).
            for i in 0..m {
                l[i + k * m] += mult * l[i + (k + 1) * m];
            }
        }

        // Commit.
        self.u = u;
        self.l = l;
        self.qcol = qcol;
        for (k, &slot) in self.qcol.iter().enumerate() {
            self.qcol_inv[slot] = k;
        }
        self.growth = growth;
        self.updates_since_refactor += 1;
        Ok(())
    }
}

/// Cyclically shift columns `q..m-1` of a column-major `m`×`m` buffer left by
/// one, moving column `q` to position `m-1`.
fn cyclic_shift_columns(buf: &mut [f64], m: usize, q: usize) {
    if q + 1 >= m {
        return;
    }
    let mut saved = vec![0.0; m];
    saved.copy_from_slice(&buf[q * m..q * m + m]);
    for j in q..m - 1 {
        let (dst, src) = (j * m, (j + 1) * m);
        buf.copy_within(src..src + m, dst);
    }
    let last = (m - 1) * m;
    buf[last..last + m].copy_from_slice(&saved);
}
