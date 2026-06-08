//! Sparse rank-1 column-replacement update — Forrest–Tomlin / Bartels–Golub–Reid.
//!
//! Replacing basis slot `leaving_slot` (column position `r`) by a new column
//! folds the spike `ρ = G⁻¹ L⁻¹ P aₙₑw` into `U`'s column `r`, then
//! re-triangularizes the resulting *bump* (`U` is upper-triangular except a
//! spike in column `r`) by **sparse Gaussian elimination with partial
//! pivoting**. Partial pivoting is what makes this correct on a sparse `U`: the
//! diagonal pivot of the spiked column may be zero, but a sub-diagonal spike
//! entry provides a nonzero pivot via a row interchange.
//!
//! The elimination's elementary operations (swaps + `row -= mult·row`) are
//! recorded as a [`FtEta`](super::sparse_factor::FtEta) and replayed on the
//! solve vector (between the `L`-solve and `U`-solve in `ftran`), so the base
//! `L` is never touched and the eta is `O(bump)` — no dense `τ`, no `O(k·n)`
//! warm-solve growth. `U` itself is updated in place.

use super::sparse_factor::{FtEta, FtOp, SparseLu};
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
    /// when the bump has no acceptable pivot (the new basis is singular).
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

        // Spike ρ = G⁻¹ L⁻¹ P (scaled entering column), in U-column space.
        let mut scaled = vec![0.0_f64; m];
        for (i, si) in scaled.iter_mut().enumerate() {
            *si = self.scale.d_row[i]
                * entering_col[self.scale.rperm[i]]
                * self.scale.d_col[leaving_slot];
        }
        let mut rho = vec![0.0_f64; m];
        self.spike_space(&scaled, &mut rho);

        let r = self.qcol_inv[leaving_slot];
        // Bump high-water mark: the highest row of the spike at/below the
        // diagonal. No entry at rows >= r means column r has no pivot.
        let h = match (r..m).rev().find(|&i| rho[i] != 0.0) {
            Some(h) => h,
            None => return Err(FeralError::SingularBasis { column: r }),
        };

        let ztol = self.params.zero_pivot_tol;
        let max_growth = self.params.max_growth;

        // Work on a clone of U so a failure leaves `self` unchanged.
        let mut u = self.u_rows.clone();
        set_column(&mut u, m, r, &rho);

        let mut ops: Vec<FtOp> = Vec::new();
        let mut growth = self.growth;

        for k in r..=h {
            // Partial pivot: among rows [k, h] with a column-k entry, take the
            // largest in magnitude.
            let mut pivot_row = k;
            let mut pivot_abs = 0.0_f64;
            for (i, urow) in u.iter().enumerate().take(h + 1).skip(k) {
                if let Some(v) = get_col(urow, k) {
                    if v.abs() > pivot_abs {
                        pivot_abs = v.abs();
                        pivot_row = i;
                    }
                }
            }
            if pivot_abs <= ztol {
                return Err(FeralError::SingularBasis { column: k });
            }
            if pivot_row != k {
                u.swap(k, pivot_row);
                ops.push(FtOp::Swap(k, pivot_row));
            }
            let pivot_data = u[k].clone();
            let pivot = get_col(&pivot_data, k).unwrap_or(0.0);

            // Eliminate the column-k entry from every lower bump row that has one.
            #[allow(clippy::needless_range_loop)]
            for i in k + 1..=h {
                if let Some(vik) = get_col(&u[i], k) {
                    let mult = vik / pivot;
                    growth = growth.max(mult.abs());
                    if growth > max_growth {
                        return Err(FeralError::NeedsRefactor);
                    }
                    u[i] = row_sub(&u[i], &pivot_data, mult, k);
                    ops.push(FtOp::Axpy {
                        target: i,
                        src: k,
                        mult,
                    });
                }
            }
        }

        // Commit.
        self.u_rows = u;
        self.etas.push(FtEta { ops });
        self.growth = growth;
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

/// Look up the value at column `c` in a column-sorted sparse row.
fn get_col(row: &[(usize, f64)], c: usize) -> Option<f64> {
    row.binary_search_by_key(&c, |&(col, _)| col)
        .ok()
        .map(|pos| row[pos].1)
}

/// Overwrite column `r` of `U` (per-row storage) with the dense spike `rho`.
fn set_column(u: &mut [Vec<(usize, f64)>], m: usize, r: usize, rho: &[f64]) {
    for (i, row) in u.iter_mut().enumerate().take(m) {
        let v = rho[i];
        match row.binary_search_by_key(&r, |&(col, _)| col) {
            Ok(pos) => {
                if v != 0.0 {
                    row[pos].1 = v;
                } else {
                    row.remove(pos);
                }
            }
            Err(pos) => {
                if v != 0.0 {
                    row.insert(pos, (r, v));
                }
            }
        }
    }
}

/// `dst − mult·src` over two column-sorted sparse rows, dropping the eliminated
/// column `drop_col` and any exact zeros. Result stays column-sorted.
fn row_sub(
    dst: &[(usize, f64)],
    src: &[(usize, f64)],
    mult: f64,
    drop_col: usize,
) -> Vec<(usize, f64)> {
    let mut out = Vec::with_capacity(dst.len() + src.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < dst.len() && j < src.len() {
        let (dc, dv) = dst[i];
        let (sc, sv) = src[j];
        if dc < sc {
            if dc != drop_col {
                out.push((dc, dv));
            }
            i += 1;
        } else if dc > sc {
            let v = -mult * sv;
            if sc != drop_col && v != 0.0 {
                out.push((sc, v));
            }
            j += 1;
        } else {
            let v = dv - mult * sv;
            if dc != drop_col && v != 0.0 {
                out.push((dc, v));
            }
            i += 1;
            j += 1;
        }
    }
    while i < dst.len() {
        let (dc, dv) = dst[i];
        if dc != drop_col {
            out.push((dc, dv));
        }
        i += 1;
    }
    while j < src.len() {
        let (sc, sv) = src[j];
        let v = -mult * sv;
        if sc != drop_col && v != 0.0 {
            out.push((sc, v));
        }
        j += 1;
    }
    out
}
