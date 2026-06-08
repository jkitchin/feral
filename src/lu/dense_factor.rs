//! Dense unsymmetric LU factorization with threshold partial pivoting.
//!
//! Maintains the invariant `P B Q = L U`, where `P` is the row permutation
//! from partial pivoting, `Q` is a column permutation (identity after a fresh
//! factorization; a rank-1 update may permute columns), `L` is unit lower
//! triangular, and `U` is upper triangular. `L` and `U` are stored as separate
//! dense `m`×`m` column-major buffers so that the rank-1
//! column-replacement update ([`super::dense_update`]) has a clean home for the
//! spike column and the Hessenberg bump it eliminates.
//!
//! The factorization kernel itself works in a packed buffer (the classic
//! right-looking LAPACK layout) and is split into `L`/`U` at the end.

use super::{LuParams, LuSingularAction};
use crate::error::FeralError;

/// Dense LU factorization of a square basis, with rank-1 update support.
#[derive(Debug, Clone)]
pub struct DenseLu {
    pub(super) m: usize,
    /// Unit lower triangular `L`, column-major `m*m` (explicit unit diagonal).
    pub(super) l: Vec<f64>,
    /// Upper triangular `U`, column-major `m*m` (strict lower part is zero).
    pub(super) u: Vec<f64>,
    /// Row permutation: `perm[k]` is the original row now in pivot row `k`
    /// (so `(P a)[k] = a[perm[k]]`).
    pub(super) perm: Vec<usize>,
    /// Inverse of `perm`: `perm_inv[orig_row] = pivot_position`.
    pub(super) perm_inv: Vec<usize>,
    /// Column permutation: `qcol[k]` is the basis slot in column position `k`
    /// (so the factorization's column `k` is basis column `qcol[k]`).
    pub(super) qcol: Vec<usize>,
    /// Inverse of `qcol`: `qcol_inv[slot] = column_position`.
    pub(super) qcol_inv: Vec<usize>,
    /// Updates applied since the last `factor`/`refactor`.
    pub(super) updates_since_refactor: usize,
    /// Running growth monitor; tripping `params.max_growth` forces a refactor.
    pub(super) growth: f64,
    pub(super) params: LuParams,
    /// Reusable length-`m` scratch buffer (no per-call allocation in solves).
    pub(super) scratch_a: Vec<f64>,
}

impl DenseLu {
    /// Factor the `m`×`m` basis given by its `m` columns (`cols[j]` is column
    /// `j`, length `m`). Computes `P B = L U` with threshold partial pivoting.
    pub fn factor(cols: &[Vec<f64>], m: usize, params: LuParams) -> Result<Self, FeralError> {
        let mut packed = vec![0.0; m * m];
        copy_columns_into(&mut packed, cols, m)?;
        let mut perm: Vec<usize> = (0..m).collect();
        factorize_packed(&mut packed, &mut perm, m, &params)?;
        let (l, u) = split_packed(&packed, m);
        let mut perm_inv = vec![0usize; m];
        for (k, &p) in perm.iter().enumerate() {
            perm_inv[p] = k;
        }
        Ok(DenseLu {
            m,
            l,
            u,
            perm,
            perm_inv,
            qcol: (0..m).collect(),
            qcol_inv: (0..m).collect(),
            updates_since_refactor: 0,
            growth: 1.0,
            params,
            scratch_a: vec![0.0; m],
        })
    }

    /// Discard all pending updates and re-factor from scratch on fresh columns.
    pub fn refactor(&mut self, cols: &[Vec<f64>]) -> Result<(), FeralError> {
        let m = self.m;
        let mut packed = vec![0.0; m * m];
        copy_columns_into(&mut packed, cols, m)?;
        for (k, p) in self.perm.iter_mut().enumerate() {
            *p = k;
        }
        factorize_packed(&mut packed, &mut self.perm, m, &self.params)?;
        let (l, u) = split_packed(&packed, m);
        self.l = l;
        self.u = u;
        for (k, &p) in self.perm.iter().enumerate() {
            self.perm_inv[p] = k;
        }
        for (k, (q, qi)) in self
            .qcol
            .iter_mut()
            .zip(self.qcol_inv.iter_mut())
            .enumerate()
        {
            *q = k;
            *qi = k;
        }
        self.updates_since_refactor = 0;
        self.growth = 1.0;
        Ok(())
    }

    /// Basis dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.m
    }

    /// Number of rank-1 updates applied since the last factor/refactor.
    #[inline]
    pub fn updates_since_refactor(&self) -> usize {
        self.updates_since_refactor
    }

    /// The row permutation `perm` (`perm[k]` = original row in pivot row `k`).
    #[inline]
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// The column permutation `qcol` (`qcol[k]` = basis slot in column `k`).
    #[inline]
    pub fn qcol(&self) -> &[usize] {
        &self.qcol
    }

    /// `L[i,j]`: unit lower triangular.
    #[inline]
    pub fn l(&self, i: usize, j: usize) -> f64 {
        self.l[i + j * self.m]
    }

    /// `U[i,j]`: upper triangular.
    #[inline]
    pub fn u(&self, i: usize, j: usize) -> f64 {
        self.u[i + j * self.m]
    }
}

/// Copy `m` columns (each length `m`) into a packed column-major buffer.
fn copy_columns_into(buf: &mut [f64], cols: &[Vec<f64>], m: usize) -> Result<(), FeralError> {
    if cols.len() != m {
        return Err(FeralError::DimensionMismatch {
            expected: m,
            got: cols.len(),
        });
    }
    for (j, col) in cols.iter().enumerate() {
        if col.len() != m {
            return Err(FeralError::DimensionMismatch {
                expected: m,
                got: col.len(),
            });
        }
        buf[j * m..j * m + m].copy_from_slice(col);
        if col.iter().any(|x| !x.is_finite()) {
            return Err(FeralError::InvalidInput(
                "LU basis column contains non-finite entries".to_string(),
            ));
        }
    }
    Ok(())
}

/// Split a packed LU buffer into explicit `L` (unit lower) and `U` (upper).
fn split_packed(packed: &[f64], m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut l = vec![0.0; m * m];
    let mut u = vec![0.0; m * m];
    for j in 0..m {
        for i in 0..m {
            let v = packed[i + j * m];
            if i > j {
                l[i + j * m] = v;
            } else {
                u[i + j * m] = v;
            }
        }
        l[j + j * m] = 1.0;
    }
    (l, u)
}

/// Right-looking outer-product LU with threshold partial pivoting, in place on
/// the packed buffer. `perm` starts as identity and accumulates row swaps.
fn factorize_packed(
    packed: &mut [f64],
    perm: &mut [usize],
    m: usize,
    params: &LuParams,
) -> Result<(), FeralError> {
    let u = params.pivot_threshold;
    let ztol = params.zero_pivot_tol;
    for k in 0..m {
        // Pivot search: max-magnitude entry in column k over rows k..m.
        let mut amax = 0.0_f64;
        let mut argmax = k;
        for i in k..m {
            let v = packed[i + k * m].abs();
            if v > amax {
                amax = v;
                argmax = i;
            }
        }
        // Threshold partial pivoting: keep the diagonal if it is within
        // threshold of the column max (and nonzero), else take the row max.
        let diag = packed[k + k * m].abs();
        let pivot_row = if diag >= u * amax && diag > ztol {
            k
        } else {
            argmax
        };

        if pivot_row != k {
            swap_rows(packed, k, pivot_row, m);
            perm.swap(k, pivot_row);
        }

        // Singularity / perturbation.
        let mut piv = packed[k + k * m];
        if piv.abs() <= ztol {
            match params.on_singular {
                LuSingularAction::Fail => {
                    return Err(FeralError::SingularBasis { column: k });
                }
                LuSingularAction::PerturbToEps { abs_floor } => {
                    let s = if piv < 0.0 { -1.0 } else { 1.0 };
                    piv = s * abs_floor.max(piv.abs());
                    packed[k + k * m] = piv;
                }
            }
        }

        // Multipliers L(k+1:, k) = A(k+1:, k) / piv.
        let inv = 1.0 / piv;
        for i in k + 1..m {
            packed[i + k * m] *= inv;
        }
        // Rank-1 trailing update A(k+1:, k+1:) -= L(k+1:, k) · U(k, k+1:).
        for j in k + 1..m {
            let ukj = packed[k + j * m];
            if ukj != 0.0 {
                for i in k + 1..m {
                    packed[i + j * m] -= packed[i + k * m] * ukj;
                }
            }
        }
    }
    Ok(())
}

/// Swap rows `a` and `b` across all `m` columns of the packed buffer.
fn swap_rows(buf: &mut [f64], a: usize, b: usize, m: usize) {
    for c in 0..m {
        buf.swap(a + c * m, b + c * m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Columns (column-major) from rows-of-the-matrix, for readable tests.
    fn cols_from_rows(rows: &[&[f64]]) -> (Vec<Vec<f64>>, usize) {
        let m = rows.len();
        let mut cols = vec![vec![0.0; m]; m];
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                cols[j][i] = v;
            }
        }
        (cols, m)
    }

    /// max_{i,j} | (P B)[i,j] - (L U)[i,j] |, the reconstruction residual.
    /// (Q = I after a fresh factorization.)
    fn reconstruction_residual(lu: &DenseLu, rows: &[&[f64]]) -> f64 {
        let m = lu.m;
        let mut worst = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                let pb = rows[lu.perm[i]][j];
                let mut prod = 0.0;
                for k in 0..m {
                    prod += lu.l(i, k) * lu.u(k, j);
                }
                worst = worst.max((pb - prod).abs());
            }
        }
        worst
    }

    #[test]
    fn factor_2x2_swap_exact() {
        // B = [[0,2],[3,4]]; partial pivoting swaps rows: P=[1,0], L=I,
        // U=[[3,4],[0,2]]. Hand-computed.
        let (cols, m) = cols_from_rows(&[&[0.0, 2.0], &[3.0, 4.0]]);
        let lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
        assert_eq!(lu.perm, vec![1, 0]);
        assert!((lu.u(0, 0) - 3.0).abs() < 1e-14);
        assert!((lu.u(0, 1) - 4.0).abs() < 1e-14);
        assert!((lu.u(1, 1) - 2.0).abs() < 1e-14);
        assert!((lu.l(1, 0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn factor_3x3_reconstruction() {
        let rows: [&[f64]; 3] = [&[2.0, 1.0, 1.0], &[4.0, 3.0, 3.0], &[8.0, 7.0, 9.0]];
        let (cols, m) = cols_from_rows(&rows);
        let lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
        assert!(reconstruction_residual(&lu, &rows) < 1e-12);
    }

    #[test]
    fn factor_random_reconstruction() {
        let m = 7;
        let mut cols = vec![vec![0.0; m]; m];
        let mut state = 0x1234_5678_u64;
        for j in 0..m {
            for i in 0..m {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
                cols[j][i] = r;
            }
            cols[j][j] += 5.0;
        }
        let rows_owned: Vec<Vec<f64>> = (0..m)
            .map(|i| (0..m).map(|j| cols[j][i]).collect())
            .collect();
        let rows: Vec<&[f64]> = rows_owned.iter().map(|r| r.as_slice()).collect();
        let lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
        assert!(reconstruction_residual(&lu, &rows) < 1e-10);
    }

    #[test]
    fn factor_singular_repeated_columns_fails() {
        let (cols, m) = cols_from_rows(&[&[1.0, 1.0], &[2.0, 2.0]]);
        let err = DenseLu::factor(&cols, m, LuParams::default());
        assert!(matches!(err, Err(FeralError::SingularBasis { .. })));
    }

    #[test]
    fn factor_singular_perturb_succeeds() {
        let (cols, m) = cols_from_rows(&[&[1.0, 1.0], &[2.0, 2.0]]);
        let params = LuParams {
            on_singular: LuSingularAction::PerturbToEps { abs_floor: 1e-10 },
            ..LuParams::default()
        };
        let lu = DenseLu::factor(&cols, m, params).expect("perturbed factor");
        assert!(lu.u(1, 1).abs() >= 1e-10);
    }

    #[test]
    fn refactor_resets_state() {
        let (cols, m) = cols_from_rows(&[&[2.0, 1.0], &[1.0, 3.0]]);
        let mut lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
        let (cols2, _) = cols_from_rows(&[&[4.0, 0.0], &[1.0, 5.0]]);
        lu.refactor(&cols2).expect("refactor");
        assert_eq!(lu.updates_since_refactor(), 0);
        let rows2: [&[f64]; 2] = [&[4.0, 0.0], &[1.0, 5.0]];
        assert!(reconstruction_residual(&lu, &rows2) < 1e-12);
    }
}
