//! Sparse unsymmetric LU factorization (left-looking, Gilbert–Peierls style).
//!
//! Factors `P A Q = L U` with threshold partial pivoting. `Q` is the
//! fill-reducing column ordering from [`SparseLuSymbolic`]; `P` is the row
//! permutation from pivoting. `L` is unit lower triangular (strict-lower stored
//! CSC, unit diagonal implicit) and `U` is upper triangular (strict-upper
//! stored CSC plus an explicit diagonal), both in pivot-position coordinates.
//!
//! Each column `k` (processing original column `qcol[k]`) is computed by a
//! forward substitution `L w = A(:,qcol[k])` in pivot order; entries landing on
//! already-pivoted rows form `U(:,k)`, and the largest remaining entry is the
//! new pivot. This is correct but not yet output-sensitive — the depth-first
//! symbolic reach that makes it O(flops) is a documented follow-up.

use super::scaling::{compute_lu_scale, LuScale};
use super::sparse_symbolic::SparseLuSymbolic;
use super::{LuParams, LuScaling, LuSingularAction};
use crate::error::FeralError;
use crate::lu::sparse_matrix::SparseColMatrix;

/// One product-form update of the `U` factor (issue #81 sparse rank-1 update).
///
/// Replacing column `q` of the current factorization by a new column yields
/// `U' = U·F` with `F = I + (τ − e_q)e_qᵀ`, where `τ` is the transformed spike.
/// We store `(q, τ)` and apply `F⁻¹` after the `U`-solve in `ftran` (and its
/// transpose in `btran`). `τ[q]` is the update's pivot / stability monitor.
#[derive(Debug, Clone)]
pub(super) struct EtaU {
    pub q: usize,
    pub tau: Vec<f64>,
}

/// Sparse LU factorization of a square basis.
#[derive(Debug, Clone)]
pub struct SparseLu {
    pub(super) m: usize,
    // L: unit lower triangular, strict-lower stored CSC, pivot-position rows.
    pub(super) l_col_ptr: Vec<usize>,
    pub(super) l_row_idx: Vec<usize>,
    pub(super) l_val: Vec<f64>,
    // U: strict-upper stored CSC (pivot-position rows) + explicit diagonal.
    pub(super) u_col_ptr: Vec<usize>,
    pub(super) u_row_idx: Vec<usize>,
    pub(super) u_val: Vec<f64>,
    pub(super) u_diag: Vec<f64>,
    /// `perm[k]` = original row in pivot position `k` (`(P a)[k] = a[perm[k]]`).
    pub(super) perm: Vec<usize>,
    /// Column order: factorization column `k` is original column `qcol[k]`.
    pub(super) qcol: Vec<usize>,
    /// Inverse of `qcol`: `qcol_inv[original_col] = column_position`.
    pub(super) qcol_inv: Vec<usize>,
    /// Product-form `U`-updates applied since the last factor/refactor.
    pub(super) etas: Vec<EtaU>,
    /// Running growth monitor (max `1/|τ_q|` over the updates).
    pub(super) growth: f64,
    pub(super) params: LuParams,
    /// Two-sided scaling of the factored matrix (identity when unscaled).
    pub(super) scale: LuScale,
    pub(super) scratch: Vec<f64>,
}

impl SparseLu {
    /// Factor `a` using the column ordering in `symbolic`.
    pub fn factor(
        a: &SparseColMatrix,
        symbolic: &SparseLuSymbolic,
        params: LuParams,
    ) -> Result<Self, FeralError> {
        let m = a.m;
        if symbolic.m != m {
            return Err(FeralError::DimensionMismatch {
                expected: m,
                got: symbolic.m,
            });
        }
        // Scaling: factor Ã = D_row Π A D_col (pattern is invariant under row
        // permutation/scaling, so the column ordering `symbolic` still applies).
        let (scale, scaled) = if params.scaling == LuScaling::None {
            (LuScale::identity(m), None)
        } else {
            let scale = compute_lu_scale(a, params.scaling)?;
            let mat = scale.apply_sparse(a)?;
            (scale, Some(mat))
        };
        let a: &SparseColMatrix = scaled.as_ref().unwrap_or(a);

        let qcol = symbolic.qcol.clone();
        let qcol_inv = symbolic.qcol_inv.clone();

        let mut w = vec![0.0_f64; m];
        let mut mark = vec![false; m];
        let mut touched: Vec<usize> = Vec::new();
        let mut pinv: Vec<isize> = vec![-1; m];
        let mut perm = vec![0usize; m]; // pivot pos -> orig row

        let mut l_col_ptr = Vec::with_capacity(m + 1);
        let mut l_row_idx: Vec<usize> = Vec::new(); // original rows, remapped later
        let mut l_val: Vec<f64> = Vec::new();
        l_col_ptr.push(0);
        let mut u_col_ptr = Vec::with_capacity(m + 1);
        let mut u_row_idx: Vec<usize> = Vec::new();
        let mut u_val: Vec<f64> = Vec::new();
        u_col_ptr.push(0);
        let mut u_diag = vec![0.0_f64; m];

        let utol = params.pivot_threshold;
        let ztol = params.zero_pivot_tol;

        for k in 0..m {
            // Scatter A(:, qcol[k]) into w.
            let (rows, vals) = a.column(qcol[k]);
            for (&i, &v) in rows.iter().zip(vals.iter()) {
                w[i] = v;
                if !mark[i] {
                    mark[i] = true;
                    touched.push(i);
                }
            }

            // Forward substitution in pivot order; collect U(:,k) entries.
            let mut u_entries: Vec<(usize, f64)> = Vec::new();
            for p in 0..k {
                let r = perm[p];
                let xp = w[r];
                if xp == 0.0 {
                    continue;
                }
                u_entries.push((p, xp));
                let (ls, le) = (l_col_ptr[p], l_col_ptr[p + 1]);
                for idx in ls..le {
                    let i = l_row_idx[idx];
                    let before = w[i];
                    w[i] = before - xp * l_val[idx];
                    if !mark[i] {
                        mark[i] = true;
                        touched.push(i);
                    }
                }
            }

            // Pivot selection among unpivoted touched rows (partial pivoting).
            let mut amax = 0.0_f64;
            let mut ipiv: isize = -1;
            for &i in touched.iter() {
                if pinv[i] < 0 {
                    let av = w[i].abs();
                    if av > amax {
                        amax = av;
                        ipiv = i as isize;
                    }
                }
            }

            let mut piv;
            let pivot_row: usize;
            if amax <= ztol {
                match params.on_singular {
                    LuSingularAction::Fail => {
                        return Err(FeralError::SingularBasis { column: k });
                    }
                    LuSingularAction::PerturbToEps { abs_floor } => {
                        // Choose any still-unpivoted row and floor the pivot.
                        let r = (0..m)
                            .find(|&i| pinv[i] < 0)
                            .ok_or(FeralError::SingularBasis { column: k })?;
                        pivot_row = r;
                        let s = if w[r] < 0.0 { -1.0 } else { 1.0 };
                        piv = s * abs_floor.max(w[r].abs());
                    }
                }
            } else {
                // Threshold partial pivoting: take the max (u=1 default).
                let _ = utol;
                pivot_row = ipiv as usize;
                piv = w[pivot_row];
                if piv.abs() <= ztol {
                    piv = if piv < 0.0 { -ztol } else { ztol };
                }
            }

            pinv[pivot_row] = k as isize;
            perm[k] = pivot_row;
            u_diag[k] = piv;
            for (p, v) in u_entries.into_iter() {
                u_row_idx.push(p);
                u_val.push(v);
            }
            u_col_ptr.push(u_row_idx.len());

            // L(:,k): unpivoted touched rows (excluding the pivot) / piv.
            let inv = 1.0 / piv;
            for &i in touched.iter() {
                if pinv[i] < 0 && w[i] != 0.0 {
                    l_row_idx.push(i); // original row, remapped after the loop
                    l_val.push(w[i] * inv);
                }
            }
            l_col_ptr.push(l_row_idx.len());

            // Clear w / mark for the next column.
            for &i in touched.iter() {
                w[i] = 0.0;
                mark[i] = false;
            }
            touched.clear();
        }

        // Remap L's stored original rows to pivot positions, then sort columns.
        let perm_inv: Vec<usize> = pinv.iter().map(|&p| p as usize).collect();
        remap_and_sort_l(&l_col_ptr, &mut l_row_idx, &mut l_val, &perm_inv, m);

        Ok(SparseLu {
            m,
            l_col_ptr,
            l_row_idx,
            l_val,
            u_col_ptr,
            u_row_idx,
            u_val,
            u_diag,
            perm,
            qcol,
            qcol_inv,
            etas: Vec::new(),
            growth: 1.0,
            params,
            scale,
            scratch: vec![0.0; m],
        })
    }

    /// Convenience: analyze + factor from dense columns.
    pub fn factor_dense_columns(
        m: usize,
        cols: &[Vec<f64>],
        params: LuParams,
    ) -> Result<Self, FeralError> {
        let a = SparseColMatrix::from_dense_columns(m, cols)?;
        let symbolic = SparseLuSymbolic::analyze(&a)?;
        SparseLu::factor(&a, &symbolic, params)
    }

    /// Basis dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.m
    }

    /// Row permutation: `perm[k]` = original row in pivot position `k`.
    #[inline]
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Column order: `qcol[k]` = original column in position `k`.
    #[inline]
    pub fn qcol(&self) -> &[usize] {
        &self.qcol
    }

    /// Total stored nonzeros in `L` and `U` (including the `U` diagonal).
    pub fn factor_nnz(&self) -> usize {
        self.l_val.len() + self.u_val.len() + self.m
    }

    /// Reconstruct dense entry `(i, j)` of `L` (pivot-position coordinates).
    pub fn l_dense(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 1.0;
        }
        let (s, e) = (self.l_col_ptr[j], self.l_col_ptr[j + 1]);
        for idx in s..e {
            if self.l_row_idx[idx] == i {
                return self.l_val[idx];
            }
        }
        0.0
    }

    /// Reconstruct dense entry `(i, j)` of `U` (pivot-position coordinates).
    pub fn u_dense(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return self.u_diag[j];
        }
        let (s, e) = (self.u_col_ptr[j], self.u_col_ptr[j + 1]);
        for idx in s..e {
            if self.u_row_idx[idx] == i {
                return self.u_val[idx];
            }
        }
        0.0
    }
}

/// Remap L's original row indices to pivot positions and sort each column.
fn remap_and_sort_l(
    col_ptr: &[usize],
    row_idx: &mut [usize],
    val: &mut [f64],
    perm_inv: &[usize],
    m: usize,
) {
    for r in row_idx.iter_mut() {
        *r = perm_inv[*r];
    }
    let mut order: Vec<usize> = Vec::new();
    for j in 0..m {
        let (s, e) = (col_ptr[j], col_ptr[j + 1]);
        order.clear();
        order.extend(s..e);
        order.sort_by_key(|&idx| row_idx[idx]);
        let rows: Vec<usize> = order.iter().map(|&idx| row_idx[idx]).collect();
        let vals: Vec<f64> = order.iter().map(|&idx| val[idx]).collect();
        row_idx[s..e].copy_from_slice(&rows);
        val[s..e].copy_from_slice(&vals);
    }
}
