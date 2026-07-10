//! Two-sided scaling for the LU basis (issue #81 robustness layer).
//!
//! [`LuScale`] holds a row permutation `Π` and diagonal row/column scalings
//! `D_row`, `D_col` such that the *scaled* matrix `Ã = D_row · Π · B · D_col`
//! is well balanced. The factorization factors `Ã`; the solves wrap the scaling
//! around the core triangular solves (see the `apply_*` helpers).
//!
//! Two strategies:
//!   * [`equilibrate_infnorm`] — Knight–Ruiz two-sided ∞-norm equilibration
//!     (`Π = I`).
//!   * [`mc64_unsymmetric`] — a maximum-weight bipartite matching that places
//!     large entries on the diagonal, reusing feral's `hungarian_match` kernel;
//!     `D_row`/`D_col` come from the matching duals. Falls back to ∞-norm
//!     equilibration when the matching is not perfect.

use super::sparse_matrix::SparseColMatrix;
use super::LuScaling;
use crate::error::FeralError;
use crate::scaling::{hungarian_match, CostGraph};

/// Row permutation + two-sided diagonal scaling. With `Ã = D_row Π B D_col`:
/// `Ã[i,j] = d_row[i] · B[rperm[i], j] · d_col[j]`.
#[derive(Debug, Clone)]
pub struct LuScale {
    /// `rperm[i]` = original row placed at scaled-row position `i`.
    pub rperm: Vec<usize>,
    /// Row scaling indexed by scaled-row position.
    pub d_row: Vec<f64>,
    /// Column scaling indexed by basis slot.
    pub d_col: Vec<f64>,
}

impl LuScale {
    /// Identity scaling (no row permutation, unit diagonals).
    pub fn identity(m: usize) -> Self {
        LuScale {
            rperm: (0..m).collect(),
            d_row: vec![1.0; m],
            d_col: vec![1.0; m],
        }
    }

    /// True when this is the identity scaling (cheap fast-path check).
    pub fn is_identity(&self) -> bool {
        self.rperm.iter().enumerate().all(|(i, &r)| i == r)
            && self.d_row.iter().all(|&x| x == 1.0)
            && self.d_col.iter().all(|&x| x == 1.0)
    }

    /// Build the scaled, row-permuted dense columns `Ã(:, j)` for factoring.
    pub fn scaled_dense_columns(&self, b: &SparseColMatrix) -> Vec<Vec<f64>> {
        let m = b.m;
        let mut rperm_inv = vec![0usize; m];
        for (i, &r) in self.rperm.iter().enumerate() {
            rperm_inv[r] = i;
        }
        let mut cols = vec![vec![0.0; m]; m];
        for (j, colj) in cols.iter_mut().enumerate() {
            let (rows, vals) = b.column(j);
            for (&r, &v) in rows.iter().zip(vals.iter()) {
                let i = rperm_inv[r];
                colj[i] = self.d_row[i] * v * self.d_col[j];
            }
        }
        cols
    }
}

impl LuScale {
    /// Build the scaled, row-permuted matrix `Ã = D_row Π B D_col` in sparse
    /// form (preserving sparsity).
    pub fn apply_sparse(&self, b: &SparseColMatrix) -> Result<SparseColMatrix, FeralError> {
        let m = b.m;
        let mut rperm_inv = vec![0usize; m];
        for (i, &r) in self.rperm.iter().enumerate() {
            rperm_inv[r] = i;
        }
        let mut sparse_cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        for (j, colj) in sparse_cols.iter_mut().enumerate() {
            let (rows, vals) = b.column(j);
            for (&r, &v) in rows.iter().zip(vals.iter()) {
                let i = rperm_inv[r];
                colj.push((i, self.d_row[i] * v * self.d_col[j]));
            }
        }
        SparseColMatrix::from_sparse_columns(m, &sparse_cols)
    }
}

/// Compute the scaling for `b` under `strategy`.
pub fn compute_lu_scale(b: &SparseColMatrix, strategy: LuScaling) -> Result<LuScale, FeralError> {
    match strategy {
        LuScaling::None => Ok(LuScale::identity(b.m)),
        LuScaling::InfNorm => Ok(equilibrate_infnorm(b, 5)),
        LuScaling::Mc64 => mc64_unsymmetric(b),
        LuScaling::Mc64ThenInfNorm => {
            let mc = mc64_unsymmetric(b)?;
            Ok(compose_with_infnorm(b, mc))
        }
    }
}

/// Knight–Ruiz two-sided ∞-norm equilibration (`Π = I`). Iterates row and
/// column ∞-norm balancing.
pub fn equilibrate_infnorm(b: &SparseColMatrix, iters: usize) -> LuScale {
    let m = b.m;
    let mut d_row = vec![1.0_f64; m];
    let mut d_col = vec![1.0_f64; m];
    for _ in 0..iters {
        let mut row_max = vec![0.0_f64; m];
        let mut col_max = vec![0.0_f64; m];
        for (j, cmj) in col_max.iter_mut().enumerate() {
            let (rows, vals) = b.column(j);
            for (&i, &v) in rows.iter().zip(vals.iter()) {
                let a = (d_row[i] * v * d_col[j]).abs();
                if a > row_max[i] {
                    row_max[i] = a;
                }
                if a > *cmj {
                    *cmj = a;
                }
            }
        }
        for (dr, &rm) in d_row.iter_mut().zip(row_max.iter()) {
            if rm > 0.0 {
                *dr = crate::scaling::kr_guarded_update(*dr, rm);
            }
        }
        for (dc, &cm) in d_col.iter_mut().zip(col_max.iter()) {
            if cm > 0.0 {
                *dc = crate::scaling::kr_guarded_update(*dc, cm);
            }
        }
    }
    LuScale {
        rperm: (0..m).collect(),
        d_row,
        d_col,
    }
}

/// Unsymmetric MC64: maximum-weight bipartite matching placing large entries on
/// the diagonal, with `D_row`/`D_col` from the matching duals. Falls back to
/// ∞-norm equilibration when the matching is not perfect.
pub fn mc64_unsymmetric(b: &SparseColMatrix) -> Result<LuScale, FeralError> {
    let m = b.m;
    if m == 0 {
        return Ok(LuScale::identity(0));
    }
    let (graph, cmax) = build_cost_graph(b);
    let matching = hungarian_match(&graph);
    if matching.n_matched != m {
        // Structurally deficient matching → fall back to equilibration.
        return Ok(equilibrate_infnorm(b, 5));
    }
    // rperm[j] = matched row of column j (puts it on the diagonal).
    let rperm = matching.perm.clone();
    // Column scaling from the column dual minus the column max.
    let mut d_col = vec![1.0; m];
    for (j, dc) in d_col.iter_mut().enumerate() {
        *dc = clamp_exp(matching.v[j] - cmax[j]).exp();
    }
    // Row scaling (indexed by scaled-row position i, whose original row is
    // rperm[i]) from the row dual.
    let mut d_row = vec![1.0; m];
    for (i, dr) in d_row.iter_mut().enumerate() {
        *dr = clamp_exp(matching.u[rperm[i]]).exp();
    }
    Ok(LuScale {
        rperm,
        d_row,
        d_col,
    })
}

/// Refine an existing scale by a few ∞-norm equilibration sweeps applied to the
/// already-scaled matrix (composes the diagonals; keeps the row permutation).
fn compose_with_infnorm(b: &SparseColMatrix, mut scale: LuScale) -> LuScale {
    let m = b.m;
    let mut rperm_inv = vec![0usize; m];
    for (i, &r) in scale.rperm.iter().enumerate() {
        rperm_inv[r] = i;
    }
    for _ in 0..3 {
        let mut row_max = vec![0.0_f64; m];
        let mut col_max = vec![0.0_f64; m];
        for (j, cmj) in col_max.iter_mut().enumerate() {
            let (rows, vals) = b.column(j);
            for (&r, &v) in rows.iter().zip(vals.iter()) {
                let i = rperm_inv[r];
                let a = (scale.d_row[i] * v * scale.d_col[j]).abs();
                if a > row_max[i] {
                    row_max[i] = a;
                }
                if a > *cmj {
                    *cmj = a;
                }
            }
        }
        for (dr, &rm) in scale.d_row.iter_mut().zip(row_max.iter()) {
            if rm > 0.0 {
                *dr = crate::scaling::kr_guarded_update(*dr, rm);
            }
        }
        for (dc, &cm) in scale.d_col.iter_mut().zip(col_max.iter()) {
            if cm > 0.0 {
                *dc = crate::scaling::kr_guarded_update(*dc, cm);
            }
        }
    }
    scale
}

/// `ln(f64::MAX) ≈ 709.78`; clamp dual sums to keep `exp` finite.
fn clamp_exp(x: f64) -> f64 {
    x.clamp(-709.0, 709.0)
}

/// Build the column-normalized log-cost bipartite graph for `hungarian_match`.
/// Costs are `cmax[j] − ln|a|` (non-negative), per the MC64 recipe.
fn build_cost_graph(b: &SparseColMatrix) -> (CostGraph, Vec<f64>) {
    let m = b.m;
    let mut col_ptr = Vec::with_capacity(m + 1);
    let mut row_idx = Vec::new();
    let mut cost = Vec::new();
    let mut cmax = vec![f64::NEG_INFINITY; m];
    col_ptr.push(0);
    // First pass: cmax per column.
    for (j, cm) in cmax.iter_mut().enumerate() {
        let (_rows, vals) = b.column(j);
        for &v in vals {
            if v != 0.0 {
                let c = v.abs().ln();
                if c > *cm {
                    *cm = c;
                }
            }
        }
    }
    // Second pass: emit non-negative costs.
    for (j, &cmax_j) in cmax.iter().enumerate() {
        let (rows, vals) = b.column(j);
        for (&i, &v) in rows.iter().zip(vals.iter()) {
            if v != 0.0 {
                let c = cmax_j - v.abs().ln();
                row_idx.push(i);
                cost.push(if c < 0.0 { 0.0 } else { c });
            }
        }
        col_ptr.push(row_idx.len());
    }
    (
        CostGraph {
            n: m,
            col_ptr,
            row_idx,
            cost,
        },
        cmax,
    )
}
