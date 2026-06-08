//! Scaling-layer tests for the LU basis engine (issue #81): ∞-norm
//! equilibration and unsymmetric MC64. Oracles are equation residuals
//! `‖Bx−a‖` (scaling must preserve correctness) plus a direct balance check on
//! the equilibrated matrix.
#![allow(clippy::needless_range_loop)]

use feral::lu::scaling::{compute_lu_scale, equilibrate_infnorm};
use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{DenseLu, LuParams, LuScaling, SparseLu, SparseLuSymbolic};

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

/// A badly row/column-scaled but nonsingular basis.
fn ill_scaled(m: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0
    };
    // Row scales spanning ~16 orders of magnitude, column scales too.
    let row_scale: Vec<f64> = (0..m).map(|i| 10f64.powi((i as i32 % 8) - 4)).collect();
    let col_scale: Vec<f64> = (0..m).map(|j| 10f64.powi(3 - (j as i32 % 7))).collect();
    let mut cols = vec![vec![0.0; m]; m];
    for j in 0..m {
        for i in 0..m {
            let base = if i == j {
                3.0 + rng().abs()
            } else if i.abs_diff(j) <= 1 {
                rng()
            } else {
                0.0
            };
            cols[j][i] = base * row_scale[i] * col_scale[j];
        }
    }
    cols
}

fn ftran_residual_dense(cols: &[Vec<f64>], m: usize, scaling: LuScaling, a: &[f64]) -> f64 {
    let params = LuParams {
        scaling,
        ..LuParams::default()
    };
    let mut lu = DenseLu::factor(cols, m, params).expect("dense factor");
    let b = feral::lu::dense_matrix::GeneralMatrix::from_columns(m, cols).expect("gen");
    let mut x = a.to_vec();
    lu.ftran(&mut x).expect("ftran");
    let mut bx = vec![0.0; m];
    b.matvec(&x, &mut bx);
    let r: Vec<f64> = bx.iter().zip(a).map(|(&p, &ai)| p - ai).collect();
    inf_norm(&r) / inf_norm(a).max(1e-300)
}

fn ftran_residual_sparse(cols: &[Vec<f64>], m: usize, scaling: LuScaling, a: &[f64]) -> f64 {
    let bmat = SparseColMatrix::from_dense_columns(m, cols).expect("matrix");
    let symbolic = SparseLuSymbolic::analyze(&bmat).expect("analyze");
    let params = LuParams {
        scaling,
        ..LuParams::default()
    };
    let mut lu = SparseLu::factor(&bmat, &symbolic, params).expect("sparse factor");
    let mut x = a.to_vec();
    lu.ftran(&mut x).expect("ftran");
    let mut bx = vec![0.0; m];
    bmat.matvec(&x, &mut bx);
    let r: Vec<f64> = bx.iter().zip(a).map(|(&p, &ai)| p - ai).collect();
    inf_norm(&r) / inf_norm(a).max(1e-300)
}

#[test]
fn dense_scaling_preserves_correctness() {
    let m = 10;
    let cols = ill_scaled(m, 0x1234);
    let a: Vec<f64> = (0..m).map(|i| 1.0 + i as f64).collect();
    for scaling in [
        LuScaling::InfNorm,
        LuScaling::Mc64,
        LuScaling::Mc64ThenInfNorm,
    ] {
        let res = ftran_residual_dense(&cols, m, scaling, &a);
        assert!(res < 1e-8, "{scaling:?}: residual {res:e}");
    }
}

#[test]
fn sparse_scaling_preserves_correctness() {
    let m = 10;
    let cols = ill_scaled(m, 0x9999);
    let a: Vec<f64> = (0..m).map(|i| 2.0 - i as f64 * 0.1).collect();
    for scaling in [
        LuScaling::InfNorm,
        LuScaling::Mc64,
        LuScaling::Mc64ThenInfNorm,
    ] {
        let res = ftran_residual_sparse(&cols, m, scaling, &a);
        assert!(res < 1e-8, "{scaling:?}: residual {res:e}");
    }
}

#[test]
fn dense_sparse_agreement_under_mc64() {
    let m = 9;
    let cols = ill_scaled(m, 0x5151);
    let a: Vec<f64> = (0..m).map(|i| 1.0 + (i % 3) as f64).collect();
    let dr = ftran_residual_dense(&cols, m, LuScaling::Mc64, &a);
    let sr = ftran_residual_sparse(&cols, m, LuScaling::Mc64, &a);
    assert!(dr < 1e-8 && sr < 1e-8, "dense {dr:e}, sparse {sr:e}");
}

#[test]
fn equilibration_balances_the_matrix() {
    // After ∞-norm equilibration every row and column ∞-norm should be ≈ 1.
    let m = 12;
    let cols = ill_scaled(m, 0xABCD);
    let b = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let scale = equilibrate_infnorm(&b, 8);
    let mut row_max = vec![0.0_f64; m];
    let mut col_max = vec![0.0_f64; m];
    for j in 0..m {
        let (rows, vals) = b.column(j);
        for (&i, &v) in rows.iter().zip(vals) {
            let a = (scale.d_row[i] * v * scale.d_col[j]).abs();
            row_max[i] = row_max[i].max(a);
            col_max[j] = col_max[j].max(a);
        }
    }
    for (i, &rm) in row_max.iter().enumerate() {
        assert!(
            (0.1..=10.0).contains(&rm),
            "row {i} max {rm:e} not balanced"
        );
    }
    for (j, &cm) in col_max.iter().enumerate() {
        assert!(
            (0.1..=10.0).contains(&cm),
            "col {j} max {cm:e} not balanced"
        );
    }
}

#[test]
fn mc64_places_large_entries_on_diagonal() {
    // After MC64 scaling + row permutation, the diagonal entries of Ã should be
    // the dominant entries (magnitude ≈ 1 by construction of the matching).
    let m = 8;
    let cols = ill_scaled(m, 0x2468);
    let b = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let scale = compute_lu_scale(&b, LuScaling::Mc64).expect("mc64");
    let scaled = scale.apply_sparse(&b).expect("apply");
    // Each diagonal entry should be present and of order 1.
    for j in 0..m {
        let (rows, vals) = scaled.column(j);
        let diag = rows
            .iter()
            .zip(vals)
            .find(|(&i, _)| i == j)
            .map(|(_, &v)| v.abs())
            .unwrap_or(0.0);
        assert!(
            diag > 1e-3,
            "col {j} diagonal {diag:e} too small after MC64"
        );
    }
}
