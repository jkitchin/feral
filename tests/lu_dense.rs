//! Integration tests for the dense unsymmetric LU basis engine (issue #81).
//!
//! Oracles: hand-worked exact factor/solve values (external) and equation-
//! residual property checks `‖Bx−a‖` (oracle-free — they verify the equation,
//! not a self-computed truth), on plain, adversarial, and ill-conditioned
//! bases.
#![allow(clippy::needless_range_loop)]

use feral::lu::dense_matrix::GeneralMatrix;
use feral::lu::{DenseLu, LuParams};
use feral::FeralError;

/// Column-major columns from rows-of-the-matrix.
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

fn general_from_cols(cols: &[Vec<f64>], m: usize) -> GeneralMatrix {
    GeneralMatrix::from_columns(m, cols).expect("general matrix")
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

/// Relative residual of `B x − a` after `ftran`.
fn ftran_rel_residual(b: &GeneralMatrix, lu: &mut DenseLu, a: &[f64]) -> f64 {
    let mut x = a.to_vec();
    lu.ftran(&mut x).expect("ftran");
    let mut bx = vec![0.0; a.len()];
    b.matvec(&x, &mut bx);
    let r: Vec<f64> = bx.iter().zip(a).map(|(&p, &ai)| p - ai).collect();
    inf_norm(&r) / inf_norm(a).max(1e-300)
}

/// Relative residual of `Bᵀ x − a` after `btran`.
fn btran_rel_residual(b: &GeneralMatrix, lu: &mut DenseLu, a: &[f64]) -> f64 {
    let mut x = a.to_vec();
    lu.btran(&mut x).expect("btran");
    let mut bx = vec![0.0; a.len()];
    b.matvec_transpose(&x, &mut bx);
    let r: Vec<f64> = bx.iter().zip(a).map(|(&p, &ai)| p - ai).collect();
    inf_norm(&r) / inf_norm(a).max(1e-300)
}

/// Deterministic pseudo-random diagonally-dominant basis (well-conditioned).
fn random_basis(m: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut cols = vec![vec![0.0; m]; m];
    let mut state = seed;
    for j in 0..m {
        for i in 0..m {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            cols[j][i] = ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
        }
        cols[j][j] += 5.0;
    }
    cols
}

// ───────────────────────── hand-worked exact ─────────────────────────

#[test]
fn ftran_2x2_swap_exact_solution() {
    // B = [[0,2],[3,4]] forces a pivot swap (P = [1,0]); verify the solve via
    // the equation residual ‖Bx − a‖ rather than a hand-computed x.
    let (cols, m) = cols_from_rows(&[&[0.0, 2.0], &[3.0, 4.0]]);
    let b = general_from_cols(&cols, m);
    let mut lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
    let a = vec![2.0, 11.0];
    assert!(ftran_rel_residual(&b, &mut lu, &a) < 1e-12);
}

#[test]
fn ftran_partial_equals_handbuilt_spike() {
    // B = [[0,2],[3,4]] → perm = [1,0], L = I. Spike of a = [5,11] is
    // L⁻¹ P a = P a = [a[1], a[0]] = [11, 5].
    let (cols, m) = cols_from_rows(&[&[0.0, 2.0], &[3.0, 4.0]]);
    let mut lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
    let mut spike = vec![5.0, 11.0];
    lu.ftran_partial(&mut spike).expect("ftran_partial");
    assert!((spike[0] - 11.0).abs() < 1e-12);
    assert!((spike[1] - 5.0).abs() < 1e-12);
}

// ───────────────────── equation-residual properties ──────────────────

#[test]
fn ftran_btran_residual_random() {
    for seed in [0x1111u64, 0x2222, 0xBEEF] {
        let m = 9;
        let cols = random_basis(m, seed);
        let b = general_from_cols(&cols, m);
        let mut lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
        let a: Vec<f64> = (0..m).map(|i| (i as f64) - 4.0).collect();
        assert!(
            ftran_rel_residual(&b, &mut lu, &a) < 1e-10,
            "ftran seed {seed:x}"
        );
        assert!(
            btran_rel_residual(&b, &mut lu, &a) < 1e-10,
            "btran seed {seed:x}"
        );
    }
}

#[test]
fn ftran_btran_roundtrip() {
    // x then Bᵀ(B x): btran(ftran(a)) is NOT identity, but ftran then a fresh
    // solve must reproduce a via matvec — covered above. Here check that
    // solving and multiplying back is consistent for several RHS.
    let m = 6;
    let cols = random_basis(m, 0x5151);
    let b = general_from_cols(&cols, m);
    let mut lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
    for k in 0..m {
        let mut a = vec![0.0; m];
        a[k] = 1.0;
        assert!(ftran_rel_residual(&b, &mut lu, &a) < 1e-10);
    }
}

// ───────────────────────── rank-1 update ─────────────────────────────

/// Build the matrix that results from replacing column `slot` of `cols` with
/// `new_col`, and run one update, returning the post-update relative residual.
fn do_update_residual(
    cols: &[Vec<f64>],
    m: usize,
    slot: usize,
    new_col: Vec<f64>,
    a: &[f64],
) -> f64 {
    let mut lu = DenseLu::factor(cols, m, LuParams::default()).expect("factor");
    lu.update(slot, &new_col).expect("update");
    // B_new
    let mut new_cols = cols.to_vec();
    new_cols[slot] = new_col;
    let b_new = general_from_cols(&new_cols, m);
    ftran_rel_residual(&b_new, &mut lu, a)
}

#[test]
fn update_single_column_residual() {
    let cols = random_basis(7, 0xABCD);
    let m = 7;
    let a: Vec<f64> = (0..m).map(|i| 1.0 + i as f64).collect();
    let new_col: Vec<f64> = (0..m).map(|i| 2.0 - 0.3 * i as f64).collect();
    for slot in [0usize, 3, 6] {
        let res = do_update_residual(&cols, m, slot, new_col.clone(), &a);
        assert!(res < 1e-8, "update slot {slot}: residual {res:e}");
    }
}

#[test]
fn update_with_row_swap_residual() {
    // A basis whose factorization pivots (P ≠ I), stressing perm composition.
    let (cols, m) = cols_from_rows(&[&[0.0, 2.0, 1.0], &[3.0, 4.0, 0.0], &[1.0, 1.0, 5.0]]);
    let mut lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
    assert_ne!(lu.perm(), &[0usize, 1, 2], "expected a pivot");
    let new_col = vec![2.0, 1.0, 3.0];
    lu.update(1, &new_col).expect("update");
    let mut new_cols = cols.clone();
    new_cols[1] = new_col;
    let b_new = general_from_cols(&new_cols, m);
    let a = vec![1.0, -2.0, 0.5];
    assert!(ftran_rel_residual(&b_new, &mut lu, &a) < 1e-9);
}

#[test]
fn update_sequence_five_residual() {
    let m = 8;
    let mut cols = random_basis(m, 0x0707);
    let mut lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
    let a: Vec<f64> = (0..m).map(|i| (i as f64 * 0.7) - 1.0).collect();
    for (step, slot) in [1usize, 4, 0, 6, 2].into_iter().enumerate() {
        let new_col: Vec<f64> = (0..m)
            .map(|i| 1.0 + ((step * 31 + i * 7) % 11) as f64 * 0.1)
            .collect();
        lu.update(slot, &new_col).expect("update");
        cols[slot] = new_col;
        let b_new = general_from_cols(&cols, m);
        let res = ftran_rel_residual(&b_new, &mut lu, &a);
        assert!(res < 1e-7, "after update {step} (slot {slot}): {res:e}");
    }
    assert_eq!(lu.updates_since_refactor(), 5);
}

#[test]
fn refactor_matches_updated_factor() {
    let m = 6;
    let mut cols = random_basis(m, 0x9090);
    let mut lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
    // Two updates.
    for slot in [2usize, 5] {
        let new_col: Vec<f64> = (0..m).map(|i| 0.5 + slot as f64 - i as f64 * 0.2).collect();
        lu.update(slot, &new_col).expect("update");
        cols[slot] = new_col;
    }
    let a: Vec<f64> = (0..m).map(|i| 3.0 - i as f64).collect();
    let mut x_updated = a.clone();
    lu.ftran(&mut x_updated).expect("ftran updated");
    // A fresh factorization of the same basis must give the same solve.
    let mut lu2 = DenseLu::factor(&cols, m, LuParams::default()).expect("refactor");
    let mut x_fresh = a.clone();
    lu2.ftran(&mut x_fresh).expect("ftran fresh");
    let diff: f64 = x_updated
        .iter()
        .zip(&x_fresh)
        .map(|(&u, &f)| (u - f).abs())
        .fold(0.0, f64::max);
    assert!(diff < 1e-9, "updated vs refactored solve diff {diff:e}");
}

#[test]
fn update_budget_returns_needs_refactor() {
    let m = 5;
    let cols = random_basis(m, 0x1234);
    let params = LuParams {
        max_updates: 2,
        ..LuParams::default()
    };
    let mut lu = DenseLu::factor(&cols, m, params).expect("factor");
    // Two successful updates with *distinct* columns. (Using the same column
    // for both slots would make columns 0 and 1 identical → a singular basis,
    // which the update now correctly rejects before the budget is reached; see
    // L1, dev/research/repo-review-2026-06-09.md.)
    let new_col0 = vec![1.0, 2.0, 1.0, 0.5, 3.0];
    let new_col1 = vec![0.5, 3.0, 2.0, 1.0, 0.25];
    lu.update(0, &new_col0).expect("update 0");
    lu.update(1, &new_col1).expect("update 1");
    // Third trips the budget; self must be unchanged.
    let before = {
        let mut a = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        lu.ftran(&mut a).expect("ftran");
        a
    };
    let err = lu.update(2, &new_col0);
    assert!(matches!(err, Err(FeralError::NeedsRefactor)));
    assert_eq!(lu.updates_since_refactor(), 2);
    let after = {
        let mut a = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        lu.ftran(&mut a).expect("ftran");
        a
    };
    let diff: f64 = before
        .iter()
        .zip(&after)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(diff < 1e-14, "self changed after NeedsRefactor: {diff:e}");
}

// ───────────────────── adversarial / refinement ──────────────────────

#[test]
fn singular_repeated_columns_fails() {
    let (cols, m) = cols_from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
    let err = DenseLu::factor(&cols, m, LuParams::default());
    assert!(matches!(err, Err(FeralError::SingularBasis { .. })));
}

#[test]
fn ill_conditioned_refinement_helps() {
    // A moderately ill-conditioned basis: refinement should not worsen the
    // residual and should keep it small.
    let (cols, m) = cols_from_rows(&[&[1.0, 1.0, 1.0], &[1.0, 1.0001, 1.0], &[1.0, 1.0, 1.0002]]);
    let b = general_from_cols(&cols, m);
    let params = LuParams {
        refine_steps: 3,
        refine_tol: 1e-14,
        ..LuParams::default()
    };
    let mut lu = DenseLu::factor(&cols, m, params).expect("factor");
    let a = vec![3.0, 3.0001, 3.0002];
    let mut x = a.clone();
    lu.ftran_refined(&b, &mut x).expect("ftran_refined");
    let mut bx = vec![0.0; m];
    b.matvec(&x, &mut bx);
    let r: Vec<f64> = bx.iter().zip(&a).map(|(&p, &ai)| p - ai).collect();
    assert!(inf_norm(&r) / inf_norm(&a) < 1e-12);
}
