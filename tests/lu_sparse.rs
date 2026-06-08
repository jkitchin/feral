//! Integration tests for the sparse unsymmetric LU basis engine (issue #81).
//!
//! Oracles: reconstruction `‖PAQ − LU‖` and equation-residual `‖Bx−a‖` (both
//! oracle-free identities), plus dense↔sparse agreement (consistency check).
#![allow(clippy::needless_range_loop)]

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{DenseLu, LuParams, LuSingularAction, SparseLu, SparseLuSymbolic};
use feral::FeralError;

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

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

/// Deterministic sparse-ish basis: a tridiagonal-plus-noise matrix.
fn banded_basis(m: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut cols = vec![vec![0.0; m]; m];
    let mut state = seed;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0
    };
    for j in 0..m {
        cols[j][j] = 4.0 + rng();
        if j > 0 {
            cols[j][j - 1] = rng();
        }
        if j + 1 < m {
            cols[j][j + 1] = rng();
        }
        // a couple of off-band entries for fill
        if j >= 2 {
            cols[j][j - 2] = 0.5 * rng();
        }
    }
    cols
}

/// max | (P A Q)[i,j] − (L U)[i,j] | over the dense reconstruction.
fn reconstruction_residual(a: &SparseColMatrix, lu: &SparseLu) -> f64 {
    let m = a.m;
    // Dense A for convenience.
    let mut dense = vec![0.0; m * m];
    for j in 0..m {
        let (rows, vals) = a.column(j);
        for (&i, &v) in rows.iter().zip(vals) {
            dense[i + j * m] = v;
        }
    }
    let perm = lu.perm();
    let qcol = lu.qcol();
    let mut worst = 0.0_f64;
    for i in 0..m {
        for j in 0..m {
            let paq = dense[perm[i] + qcol[j] * m];
            let mut prod = 0.0;
            for k in 0..m {
                prod += lu.l_dense(i, k) * lu.u_dense(k, j);
            }
            worst = worst.max((paq - prod).abs());
        }
    }
    worst
}

fn ftran_rel_residual(b: &SparseColMatrix, lu: &mut SparseLu, a: &[f64]) -> f64 {
    let mut x = a.to_vec();
    lu.ftran(&mut x).expect("ftran");
    let mut bx = vec![0.0; a.len()];
    b.matvec(&x, &mut bx);
    let r: Vec<f64> = bx.iter().zip(a).map(|(&p, &ai)| p - ai).collect();
    inf_norm(&r) / inf_norm(a).max(1e-300)
}

fn btran_rel_residual(b: &SparseColMatrix, lu: &mut SparseLu, a: &[f64]) -> f64 {
    let mut x = a.to_vec();
    lu.btran(&mut x).expect("btran");
    let mut bx = vec![0.0; a.len()];
    b.matvec_transpose(&x, &mut bx);
    let r: Vec<f64> = bx.iter().zip(a).map(|(&p, &ai)| p - ai).collect();
    inf_norm(&r) / inf_norm(a).max(1e-300)
}

#[test]
fn sparse_factor_reconstruction() {
    for seed in [0x11u64, 0xA5, 0xF0] {
        let m = 12;
        let cols = banded_basis(m, seed);
        let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
        let symbolic = SparseLuSymbolic::analyze(&a).expect("analyze");
        let lu = SparseLu::factor(&a, &symbolic, LuParams::default()).expect("factor");
        let res = reconstruction_residual(&a, &lu);
        assert!(res < 1e-10, "seed {seed:x}: ‖PAQ−LU‖ = {res:e}");
    }
}

#[test]
fn sparse_ftran_btran_residual() {
    let m = 14;
    let cols = banded_basis(m, 0x99);
    let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let symbolic = SparseLuSymbolic::analyze(&a).expect("analyze");
    let mut lu = SparseLu::factor(&a, &symbolic, LuParams::default()).expect("factor");
    let rhs: Vec<f64> = (0..m).map(|i| 1.0 + i as f64).collect();
    assert!(ftran_rel_residual(&a, &mut lu, &rhs) < 1e-10);
    assert!(btran_rel_residual(&a, &mut lu, &rhs) < 1e-10);
}

#[test]
fn sparse_natural_ordering_works() {
    let m = 10;
    let cols = banded_basis(m, 0x42);
    let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let symbolic = SparseLuSymbolic::natural(m);
    let mut lu = SparseLu::factor(&a, &symbolic, LuParams::default()).expect("factor");
    assert!(reconstruction_residual(&a, &lu) < 1e-10);
    let rhs: Vec<f64> = (0..m).map(|i| 2.0 - i as f64 * 0.3).collect();
    assert!(ftran_rel_residual(&a, &mut lu, &rhs) < 1e-10);
}

#[test]
fn dense_sparse_agreement() {
    // The same basis solved through both paths must agree.
    let m = 9;
    let cols = banded_basis(m, 0x7777);
    let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let symbolic = SparseLuSymbolic::analyze(&a).expect("analyze");
    let mut sparse = SparseLu::factor(&a, &symbolic, LuParams::default()).expect("sparse");
    let mut dense = DenseLu::factor(&cols, m, LuParams::default()).expect("dense");

    let rhs: Vec<f64> = (0..m).map(|i| (i as f64 * 1.3) - 2.0).collect();
    let mut xs = rhs.clone();
    let mut xd = rhs.clone();
    sparse.ftran(&mut xs).expect("sparse ftran");
    dense.ftran(&mut xd).expect("dense ftran");
    let diff: f64 = xs
        .iter()
        .zip(&xd)
        .map(|(&s, &d)| (s - d).abs())
        .fold(0.0, f64::max);
    assert!(diff < 1e-9, "ftran dense vs sparse diff {diff:e}");

    let mut ys = rhs.clone();
    let mut yd = rhs.clone();
    sparse.btran(&mut ys).expect("sparse btran");
    dense.btran(&mut yd).expect("dense btran");
    let diffb: f64 = ys
        .iter()
        .zip(&yd)
        .map(|(&s, &d)| (s - d).abs())
        .fold(0.0, f64::max);
    assert!(diffb < 1e-9, "btran dense vs sparse diff {diffb:e}");
}

#[test]
fn sparse_singular_fails() {
    // Two identical columns → structurally singular.
    let (cols, m) = cols_from_rows(&[&[1.0, 1.0, 0.0], &[2.0, 2.0, 0.0], &[0.0, 0.0, 3.0]]);
    let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let symbolic = SparseLuSymbolic::natural(m);
    let err = SparseLu::factor(&a, &symbolic, LuParams::default());
    assert!(matches!(err, Err(FeralError::SingularBasis { .. })));
}

#[test]
fn sparse_perturb_succeeds() {
    let (cols, m) = cols_from_rows(&[&[1.0, 1.0, 0.0], &[2.0, 2.0, 0.0], &[0.0, 0.0, 3.0]]);
    let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let symbolic = SparseLuSymbolic::natural(m);
    let params = LuParams {
        on_singular: LuSingularAction::PerturbToEps { abs_floor: 1e-10 },
        ..LuParams::default()
    };
    let lu = SparseLu::factor(&a, &symbolic, params);
    assert!(lu.is_ok());
}

#[test]
fn sparse_update_single_residual() {
    let m = 12;
    let mut cols = banded_basis(m, 0x33);
    let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let symbolic = SparseLuSymbolic::analyze(&a).expect("analyze");
    let mut lu = SparseLu::factor(&a, &symbolic, LuParams::default()).expect("factor");
    let rhs: Vec<f64> = (0..m).map(|i| 1.0 + (i as f64) * 0.4).collect();
    let new_col: Vec<f64> = (0..m).map(|i| 0.5 + ((i * 3) % 5) as f64).collect();
    for &slot in &[0usize, 5, 11] {
        let mut lu_k = lu.clone();
        lu_k.update(slot, &new_col).expect("update");
        let mut new_cols = cols.clone();
        new_cols[slot] = new_col.clone();
        let b_new = SparseColMatrix::from_dense_columns(m, &new_cols).expect("b_new");
        assert!(
            ftran_rel_residual(&b_new, &mut lu_k, &rhs) < 1e-8,
            "slot {slot}"
        );
    }
    // Chain of updates on the live factor.
    let a0 = rhs.clone();
    for (step, &slot) in [2usize, 7, 4, 9].iter().enumerate() {
        let nc: Vec<f64> = (0..m)
            .map(|i| 1.0 + ((step + i) % 6) as f64 * 0.3)
            .collect();
        lu.update(slot, &nc).expect("chain update");
        cols[slot] = nc;
        let b_new = SparseColMatrix::from_dense_columns(m, &cols).expect("b_new");
        let res = ftran_rel_residual(&b_new, &mut lu, &a0);
        assert!(res < 1e-7, "chain step {step}: {res:e}");
    }
    assert_eq!(lu.updates_since_refactor(), 4);
}

#[test]
fn sparse_update_budget_and_refactor() {
    let m = 8;
    let cols = banded_basis(m, 0x55);
    let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let symbolic = SparseLuSymbolic::analyze(&a).expect("analyze");
    let params = LuParams {
        max_updates: 2,
        ..LuParams::default()
    };
    let mut lu = SparseLu::factor(&a, &symbolic, params).expect("factor");
    let nc0: Vec<f64> = (0..m).map(|i| 1.0 + (i % 3) as f64).collect();
    let nc1: Vec<f64> = (0..m).map(|i| 2.0 - (i % 4) as f64 * 0.5).collect();
    let nc2: Vec<f64> = (0..m).map(|i| 0.5 + (i % 5) as f64 * 0.3).collect();
    lu.update(0, &nc0).expect("u1");
    lu.update(1, &nc1).expect("u2");
    let err = lu.update(2, &nc2);
    assert!(matches!(err, Err(FeralError::NeedsRefactor)));
    assert_eq!(lu.updates_since_refactor(), 2);
    // Refactor clears the eta chain.
    lu.refactor(&a, &symbolic).expect("refactor");
    assert_eq!(lu.updates_since_refactor(), 0);
}

#[test]
fn sparse_update_matches_dense() {
    // The same update through both engines must agree.
    let m = 9;
    let cols = banded_basis(m, 0xDD);
    let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let symbolic = SparseLuSymbolic::analyze(&a).expect("analyze");
    let mut sparse = SparseLu::factor(&a, &symbolic, LuParams::default()).expect("sparse");
    let mut dense = DenseLu::factor(&cols, m, LuParams::default()).expect("dense");
    let new_col: Vec<f64> = (0..m).map(|i| 2.0 - 0.25 * i as f64).collect();
    sparse.update(3, &new_col).expect("sparse update");
    dense.update(3, &new_col).expect("dense update");
    let rhs: Vec<f64> = (0..m).map(|i| 1.0 + i as f64).collect();
    let mut xs = rhs.clone();
    let mut xd = rhs.clone();
    sparse.ftran(&mut xs).expect("sparse ftran");
    dense.ftran(&mut xd).expect("dense ftran");
    let diff: f64 = xs
        .iter()
        .zip(&xd)
        .map(|(&s, &d)| (s - d).abs())
        .fold(0.0, f64::max);
    assert!(diff < 1e-9, "post-update dense vs sparse diff {diff:e}");
}

/// Build a diagonally-dominant tridiagonal `n`×`n` basis (≈3n nonzeros).
fn tridiagonal(n: usize) -> SparseColMatrix {
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (j, col) in cols.iter_mut().enumerate() {
        if j > 0 {
            col.push((j - 1, -1.0));
        }
        col.push((j, 4.0));
        if j + 1 < n {
            col.push((j + 1, -1.0));
        }
    }
    SparseColMatrix::from_sparse_columns(n, &cols).expect("tridiagonal")
}

#[test]
fn factor_traversal_is_subquadratic() {
    // Deterministic (timing-free) scalability guard: on a no-fill tridiagonal
    // basis the Gilbert–Peierls reach work is O(nnz) = O(n). Doubling n must
    // (well) less than quadruple the reach work; the pre-reach O(n²) scan would
    // have quadrupled it. We allow a 3× margin (linear ≈ 2×).
    let work = |n: usize| -> usize {
        let a = tridiagonal(n);
        let sym = SparseLuSymbolic::natural(n);
        let lu = SparseLu::factor(&a, &sym, LuParams::default()).expect("factor");
        // Sanity: tridiagonal has no fill.
        assert_eq!(lu.factor_nnz(), a.nnz());
        lu.reach_visits()
    };
    let w1 = work(2000);
    let w2 = work(4000);
    let w3 = work(8000);
    assert!(
        w2 < 3 * w1,
        "2000→4000 reach work {w1} → {w2} not sub-quadratic"
    );
    assert!(
        w3 < 3 * w2,
        "4000→8000 reach work {w2} → {w3} not sub-quadratic"
    );
}

#[test]
fn sparse_refinement_converges() {
    let m = 11;
    let cols = banded_basis(m, 0xABCDEF);
    let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let symbolic = SparseLuSymbolic::analyze(&a).expect("analyze");
    let params = LuParams {
        refine_steps: 2,
        refine_tol: 1e-14,
        ..LuParams::default()
    };
    let mut lu = SparseLu::factor(&a, &symbolic, params).expect("factor");
    let rhs: Vec<f64> = (0..m).map(|i| 1.0 + (i % 3) as f64).collect();
    let mut x = rhs.clone();
    lu.ftran_refined(&a, &mut x).expect("ftran_refined");
    let mut bx = vec![0.0; m];
    a.matvec(&x, &mut bx);
    let r: Vec<f64> = bx.iter().zip(&rhs).map(|(&p, &ai)| p - ai).collect();
    assert!(inf_norm(&r) / inf_norm(&rhs) < 1e-12);
}
