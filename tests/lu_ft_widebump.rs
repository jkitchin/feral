//! Issue #87 regression: wide-bump dense-column updates.
//!
//! A tridiagonal basis (sparse `L`,`U`) is repeatedly updated by replacing an
//! early slot with a *dense* column. The spike is then dense and the Forrest–
//! Tomlin bump spans the whole trailing factor — the regime that drove the old
//! re-triangularization into its O(bump²) worst case (`autocorr_bern`,
//! `casctanks`). This asserts the logical-permutation FT update stays correct:
//! `ftran` AND `btran` residuals against the live basis after every update in a
//! long chain, with no refactor in between.

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{LuParams, SparseLu, SparseLuSymbolic};

fn tridiag_cols(m: usize) -> Vec<Vec<f64>> {
    let mut cols = vec![vec![0.0; m]; m];
    for (j, col) in cols.iter_mut().enumerate() {
        col[j] = 4.0;
        if j > 0 {
            col[j - 1] = -1.0;
        }
        if j + 1 < m {
            col[j + 1] = -1.0;
        }
    }
    cols
}

fn matvec(basis: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let m = x.len();
    let mut y = vec![0.0; m];
    for (j, xj) in x.iter().enumerate() {
        for (i, yi) in y.iter_mut().enumerate() {
            *yi += basis[j][i] * xj;
        }
    }
    y
}

fn matvec_t(basis: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let m = x.len();
    let mut y = vec![0.0; m];
    for (j, bj) in basis.iter().enumerate() {
        let mut acc = 0.0;
        for (i, &v) in bj.iter().enumerate() {
            acc += v * x[i];
        }
        y[j] = acc;
    }
    y
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn wide_bump_dense_update_chain_ftran_btran_residuals() {
    let m = 60;
    let cols = tridiag_cols(m);
    let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let sym = SparseLuSymbolic::natural(m);
    let params = LuParams {
        max_updates: 100_000,
        max_growth: 1e30,
        ..LuParams::default()
    };
    let mut lu = SparseLu::factor(&a, &sym, params).expect("factor");
    let mut basis = cols;

    let mut applied = 0usize;
    for s in 0..40usize {
        let slot = s % 5; // early slots ⇒ wide bumps
        let mut col = vec![0.0; m];
        for (i, ci) in col.iter_mut().enumerate() {
            *ci = 0.3 + ((i * 11 + s * 7) % 9) as f64 * 0.2;
        }
        col[slot] = 80.0 + (s % 11) as f64; // dominant diagonal ⇒ nonsingular
        if lu.update(slot, &col).is_err() {
            continue;
        }
        basis[slot] = col;
        applied += 1;

        // ftran: B x = rhs  ⇒  residual ‖B x − rhs‖∞.
        let rhs: Vec<f64> = (0..m).map(|i| 1.0 + (i % 4) as f64).collect();
        let mut x = rhs.clone();
        lu.ftran(&mut x).expect("ftran");
        let fres = max_abs_diff(&matvec(&basis, &x), &rhs);
        assert!(fres < 1e-7, "update {s}: ftran residual {fres:.3e}");

        // btran: Bᵀ y = rhs  ⇒  residual ‖Bᵀ y − rhs‖∞.
        let mut y = rhs.clone();
        lu.btran(&mut y).expect("btran");
        let bres = max_abs_diff(&matvec_t(&basis, &y), &rhs);
        assert!(bres < 1e-7, "update {s}: btran residual {bres:.3e}");
    }
    assert!(
        applied >= 30,
        "chain should apply most updates, got {applied}"
    );
}

/// A wide-bump update that trips the growth budget must roll back cleanly —
/// restoring both `U` and the `uperm` rank order — so the factorization is
/// unchanged and still solves the pre-update basis correctly.
#[test]
fn wide_bump_growth_rollback_leaves_self_solvable() {
    let m = 40;
    let cols = tridiag_cols(m);
    let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
    let sym = SparseLuSymbolic::natural(m);
    // A tiny growth budget so a dense-spike update is rejected.
    let params = LuParams {
        max_updates: 100_000,
        max_growth: 1.0 + 1e-9,
        ..LuParams::default()
    };
    let mut lu = SparseLu::factor(&a, &sym, params).expect("factor");

    // Baseline solve on the original basis.
    let rhs: Vec<f64> = (0..m).map(|i| 1.0 + (i % 4) as f64).collect();
    let mut x0 = rhs.clone();
    lu.ftran(&mut x0).expect("ftran");
    let base_res = max_abs_diff(&matvec(&cols, &x0), &rhs);
    assert!(base_res < 1e-9, "baseline ftran residual {base_res:.3e}");

    // A dense-spike update that should exceed the growth budget.
    let mut col = vec![0.0; m];
    for (i, ci) in col.iter_mut().enumerate() {
        *ci = 1.0 + (i % 3) as f64;
    }
    col[0] = 5.0;
    let err = lu.update(0, &col);
    assert!(
        matches!(err, Err(feral::error::FeralError::NeedsRefactor)),
        "tiny growth budget should reject the wide-bump update, got {err:?}"
    );

    // Self unchanged: the original basis still solves to the same answer.
    let mut x1 = rhs.clone();
    lu.ftran(&mut x1).expect("ftran after rollback");
    let res = max_abs_diff(&matvec(&cols, &x1), &rhs);
    assert!(res < 1e-9, "post-rollback ftran residual {res:.3e}");
    assert_eq!(x0, x1, "rollback must leave the factor bit-identical");
}

/// The FT update records a *pivotal-row-local* eta, not the O(bump²) eta of a
/// full bump re-triangularization. For a tridiagonal base the dense spike folds
/// into the chain factor, so the eta is O(m) (the pivotal row cascades the whole
/// chain) — but it must stay **sub-quadratic**. Across a 16× span in m the eta
/// must grow far slower than the ~256× a quadratic eta would (the old code's eta
/// was ≈ m²/2: 30k at m=250, 2.1M at m=4000).
#[test]
fn wide_bump_eta_is_subquadratic_in_m() {
    let params = LuParams {
        max_updates: 100_000,
        max_growth: 1e30,
        ..LuParams::default()
    };
    let ms = [64usize, 256, 1024];
    let mut max_eta: Vec<usize> = Vec::new();
    for &m in &ms {
        let cols = tridiag_cols(m);
        let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
        let sym = SparseLuSymbolic::natural(m);
        let mut lu = SparseLu::factor(&a, &sym, params.clone()).expect("factor");
        let mut me = 0usize;
        for s in 0..8usize {
            let slot = s % 3;
            let mut col = vec![0.0; m];
            for (i, ci) in col.iter_mut().enumerate() {
                *ci = 0.4 + ((i * 7 + s) % 5) as f64 * 0.2;
            }
            col[slot] = 50.0 + s as f64;
            if lu.update(slot, &col).is_ok() {
                me = me.max(lu.last_eta_ops());
            }
        }
        max_eta.push(me);
    }
    // m grows 16× (64 → 1024). A quadratic eta would grow ~256×; assert the
    // observed growth is well under that (≤ 32×, comfortably between linear and
    // quadratic), proving the O(bump²) re-triangularization is gone.
    let m_ratio = (ms[2] / ms[0]) as f64; // 16
    let eta_ratio = max_eta[2] as f64 / max_eta[0].max(1) as f64;
    assert!(
        eta_ratio <= 2.0 * m_ratio,
        "eta scaled ~quadratically with m (regression): {max_eta:?}, ratio {eta_ratio:.1}× over {m_ratio}× in m"
    );
}
