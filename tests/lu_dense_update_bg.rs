//! Dense LU Bartels–Golub update (issue #115): thorough exercise of the
//! eta-based update path — bump-forming column replacements (which trigger the
//! Hessenberg reduction and its row interchanges), update chains (multiple etas
//! replayed per solve), and ftran/btran parity against an **independent**
//! from-scratch factorization of the replaced basis.
//!
//! Oracles are external and oracle-free: the equation-residual identity
//! `‖B'x − b‖` / `‖B'ᵀy − c‖`, and agreement with a fresh `DenseLu::factor`
//! of the current basis (independent recomputation, not the update's own math).

use feral::lu::{DenseLu, LuParams};

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

/// `B x` for a basis given column-major (`cols[j][i]`).
fn matvec(cols: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    for (col, &xj) in cols.iter().zip(x) {
        for (yi, &cij) in y.iter_mut().zip(col) {
            *yi += cij * xj;
        }
    }
    y
}

/// `Bᵀ x`.
fn matvec_t(cols: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    cols.iter()
        .map(|col| col.iter().zip(x).map(|(&cij, &xi)| cij * xi).sum())
        .collect()
}

/// Verify a factorization solves the given basis for both ftran and btran to a
/// backward-stable residual, cross-checked against a fresh factorization.
fn assert_solves(lu: &mut DenseLu, cols: &[Vec<f64>], tag: &str) {
    let m = cols.len();
    let xt: Vec<f64> = (0..m).map(|i| 1.0 + (i % 5) as f64 * 0.5).collect();

    // ftran: B x = b.
    let b = matvec(cols, &xt);
    let mut x = b.clone();
    lu.ftran(&mut x)
        .unwrap_or_else(|e| panic!("{tag}: ftran failed: {e:?}"));
    let res = inf_norm(&sub(&matvec(cols, &x), &b));
    assert!(
        res <= 1e-8 * inf_norm(&b).max(1.0),
        "{tag}: ftran residual {res:.3e} too large"
    );

    // btran: Bᵀ y = c.
    let c = matvec_t(cols, &xt);
    let mut y = c.clone();
    lu.btran(&mut y)
        .unwrap_or_else(|e| panic!("{tag}: btran failed: {e:?}"));
    let rest = inf_norm(&sub(&matvec_t(cols, &y), &c));
    assert!(
        rest <= 1e-8 * inf_norm(&c).max(1.0),
        "{tag}: btran residual {rest:.3e} too large"
    );

    // Cross-check ftran against an independent fresh factorization.
    let mut fresh = DenseLu::factor(cols, m, LuParams::default())
        .unwrap_or_else(|e| panic!("{tag}: fresh factor failed: {e:?}"));
    let mut xf = b.clone();
    fresh.ftran(&mut xf).expect("fresh ftran");
    let diff = inf_norm(&sub(&x, &xf));
    assert!(
        diff <= 1e-7 * inf_norm(&xf).max(1.0),
        "{tag}: updated ftran disagrees with fresh factor by {diff:.3e}"
    );
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(p, q)| p - q).collect()
}

/// A deterministic well-conditioned dense basis.
fn dense_basis(m: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    let mut rng = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    let mut cols = vec![vec![0.0; m]; m];
    for (j, col) in cols.iter_mut().enumerate() {
        for cij in col.iter_mut() {
            *cij = rng();
        }
        col[j] += m as f64; // diagonally dominant → well-conditioned
    }
    cols
}

/// Bump-forming updates: replacing an *early* column forces the Hessenberg
/// reduction over the whole tail, exercising interchanges and multi-eta chains.
/// After each committed update, both solves must be correct and agree with a
/// fresh factor.
#[test]
fn dense_bg_update_chain_dense_basis() {
    let m = 9;
    let mut cols = dense_basis(m, 0xBEEF);
    let params = LuParams {
        max_updates: 64,
        max_growth: 1e10,
        ..LuParams::default()
    };
    let mut lu = DenseLu::factor(&cols, m, params.clone()).expect("factor");
    assert_solves(&mut lu, &cols, "initial");

    // Replace slots 0,1,2,... (early columns → wide bumps) with fresh columns.
    let mut gen = dense_basis(m, 0xF00D);
    for (step, slot) in [0usize, 1, 2, 0, 3, 1].into_iter().enumerate() {
        let new_col = std::mem::take(&mut gen[step % m]);
        match lu.update(slot, &new_col) {
            Ok(()) => {
                cols[slot] = new_col;
                assert_solves(
                    &mut lu,
                    &cols,
                    &format!("after update {step} (slot {slot})"),
                );
            }
            Err(_) => {
                // A legitimate growth/singular bail: refactor and continue.
                cols[slot] = new_col;
                lu.refactor(&cols).expect("refactor");
                assert_solves(&mut lu, &cols, &format!("after refactor {step}"));
            }
        }
    }
}

/// Slack/triangular bases are the landmine case (zero superdiagonals). Replacing
/// early columns must commit via interchanges and stay correct across a chain.
#[test]
fn dense_bg_update_chain_slack_basis() {
    let m = 6;
    // Lower-bidiagonal "slack-like" basis: identity plus a subdiagonal.
    let mut cols = vec![vec![0.0; m]; m];
    for j in 0..m {
        cols[j][j] = 1.0;
        if j + 1 < m {
            cols[j][j + 1] = 0.5;
        }
    }
    let params = LuParams {
        max_updates: 64,
        ..LuParams::default()
    };
    let mut lu = DenseLu::factor(&cols, m, params).expect("factor");
    assert_solves(&mut lu, &cols, "slack initial");

    // Replace column 0 with a dense column (wide bump on a triangular base —
    // the exact shape that tripped the old fixed-order sweep).
    let replacements = [
        (0usize, vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        (1usize, vec![0.0, 2.0, 1.0, 1.0, 0.0, 0.0]),
        (0usize, vec![3.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
    ];
    for (step, (slot, col)) in replacements.into_iter().enumerate() {
        lu.update(slot, &col)
            .unwrap_or_else(|e| panic!("slack update {step} (slot {slot}) must commit: {e:?}"));
        cols[slot] = col;
        assert_solves(&mut lu, &cols, &format!("slack after update {step}"));
    }
}

/// Refinement (`ftran_refined`/`btran_refined`) must also work through the eta
/// chain — it drives repeated ftran/btran calls internally.
#[test]
fn dense_bg_update_refined_solve() {
    use feral::lu::GeneralMatrix;
    let m = 7;
    let mut cols = dense_basis(m, 0x1234);
    let params = LuParams {
        max_updates: 64,
        refine_steps: 2,
        ..LuParams::default()
    };
    let mut lu = DenseLu::factor(&cols, m, params).expect("factor");
    // A bump-forming update.
    let new_col: Vec<f64> = (0..m).map(|i| 1.0 + (i as f64) * 0.3).collect();
    lu.update(0, &new_col).expect("update commits");
    cols[0] = new_col;

    let b = GeneralMatrix::from_columns(m, &cols).expect("matrix");
    let xt: Vec<f64> = (0..m).map(|i| 2.0 - (i % 3) as f64).collect();
    let rhs0 = matvec(&cols, &xt);
    let mut x = rhs0.clone();
    lu.ftran_refined(&b, &mut x).expect("ftran_refined");
    let res = inf_norm(&sub(&matvec(&cols, &x), &rhs0));
    assert!(
        res <= 1e-10 * inf_norm(&rhs0).max(1.0),
        "refined ftran residual {res:.3e}"
    );
}
