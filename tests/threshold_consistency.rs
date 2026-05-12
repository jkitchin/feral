//! Tests for the pivot threshold consistency between factor and solve.
//!
//! Before this fix, `factor` flagged a pivot as numerically zero when
//! `|d| <= zero_tol = 100·eps ≈ 2.22e-14` but the dense `solve` divided
//! by it whenever `|d| > eps·1e-10 ≈ 2.22e-26`. Pivots in the band
//! [2.22e-26, 2.22e-14] were counted as zero in inertia AND divided by
//! in solve, producing catastrophic error on rank-deficient matrices.
//!
//! Symptom case: POLAK6_0021 (κ ≈ 1e46) had residual 8.97e-1 before the
//! fix and 4.6e-17 (machine precision) after.
//!
//! See dev/plans/threshold-mismatch-fix.md.

#![allow(clippy::needless_range_loop)]
use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::{solve_sparse, solve_sparse_refined};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    factor, read_mtx, read_sidecar, solve, solve_refined, BunchKaufmanParams, CscMatrix,
    SymmetricMatrix, ZeroPivotAction,
};
use std::path::Path;

fn ldlt_params() -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    }
}

fn sparse_params() -> NumericParams {
    NumericParams::with_bk(ldlt_params())
}

fn rel_residual_dense(a: &SymmetricMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        let r = ax[i] - b[i];
        rs += r * r;
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

fn rel_residual_csc(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        let r = ax[i] - b[i];
        rs += r * r;
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

#[test]
fn factors_carry_zero_tol_from_params() {
    // Verify the fields exist and are populated from BunchKaufmanParams.
    let mut mat = SymmetricMatrix::zeros(2);
    mat.set(0, 0, 1.0);
    mat.set(1, 1, 1.0);
    let params = ldlt_params();
    let (factors, _) = factor(&mat, &params).expect("factor");
    assert_eq!(factors.zero_tol, params.zero_tol);
    assert_eq!(factors.zero_tol_2x2, params.zero_tol_2x2);
}

#[test]
fn dense_solve_skips_zero_pivots_rank_deficient() {
    // 3×3 rank-2 matrix:
    //   [ 2  1  0 ]
    //   [ 1  1  1 ]
    //   [ 0  1  2 ]
    // Determinant = 2(1·2−1·1) − 1(1·2−1·0) + 0 = 2 − 2 = 0.
    // Eigenvalues are 0, 2, 3. After BK factorization one D pivot is
    // numerically zero (within zero_tol of zero from elimination round-off).
    // ForceAccept counts it as zero in inertia. The solve must NOT divide
    // by that pivot — the threshold-mismatch fix makes it skip cleanly.
    //
    // Inertia: 2 positive, 0 negative, 1 zero.
    let mut mat = SymmetricMatrix::zeros(3);
    mat.set(0, 0, 2.0);
    mat.set(1, 0, 1.0);
    mat.set(1, 1, 1.0);
    mat.set(2, 1, 1.0);
    mat.set(2, 2, 2.0);

    let (factors, inertia) = factor(&mat, &ldlt_params()).expect("factor");
    // Equilibration may slightly perturb the round-off, so accept either
    // (3,0,0) "got lucky and counted as positive" or (2,0,1) "force-accepted".
    // The interesting case is the (2,0,1) one — verify the solve regardless.
    assert!(
        inertia.zero <= 1 && inertia.positive >= 2,
        "unexpected inertia {} for rank-2 matrix",
        inertia
    );

    // Pick a RHS in the column space of the matrix: A · [1; 1; 1] = [3; 3; 3]
    let rhs = vec![3.0, 3.0, 3.0];
    let x = solve(&factors, &rhs).expect("solve");

    // No component should explode from dividing by a tiny pivot
    for i in 0..3 {
        assert!(
            x[i].abs() < 1e10,
            "x[{}] = {} — likely divided by a force-accepted zero pivot",
            i,
            x[i]
        );
    }

    // Residual should be small (RHS is in column space)
    let res = rel_residual_dense(&mat, &x, &rhs);
    assert!(
        res < 1e-10,
        "rank-deficient residual {:.3e} too large — likely a divide-by-tiny-pivot",
        res
    );
}

#[test]
fn sparse_solve_skips_zero_pivots_rank_deficient() {
    // Same matrix as the dense test, via the sparse path.
    let csc = CscMatrix::from_triplets(
        3,
        &[0, 1, 1, 2, 2],
        &[0, 0, 1, 1, 2],
        &[2.0, 1.0, 1.0, 1.0, 2.0],
    )
    .unwrap();
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (factors, inertia) =
        factorize_multifrontal(&csc, &sym, &sparse_params()).expect("sparse factor");
    assert!(
        inertia.zero <= 1 && inertia.positive >= 2,
        "unexpected sparse inertia {}",
        inertia
    );

    let rhs = vec![3.0, 3.0, 3.0];
    let x = solve_sparse(&factors, &rhs).expect("solve_sparse");

    for i in 0..3 {
        assert!(
            x[i].abs() < 1e10,
            "sparse x[{}] = {} — likely divided by a force-accepted zero pivot",
            i,
            x[i]
        );
    }

    let res = rel_residual_csc(&csc, &x, &rhs);
    assert!(
        res < 1e-10,
        "sparse rank-deficient residual {:.3e} too large",
        res
    );
}

#[test]
fn refinement_does_not_amplify_error_on_rank_deficient_matrix() {
    // Same rank-deficient matrix used above. Both solve and solve_refined
    // should produce a small residual; best-iterate guarantees refinement
    // is no worse than unrefined.
    let mut mat = SymmetricMatrix::zeros(3);
    mat.set(0, 0, 2.0);
    mat.set(1, 0, 1.0);
    mat.set(1, 1, 1.0);
    mat.set(2, 1, 1.0);
    mat.set(2, 2, 2.0);

    let (factors, _) = factor(&mat, &ldlt_params()).expect("factor");
    let rhs = vec![3.0, 3.0, 3.0];

    let x_un = solve(&factors, &rhs).expect("solve");
    let x_ref = solve_refined(&mat, &factors, &rhs).expect("solve_refined");

    let r_un = rel_residual_dense(&mat, &x_un, &rhs);
    let r_ref = rel_residual_dense(&mat, &x_ref, &rhs);

    // Best-iterate refinement guarantees: r_ref <= r_un (with FP slop).
    assert!(
        r_ref <= r_un + 1e-15,
        "refinement amplified error: unrefined {:.3e}, refined {:.3e}",
        r_un,
        r_ref
    );

    // Both should be at machine precision (RHS is in the column space).
    assert!(r_ref < 1e-12, "refined residual {:.3e} too large", r_ref);
}

#[test]
#[ignore]
fn polak6_0021_residual_after_threshold_fix() {
    // Real-world regression: POLAK6_0021 had residual 8.97e-1 before
    // the threshold-mismatch fix and 4.6e-17 after. Gated #[ignore]d
    // because the data file is not committed.
    let mtx_path = Path::new("data/matrices/kkt/POLAK6/POLAK6_0021.mtx");
    let json_path = Path::new("data/matrices/kkt/POLAK6/POLAK6_0021.json");
    if !mtx_path.exists() || !json_path.exists() {
        eprintln!("SKIP: {} not found", mtx_path.display());
        return;
    }

    let mtx = read_mtx(mtx_path).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let dense = mtx.to_dense();
    let sc = read_sidecar(json_path).expect("read sidecar");
    let rhs = sc.finite_rhs().expect("finite rhs");

    // Dense path
    let (dfac, _) = factor(&dense, &ldlt_params()).expect("dense factor");
    let xd = solve_refined(&dense, &dfac, &rhs).expect("dense solve_refined");
    let rd = rel_residual_csc(&csc, &xd, &rhs);
    assert!(
        rd < 1e-6,
        "POLAK6_0021 dense residual = {:.3e} — threshold-mismatch fix regressed",
        rd
    );

    // Sparse path
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (sfac, _) = factorize_multifrontal(&csc, &sym, &sparse_params()).expect("sparse factor");
    let xs = solve_sparse_refined(&csc, &sfac, &rhs).expect("solve_sparse_refined");
    let rs = rel_residual_csc(&csc, &xs, &rhs);
    assert!(
        rs < 1e-6,
        "POLAK6_0021 sparse residual = {:.3e} — threshold-mismatch fix regressed",
        rs
    );
}

#[test]
fn factor_inertia_force_accept_implies_solve_skip_invariant() {
    // Invariant: every pivot counted as zero in inertia must satisfy
    // |d_diag[k]| <= factors.zero_tol. This is the property that the
    // threshold-mismatch fix relies on — solve uses factors.zero_tol to
    // decide whether to divide, so if factor counted a pivot as zero
    // it must be at-or-below the same threshold.
    //
    // Use a 4x4 block-diagonal matrix with two rank-1 blocks of [[1,1],[1,1]].
    // Each block has eigenvalues 0 and 2. Inertia: (2, 0, 2).
    let mut mat = SymmetricMatrix::zeros(4);
    mat.set(0, 0, 1.0);
    mat.set(1, 0, 1.0);
    mat.set(1, 1, 1.0);
    mat.set(2, 2, 1.0);
    mat.set(3, 2, 1.0);
    mat.set(3, 3, 1.0);

    let (factors, inertia) = factor(&mat, &ldlt_params()).expect("factor");
    // Equilibration may turn this into 2 positives + 2 zeros, or detect a
    // 2x2 block. Either way, the count of zero pivots should be 2.
    assert_eq!(
        inertia.zero + inertia.positive,
        4,
        "got inertia {}",
        inertia
    );
    assert_eq!(inertia.negative, 0, "got inertia {}", inertia);

    // Bulk invariant: the count of d_diag entries with |d| <= zero_tol
    // is at least inertia.zero.
    let n_below: usize = factors
        .d_diag
        .iter()
        .filter(|d| d.abs() <= factors.zero_tol)
        .count();
    assert!(
        n_below >= inertia.zero,
        "{} d_diag entries below zero_tol but inertia.zero = {}",
        n_below,
        inertia.zero
    );
}
