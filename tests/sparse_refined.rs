//! Tests for `solve_sparse_refined` — iterative refinement on the sparse path.
//!
//! Mirrors the existing dense `solve_refined` tests in tests/property_tests.rs
//! and tests/dense_ldlt.rs. See FERAL-PROJECT-SPEC.md §1709 for the
//! Phase 1b solve convention requiring refinement on all KKT solves.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::{solve_sparse, solve_sparse_refined};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{factor, solve_refined, BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    })
}

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
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
fn solve_sparse_refined_matches_dense_refined_bordered_kkt() {
    // Same hand-built bordered KKT used in sparse_postorder.rs.
    // Both refined paths should produce equivalent answers to machine
    // precision.
    let csc = CscMatrix::from_triplets(
        4,
        &[0, 3, 1, 3, 2, 3, 3],
        &[0, 0, 1, 1, 2, 2, 3],
        &[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0],
    )
    .unwrap();
    let rhs = vec![1.0, 2.0, 3.0, 6.0];

    // Dense refined
    let dense = csc.to_dense();
    let (dfac, _) = factor(&dense, &ldlt_params().bk).expect("dense factor");
    let xd = solve_refined(&dense, &dfac, &rhs).expect("dense solve_refined");

    // Sparse refined
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (sfac, _) = factorize_multifrontal(&csc, &sym, &ldlt_params()).expect("sparse factor");
    let xs = solve_sparse_refined(&csc, &sfac, &rhs).expect("solve_sparse_refined");

    let mut max_diff = 0.0f64;
    for i in 0..4 {
        max_diff = max_diff.max((xd[i] - xs[i]).abs());
    }
    assert!(
        max_diff < 1e-12,
        "dense and sparse refined solutions differ by {:.3e}",
        max_diff
    );

    // Both should give a small residual against the original system.
    assert!(rel_residual(&csc, &xs, &rhs) < 1e-12);
}

#[test]
fn solve_sparse_refined_improves_residual_on_ill_conditioned() {
    // Two-constraint bordered KKT with very ill-conditioned diagonals
    // (slack diagonal = 1e-12 and constraint diagonal = -1e-10).
    // The first sparse solve will leave a non-trivial residual; refinement
    // should reduce it.
    let csc = CscMatrix::from_triplets(
        6,
        &[0, 4, 1, 5, 2, 4, 3, 5, 4, 5],
        &[0, 0, 1, 1, 2, 2, 3, 3, 4, 5],
        &[
            100.0, -1.0, 100.0, -1.0, 1e-12, 1.0, 1e-12, 1.0, -1e-10, -1e-10,
        ],
    )
    .unwrap();
    let rhs = vec![1.0, 2.0, 0.0, 0.0, -50.0, -75.0];

    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (sfac, _) = factorize_multifrontal(&csc, &sym, &ldlt_params()).expect("sparse factor");

    let x_unrefined = solve_sparse(&sfac, &rhs).expect("solve_sparse");
    let res_unrefined = rel_residual(&csc, &x_unrefined, &rhs);

    let x_refined = solve_sparse_refined(&csc, &sfac, &rhs).expect("solve_sparse_refined");
    let res_refined = rel_residual(&csc, &x_refined, &rhs);

    // Refined residual should be at least as good as unrefined and ideally
    // much better. Use a generous bound to avoid flakiness on borderline
    // BK pivot decisions.
    assert!(
        res_refined <= res_unrefined + 1e-15,
        "refinement made residual worse: unrefined {:.3e}, refined {:.3e}",
        res_unrefined,
        res_refined
    );
    // Refined should be at machine-precision-ish for this matrix.
    assert!(
        res_refined < 1e-10,
        "refined residual {:.3e} is too large for this matrix",
        res_refined
    );
}

#[test]
fn solve_sparse_refined_well_conditioned_no_change() {
    // SPD diagonal: refinement should converge in 0 steps and the answer
    // should equal the unrefined solve to machine precision.
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
    let rhs = vec![4.0, 9.0, 25.0];

    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (sfac, _) = factorize_multifrontal(&csc, &sym, &ldlt_params()).expect("sparse factor");

    let x_un = solve_sparse(&sfac, &rhs).expect("solve_sparse");
    let x_ref = solve_sparse_refined(&csc, &sfac, &rhs).expect("solve_sparse_refined");

    for i in 0..3 {
        assert!(
            (x_un[i] - x_ref[i]).abs() < 1e-15,
            "well-conditioned refined diff at i={}: {:.3e}",
            i,
            (x_un[i] - x_ref[i]).abs()
        );
    }

    // Sanity: x = [2, 3, 5]
    assert!((x_ref[0] - 2.0).abs() < 1e-14);
    assert!((x_ref[1] - 3.0).abs() < 1e-14);
    assert!((x_ref[2] - 5.0).abs() < 1e-14);
}

#[test]
fn solve_sparse_refined_residual_monotone_on_singular_matrix() {
    // Singular 3x3 matrix that triggers ForceAccept on a zero pivot.
    // Without the residual-monotone guard, refinement would amplify the
    // error in dx = A⁻¹·r and leave the result worse than the unrefined
    // solve. With the guard, the refined residual must be no worse than
    // the unrefined residual.
    //
    //   [ 1e-16  0   1 ]
    //   [   0    1   0 ]
    //   [   1    0   0 ]
    //
    // The (0,0) pivot is essentially zero. ForceAccept ignores it; the
    // resulting "factorization" is unable to faithfully represent A⁻¹.
    let csc =
        CscMatrix::from_triplets(3, &[0, 2, 1, 2], &[0, 0, 1, 2], &[1e-16, 1.0, 1.0, 0.0]).unwrap();
    let rhs = vec![1.0, 2.0, 3.0];

    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (sfac, _) = factorize_multifrontal(&csc, &sym, &ldlt_params()).expect("sparse factor");

    let x_un = solve_sparse(&sfac, &rhs).expect("solve_sparse");
    let res_un = rel_residual(&csc, &x_un, &rhs);

    let x_ref = solve_sparse_refined(&csc, &sfac, &rhs).expect("solve_sparse_refined");
    let res_ref = rel_residual(&csc, &x_ref, &rhs);

    // Guard property: refined residual must be ≤ unrefined residual.
    // Without the guard this would fail spectacularly on this kind of
    // matrix.
    assert!(
        res_ref <= res_un + 1e-15,
        "residual-monotone guard failed: unrefined {:.3e}, refined {:.3e}",
        res_un,
        res_ref
    );
}

#[test]
fn solve_sparse_refined_dimension_mismatch_returns_error() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0]).unwrap();
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).unwrap();
    let (sfac, _) = factorize_multifrontal(&csc, &sym, &ldlt_params()).unwrap();
    let rhs = vec![1.0, 2.0]; // wrong length
    let result = solve_sparse_refined(&csc, &sfac, &rhs);
    assert!(result.is_err());
}
