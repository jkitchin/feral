//! Exact tests for dense LDLᵀ factorization with Bunch-Kaufman pivoting.
//!
//! Test matrices are designed to exercise every branch of the BK pivot
//! selection algorithm (research note Section 9.1) and to verify exact
//! inertia counting for 2×2 blocks.

use feral::{
    factor, solve, solve_refined, BunchKaufmanParams, FeralError, Inertia, SymmetricMatrix,
    ZeroPivotAction,
};

/// Helper: create a SymmetricMatrix from a row-major 2D array (lower triangle).
fn sym_from_dense(rows: &[&[f64]]) -> SymmetricMatrix {
    let n = rows.len();
    let mut mat = SymmetricMatrix::zeros(n);
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if j <= i {
                mat.set(i, j, val);
            }
        }
    }
    mat
}

/// Helper: params with equilibration disabled (d_eq = identity) for testing BK in isolation.
/// We use a very well-conditioned matrix so equilibration is close to identity.
fn default_params() -> BunchKaufmanParams {
    BunchKaufmanParams::default()
}

/// Helper: verify P·L·D·Lᵀ·Pᵀ = D_eq·A·D_eq by computing the product and comparing.
fn verify_factorization(mat: &SymmetricMatrix, factors: &feral::Factors, tol: f64) {
    let n = factors.n;

    // Reconstruct P·L·D·Lᵀ·Pᵀ
    // First: L·D·Lᵀ
    let mut ldlt = vec![0.0; n * n];

    // Build D as a full matrix
    let mut d_full = vec![0.0; n * n];
    let mut k = 0;
    while k < n {
        if k + 1 < n && factors.d_subdiag[k] != 0.0 {
            // 2×2 block
            d_full[k * n + k] = factors.d_diag[k];
            d_full[k * n + (k + 1)] = factors.d_subdiag[k];
            d_full[(k + 1) * n + k] = factors.d_subdiag[k];
            d_full[(k + 1) * n + (k + 1)] = factors.d_diag[k + 1];
            k += 2;
        } else {
            d_full[k * n + k] = factors.d_diag[k];
            k += 1;
        }
    }

    // Compute L·D
    let mut ld = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..n {
                sum += factors.l[p * n + i] * d_full[p * n + j];
            }
            ld[j * n + i] = sum;
        }
    }

    // Compute (L·D)·Lᵀ
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..n {
                sum += ld[p * n + i] * factors.l[p * n + j]; // Lᵀ[p,j] = L[j,p]
            }
            ldlt[j * n + i] = sum;
        }
    }

    // Apply permutation: P·(LDLᵀ)·Pᵀ
    // Result[perm[i], perm[j]] = ldlt[i, j]
    // Or equivalently: result[i,j] = ldlt[perm_inv[i], perm_inv[j]]
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let pi = factors.perm_inv[i];
            let pj = factors.perm_inv[j];
            result[j * n + i] = ldlt[pj * n + pi];
        }
    }

    // Compare with D_eq·A·D_eq
    for i in 0..n {
        for j in 0..=i {
            let expected = factors.d_eq[i] * mat.get(i, j) * factors.d_eq[j];
            let got = result[j * n + i];
            let err = (expected - got).abs();
            let scale = expected.abs().max(got.abs()).max(1e-15);
            assert!(
                err / scale < tol,
                "factorization mismatch at ({},{}): expected {}, got {}, err/scale = {}",
                i,
                j,
                expected,
                got,
                err / scale
            );
        }
    }
}

/// Helper: verify that Ax = b by computing the residual.
fn verify_solve(mat: &SymmetricMatrix, x: &[f64], rhs: &[f64], tol: f64) {
    let n = mat.n;
    let mut ax = vec![0.0; n];
    mat.symv(x, &mut ax);

    let rhs_norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = if rhs_norm > 0.0 { rhs_norm } else { 1.0 };

    for i in 0..n {
        let err = (ax[i] - rhs[i]).abs();
        assert!(
            err / scale < tol,
            "solve residual at row {}: |Ax-b| = {}, scale = {}",
            i,
            err,
            scale
        );
    }
}

// =======================================================================
// Test 1: 1×1 pivot, no swap (Test 3 passes: |A[0,0]| >= α·γ₀)
// =======================================================================
#[test]
fn test_1x1_pivot_no_swap() {
    // A diagonal-dominant matrix where the first diagonal is large enough
    // relative to the off-diagonal column maximum.
    // α ≈ 0.6404, so |A[0,0]| >= α·γ₀ means diagonal dominates.
    let mat = sym_from_dense(&[&[4.0], &[1.0, 3.0], &[0.5, 0.5, 2.0]]);

    let params = default_params();
    let (factors, inertia) = factor(&mat, &params).ok().expect("factor failed");

    // All eigenvalues should be positive (SPD-like)
    assert_eq!(inertia, Inertia::new(3, 0, 0), "expected (3,0,0) inertia");
    verify_factorization(&mat, &factors, 1e-12);

    // Test solve
    let rhs = vec![1.0, 2.0, 3.0];
    let x = solve(&factors, &rhs).ok().expect("solve failed");
    verify_solve(&mat, &x, &rhs, 1e-12);
}

// =======================================================================
// Test 2: 1×1 pivot with swap (Test 5 passes: |A[r,r]| >= α·γᵣ)
// =======================================================================
#[test]
fn test_1x1_pivot_with_swap() {
    // A[0,0] is small, but A[r,r] (where r is the column-max row) is large.
    // This forces Test 3 to fail but Test 5 to pass.
    let mat = sym_from_dense(&[&[0.1], &[0.5, 5.0], &[0.3, 0.2, 3.0]]);

    let params = default_params();
    let (factors, inertia) = factor(&mat, &params).ok().expect("factor failed");

    assert_eq!(inertia, Inertia::new(3, 0, 0));
    verify_factorization(&mat, &factors, 1e-12);

    let rhs = vec![1.0, 2.0, 3.0];
    let x = solve(&factors, &rhs).ok().expect("solve failed");
    verify_solve(&mat, &x, &rhs, 1e-12);
}

// =======================================================================
// Test 3: 2×2 pivot (all 1×1 tests fail)
// =======================================================================
#[test]
fn test_2x2_pivot() {
    // An indefinite matrix where the off-diagonal dominates, forcing a 2×2 pivot.
    // A = [[0.01, 5.0], [5.0, 0.02]]
    // γ₀ = 5.0, |A[0,0]| = 0.01 << α·5.0 = 3.2, so Test 3 fails.
    // γᵣ for row 1: max off-diag in row 1 = 5.0. |A[1,1]| = 0.02 << α·5.0, Test 5 fails.
    // Test 6: |A[0,0]|·γᵣ = 0.01·5.0 = 0.05 < α·γ₀² = 0.6404·25 = 16.01. Fails.
    // → 2×2 pivot.
    let mat = sym_from_dense(&[&[0.01], &[5.0, 0.02]]);

    let params = default_params();
    let (factors, inertia) = factor(&mat, &params).ok().expect("factor failed");

    // det = 0.01*0.02 - 25 = -24.9998 < 0 → inertia (1, 1, 0)
    assert_eq!(
        inertia,
        Inertia::new(1, 1, 0),
        "2x2 block must be indefinite"
    );
    verify_factorization(&mat, &factors, 1e-12);

    let rhs = vec![1.0, 2.0];
    let x = solve(&factors, &rhs).ok().expect("solve failed");
    verify_solve(&mat, &x, &rhs, 1e-10);
}

// =======================================================================
// Test 4: 2×2 block with positive diagonals but negative determinant
// (Inertia-critical: must NOT be counted as (2,0,0))
// =======================================================================
#[test]
fn test_2x2_inertia_positive_diag_negative_det() {
    // D block = [[1, 3], [3, 2]], det = 1·2 − 9 = −7 < 0
    // Both diagonals positive, but inertia is (1, 1, 0), NOT (2, 0, 0).
    // We construct a matrix that produces this D block.
    let mat = sym_from_dense(&[&[1.0], &[3.0, 2.0]]);

    let params = default_params();
    let (factors, inertia) = factor(&mat, &params).ok().expect("factor failed");

    assert_eq!(
        inertia,
        Inertia::new(1, 1, 0),
        "2x2 with positive diags but det<0 must be (1,1,0)"
    );
    verify_factorization(&mat, &factors, 1e-12);
}

// =======================================================================
// Test 5: Small KKT matrix with known inertia (n, m, 0)
// =======================================================================
#[test]
fn test_kkt_structure() {
    // KKT matrix: [[H, Jᵀ], [J, -δI]]
    // H = [[2, 0], [0, 2]] (positive definite Hessian, n=2)
    // J = [[1, 1]] (one constraint, m=1)
    // δ = 1e-8 (small regularization)
    // Expected inertia: (2, 1, 0)
    //
    // Full matrix (3×3):
    // [ 2    0    1  ]
    // [ 0    2    1  ]
    // [ 1    1   -1e-8]
    let mat = sym_from_dense(&[&[2.0], &[0.0, 2.0], &[1.0, 1.0, -1e-8]]);

    let params = default_params();
    let (factors, inertia) = factor(&mat, &params).ok().expect("factor failed");

    assert_eq!(
        inertia,
        Inertia::new(2, 1, 0),
        "KKT should have inertia (n=2, m=1, 0)"
    );
    verify_factorization(&mat, &factors, 1e-6); // KKT: κ ≈ 1/δ = 1e8, expect ~n*κ*eps ≈ 1e-7

    let rhs = vec![1.0, 2.0, 0.5];
    let x = solve(&factors, &rhs).ok().expect("solve failed");
    verify_solve(&mat, &x, &rhs, 1e-4); // KKT is ill-conditioned (κ ≈ 1/δ)
}

// =======================================================================
// Test 6: Identity matrix
// =======================================================================
#[test]
fn test_identity() {
    let n = 5;
    let mut mat = SymmetricMatrix::zeros(n);
    for i in 0..n {
        mat.set(i, i, 1.0);
    }

    let params = default_params();
    let (factors, inertia) = factor(&mat, &params).ok().expect("factor failed");

    assert_eq!(inertia, Inertia::new(5, 0, 0));
    verify_factorization(&mat, &factors, 1e-14);

    let rhs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = solve(&factors, &rhs).ok().expect("solve failed");
    verify_solve(&mat, &x, &rhs, 1e-14);
}

// =======================================================================
// Test 7: Negative definite matrix
// =======================================================================
#[test]
fn test_negative_definite() {
    let mat = sym_from_dense(&[&[-4.0], &[-1.0, -3.0], &[-0.5, -0.5, -2.0]]);

    let params = default_params();
    let (factors, inertia) = factor(&mat, &params).ok().expect("factor failed");

    assert_eq!(inertia, Inertia::new(0, 3, 0));
    verify_factorization(&mat, &factors, 1e-12);

    let rhs = vec![1.0, 2.0, 3.0];
    let x = solve(&factors, &rhs).ok().expect("solve failed");
    verify_solve(&mat, &x, &rhs, 1e-12);
}

// =======================================================================
// Test 8: Diagonal matrix (no pivoting needed)
// =======================================================================
#[test]
fn test_diagonal() {
    let mat = sym_from_dense(&[
        &[3.0],
        &[0.0, -2.0],
        &[0.0, 0.0, 1.0],
        &[0.0, 0.0, 0.0, -4.0],
    ]);

    let params = default_params();
    let (factors, inertia) = factor(&mat, &params).ok().expect("factor failed");

    assert_eq!(inertia, Inertia::new(2, 2, 0));
    verify_factorization(&mat, &factors, 1e-14);

    let rhs = vec![6.0, -4.0, 1.0, -8.0];
    let x = solve(&factors, &rhs).ok().expect("solve failed");
    // x should be [2, 2, 1, 2]
    verify_solve(&mat, &x, &rhs, 1e-14);
}

// =======================================================================
// Test 9: ZeroPivotAction::Fail on singular matrix
// =======================================================================
#[test]
fn test_zero_pivot_fail() {
    // Singular matrix
    let mat = sym_from_dense(&[&[1.0], &[1.0, 1.0]]);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::Fail,
        ..BunchKaufmanParams::default()
    };

    let result = factor(&mat, &params);
    assert!(
        matches!(result, Err(FeralError::NumericallyRankDeficient)),
        "singular matrix should fail with Fail action"
    );
}

// =======================================================================
// Test 10: ZeroPivotAction::ForceAccept with solve_refined
// =======================================================================
#[test]
fn test_force_accept_with_refinement() {
    // Exactly singular: rows are identical → rank 1, one zero eigenvalue.
    // Equilibration can't fix this because the matrix is truly rank-deficient.
    let mat = sym_from_dense(&[&[1.0], &[1.0, 1.0]]);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };

    let (factors, inertia) = factor(&mat, &params).ok().expect("factor failed");
    assert!(factors.needs_refinement, "should flag for refinement");
    assert_eq!(inertia.zero, 1, "should have one zero eigenvalue");

    // solve_refined should handle this without panicking
    let rhs = vec![1.0, 1.0];
    let x = solve_refined(&mat, &factors, &rhs)
        .ok()
        .expect("solve_refined failed");
    assert_eq!(x.len(), 2);
}

// =======================================================================
// Test 11: 4×4 indefinite matrix requiring mixed pivot sizes
// =======================================================================
#[test]
fn test_mixed_pivots_4x4() {
    // A 4×4 matrix that requires both 1×1 and 2×2 pivots.
    // Block structure designed to produce mixed pivot types.
    let mat = sym_from_dense(&[
        &[10.0],
        &[1.0, 0.01],
        &[0.5, 5.0, 0.02],
        &[0.1, 0.1, 0.1, 8.0],
    ]);

    let params = default_params();
    let (factors, inertia) = factor(&mat, &params).ok().expect("factor failed");

    // Total inertia should sum to 4
    assert_eq!(inertia.total(), 4, "inertia must sum to n=4");
    verify_factorization(&mat, &factors, 1e-10);

    let rhs = vec![1.0, 2.0, 3.0, 4.0];
    let x = solve(&factors, &rhs).ok().expect("solve failed");
    verify_solve(&mat, &x, &rhs, 1e-10);
}

// =======================================================================
// Test 12: 1×1 matrix edge case
// =======================================================================
#[test]
fn test_1x1_matrix() {
    let mat = sym_from_dense(&[&[7.0]]);

    let params = default_params();
    let (factors, inertia) = factor(&mat, &params).ok().expect("factor failed");

    assert_eq!(inertia, Inertia::new(1, 0, 0));

    let rhs = vec![14.0];
    let x = solve(&factors, &rhs).ok().expect("solve failed");
    verify_solve(&mat, &x, &rhs, 1e-14);
}

// =======================================================================
// Test 13: Input validation
// =======================================================================
#[test]
fn test_zero_dimension_rejected() {
    let mat = SymmetricMatrix::zeros(0);
    let result = factor(&mat, &BunchKaufmanParams::default());
    assert!(matches!(result, Err(FeralError::InvalidInput(_))));
}

#[test]
fn test_nan_rejected() {
    let mut mat = SymmetricMatrix::zeros(2);
    mat.set(0, 0, 1.0);
    mat.set(1, 0, f64::NAN);
    mat.set(1, 1, 2.0);
    let result = factor(&mat, &BunchKaufmanParams::default());
    assert!(matches!(result, Err(FeralError::InvalidInput(_))));
}

#[test]
fn test_solve_dimension_mismatch() {
    let mat = sym_from_dense(&[&[2.0], &[0.0, 3.0]]);
    let (factors, _) = factor(&mat, &BunchKaufmanParams::default())
        .ok()
        .expect("factor failed");
    let result = solve(&factors, &[1.0, 2.0, 3.0]);
    assert!(matches!(
        result,
        Err(FeralError::DimensionMismatch {
            expected: 2,
            got: 3
        })
    ));
}

// =======================================================================
// Test 14: Larger KKT system (5 variables, 2 constraints)
// =======================================================================
#[test]
fn test_kkt_5x2() {
    // H = 5×5 diagonal [2, 3, 1, 4, 2]
    // J = 2×5 = [[1,0,1,0,0], [0,1,0,1,1]]
    // δ = 1e-8
    // Size: 7×7, expected inertia (5, 2, 0)
    let n = 7;
    let mut mat = SymmetricMatrix::zeros(n);

    // Hessian block (diagonal)
    mat.set(0, 0, 2.0);
    mat.set(1, 1, 3.0);
    mat.set(2, 2, 1.0);
    mat.set(3, 3, 4.0);
    mat.set(4, 4, 2.0);

    // Jacobian entries (rows 5-6, cols 0-4)
    mat.set(5, 0, 1.0); // J[0,0]
    mat.set(5, 2, 1.0); // J[0,2]
    mat.set(6, 1, 1.0); // J[1,1]
    mat.set(6, 3, 1.0); // J[1,3]
    mat.set(6, 4, 1.0); // J[1,4]

    // Regularization block
    mat.set(5, 5, -1e-8);
    mat.set(6, 6, -1e-8);

    let params = default_params();
    let (factors, inertia) = factor(&mat, &params).ok().expect("factor failed");

    assert_eq!(
        inertia,
        Inertia::new(5, 2, 0),
        "KKT(5,2) should have inertia (5, 2, 0)"
    );
    verify_factorization(&mat, &factors, 1e-8);

    let rhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.1, 0.2];
    let x = solve(&factors, &rhs).ok().expect("solve failed");
    verify_solve(&mat, &x, &rhs, 1e-4);
}

// =======================================================================
// Test 15: Benchmark harness output verification
// =======================================================================
#[test]
fn test_bench_harness_output() {
    let output = std::process::Command::new("cargo")
        .args(["run", "--bin", "bench"])
        .output()
        .expect("failed to run bench binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("FERAL benchmark harness"),
        "missing harness header"
    );
    assert!(
        stdout.contains("not found"),
        "should report config not found"
    );
    assert!(
        stdout.contains("0 matrices benchmarked"),
        "should report 0 matrices"
    );
}
