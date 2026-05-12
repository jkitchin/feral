//! KKT-specific hardening tests targeting real IPM failure modes:
//!
//! 1. Indefinite Hessian (nonconvex NLP)
//! 2. Rank-deficient Jacobian (degenerate constraints)
//! 3. Barrier-scaled diagonal (z/x terms spanning many orders of magnitude)
//! 4. Jacobian-dominated structure (large J, small H)
//! 5. Reconstruction accuracy diagnosis for ill-conditioned KKT

#![allow(clippy::needless_range_loop)]
use feral::{
    factor, solve, solve_refined, BunchKaufmanParams, Inertia, SymmetricMatrix, ZeroPivotAction,
};

fn check_solve(mat: &SymmetricMatrix, x: &[f64], rhs: &[f64], tol: f64) {
    let n = mat.n;
    let mut ax = vec![0.0; n];
    mat.symv(x, &mut ax);
    let rhs_norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-15);
    let resid: f64 = (0..n).map(|i| (ax[i] - rhs[i]).powi(2)).sum::<f64>().sqrt();
    assert!(
        resid / rhs_norm < tol,
        "solve residual: {:.2e} (tol {:.2e})",
        resid / rhs_norm,
        tol
    );
}

// =======================================================================
// Test 1: Indefinite Hessian (nonconvex NLP)
// =======================================================================
// In nonconvex NLP, the Lagrangian Hessian can have negative eigenvalues.
// Without primal perturbation, the expected inertia is NOT (n, m, 0).
// FERAL must report the actual inertia so the IPM layer can detect the
// wrong inertia and add δ_w perturbation.
#[test]
fn test_kkt_indefinite_hessian_no_perturbation() {
    // H = [[2, 0], [0, -3]] — one positive, one negative eigenvalue
    // J = [[1, 1]] (one constraint)
    // δ = 1e-8
    //
    // Full 3×3 KKT:
    // [ 2    0    1  ]
    // [ 0   -3    1  ]
    // [ 1    1   -1e-8]
    //
    // Without perturbation: H has inertia (1,1), so KKT inertia is NOT (2,1,0).
    // The IPM layer would detect this and add δ_w to the H diagonal.
    let mut mat = SymmetricMatrix::zeros(3);
    mat.set(0, 0, 2.0);
    mat.set(1, 1, -3.0);
    mat.set(2, 0, 1.0);
    mat.set(2, 1, 1.0);
    mat.set(2, 2, -1e-8);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    let (factors, inertia) = factor(&mat, &params).expect("factor failed");

    // The matrix has eigenvalues: ~2.35, ~-0.35, ~-3.0 (approx)
    // So inertia should be (1, 2, 0) — NOT the desired (2, 1, 0).
    // This is the signal to the IPM that the Hessian needs perturbation.
    assert_eq!(
        inertia,
        Inertia::new(1, 2, 0),
        "indefinite H should give wrong inertia for KKT — signals need for perturbation"
    );
    assert_eq!(inertia.total(), 3);

    // Solve should still work (matrix is nonsingular)
    let rhs = vec![1.0, 2.0, 0.5];
    let x = solve_refined(&mat, &factors, &rhs).expect("solve failed");
    check_solve(&mat, &x, &rhs, 1e-6);
}

// After the IPM adds perturbation δ_w to the H diagonal, the inertia should
// become (n, m, 0).
#[test]
fn test_kkt_indefinite_hessian_with_perturbation() {
    // Same matrix but with δ_w = 4.0 added to H diagonal
    // H_perturbed = [[2+4, 0], [0, -3+4]] = [[6, 0], [0, 1]] — now SPD
    let mut mat = SymmetricMatrix::zeros(3);
    mat.set(0, 0, 6.0); // 2 + δ_w
    mat.set(1, 1, 1.0); // -3 + δ_w
    mat.set(2, 0, 1.0);
    mat.set(2, 1, 1.0);
    mat.set(2, 2, -1e-8);

    let params = BunchKaufmanParams::default();
    let (factors, inertia) = factor(&mat, &params).expect("factor failed");

    assert_eq!(
        inertia,
        Inertia::new(2, 1, 0),
        "perturbed KKT should have correct inertia (n=2, m=1, 0)"
    );

    let rhs = vec![1.0, 2.0, 0.5];
    let x = solve(&factors, &rhs).expect("solve failed");
    check_solve(&mat, &x, &rhs, 1e-6);
}

// =======================================================================
// Test 2: Rank-deficient Jacobian (degenerate constraints)
// =======================================================================
#[test]
fn test_kkt_rank_deficient_jacobian() {
    // Two constraints, but J rows are linearly dependent:
    // J = [[1, 1, 0], [2, 2, 0]] — rank 1, not 2
    // H = diag(2, 3, 1) (SPD)
    // δ = 1e-8
    //
    // The constraint block is rank-deficient → one zero pivot expected.
    let n = 5; // 3 vars + 2 constraints
    let mut mat = SymmetricMatrix::zeros(n);

    // H block
    mat.set(0, 0, 2.0);
    mat.set(1, 1, 3.0);
    mat.set(2, 2, 1.0);

    // J block (rank deficient: row 2 = 2 * row 1)
    mat.set(3, 0, 1.0);
    mat.set(3, 1, 1.0);
    mat.set(4, 0, 2.0);
    mat.set(4, 1, 2.0);

    // -δI constraint regularization
    mat.set(3, 3, -1e-8);
    mat.set(4, 4, -1e-8);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    let (factors, inertia) = factor(&mat, &params).expect("factor failed");

    assert_eq!(inertia.total(), n);
    // With rank-deficient J and tiny δ, we expect a near-zero pivot.
    // The exact inertia depends on the interaction between J's rank deficiency
    // and the regularization. Just verify the factorization completes and
    // the solve produces a reasonable answer.

    let rhs = vec![1.0, 2.0, 3.0, 0.1, 0.2];
    let x = solve_refined(&mat, &factors, &rhs).expect("solve failed");
    check_solve(&mat, &x, &rhs, 1.0); // Very loose — system is nearly singular
}

// =======================================================================
// Test 3: Barrier-scaled diagonal (z/x terms)
// =======================================================================
// In an IPM, the (1,1) block of the KKT has H + Σ where Σ = diag(z_i/x_i).
// At early iterations, z_i/x_i ≈ 1. Near convergence, some are 1e-15 and
// others are 1e15, creating extreme diagonal scaling within the H block.
#[test]
fn test_kkt_barrier_scaling() {
    let n_var = 6;
    let n_con = 2;
    let n = n_var + n_con;

    let mut mat = SymmetricMatrix::zeros(n);

    // H block: small base values + barrier terms z/x on diagonal
    // Simulating near-convergence: some variables are at bounds (z/x large),
    // some are free (z/x small)
    let barrier_terms = [1e15, 1e-15, 1e10, 1e-10, 1.0, 1.0];
    for i in 0..n_var {
        mat.set(i, i, 1.0 + barrier_terms[i]); // H_ii + z_i/x_i
    }
    // Some off-diagonal H entries
    mat.set(1, 0, 0.5);
    mat.set(3, 2, -0.3);
    mat.set(5, 4, 0.1);

    // Jacobian
    mat.set(6, 0, 1.0);
    mat.set(6, 2, 1.0);
    mat.set(6, 4, 1.0);
    mat.set(7, 1, 1.0);
    mat.set(7, 3, 1.0);
    mat.set(7, 5, 1.0);

    // Constraint regularization
    mat.set(6, 6, -1e-8);
    mat.set(7, 7, -1e-8);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    let (factors, inertia) = factor(&mat, &params).expect("barrier-scale factor failed");

    assert_eq!(inertia.total(), n);
    // H + Σ should be positive definite (barrier terms dominate), so expect (n_var, n_con, 0)
    assert_eq!(
        inertia,
        Inertia::new(n_var, n_con, 0),
        "barrier-scaled KKT: expected ({}, {}, 0), got {}",
        n_var,
        n_con,
        inertia
    );

    let rhs = vec![1.0; n];
    let x = solve_refined(&mat, &factors, &rhs).expect("barrier-scale solve failed");
    // Very ill-conditioned — just check it doesn't blow up
    check_solve(&mat, &x, &rhs, 1e2);
}

// =======================================================================
// Test 4: Jacobian-dominated structure
// =======================================================================
// When J entries are much larger than H entries, the BK algorithm may
// need many 2×2 pivots at the H/J boundary.
#[test]
fn test_kkt_jacobian_dominated() {
    let n_var = 4;
    let n_con = 2;
    let n = n_var + n_con;

    let mut mat = SymmetricMatrix::zeros(n);

    // Tiny H block (small Hessian)
    for i in 0..n_var {
        mat.set(i, i, 0.01);
    }

    // Large Jacobian
    mat.set(4, 0, 100.0);
    mat.set(4, 1, 50.0);
    mat.set(5, 2, 100.0);
    mat.set(5, 3, 50.0);

    // Constraint regularization
    mat.set(4, 4, -1e-8);
    mat.set(5, 5, -1e-8);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    let (factors, inertia) = factor(&mat, &params).expect("J-dominated factor failed");

    assert_eq!(inertia.total(), n);
    // With very small H and large J, the BK algorithm should still produce
    // correct inertia. The Schur complement of J block onto H block
    // should make the effective H negative, so we might NOT get (n_var, n_con, 0).
    // What matters is that inertia is reported correctly.

    let rhs = vec![1.0; n];
    let x = solve_refined(&mat, &factors, &rhs).expect("J-dominated solve failed");
    check_solve(&mat, &x, &rhs, 1e-2);
}

// =======================================================================
// Test 5: Diagnose reconstruction accuracy for KKT
// =======================================================================
// This test specifically exercises the case that showed 0.22 reconstruction
// error in property tests, but with diagnostic output.
#[test]
fn test_kkt_reconstruction_accuracy() {
    // Replicate the random KKT generator from property_tests with seed 999
    // for size (8, 3) to reproduce the failing case.
    struct Rng {
        state: u64,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self { state: seed.max(1) }
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            let t = (self.next_u64() as f64) / (u64::MAX as f64);
            lo + t * (hi - lo)
        }
    }

    let mut rng = Rng::new(999);
    let params = BunchKaufmanParams::default();

    // Skip through smaller sizes to reach (8,3) with the same RNG state
    for &(nv, nc) in &[(3, 1), (5, 2)] {
        for _ in 0..3 {
            let n = nv + nc;
            // Burn RNG state for this matrix
            for _ in 0..(nv * nv + nv * nc + n) {
                rng.uniform(-1.0, 1.0);
            }
        }
    }

    // Now generate the (8, 3) KKT matrix
    let n_var = 8;
    let n_con = 3;
    let n = n_var + n_con;
    let mut mat = SymmetricMatrix::zeros(n);

    for i in 0..n_var {
        for j in 0..=i {
            let val = rng.uniform(-1.0, 1.0);
            if i == j {
                mat.set(i, j, val.abs() + 1.0);
            } else {
                mat.set(i, j, val * 0.3);
            }
        }
    }
    for i in 0..n_var {
        let old = mat.get(i, i);
        mat.set(i, i, old + n_var as f64 * 0.5);
    }
    for i in 0..n_con {
        for j in 0..n_var {
            mat.set(n_var + i, j, rng.uniform(-2.0, 2.0));
        }
    }
    let delta = 1e-8;
    for i in 0..n_con {
        mat.set(n_var + i, n_var + i, -delta);
    }

    let (factors, inertia) = factor(&mat, &params).expect("factor failed");

    assert_eq!(inertia, Inertia::new(n_var, n_con, 0), "KKT (8,3) inertia");
    assert_eq!(inertia.total(), n);

    // The key test: does solve actually produce correct answers?
    // Even if P·L·D·Lᵀ·Pᵀ reconstruction has large error (due to O(n³) FP
    // error amplified by κ≈1e8), the solve should be accurate.
    let rhs: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3 + 1.0).collect();
    let x = solve(&factors, &rhs).expect("solve failed");
    check_solve(&mat, &x, &rhs, 1e-3);

    // Also verify with refinement
    let x_ref = solve_refined(&mat, &factors, &rhs).expect("solve_refined failed");
    check_solve(&mat, &x_ref, &rhs, 1e-3);
}

// =======================================================================
// Test 6: Full IPM trajectory simulation
// =======================================================================
// Simulates what POUNCE would do: factorize, check inertia, perturb if wrong,
// refactorize. Tests the full interaction loop.
#[test]
fn test_kkt_inertia_correction_loop() {
    let n_var = 3;
    let n_con = 1;
    let n = n_var + n_con;

    // H is indefinite: eigenvalues ~ 5, 2, -1
    let mut mat = SymmetricMatrix::zeros(n);
    mat.set(0, 0, 5.0);
    mat.set(1, 0, 0.5);
    mat.set(1, 1, 2.0);
    mat.set(2, 0, 0.1);
    mat.set(2, 1, -0.2);
    mat.set(2, 2, -1.0); // negative eigenvalue

    mat.set(3, 0, 1.0);
    mat.set(3, 1, 1.0);
    mat.set(3, 2, 1.0);
    mat.set(3, 3, -1e-8);

    let params = BunchKaufmanParams::default();

    // First factorization: wrong inertia expected
    let (_, inertia1) = factor(&mat, &params).expect("first factor failed");
    let correct_inertia = Inertia::new(n_var, n_con, 0);

    if inertia1 != correct_inertia {
        // IPM would add perturbation. Try increasing δ_w.
        let mut delta_w = 1e-4;
        let mut found_correct = false;

        for _ in 0..20 {
            let mut perturbed = SymmetricMatrix::zeros(n);
            for i in 0..n {
                for j in 0..=i {
                    perturbed.set(i, j, mat.get(i, j));
                }
            }
            // Add δ_w to primal diagonal
            for i in 0..n_var {
                let old = perturbed.get(i, i);
                perturbed.set(i, i, old + delta_w);
            }

            let (factors, inertia) = factor(&perturbed, &params).expect("perturbed factor failed");

            if inertia == correct_inertia {
                // Success — verify solve works
                let rhs = vec![1.0, 2.0, 3.0, 0.5];
                let x = solve(&factors, &rhs).expect("solve failed");
                check_solve(&perturbed, &x, &rhs, 1e-6);
                found_correct = true;
                break;
            }

            delta_w *= 8.0; // Ipopt-style escalation
        }

        assert!(
            found_correct,
            "inertia correction loop should find correct inertia"
        );
    }
}

// =======================================================================
// Test 7: Larger KKT with realistic Hessian structure
// =======================================================================
#[test]
fn test_kkt_realistic_hessian() {
    // Sparse-ish Hessian: tridiagonal + barrier terms
    let n_var = 20;
    let n_con = 5;
    let n = n_var + n_con;

    let mut mat = SymmetricMatrix::zeros(n);

    // Tridiagonal H with barrier terms
    for i in 0..n_var {
        // Barrier term: large for variables near bounds, small for free
        let barrier = if i % 3 == 0 { 1e6 } else { 1.0 };
        mat.set(i, i, 2.0 + barrier);
        if i + 1 < n_var {
            mat.set(i + 1, i, -0.5);
        }
    }

    // Jacobian: each constraint involves 4 consecutive variables
    for c in 0..n_con {
        let start = c * 4;
        for j in start..(start + 4).min(n_var) {
            mat.set(n_var + c, j, 1.0 + (j as f64) * 0.1);
        }
        mat.set(n_var + c, n_var + c, -1e-8);
    }

    let params = BunchKaufmanParams::default();
    let (factors, inertia) = factor(&mat, &params).expect("realistic KKT factor failed");

    assert_eq!(
        inertia,
        Inertia::new(n_var, n_con, 0),
        "realistic KKT inertia"
    );
    assert_eq!(inertia.total(), n);

    let rhs: Vec<f64> = (0..n).map(|i| ((i + 1) as f64) * 0.1).collect();
    let x = solve(&factors, &rhs).expect("realistic KKT solve failed");
    check_solve(&mat, &x, &rhs, 1e-4);
}
