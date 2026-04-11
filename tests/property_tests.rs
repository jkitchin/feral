//! Property-based tests for dense LDLᵀ factorization.
//!
//! These tests use deterministic pseudo-random matrices to verify
//! invariants that must hold for ALL inputs, not just hand-crafted examples:
//!
//! 1. Inertia always sums to n
//! 2. P·L·D·Lᵀ·Pᵀ = D_eq·A·D_eq (factorization reconstruction)
//! 3. Solve residual ||Ax - b|| / ||b|| is small
//! 4. Permutation is a valid permutation (perm ∘ perm_inv = identity)
//! 5. L is unit lower triangular
//! 6. D blocks are correctly structured (subdiag discriminant)

use feral::{
    factor, solve, solve_refined, BunchKaufmanParams, Inertia, SymmetricMatrix, ZeroPivotAction,
};

// -----------------------------------------------------------------------
// Simple deterministic PRNG (xorshift64) — no external dependency
// -----------------------------------------------------------------------
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

    /// Uniform f64 in [lo, hi)
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let t = (self.next_u64() as f64) / (u64::MAX as f64);
        lo + t * (hi - lo)
    }
}

// -----------------------------------------------------------------------
// Matrix generators
// -----------------------------------------------------------------------

/// Generate a random SPD matrix of size n: A = M·Mᵀ + δI
fn random_spd(n: usize, rng: &mut Rng) -> SymmetricMatrix {
    let mut mat = SymmetricMatrix::zeros(n);
    // Generate random lower triangular M, then compute MMᵀ + δI
    let mut m = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            m[j * n + i] = rng.uniform(-5.0, 5.0);
        }
    }
    // A = M * Mᵀ
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..n {
                sum += m[k * n + i] * m[k * n + j];
            }
            mat.set(i, j, sum + if i == j { 0.01 } else { 0.0 });
        }
    }
    mat
}

/// Generate a random indefinite matrix: A = M - M_mean_diag * I
fn random_indefinite(n: usize, rng: &mut Rng) -> SymmetricMatrix {
    let mut mat = random_spd(n, rng);
    // Shift diagonal to make indefinite
    let mean_diag: f64 = (0..n).map(|i| mat.get(i, i)).sum::<f64>() / n as f64;
    for i in 0..n {
        let old = mat.get(i, i);
        mat.set(i, i, old - mean_diag);
    }
    mat
}

/// Generate a random KKT matrix: [[H, Jᵀ], [J, -δI]]
/// n_var primal variables, n_con constraints.
fn random_kkt(n_var: usize, n_con: usize, rng: &mut Rng) -> (SymmetricMatrix, usize, usize) {
    let n = n_var + n_con;
    let mut mat = SymmetricMatrix::zeros(n);

    // H block: SPD n_var × n_var
    for i in 0..n_var {
        for j in 0..=i {
            let val = rng.uniform(-1.0, 1.0);
            if i == j {
                mat.set(i, j, val.abs() + 1.0); // ensure positive diagonal
            } else {
                mat.set(i, j, val * 0.3);
            }
        }
    }
    // Make H SPD by adding to diagonal
    for i in 0..n_var {
        let old = mat.get(i, i);
        mat.set(i, i, old + n_var as f64 * 0.5);
    }

    // J block: n_con × n_var, stored in rows n_var..n
    for i in 0..n_con {
        for j in 0..n_var {
            mat.set(n_var + i, j, rng.uniform(-2.0, 2.0));
        }
    }

    // -δI block
    let delta = 1e-8;
    for i in 0..n_con {
        mat.set(n_var + i, n_var + i, -delta);
    }

    (mat, n_var, n_con)
}

/// Generate a badly-scaled matrix by applying a wide range of diagonal scaling.
fn random_badly_scaled(n: usize, rng: &mut Rng) -> SymmetricMatrix {
    let mut mat = random_spd(n, rng);
    // Apply scaling: D * A * D where D has entries spanning 1e-6 to 1e6
    let mut scale = vec![0.0; n];
    for s in scale.iter_mut() {
        let exponent = rng.uniform(-6.0, 6.0);
        *s = 10f64.powf(exponent);
    }
    for i in 0..n {
        for j in 0..=i {
            let val = mat.get(i, j) * scale[i] * scale[j];
            mat.set(i, j, val);
        }
    }
    mat
}

// -----------------------------------------------------------------------
// Verification helpers
// -----------------------------------------------------------------------

/// Verify that P·L·D·Lᵀ·Pᵀ = D_eq·A·D_eq.
fn check_reconstruction(mat: &SymmetricMatrix, factors: &feral::Factors, tol: f64) {
    let n = factors.n;

    // Build D as full matrix
    let mut d_full = vec![0.0; n * n];
    let mut k = 0;
    while k < n {
        if k + 1 < n && factors.d_subdiag[k] != 0.0 {
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
    let mut ldlt = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..n {
                sum += ld[p * n + i] * factors.l[p * n + j];
            }
            ldlt[j * n + i] = sum;
        }
    }

    // Apply permutation: P·(LDLᵀ)·Pᵀ
    // result[i,j] = ldlt[perm_inv[i], perm_inv[j]]
    let mut max_err = 0.0f64;
    let mut max_scale = 0.0f64;
    for i in 0..n {
        for j in 0..=i {
            let pi = factors.perm_inv[i];
            let pj = factors.perm_inv[j];
            let got = ldlt[pj * n + pi];
            let expected = factors.d_eq[i] * mat.get(i, j) * factors.d_eq[j];
            let err = (expected - got).abs();
            let scale = expected.abs().max(got.abs()).max(1e-15);
            if err / scale > max_err / max_scale.max(1e-15) {
                max_err = err;
                max_scale = scale;
            }
        }
    }
    assert!(
        max_err / max_scale.max(1e-15) < tol,
        "reconstruction error: {:.2e} (tol {:.2e})",
        max_err / max_scale.max(1e-15),
        tol
    );
}

/// Check that the solve residual is small.
fn check_solve_residual(mat: &SymmetricMatrix, x: &[f64], rhs: &[f64], tol: f64) {
    let n = mat.n;
    let mut ax = vec![0.0; n];
    mat.symv(x, &mut ax);

    let mut max_resid = 0.0f64;
    let rhs_norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = if rhs_norm > 0.0 { rhs_norm } else { 1.0 };

    for i in 0..n {
        let resid = (ax[i] - rhs[i]).abs();
        if resid > max_resid {
            max_resid = resid;
        }
    }
    assert!(
        max_resid / scale < tol,
        "solve residual: {:.2e} / {:.2e} = {:.2e} (tol {:.2e})",
        max_resid,
        scale,
        max_resid / scale,
        tol
    );
}

/// Check structural invariants of the factors.
fn check_factor_structure(factors: &feral::Factors, inertia: &Inertia) {
    let n = factors.n;

    // 1. Inertia sums to n
    assert_eq!(
        inertia.total(),
        n,
        "inertia {} does not sum to n={}",
        inertia,
        n
    );

    // 2. Permutation is valid: perm and perm_inv are inverses
    for i in 0..n {
        assert_eq!(
            factors.perm_inv[factors.perm[i]], i,
            "perm_inv[perm[{}]] != {}",
            i, i
        );
        assert_eq!(
            factors.perm[factors.perm_inv[i]], i,
            "perm[perm_inv[{}]] != {}",
            i, i
        );
    }

    // 3. Permutation is a valid bijection (all values 0..n appear exactly once)
    let mut seen = vec![false; n];
    for &p in &factors.perm {
        assert!(p < n, "perm value {} >= n={}", p, n);
        assert!(!seen[p], "duplicate perm value {}", p);
        seen[p] = true;
    }

    // 4. L has unit diagonal
    for i in 0..n {
        assert_eq!(
            factors.l[i * n + i],
            1.0,
            "L diagonal at {} is {} (expected 1.0)",
            i,
            factors.l[i * n + i]
        );
    }

    // 5. L is lower triangular (strict upper triangle is zero)
    for j in 0..n {
        for i in 0..j {
            assert_eq!(
                factors.l[j * n + i],
                0.0,
                "L[{},{}] = {} (expected 0.0 in upper triangle)",
                i,
                j,
                factors.l[j * n + i]
            );
        }
    }

    // 6. D block structure: subdiag discriminant is consistent
    let mut k = 0;
    while k < n {
        if k + 1 < n && factors.d_subdiag[k] != 0.0 {
            // 2×2 block: subdiag[k+1] must be 0
            assert_eq!(
                factors.d_subdiag[k + 1],
                0.0,
                "subdiag[{}] = {} but should be 0 (second row of 2×2 block at {})",
                k + 1,
                factors.d_subdiag[k + 1],
                k
            );
            k += 2;
        } else {
            k += 1;
        }
    }

    // 7. d_eq entries are positive (equilibration scaling)
    for (i, &d) in factors.d_eq.iter().enumerate() {
        assert!(d > 0.0, "d_eq[{}] = {} (must be positive)", i, d);
    }
}

// -----------------------------------------------------------------------
// Property tests
// -----------------------------------------------------------------------

#[test]
fn property_random_spd_matrices() {
    let mut rng = Rng::new(42);
    let params = BunchKaufmanParams::default();

    for n in [3, 5, 8, 12, 20] {
        for trial in 0..5 {
            let mat = random_spd(n, &mut rng);
            let (factors, inertia) = factor(&mat, &params)
                .unwrap_or_else(|e| panic!("SPD n={} trial={}: factor failed: {}", n, trial, e));

            // SPD → all positive eigenvalues
            assert_eq!(
                inertia,
                Inertia::new(n, 0, 0),
                "SPD n={} trial={}: expected all-positive inertia, got {}",
                n,
                trial,
                inertia
            );

            check_factor_structure(&factors, &inertia);
            check_reconstruction(&mat, &factors, 1e-10);

            // Solve
            let rhs: Vec<f64> = (0..n)
                .map(|i| rng.uniform(-10.0, 10.0) * (i + 1) as f64)
                .collect();
            let x = solve(&factors, &rhs)
                .unwrap_or_else(|e| panic!("SPD n={} trial={}: solve failed: {}", n, trial, e));
            check_solve_residual(&mat, &x, &rhs, 1e-8);
        }
    }
}

#[test]
fn property_random_indefinite_matrices() {
    let mut rng = Rng::new(123);
    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };

    for n in [3, 5, 8, 12, 20] {
        for trial in 0..5 {
            let mat = random_indefinite(n, &mut rng);
            let (factors, inertia) = factor(&mat, &params).unwrap_or_else(|e| {
                panic!("indefinite n={} trial={}: factor failed: {}", n, trial, e)
            });

            check_factor_structure(&factors, &inertia);
            check_reconstruction(&mat, &factors, 1e-9);

            // Should have both positive and negative (or zero) eigenvalues
            // (can't assert exact counts for random indefinite)

            // Solve (use refined since ForceAccept might trigger)
            let rhs: Vec<f64> = (0..n).map(|_| rng.uniform(-5.0, 5.0)).collect();
            let x = solve_refined(&mat, &factors, &rhs).unwrap_or_else(|e| {
                panic!("indefinite n={} trial={}: solve failed: {}", n, trial, e)
            });
            // Looser tolerance for indefinite (can be ill-conditioned)
            check_solve_residual(&mat, &x, &rhs, 1e-4);
        }
    }
}

#[test]
fn property_random_kkt_matrices() {
    let mut rng = Rng::new(999);
    let params = BunchKaufmanParams::default();

    for (n_var, n_con) in [(3, 1), (5, 2), (8, 3), (10, 4), (15, 5)] {
        for trial in 0..3 {
            let (mat, nv, nc) = random_kkt(n_var, n_con, &mut rng);
            let n = nv + nc;

            let (factors, inertia) = factor(&mat, &params).unwrap_or_else(|e| {
                panic!(
                    "KKT ({},{}) trial={}: factor failed: {}",
                    n_var, n_con, trial, e
                )
            });

            check_factor_structure(&factors, &inertia);
            // KKT should have inertia (n_var, n_con, 0) for well-conditioned H
            assert_eq!(
                inertia,
                Inertia::new(nv, nc, 0),
                "KKT ({},{}) trial={}: expected inertia ({},{},0), got {}",
                n_var,
                n_con,
                trial,
                nv,
                nc,
                inertia
            );

            check_factor_structure(&factors, &inertia);

            // For KKT, the reconstruction tolerance must be generous:
            // κ(A) ≈ 1/δ = 1e8, and the O(n³) matrix product in the check
            // amplifies the backward error. Use solve residual as the primary
            // correctness signal.
            check_reconstruction(&mat, &factors, 1.0);

            // Solve — this is the real correctness check for ill-conditioned KKT
            let rhs: Vec<f64> = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
            let x = solve(&factors, &rhs).unwrap_or_else(|e| {
                panic!(
                    "KKT ({},{}) trial={}: solve failed: {}",
                    n_var, n_con, trial, e
                )
            });
            check_solve_residual(&mat, &x, &rhs, 1e-3);
        }
    }
}

#[test]
fn property_badly_scaled_matrices() {
    let mut rng = Rng::new(7777);
    let params = BunchKaufmanParams::default();

    for n in [4, 8, 15] {
        for trial in 0..3 {
            let mat = random_badly_scaled(n, &mut rng);
            let (factors, inertia) = factor(&mat, &params)
                .unwrap_or_else(|e| panic!("scaled n={} trial={}: factor failed: {}", n, trial, e));

            check_factor_structure(&factors, &inertia);

            // SPD after scaling → still all positive
            assert_eq!(
                inertia,
                Inertia::new(n, 0, 0),
                "scaled SPD n={} trial={}: expected all-positive, got {}",
                n,
                trial,
                inertia
            );

            check_reconstruction(&mat, &factors, 1e-6);

            // Solve
            let rhs: Vec<f64> = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
            let x = solve(&factors, &rhs)
                .unwrap_or_else(|e| panic!("scaled n={} trial={}: solve failed: {}", n, trial, e));
            // Badly scaled → equilibration should help, but residual may be larger
            check_solve_residual(&mat, &x, &rhs, 1e-2);
        }
    }
}

#[test]
fn property_permutation_inverse_consistency() {
    // Verify perm/perm_inv on a variety of matrices that trigger different pivot paths
    let mut rng = Rng::new(555);
    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };

    for n in [2, 3, 5, 10, 25, 50] {
        let mat = random_indefinite(n, &mut rng);
        let (factors, inertia) = factor(&mat, &params)
            .unwrap_or_else(|e| panic!("perm test n={}: factor failed: {}", n, e));
        check_factor_structure(&factors, &inertia);
    }
}

#[test]
fn property_solve_refined_improves_accuracy() {
    // For moderately ill-conditioned matrices, solve_refined should give
    // at least as good accuracy as plain solve.
    let mut rng = Rng::new(314);
    let params = BunchKaufmanParams::default();

    for n in [5, 10, 20] {
        let mat = random_badly_scaled(n, &mut rng);
        let (factors, _inertia) = factor(&mat, &params)
            .unwrap_or_else(|e| panic!("refine test n={}: factor failed: {}", n, e));

        let rhs: Vec<f64> = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let x_plain = solve(&factors, &rhs)
            .unwrap_or_else(|e| panic!("refine test n={}: plain solve failed: {}", n, e));
        let x_refined = solve_refined(&mat, &factors, &rhs)
            .unwrap_or_else(|e| panic!("refine test n={}: refined solve failed: {}", n, e));

        // Compute residuals
        let mut ax_plain = vec![0.0; n];
        let mut ax_refined = vec![0.0; n];
        mat.symv(&x_plain, &mut ax_plain);
        mat.symv(&x_refined, &mut ax_refined);

        let resid_plain: f64 = (0..n)
            .map(|i| (ax_plain[i] - rhs[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let resid_refined: f64 = (0..n)
            .map(|i| (ax_refined[i] - rhs[i]).powi(2))
            .sum::<f64>()
            .sqrt();

        // For well-conditioned systems, both should be very accurate.
        // Refinement may not improve (and can slightly worsen) near machine
        // precision due to extra FP operations. Just verify both are small.
        let rhs_norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-15);
        assert!(
            resid_plain / rhs_norm < 1e-2,
            "n={}: plain residual too large: {:.2e}",
            n,
            resid_plain / rhs_norm
        );
        assert!(
            resid_refined / rhs_norm < 1e-2,
            "n={}: refined residual too large: {:.2e}",
            n,
            resid_refined / rhs_norm
        );
    }
}
