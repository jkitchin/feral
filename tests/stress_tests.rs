//! Stress tests for dense LDLᵀ: larger matrices, badly-scaled systems,
//! LAPACK extension pivot path, arrow/bordered diagonal structure, and
//! matrices requiring many 2×2 pivots.

use feral::{
    factor, solve, solve_refined, BunchKaufmanParams, Inertia, SymmetricMatrix, ZeroPivotAction,
};

// -----------------------------------------------------------------------
// Simple deterministic PRNG (xorshift64)
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

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let t = (self.next_u64() as f64) / (u64::MAX as f64);
        lo + t * (hi - lo)
    }
}

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

fn check_inertia_sums_to_n(inertia: &Inertia, n: usize) {
    assert_eq!(
        inertia.total(),
        n,
        "inertia {} does not sum to n={}",
        inertia,
        n
    );
}

// =======================================================================
// Test: LAPACK extension (Test 6) — designed matrix
// =======================================================================
// Test 6 passes when: Tests 3 and 5 fail, but |A[0,0]|·γᵣ >= α·γ₀²
//
// We need:
//   |A[0,0]| < α·γ₀          (Test 3 fails)
//   |A[r,r]| < α·γᵣ          (Test 5 fails)
//   |A[0,0]|·γᵣ >= α·γ₀²     (Test 6 passes)
//
// This means γᵣ must be large (much larger than γ₀), while A[0,0] is moderate.
#[test]
fn test_lapack_extension_branch() {
    // Construct a 4×4 matrix that hits Test 6.
    //
    // Column 0 off-diagonal max (γ₀): row r=1 with |A[1,0]| = 2.0
    // So γ₀ = 2.0, r = 1.
    // Test 3: |A[0,0]| = 1.0. α·γ₀ = 0.6404·2 = 1.28. 1.0 < 1.28. FAILS. ✓
    //
    // Row 1 symmetric max (γᵣ): we need max off-diag in full row 1.
    //   row 1 entries: A[1,0]=2.0, A[2,1]=0.1, A[3,1]=0.1
    //   γᵣ = max(2.0, 0.1, 0.1) = 2.0
    // Test 5: |A[1,1]| = 0.5. α·γᵣ = 0.6404·2 = 1.28. 0.5 < 1.28. FAILS. ✓
    //
    // Test 6: |A[0,0]|·γᵣ = 1.0·2.0 = 2.0. α·γ₀² = 0.6404·4 = 2.5616.
    // 2.0 < 2.5616. FAILS too. Need to adjust.
    //
    // Revised: make γᵣ much larger by adding a large entry in row r.
    // A[3,1] = 10.0 → γᵣ = 10.0
    // Test 5: |A[1,1]| = 0.5. α·γᵣ = 0.6404·10 = 6.4. 0.5 < 6.4. FAILS. ✓
    // Test 6: |A[0,0]|·γᵣ = 1.0·10 = 10. α·γ₀² = 0.6404·4 = 2.56. 10 >= 2.56. PASSES! ✓
    let mut mat = SymmetricMatrix::zeros(4);
    mat.set(0, 0, 1.0);
    mat.set(1, 0, 2.0);
    mat.set(1, 1, 0.5);
    mat.set(2, 0, 0.1);
    mat.set(2, 1, 0.1);
    mat.set(2, 2, 3.0);
    mat.set(3, 0, 0.1);
    mat.set(3, 1, 10.0);
    mat.set(3, 2, 0.1);
    mat.set(3, 3, 4.0);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    let (factors, inertia) = factor(&mat, &params).expect("factor failed");

    check_inertia_sums_to_n(&inertia, 4);
    // The matrix is indefinite due to the large A[3,1]=10.0 entry.
    // The purpose of this test is to exercise Test 6, not to verify definiteness.
    assert_eq!(
        inertia,
        Inertia::new(3, 1, 0),
        "expected (3,1,0) for this indefinite matrix"
    );

    let rhs = vec![1.0, 2.0, 3.0, 4.0];
    let x = solve_refined(&mat, &factors, &rhs).expect("solve failed");
    check_solve(&mat, &x, &rhs, 1e-8);
}

// =======================================================================
// Test: Arrow matrix (bordered diagonal) — common in decomposition NLP
// =======================================================================
#[test]
fn test_arrow_matrix() {
    // Arrow matrix: diagonal + dense last row/column
    // [ d₀  0   0   ...  a₀ ]
    // [ 0   d₁  0   ...  a₁ ]
    // [ ...                   ]
    // [ a₀  a₁  a₂  ...  d_n]
    let n = 20;
    let mut mat = SymmetricMatrix::zeros(n);
    let mut rng = Rng::new(42);

    // Diagonal entries
    for i in 0..(n - 1) {
        mat.set(i, i, rng.uniform(2.0, 10.0));
    }
    // Arrow entries (last row/column)
    for i in 0..(n - 1) {
        let val = rng.uniform(-1.0, 1.0);
        mat.set(n - 1, i, val);
    }
    // Last diagonal: make it large enough to keep matrix SPD
    mat.set(n - 1, n - 1, 20.0);

    let params = BunchKaufmanParams::default();
    let (factors, inertia) = factor(&mat, &params).expect("arrow factor failed");

    check_inertia_sums_to_n(&inertia, n);
    assert_eq!(inertia, Inertia::new(n, 0, 0), "arrow should be SPD");

    let rhs: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let x = solve(&factors, &rhs).expect("arrow solve failed");
    check_solve(&mat, &x, &rhs, 1e-10);
}

// =======================================================================
// Test: Matrix requiring all 2×2 pivots (maximally indefinite)
// =======================================================================
#[test]
fn test_all_2x2_pivots() {
    // Construct a matrix where every step selects a 2×2 pivot:
    // Block diagonal of 2×2 indefinite blocks [[ε, c], [c, ε]].
    let n = 10; // 5 blocks of size 2
    let mut mat = SymmetricMatrix::zeros(n);

    for blk in 0..5 {
        let i = blk * 2;
        mat.set(i, i, 0.001);
        mat.set(i + 1, i, 5.0 + blk as f64);
        mat.set(i + 1, i + 1, 0.002);
    }

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    let (factors, inertia) = factor(&mat, &params).expect("all-2x2 factor failed");

    // Each 2×2 block has det ≈ -25 < 0 → inertia (1,1,0) per block
    assert_eq!(
        inertia,
        Inertia::new(5, 5, 0),
        "5 indefinite 2×2 blocks → (5,5,0)"
    );
    check_inertia_sums_to_n(&inertia, n);

    let rhs: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 + 1.0).collect();
    let x = solve(&factors, &rhs).expect("all-2x2 solve failed");
    check_solve(&mat, &x, &rhs, 1e-10);
}

// =======================================================================
// Test: Large SPD matrix (n=50)
// =======================================================================
#[test]
fn test_large_spd_50() {
    let n = 50;
    let mut rng = Rng::new(12345);
    let mut mat = SymmetricMatrix::zeros(n);

    // Generate SPD: A = M·Mᵀ + I
    let mut m = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            m[j * n + i] = rng.uniform(-1.0, 1.0);
        }
    }
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..n {
                sum += m[k * n + i] * m[k * n + j];
            }
            mat.set(i, j, sum + if i == j { 1.0 } else { 0.0 });
        }
    }

    let params = BunchKaufmanParams::default();
    let (factors, inertia) = factor(&mat, &params).expect("n=50 factor failed");

    assert_eq!(inertia, Inertia::new(n, 0, 0));
    check_inertia_sums_to_n(&inertia, n);

    let rhs: Vec<f64> = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let x = solve(&factors, &rhs).expect("n=50 solve failed");
    check_solve(&mat, &x, &rhs, 1e-8);
}

// =======================================================================
// Test: Large indefinite matrix (n=100)
// =======================================================================
#[test]
fn test_large_indefinite_100() {
    let n = 100;
    let mut rng = Rng::new(99999);
    let mut mat = SymmetricMatrix::zeros(n);

    // Generate indefinite: A = M·Mᵀ + I - shift*I
    let mut m = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            m[j * n + i] = rng.uniform(-1.0, 1.0);
        }
    }
    let shift = 25.0; // shift large enough to make many eigenvalues negative
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..n {
                sum += m[k * n + i] * m[k * n + j];
            }
            mat.set(i, j, sum + if i == j { 1.0 - shift } else { 0.0 });
        }
    }

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    let (factors, inertia) = factor(&mat, &params).expect("n=100 factor failed");

    check_inertia_sums_to_n(&inertia, n);
    // Should have both positive and negative eigenvalues
    assert!(inertia.positive > 0, "expected some positive eigenvalues");
    assert!(inertia.negative > 0, "expected some negative eigenvalues");

    let rhs: Vec<f64> = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let x = solve_refined(&mat, &factors, &rhs).expect("n=100 solve failed");
    check_solve(&mat, &x, &rhs, 1e-4);
}

// =======================================================================
// Test: Badly-scaled SPD (scaling range 1e-8 to 1e8)
// =======================================================================
#[test]
fn test_extreme_scaling() {
    let n = 15;
    let mut rng = Rng::new(2024);

    // Start with well-conditioned SPD
    let mut mat = SymmetricMatrix::zeros(n);
    let mut m = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            m[j * n + i] = rng.uniform(-2.0, 2.0);
        }
    }
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..n {
                sum += m[k * n + i] * m[k * n + j];
            }
            mat.set(i, j, sum + if i == j { 0.1 } else { 0.0 });
        }
    }

    // Apply extreme scaling: D·A·D where D spans 1e-8 to 1e8
    let mut scale = vec![0.0; n];
    for (idx, s) in scale.iter_mut().enumerate() {
        // Deterministic scaling: first half small, second half large
        let exponent = -8.0 + 16.0 * (idx as f64) / ((n - 1) as f64);
        *s = 10f64.powf(exponent);
    }
    for i in 0..n {
        for j in 0..=i {
            let val = mat.get(i, j) * scale[i] * scale[j];
            mat.set(i, j, val);
        }
    }

    let params = BunchKaufmanParams::default();
    let (factors, inertia) = factor(&mat, &params).expect("extreme-scale factor failed");

    assert_eq!(
        inertia,
        Inertia::new(n, 0, 0),
        "scaled SPD should remain all-positive"
    );

    let rhs: Vec<f64> = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
    // With 1e-8 to 1e8 scaling, κ ≈ 1e16 before equilibration.
    // Even with equilibration, residual can be large. Use solve_refined.
    let x = solve_refined(&mat, &factors, &rhs).expect("extreme-scale solve failed");
    check_solve(&mat, &x, &rhs, 1.0);
}

// =======================================================================
// Test: Tridiagonal matrix (common in 1D discretizations)
// =======================================================================
#[test]
fn test_tridiagonal() {
    let n = 30;
    let mut mat = SymmetricMatrix::zeros(n);

    // Classic: 2 on diagonal, -1 on sub/super diagonal
    for i in 0..n {
        mat.set(i, i, 2.0);
        if i + 1 < n {
            mat.set(i + 1, i, -1.0);
        }
    }

    let params = BunchKaufmanParams::default();
    let (factors, inertia) = factor(&mat, &params).expect("tridiag factor failed");

    assert_eq!(inertia, Inertia::new(n, 0, 0), "tridiag should be SPD");

    let rhs: Vec<f64> = (0..n)
        .map(|i| if i == 0 || i == n - 1 { 1.0 } else { 0.0 })
        .collect();
    let x = solve(&factors, &rhs).expect("tridiag solve failed");
    check_solve(&mat, &x, &rhs, 1e-12);
}

// =======================================================================
// Test: KKT with increasing constraint count (trajectory-like)
// =======================================================================
#[test]
fn test_kkt_trajectory() {
    // Simulates an IPM trajectory: same H, same J, but δ shrinks
    let n_var = 5;
    let n_con = 2;
    let n = n_var + n_con;

    let mut base = SymmetricMatrix::zeros(n);
    // H block: diagonal SPD
    for i in 0..n_var {
        base.set(i, i, (i + 1) as f64 * 2.0);
    }
    // J block
    base.set(5, 0, 1.0);
    base.set(5, 1, 0.5);
    base.set(5, 2, 1.0);
    base.set(6, 2, 1.0);
    base.set(6, 3, 0.5);
    base.set(6, 4, 1.0);

    let params = BunchKaufmanParams::default();

    // Test across a range of δ values (simulating barrier parameter decrease)
    for &delta in &[1e-2, 1e-4, 1e-6, 1e-8, 1e-10] {
        let mut mat = SymmetricMatrix::zeros(n);
        // Copy base
        for i in 0..n {
            for j in 0..=i {
                mat.set(i, j, base.get(i, j));
            }
        }
        // Set constraint regularization
        mat.set(5, 5, -delta);
        mat.set(6, 6, -delta);

        let (factors, inertia) = factor(&mat, &params)
            .unwrap_or_else(|e| panic!("KKT trajectory δ={:.0e}: factor failed: {}", delta, e));

        assert_eq!(
            inertia,
            Inertia::new(n_var, n_con, 0),
            "KKT trajectory δ={:.0e}: wrong inertia {}",
            delta,
            inertia
        );

        let rhs = vec![1.0; n];
        let x = solve(&factors, &rhs)
            .unwrap_or_else(|e| panic!("KKT trajectory δ={:.0e}: solve failed: {}", delta, e));

        // Tolerance scales with 1/δ (the condition number)
        let tol = (1.0 / delta) * (n as f64) * f64::EPSILON * 100.0;
        check_solve(&mat, &x, &rhs, tol.max(1e-10));
    }
}
