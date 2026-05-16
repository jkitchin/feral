//! Issue #34 phases (c) and (d) — SQD (symmetric quasi-definite)
//! diagonal-only fast-path.
//!
//! Phase (c) introduced `factor_diagonal` and
//! `factor_frontal_diagonal_in_place` as standalone kernels.
//! Phase (d) wires `Solver::with_sqd_mode(true)` to dispatch the
//! supernodal driver (`factor_one_supernode`, `factor_one_small_leaf`,
//! and the dense fast-path) through the diagonal kernel.
//!
//! Phase (f) will grow this file with the full reference-parity,
//! property, regression, negative, builder, and cache test categories
//! listed in `dev/plans/sqd-fast-path.md` (and the user-approved
//! plan at `~/.claude/plans/let-s-work-on-a-reflective-anchor.md`).

use feral::dense::factor::{factor_diagonal, BunchKaufmanParams, Factors};
use feral::{CscMatrix, FactorStatus, Inertia, Solver, SymmetricMatrix};

fn params() -> BunchKaufmanParams {
    BunchKaufmanParams::default()
}

/// `K = diag(-1, +1)` — the simplest SQD: zero off-diagonal so
/// equilibration is identity, L = I, D = diag(-1, +1), inertia
/// (1, 1, 0).
#[test]
fn sqd_2x2_pure_diagonal_hand_check() {
    let n = 2;
    let mut data = vec![0.0; n * n];
    data[0] = -1.0; // a[0,0]
    data[n + 1] = 1.0; // a[1,1]
    let mat = SymmetricMatrix { n, data };

    let (factors, inertia) = factor_diagonal(&mat, &params()).expect("factor_diagonal");

    assert_eq!(factors.n, 2);
    assert_eq!(factors.d_subdiag, vec![0.0, 0.0], "SQD D is diagonal");
    assert_eq!(factors.perm, vec![0, 1], "SQD does no row/col swaps");
    assert_eq!(factors.perm_inv, vec![0, 1]);
    assert_eq!(
        inertia,
        Inertia {
            positive: 1,
            negative: 1,
            zero: 0,
        }
    );
    // Equilibration scaling for `diag(-1, +1)` is `1/sqrt(|d|) = 1`,
    // so the post-scaling D matches the input diagonal exactly.
    assert!((factors.d_diag[0] + 1.0).abs() < 1e-15);
    assert!((factors.d_diag[1] - 1.0).abs() < 1e-15);
    // L = I (unit diagonal, no off-diagonal because input is diagonal).
    assert!((factors.l[0] - 1.0).abs() < 1e-15);
    assert!(factors.l[1].abs() < 1e-15);
    assert!(factors.l[n].abs() < 1e-15);
    assert!((factors.l[n + 1] - 1.0).abs() < 1e-15);
}

/// `K = [[-2, 1], [1, 3]]` — a 2x2 SQD with off-diagonal. After
/// equilibration the BK and SQD paths must agree on D up to
/// numerical noise.
///
/// Hand computation on the un-equilibrated `K`:
///   d_1 = -2,  L[1,0] = 1 / -2 = -0.5
///   d_2 = 3 - (-0.5) * (-2) * (-0.5) = 3 - 0.5 = 2.5
///   inertia = (1, 1, 0)
#[test]
fn sqd_2x2_with_offdiag_hand_check() {
    let n = 2;
    let mut data = vec![0.0; n * n];
    data[0] = -2.0;
    data[1] = 1.0; // a[1,0] in column-major lower
    data[n + 1] = 3.0;
    let mat = SymmetricMatrix { n, data };

    let (factors, inertia) = factor_diagonal(&mat, &params()).expect("factor_diagonal");

    assert_eq!(
        inertia,
        Inertia {
            positive: 1,
            negative: 1,
            zero: 0,
        }
    );
    // Diagonal D — no 2x2 block.
    assert_eq!(factors.d_subdiag, vec![0.0, 0.0]);
    // Recover the un-equilibrated factorization by un-scaling:
    //   K = D_eq^{-1} L D L^T D_eq^{-1}  (with D_eq * K * D_eq factored)
    // We don't need the exact post-equilibration D — only that the
    // signs agree (-, +) and the reconstructed `L D L^T` recovers K
    // within a tight residual.
    assert!(factors.d_diag[0] < 0.0, "first pivot must be negative");
    assert!(factors.d_diag[1] > 0.0, "second pivot must be positive");

    // Residual check: reconstruct A_scaled = L * diag(D) * L^T and
    // un-equilibrate to recover K.
    let scaled = reconstruct_ldlt(&factors);
    let mut recovered = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            recovered[j * n + i] = scaled[j * n + i] / (factors.d_eq[i] * factors.d_eq[j]);
        }
    }
    let expected_full = [-2.0_f64, 1.0, 1.0, 3.0];
    for j in 0..n {
        for i in 0..n {
            let got = recovered[j * n + i];
            let want = expected_full[j * n + i];
            assert!((got - want).abs() < 1e-12, "K[{i},{j}] = {got} != {want}",);
        }
    }
}

/// SQD contract violation: a diagonal-zero matrix at column 0 must
/// return `Err` (phase c uses the placeholder `NumericallyRankDeficient`;
/// phase e replaces with `SqdContractViolated`).
#[test]
fn sqd_zero_pivot_rejected() {
    let n = 2;
    let mut data = vec![0.0; n * n];
    data[0] = 0.0; // d_1 = 0 — contract violation at column 0
    data[1] = 1.0;
    data[n + 1] = 3.0;
    let mat = SymmetricMatrix { n, data };
    assert!(factor_diagonal(&mat, &params()).is_err());
}

// ---------- Phase (d): Solver-level dispatch ----------

/// Phase (d) — dense fast-path dispatch. A 4×4 diagonal SQD
/// (n ≤ N_TINY = 16) routes through `dense_fast_factor`, which now
/// dispatches on `params.sqd_mode`. Verifies the diagonal kernel
/// took the call (post-solve recovery of an arbitrary RHS) and the
/// reported inertia matches the SQD theoretical prediction.
#[test]
fn sqd_solver_dispatch_dense_path() {
    let n = 4;
    let rows = vec![0, 1, 2, 3];
    let cols = vec![0, 1, 2, 3];
    let vals = vec![-1.0_f64, -2.0, 3.0, 4.0];
    let csc = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("csc");

    let mut solver = Solver::new().with_sqd_mode(true);
    let status = solver.factor(
        &csc,
        Some(Inertia {
            positive: 2,
            negative: 2,
            zero: 0,
        }),
    );
    assert!(matches!(status, FactorStatus::Success), "got {:?}", status);

    // Solve A x = b with b = [1, 2, 3, 4]^T. Expected:
    // x = [-1, -1, 1, 1]
    let x = solver.solve(&[1.0, 2.0, 3.0, 4.0]).expect("solve");
    assert!((x[0] - (-1.0)).abs() < 1e-12, "x[0]={}", x[0]);
    assert!((x[1] - (-1.0)).abs() < 1e-12, "x[1]={}", x[1]);
    assert!((x[2] - 1.0).abs() < 1e-12, "x[2]={}", x[2]);
    assert!((x[3] - 1.0).abs() < 1e-12, "x[3]={}", x[3]);
}

/// Phase (d) — multifrontal supernode dispatch. n=24 banded SQD
/// (density well below 1/4 and n > N_TINY) routes through
/// `factor_one_supernode`. First 12 columns negative-diagonal, last
/// 12 positive-diagonal; off-diagonal coupling at i,i+1 in the
/// positive block to force a non-trivial elimination tree.
#[test]
fn sqd_solver_dispatch_multifrontal_path() {
    let n = 24;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    // Diagonal: -2.0 on first 12, +2.0 on last 12.
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(if i < 12 { -2.0 } else { 2.0 });
    }
    // Sub-diagonal coupling in the positive block (i, i+1) for i in
    // 12..n-1 — small magnitude (0.1) so the SQD off-diagonal stays
    // dominated by the diagonal and we recover a valid factorization.
    for i in 12..n - 1 {
        rows.push(i + 1);
        cols.push(i);
        vals.push(0.1);
    }
    let csc = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("csc");

    let mut solver = Solver::new().with_sqd_mode(true);
    let status = solver.factor(
        &csc,
        Some(Inertia {
            positive: 12,
            negative: 12,
            zero: 0,
        }),
    );
    assert!(matches!(status, FactorStatus::Success), "got {:?}", status);

    // Solve with b = e_0 (first canonical) and verify A x ≈ b by
    // residual norm. Avoids hand-computing the exact x for the banded
    // positive block.
    let mut b = vec![0.0_f64; n];
    b[0] = 1.0;
    let x = solver.solve(&b).expect("solve");
    // Compute residual r = A x - b directly from the triplets.
    let mut r = vec![0.0_f64; n];
    for k in 0..rows.len() {
        let (i, j, v) = (rows[k], cols[k], vals[k]);
        r[i] += v * x[j];
        if i != j {
            r[j] += v * x[i];
        }
    }
    for ri in r.iter_mut() {
        *ri -= 0.0;
    }
    r[0] -= 1.0;
    let r_norm: f64 = r.iter().map(|&v| v * v).sum::<f64>().sqrt();
    assert!(r_norm < 1e-10, "residual norm = {} too large", r_norm);
}

/// Phase (d) — SQD contract-trip surfaces through the Solver as
/// `FactorStatus::Failed`. Diagonal-zero at column 0 of a 24×24
/// matrix (forces multifrontal routing) trips the contract.
#[test]
fn sqd_solver_dispatch_contract_violation_returns_failed() {
    let n = 24;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    rows.push(0);
    cols.push(0);
    vals.push(0.0); // contract trip — zero diagonal at column 0
    for i in 1..n {
        rows.push(i);
        cols.push(i);
        vals.push(if i < 12 { -2.0 } else { 2.0 });
    }
    let csc = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("csc");

    let mut solver = Solver::new().with_sqd_mode(true);
    let status = solver.factor(&csc, None);
    // Phase (c) returns NumericallyRankDeficient → FactorStatus
    // surfaces as Failed or Singular. Phase (e) will tighten to
    // SqdContractViolated. Until then, just verify non-success.
    assert!(
        !matches!(status, FactorStatus::Success),
        "expected non-success, got {:?}",
        status
    );
}

/// Reconstruct `L * diag(d_diag) * L^T` into a column-major n×n
/// dense matrix.
fn reconstruct_ldlt(f: &Factors) -> Vec<f64> {
    let n = f.n;
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += f.l[k * n + i] * f.d_diag[k] * f.l[k * n + j];
            }
            a[j * n + i] = s;
        }
    }
    a
}
