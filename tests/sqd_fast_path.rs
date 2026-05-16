//! Issue #34 phase (c) — unit smoke test for the SQD
//! (symmetric quasi-definite) diagonal-only kernel.
//!
//! Phase (c) introduces `factor_diagonal` and
//! `factor_frontal_diagonal_in_place` but leaves them uncalled by
//! production code (phase (d) wires `factor_one_supernode` dispatch).
//! This file holds the minimum coverage that proves the kernel
//! builds, returns the documented shape, and matches a hand-computed
//! 2x2 SQD example.
//!
//! Phase (f) will grow this file with the full reference-parity,
//! property, regression, negative, builder, and cache test categories
//! listed in `dev/plans/sqd-fast-path.md` (and the user-approved
//! plan at `~/.claude/plans/let-s-work-on-a-reflective-anchor.md`).

use feral::dense::factor::{factor_diagonal, BunchKaufmanParams, Factors};
use feral::{Inertia, SymmetricMatrix};

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
