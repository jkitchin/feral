//! Phase 2.2.2 tests: column-relative pivot-threshold rejection
//! (scaling-aware pivot rejection).
//!
//! These tests are the acceptance gate for
//! `dev/plans/scaling-aware-pivot-rejection.md`. They verify that:
//!
//!   A. A 1×1 pivot that is tiny relative to its column max is
//!      rejected under `pivot_threshold = 0.01` but accepted under
//!      `pivot_threshold = 0.0` (backward compat).
//!   B. `pivot_threshold = 0.01` is inactive on a well-conditioned
//!      SPD matrix — all pivots pass and inertia is clean.
//!   C. `pivot_threshold = 0.0` reproduces the Phase 1 default
//!      behavior exactly (same factors, same inertia, same solution).
//!   D. The Duff-Reid 2×2 growth bound accepts a 2×2 block at
//!      `u = 0.01` and rejects the same block at `u = 0.1`.
//!
//! Oracles are hand-computed from the MUMPS
//! (`dfac_front_aux.F:1494-1606`) and SSIDS
//! (`block_ldlt.hxx::test_2x2`) acceptance formulas.

use feral::{factor, solve, BunchKaufmanParams, Factors, SymmetricMatrix, ZeroPivotAction};

/// Count d_diag entries that were force-accepted as zero.
fn n_zero_pivots(f: &Factors) -> usize {
    f.d_diag.iter().filter(|d| d.abs() <= f.zero_tol).count()
}

/// Build the 3×3 matrix used in Test A:
///
///     [  1.0   0.5   0.0 ]
///     [  0.5   1e-12 0.3 ]
///     [  0.0   0.3   1.0 ]
///
/// Row 1 has `|a_11| = 1e-12` but column max `max(|0.5|, |0.3|) = 0.5`.
/// With `u = 0.01`, threshold = `0.005` — the tiny pivot fails the column
/// test and must be rejected. With `u = 0.0`, the absolute-only floor
/// `zero_tol = eps` accepts the pivot and the solve amplifies.
fn test_a_matrix() -> SymmetricMatrix {
    let mut mat = SymmetricMatrix::zeros(3);
    mat.set(0, 0, 1.0);
    mat.set(1, 0, 0.5);
    mat.set(1, 1, 1e-12);
    mat.set(2, 0, 0.0);
    mat.set(2, 1, 0.3);
    mat.set(2, 2, 1.0);
    mat
}

#[test]
fn threshold_rejects_tiny_1x1_pivot() {
    let mat = test_a_matrix();
    let rhs = vec![1.0, 1.0, 1.0];

    // Case 1: pivot_threshold = 0.0 → no column-relative rejection, tiny
    // pivot is accepted (absolute zero_tol = eps passes it through) and
    // the solve blows up because the rank-1 update divides by 1e-12.
    let params0 = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.0,
        ..BunchKaufmanParams::default()
    };
    let (f0, _) = factor(&mat, &params0).expect("factor u=0");
    let n_zero_baseline = n_zero_pivots(&f0);

    // Case 2: pivot_threshold = 0.01 → tiny pivot rejected, counted as
    // zero via ForceAccept. The number of zero pivots must strictly
    // exceed the baseline.
    let params1 = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    let (f1, inertia1) = factor(&mat, &params1).expect("factor u=0.01");
    let n_zero_thresholded = n_zero_pivots(&f1);

    assert!(
        n_zero_thresholded > n_zero_baseline,
        "expected at least one additional zero pivot under u=0.01 \
         (baseline {}, threshold {})",
        n_zero_baseline,
        n_zero_thresholded
    );
    assert!(
        inertia1.zero >= 1,
        "expected inertia.zero >= 1 under u=0.01, got {}",
        inertia1
    );

    // Sanity: the threshold solve must not amplify catastrophically.
    let x1 = solve(&f1, &rhs).expect("solve u=0.01");
    for (i, xi) in x1.iter().enumerate() {
        assert!(
            xi.abs() < 1e6,
            "threshold solve x[{}] = {:.3e} — likely divided by a tiny pivot",
            i,
            xi
        );
    }
}

#[test]
fn threshold_inactive_on_well_conditioned() {
    // Strictly diagonally-dominant 4×4 SPD matrix. All pivots are O(1),
    // column maxes are O(1), so |a_kk| / col_max is well above 0.01.
    // Inertia must be (4, 0, 0) and the solve must match the known answer
    // computed from the input.
    let mut mat = SymmetricMatrix::zeros(4);
    mat.set(0, 0, 4.0);
    mat.set(1, 0, 1.0);
    mat.set(1, 1, 4.0);
    mat.set(2, 1, 1.0);
    mat.set(2, 2, 4.0);
    mat.set(3, 2, 1.0);
    mat.set(3, 3, 4.0);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    let (f, inertia) = factor(&mat, &params).expect("factor");

    assert_eq!(
        inertia.positive, 4,
        "expected 4 positive pivots, got {}",
        inertia
    );
    assert_eq!(inertia.negative, 0);
    assert_eq!(inertia.zero, 0);

    // Solve A x = b and verify via residual that no pivot was rejected
    // (if any pivot had been force-accepted as zero, the solve would
    // leave that position unchanged and the residual would be large).
    let rhs = vec![1.0, 2.0, 3.0, 4.0];
    let x = solve(&f, &rhs).expect("solve");
    let mut ax = vec![0.0; 4];
    mat.symv(&x, &mut ax);
    let mut rss = 0.0;
    let mut bss = 0.0;
    for i in 0..4 {
        let r = ax[i] - rhs[i];
        rss += r * r;
        bss += rhs[i] * rhs[i];
    }
    let rel = (rss / bss).sqrt();
    assert!(
        rel < 1e-12,
        "well-conditioned residual {:.3e} too large",
        rel
    );
}

#[test]
fn threshold_zero_reproduces_default() {
    // Backward-compat gate: pivot_threshold = 0.0 must produce the same
    // factors and inertia as the Phase 1 default (no threshold check).
    // We use the Test A rank-deficient matrix so both paths actually
    // exercise the acceptance clause.
    let mat = test_a_matrix();

    let p_default = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    let p_explicit_zero = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.0,
        ..BunchKaufmanParams::default()
    };

    let (fa, ia) = factor(&mat, &p_default).expect("factor default");
    let (fb, ib) = factor(&mat, &p_explicit_zero).expect("factor u=0");

    assert_eq!(ia, ib, "inertia mismatch between default and u=0.0");
    assert_eq!(fa.n, fb.n);
    for k in 0..fa.n {
        assert_eq!(
            fa.d_diag[k], fb.d_diag[k],
            "d_diag[{}] mismatch: {} vs {}",
            k, fa.d_diag[k], fb.d_diag[k]
        );
        assert_eq!(
            fa.d_subdiag[k], fb.d_subdiag[k],
            "d_subdiag[{}] mismatch: {} vs {}",
            k, fa.d_subdiag[k], fb.d_subdiag[k]
        );
    }
    for k in 0..fa.n * fa.n {
        assert_eq!(
            fa.l[k], fb.l[k],
            "L[{}] mismatch: {} vs {}",
            k, fa.l[k], fb.l[k]
        );
    }
}

#[test]
fn duff_reid_2x2_growth_bound() {
    // Construct a 4×4 symmetric matrix where the 2×2 pivot decision on
    // rows {0,1} depends on the Duff-Reid growth bound, not the absolute
    // determinant floor.
    //
    // Upper-left 2×2 block:
    //     a11 = 0.01, a22 = -0.01, a21 = 0.1
    // |det| = a11*a22 - a21^2 = -0.0001 - 0.01 = 0.0101
    //
    // Trailing 2 rows couple to both pivot columns with magnitude 1, so
    // after the 2×2 picks, the out-of-block column maxes are
    //     RMAX = max |a[i, 0]| for i >= 2 = 1.0
    //     TMAX = max |a[i, 1]| for i >= 2 = 1.0
    //     AMAX = |a21| = 0.1          (the 2×2 off-diagonal)
    //
    // MUMPS growth bound (dfac_front_aux.F:1599-1606):
    //   reject iff (|a22|*RMAX + AMAX*TMAX)*u  >  |det|
    //   reject iff (0.01 + 0.1)*u = 0.11*u > 0.0101
    //   equality at u ≈ 0.0918.
    //
    // So u = 0.01 → 0.0011 ≤ 0.0101 → accept.
    //    u = 0.1  → 0.011  > 0.0101 → reject.
    //
    // On rejection, feral should count an extra zero pivot (via the 1×1
    // fall-through, which the column-relative 1×1 threshold will also
    // reject because a11 = 0.01 but col_max = max(0.1, 1, 1) = 1 so
    // |a11| < 0.1·1 = 0.1 fails at u=0.1; similarly for a22).
    let mut mat = SymmetricMatrix::zeros(4);
    mat.set(0, 0, 0.01);
    mat.set(1, 0, 0.1);
    mat.set(1, 1, -0.01);
    mat.set(2, 0, 1.0);
    mat.set(2, 1, 1.0);
    mat.set(2, 2, 3.0);
    mat.set(3, 0, 1.0);
    mat.set(3, 1, 1.0);
    mat.set(3, 2, 0.5);
    mat.set(3, 3, 3.0);

    let params_accept = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    let params_reject = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.1,
        ..BunchKaufmanParams::default()
    };

    let (f_accept, _) = factor(&mat, &params_accept).expect("factor u=0.01");
    let (f_reject, _) = factor(&mat, &params_reject).expect("factor u=0.1");

    let z_accept = n_zero_pivots(&f_accept);
    let z_reject = n_zero_pivots(&f_reject);

    assert!(
        z_reject > z_accept,
        "expected u=0.1 to reject more pivots than u=0.01 \
         (accept zero-count {}, reject zero-count {})",
        z_accept,
        z_reject
    );
}
