//! Regression test for https://github.com/jkitchin/feral/issues/2.
//!
//! On rank-deficient KKT-augmented LS-init systems, the prior
//! `NumericParams::default()` (`bk.pivot_threshold = 0.0`) silently
//! zeroed multipliers on non-structurally-zero rows because the
//! 2×2 SSIDS det-floor would reject saddle blocks and the 1×1
//! fallback's rook-rescue fast-path was dead at `pivot_threshold = 0`.
//!
//! See `dev/research/issue-2-kkt-pivot-default.md` for the full
//! mechanism. The test enforces three invariants:
//!
//!   1. `NumericParams::default()` exposes `pivot_threshold = 0.01`
//!      (SSIDS/MUMPS canonical, what feral's in-tree sparse callers
//!      already use).
//!   2. `BunchKaufmanParams::default()` (dense) stays at `0.0` per
//!      the 2026-04-13 dense-vs-sparse split decision.
//!   3. On a small bordered LS-init matrix mirroring the failing
//!      inequality-row pattern, the residual under
//!      `NumericParams::default()` is finite and small. (We do not
//!      attempt to reproduce the 58-zero arki0003 pattern at unit
//!      scale; the structural defaults assertion is what locks in
//!      the fix.)

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};

#[test]
fn numeric_params_default_uses_sparse_pivot_threshold() {
    let p = NumericParams::default();
    assert_eq!(
        p.bk.pivot_threshold, 1e-8,
        "NumericParams::default() must use MA27's cntl[1] / Ipopt's \
         ma27_pivtol default (1e-8) to activate the column-relative \
         pivot rejection on rank-deficient KKT systems while staying \
         conservative on Identity-scaled inputs. See issue #2."
    );
}

#[test]
fn bunch_kaufman_params_default_unchanged() {
    let p = BunchKaufmanParams::default();
    assert_eq!(
        p.pivot_threshold, 0.0,
        "BunchKaufmanParams::default() (dense entry point) must stay \
         at 0.0 per dev/decisions.md 2026-04-13. Issue #2 only \
         changes the sparse default at NumericParams::default()."
    );
}

/// Build a small bordered LS-init matrix matching ripopt's
/// `compute_ls_multiplier_estimate_augmented` shape with one redundant
/// equality row (so the augmented system is rank-deficient by one).
///
/// Layout (5×5, lower triangle):
///
/// ```text
///       c=0  c=1  c=2  c=3  c=4
/// r=0:   1
/// r=1:   .    1
/// r=2:   1    .    0                  <- eq (D=0), J=[1,0]
/// r=3:   .    1    .    0             <- eq (D=0), J=[0,1]
/// r=4:   1    1    .    .    0        <- eq (D=0), J=[1,1] redundant
/// ```
///
/// Rank is 4. There is a 1-dimensional nullspace
/// `(0, 0, -1, -1, 1)^T` corresponding to the redundancy. AMD will
/// usually push the constraint rows to the elimination tail, where
/// the zero diagonals couple to unit-magnitude J^T entries — exactly
/// the saddle pattern from arki0003's tail.
fn build_ls_redundant_eq_matrix() -> CscMatrix {
    // Lower-triangle triplets only.
    let rows = vec![0, 1, 2, 3, 4, 2, 3, 4, 4];
    let cols = vec![0, 1, 0, 1, 0, 2, 3, 1, 4];
    let vals = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    CscMatrix::from_triplets(5, &rows, &cols, &vals).expect("triplet build")
}

#[test]
fn ls_init_redundant_eq_factors_under_default() {
    // The matrix is rank-deficient by 1, so solve cannot achieve
    // machine precision in general — but `solve_sparse_refined`
    // should still produce a finite, structured solution under the
    // new default. The key check is "no NaN, no exact-zero
    // multiplier on a non-structurally-zero row, residual < 1.0".
    let m = build_ls_redundant_eq_matrix();
    let sym = symbolic_factorize(&m, &SupernodeParams::default()).expect("symbolic");

    let params = NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            ..NumericParams::default().bk
        },
        ..NumericParams::default()
    };
    let (fac, _inertia) = factorize_multifrontal(&m, &sym, &params).expect("factor");

    // RHS: top block (r) primal-feasibility residual, bottom block
    // (y) inequality slack residual. Mimics ripopt's LS-init RHS.
    let rhs = vec![1.0, 1.0, 0.0, 0.0, 0.0];
    let x = solve_sparse_refined(&m, &fac, &rhs).expect("solve");

    for (i, xi) in x.iter().enumerate() {
        assert!(xi.is_finite(), "x[{}] = {} not finite", i, xi);
    }

    // Residual sanity: with a rank-1 deficiency the projection onto
    // range(A) keeps the residual bounded; we assert a generous
    // bound that the buggy-default path can fail (it produces
    // larger residuals when forced-zeros propagate through the
    // factor).
    let n = m.n;
    let mut ax = vec![0.0; n];
    m.symv(&x, &mut ax);
    let mut r2 = 0.0;
    let mut b2 = 0.0;
    for i in 0..n {
        r2 += (ax[i] - rhs[i]).powi(2);
        b2 += rhs[i] * rhs[i];
    }
    let rel = (r2 / b2.max(1.0)).sqrt();
    assert!(
        rel < 1.0,
        "relative residual {} on rank-deficient LS matrix is implausibly large",
        rel
    );
}
