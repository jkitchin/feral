//! Phase 2.3 kernel-level tests for delayed pivoting in `factor_frontal`.
//!
//! These tests pin down the contract of the new `may_delay` parameter and
//! the `nelim` / `n_delayed` fields on `FrontalFactors`:
//!
//!   1. `factor_frontal_delays_first_pivot_when_may_delay` — a frontal
//!      whose only strong entries live in the non-fully-summed trailing
//!      rows has no valid BK pivot inside the fully-summed block. With
//!      `may_delay = true`, the kernel breaks on the very first column
//!      and returns `nelim = 0` with the full trailing block preserved.
//!   2. `factor_frontal_root_force_accepts_without_delay` — the same
//!      frontal with `may_delay = false` falls through to the existing
//!      `ZeroPivotAction::ForceAccept` path and returns `nelim == ncol`
//!      with `inertia.zero == ncol`. This is the root-supernode contract.
//!   3. `factor_frontal_partial_elim_with_delay` — a 5×5 block-diagonal
//!      frontal where columns 0 and 1 factor cleanly but columns 2 and 3
//!      cannot pivot without swapping in a trailing row. With
//!      `may_delay = true`, the kernel eliminates the first two columns
//!      and delays the last two, producing `nelim = 2`, `n_delayed = 2`,
//!      and a 3×3 contribution block whose top-left 2×2 is the delayed
//!      pivot pair.
//!
//! The contribution-block content is sanity-checked against the raw
//! input entries (cols 2..5 are untouched by the first two rank-1
//! updates because columns 0 and 1 are block-diagonal), so the test is
//! an independent oracle — it does not depend on any internal
//! book-keeping the implementation might change.
//!
//! Integration-level tests (delayed pivots propagating to the parent
//! supernode) arrive with Step 4 of the Phase 2.3 plan.

use feral::dense::factor::factor_frontal;
use feral::{BunchKaufmanParams, SymmetricMatrix, ZeroPivotAction};

fn delay_params() -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    }
}

/// 4×4 frontal with ncol=2:
///
///     [ 1e-14   1e-14  |  1.0    1.0  ]
///     [ 1e-14   1e-14  |  1.0    1.0  ]
///     [ 1.0     1.0    | 10.0    0.0  ]
///     [ 1.0     1.0    |  0.0   10.0  ]
///
/// BK at k=0 finds gamma0 = 1.0 at r = 2, but r is NOT fully-summed
/// (2 >= ncol), so the r-swap path is skipped. The LAPACK extension
/// akk*gamma_r = 1e-14 < alpha*gamma0^2 also fails. The 2×2 path needs
/// r fully-summed — it doesn't apply either. The kernel falls into the
/// last-resort 1×1 at k, where try_reject_1x1_frontal sees
/// |d| = 1e-14 ≤ 0.01 * 1.0 = 0.01 and rejects.
fn trailing_dominated_frontal() -> SymmetricMatrix {
    let mut mat = SymmetricMatrix::zeros(4);
    // Fully-summed block (tiny)
    mat.set(0, 0, 1e-14);
    mat.set(1, 0, 1e-14);
    mat.set(1, 1, 1e-14);
    // Fully-summed × trailing block
    mat.set(2, 0, 1.0);
    mat.set(2, 1, 1.0);
    mat.set(3, 0, 1.0);
    mat.set(3, 1, 1.0);
    // Trailing block (diagonal, well-conditioned)
    mat.set(2, 2, 10.0);
    mat.set(3, 3, 10.0);
    mat
}

#[test]
fn factor_frontal_delays_first_pivot_when_may_delay() {
    let mat = trailing_dominated_frontal();
    let params = delay_params();

    let ff = factor_frontal(&mat, 2, true, &params).expect("factor_frontal");

    // Nothing eliminated — the parent supernode will retry these columns.
    assert_eq!(ff.nelim, 0, "expected zero eliminations");
    assert_eq!(ff.ncol, 2, "ncol preserved from input");
    assert_eq!(ff.n_delayed, 2, "both fully-summed columns delayed");
    assert_eq!(ff.contrib_dim, 4, "contrib captures the full frontal");

    // Inertia should be all zeros: no pivots committed, no ForceAccept
    // fired, no needs_refinement flag.
    assert_eq!(ff.inertia.positive, 0);
    assert_eq!(ff.inertia.negative, 0);
    assert_eq!(ff.inertia.zero, 0);
    assert!(!ff.needs_refinement, "delay path must not flag refinement");

    // L and D are sized to zero eliminations.
    assert_eq!(ff.l.len(), 0);
    assert_eq!(ff.d_diag.len(), 0);
    assert_eq!(ff.d_subdiag.len(), 0);

    // Contribution block must preserve the frontal data verbatim
    // (nothing was updated because nothing was eliminated).
    // Column-major, lower triangle only, dim = 4.
    let cdim = ff.contrib_dim;
    let get = |i: usize, j: usize| -> f64 {
        let (ii, jj) = if i >= j { (i, j) } else { (j, i) };
        ff.contrib[jj * cdim + ii]
    };
    assert_eq!(get(0, 0), 1e-14);
    assert_eq!(get(1, 0), 1e-14);
    assert_eq!(get(1, 1), 1e-14);
    assert_eq!(get(2, 0), 1.0);
    assert_eq!(get(2, 1), 1.0);
    assert_eq!(get(3, 0), 1.0);
    assert_eq!(get(3, 1), 1.0);
    assert_eq!(get(2, 2), 10.0);
    assert_eq!(get(3, 3), 10.0);
}

#[test]
fn factor_frontal_root_force_accepts_without_delay() {
    let mat = trailing_dominated_frontal();
    let params = delay_params();

    // may_delay = false is the root-supernode contract. ForceAccept
    // must fire, flushing both tiny pivots as zeros.
    let ff = factor_frontal(&mat, 2, false, &params).expect("factor_frontal");

    assert_eq!(ff.nelim, 2, "root eliminates all attempted columns");
    assert_eq!(ff.ncol, 2);
    assert_eq!(ff.n_delayed, 0, "no delay path taken");
    assert_eq!(ff.contrib_dim, 2, "contrib is the 2×2 trailing block");
    assert_eq!(ff.inertia.zero, 2, "both pivots counted as zero");
    assert_eq!(ff.inertia.positive, 0);
    assert_eq!(ff.inertia.negative, 0);
    assert!(ff.needs_refinement, "ForceAccept must flag refinement");
    assert_eq!(ff.d_diag.len(), 2);
}

/// 5×5 frontal with ncol=4 and a block-diagonal split between the first
/// two columns (which factor trivially) and the remaining delayed pair:
///
///     [ 2.0    0      0      0      0   ]
///     [ 0      3.0    0      0      0   ]
///     [ 0      0      1e-14  1.0    100 ]
///     [ 0      0      1.0    1e-14  100 ]
///     [ 0      0      100    100    1e6 ]
///
/// Columns 0 and 1 have `gamma0 = 0` (nothing below the diagonal), so
/// BK counts them as positive 1×1 pivots without ever touching
/// `try_reject_1x1_frontal`. At k=2 the column max is 100 (in row 4,
/// non-fully-summed) so the 1×1 fallback rejects the 1e-14 diagonal
/// and — with `may_delay = true` — the kernel breaks. Because columns
/// 0 and 1 are block-diagonal with respect to columns 2..=4, the
/// rank-1 updates for the first two eliminations are no-ops on the
/// trailing block, so the contribution block equals the raw 3×3
/// bottom-right submatrix.
fn block_diagonal_partial_frontal() -> SymmetricMatrix {
    let mut mat = SymmetricMatrix::zeros(5);
    mat.set(0, 0, 2.0);
    mat.set(1, 1, 3.0);
    mat.set(2, 2, 1e-14);
    mat.set(3, 2, 1.0);
    mat.set(3, 3, 1e-14);
    mat.set(4, 2, 100.0);
    mat.set(4, 3, 100.0);
    mat.set(4, 4, 1e6);
    mat
}

#[test]
fn factor_frontal_partial_elim_with_delay() {
    let mat = block_diagonal_partial_frontal();
    let params = delay_params();

    let ff = factor_frontal(&mat, 4, true, &params).expect("factor_frontal");

    assert_eq!(ff.nelim, 2, "columns 0..=1 factored, 2..=3 delayed");
    assert_eq!(ff.ncol, 4);
    assert_eq!(ff.n_delayed, 2, "two delayed fully-summed columns");
    assert_eq!(ff.contrib_dim, 3, "contrib = (nrow - nelim) = 3");
    assert_eq!(ff.inertia.positive, 2);
    assert_eq!(ff.inertia.negative, 0);
    assert_eq!(ff.inertia.zero, 0);

    // L has (nrow × nelim) = 5 × 2 shape with unit diagonals at
    // positions (0,0) and (1,1) and zero sub-diagonal entries (the
    // trivial block-diagonal columns).
    let nrow = ff.nrow;
    let l_at = |i: usize, j: usize| ff.l[j * nrow + i];
    assert_eq!(ff.l.len(), nrow * ff.nelim);
    assert_eq!(l_at(0, 0), 1.0, "L[0,0] unit diagonal");
    assert_eq!(l_at(1, 1), 1.0, "L[1,1] unit diagonal");
    for i in 1..nrow {
        assert_eq!(l_at(i, 0), 0.0, "L[{},0] should be zero", i);
    }
    for i in 2..nrow {
        assert_eq!(l_at(i, 1), 0.0, "L[{},1] should be zero", i);
    }
    assert_eq!(ff.d_diag.len(), 2);
    assert_eq!(ff.d_diag[0], 2.0);
    assert_eq!(ff.d_diag[1], 3.0);

    // Contribution block (3×3, lower triangle). Because cols 0,1 are
    // block-diagonal w.r.t. cols 2..=4, no rank-1 update touches the
    // trailing block — contrib equals the raw bottom-right 3×3.
    let cdim = ff.contrib_dim;
    let get = |i: usize, j: usize| -> f64 {
        let (ii, jj) = if i >= j { (i, j) } else { (j, i) };
        ff.contrib[jj * cdim + ii]
    };
    assert_eq!(get(0, 0), 1e-14, "delayed diag col 2");
    assert_eq!(get(1, 0), 1.0, "delayed off-diag (3,2)");
    assert_eq!(get(1, 1), 1e-14, "delayed diag col 3");
    assert_eq!(get(2, 0), 100.0, "cross-block (4,2)");
    assert_eq!(get(2, 1), 100.0, "cross-block (4,3)");
    assert_eq!(get(2, 2), 1e6, "trailing diag col 4");
}
