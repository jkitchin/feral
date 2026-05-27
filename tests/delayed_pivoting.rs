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
//!   2. `factor_frontal_root_force_accepts_without_delay` — a frontal
//!      whose fully-summed diagonal is exactly zero (rank-deficient),
//!      factored with `may_delay = false`, must fall through to the
//!      `ZeroPivotAction::ForceAccept` path and return `nelim == ncol`
//!      with `inertia.zero == ncol`. For small-but-nonzero pivots the
//!      root path accepts them with their correct sign instead (see
//!      test 2b). This is the root-supernode contract.
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
//! Integration-level tests — Phase 2.3 Step 5:
//!
//!   4. `factorize_multifrontal_delays_propagate_to_parent` — a 4×4
//!      arrow KKT-like matrix with AMD ordering producing multiple
//!      supernodes under `nemin = 1`. Column 0 has a tiny diagonal
//!      (1e-3) and a dominant off-diagonal (1.0) in row 3, so the
//!      column-relative test at threshold 0.01 rejects it. With
//!      delays enabled, column 0 is left un-eliminated at its leaf
//!      supernode and must re-enter pivot search at the arrow-apex
//!      supernode (which is the parent via the elimination tree).
//!      The test pins `n_delayed_in == 1` at the root.
//!   5. `factorize_multifrontal_delayed_pivot_succeeds_at_parent` —
//!      the same matrix, with an additional end-to-end inertia check
//!      against a dense LDLᵀ oracle to prove the delayed pivot path
//!      reaches the same factorization as the dense path.

#![allow(clippy::int_plus_one)]
use feral::dense::factor::{factor, factor_frontal};
use feral::numeric::factorize::factorize_multifrontal_supernodal;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, SymmetricMatrix, ZeroPivotAction};

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

/// Variant of `trailing_dominated_frontal` where the fully-summed
/// diagonal is *exactly* zero. At the root (may_delay=false) these
/// pivots are below `zero_tol = f64::EPSILON` and go through the
/// strict-zero ForceAccept path. Issue #54 (SSIDS alignment): that
/// path now routes both bit-exact `0.0` pivots into `inertia.zero`
/// (matching SSIDS `NumericSubtree.hxx:259-267` and MA57).
fn trailing_dominated_zero_frontal() -> SymmetricMatrix {
    let mut mat = SymmetricMatrix::zeros(4);
    // Fully-summed block is exactly zero — rank-deficient by construction.
    mat.set(2, 0, 1.0);
    mat.set(2, 1, 1.0);
    mat.set(3, 0, 1.0);
    mat.set(3, 1, 1.0);
    mat.set(2, 2, 10.0);
    mat.set(3, 3, 10.0);
    mat
}

#[test]
fn factor_frontal_root_force_accepts_without_delay() {
    let mat = trailing_dominated_zero_frontal();
    let params = delay_params();

    // may_delay = false is the root-supernode contract. With a
    // genuinely zero diagonal (below zero_tol), the strict-zero
    // ForceAccept path fires. Issue #54 (SSIDS alignment): both
    // bit-exact 0.0 pivots are counted into `zero`, matching SSIDS /
    // MA57.
    let ff = factor_frontal(&mat, 2, false, &params).expect("factor_frontal");

    assert_eq!(ff.nelim, 2, "root eliminates all attempted columns");
    assert_eq!(ff.ncol, 2);
    assert_eq!(ff.n_delayed, 0, "no delay path taken");
    assert_eq!(ff.contrib_dim, 2, "contrib is the 2×2 trailing block");
    assert_eq!(
        ff.inertia.zero, 2,
        "issue #54 (SSIDS): both bit-exact 0.0 pivots → `zero`",
    );
    assert_eq!(ff.inertia.positive, 0);
    assert_eq!(
        ff.inertia.negative, 0,
        "no negative pivots — zeros are in `zero`"
    );
    assert!(ff.needs_refinement, "ForceAccept must flag refinement");
    assert_eq!(ff.d_diag.len(), 2);
}

#[test]
fn factor_frontal_root_accepts_small_pivot_with_sign() {
    // Same structure as trailing_dominated_frontal but the first
    // fully-summed diagonal is -1e-8 (small and *negative*). At the
    // root, the column-relative test |d| >= u*col_max fails
    // (|−1e-8| < 0.01 · 1.0), but −1e-8 is well above
    // zero_tol = f64::EPSILON ≈ 2.2e-16, so the pivot is accepted
    // with its correct sign. This is the SSIDS/MUMPS convention:
    // a small-but-clearly-nonzero pivot contributes to the negative
    // inertia, not the zero count. Without this fix a DEGENLPA-
    // style KKT would mis-report (n+, n−, n0) = (20, 14, 1) instead
    // of the true (20, 15, 0).
    let mut mat = SymmetricMatrix::zeros(4);
    mat.set(0, 0, -1e-8);
    mat.set(1, 1, 5.0);
    mat.set(2, 0, 1.0);
    mat.set(3, 0, 1.0);
    mat.set(2, 2, 10.0);
    mat.set(3, 3, 10.0);
    let params = delay_params();

    let ff = factor_frontal(&mat, 2, false, &params).expect("factor_frontal");

    assert_eq!(ff.nelim, 2, "root eliminates all attempted columns");
    assert_eq!(ff.ncol, 2);
    assert_eq!(ff.n_delayed, 0, "no delay path taken");
    assert_eq!(
        ff.inertia.negative, 1,
        "small negative pivot counted as negative, not zero"
    );
    assert_eq!(
        ff.inertia.positive, 1,
        "clean positive pivot counted correctly"
    );
    assert_eq!(ff.inertia.zero, 0, "no zero pivots — both above zero_tol");
    assert!(
        ff.needs_refinement,
        "small-but-accepted pivot still requires iterative refinement"
    );
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

// ---------------------------------------------------------------------
// Integration tests — Phase 2.3 Step 5 (parent-side delay assembly).
// ---------------------------------------------------------------------

/// Assemble a 4×4 arrow-apex matrix where column 0 has a tiny diagonal
/// and a dominant off-diagonal in the arrow apex row. The extra
/// off-diagonal `(2,1) = 0.5` breaks the leaf degree tie so column 0
/// is *uniquely* the minimum-degree column (degree 1) — every AMD
/// tie-breaking rule selects it first, making pivot 0 the tiny-pivot
/// leaf. With `pivot_threshold = 0.01`:
///
///   * column 0's relative test (|1e-3| vs 0.01 · 1.0 = 0.01) fails,
///     so the leaf supernode at pivot 0 must delay its column;
///   * pivots 1, 2, 3 pass cleanly and amalgamate into one root
///     supernode (fundamental-supernode detection + SSIDS
///     trivial-chain merge with `nemin=1`);
///   * the root supernode inherits column 0 as a delayed fully-
///     summed column in its frontal.
///
/// Historical note: the original fixture had `(2,1) = 0` (three
/// equal-degree leaves). That worked with the in-tree AMD's
/// ascending-index tie-breaker — column 0 ended up at pivot 0 — but
/// feral-amd (the current default) uses a different tie-breaker and
/// ordered column 0 at pivot 2, which then merged into the root
/// supernode under SSIDS trivial-chain amalgamation. See
/// `dev/journal/2026-04-18-03.org` 10:30 entry.
///
///     A = [ 1e-3   0     0     1   ]
///         [ 0     10    0.5    1   ]
///         [ 0    0.5   10     1   ]
///         [ 1     1     1     2   ]
fn arrow_apex_matrix() -> CscMatrix {
    CscMatrix::from_triplets(
        4,
        // (row, col) lower-triangle entries only
        &[0, 3, 1, 2, 3, 2, 3, 3],
        &[0, 0, 1, 1, 1, 2, 2, 3],
        &[1e-3, 1.0, 10.0, 0.5, 1.0, 10.0, 1.0, 2.0],
    )
    .expect("build arrow matrix")
}

fn delay_sparse_params() -> SupernodeParams {
    SupernodeParams {
        nemin: 1,
        ..Default::default()
    }
}

fn delay_numeric_params() -> feral::numeric::factorize::NumericParams {
    feral::numeric::factorize::NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        },
        scaling: feral::scaling::ScalingStrategy::Identity,
        small_leaf: feral::numeric::factorize::SmallLeafBatch::default(),
        profiler: None,
        parallel_telemetry: None,
        fma: false,
        allow_delayed_pivots: true,
        cascade_break_ratio: None,
        cascade_break_eps: None,
        min_parallel_flops: None,
        sqd_mode: false,
        static_pivot_threshold: None,
        warn_partial_singular: false,
        pattern_reused_hint: false,
    }
}

#[test]
fn factorize_multifrontal_delays_propagate_to_parent() {
    let m = arrow_apex_matrix();
    let sym = symbolic_factorize(&m, &delay_sparse_params()).expect("symbolic");
    let params = delay_numeric_params();
    let (factors, _inertia) = factorize_multifrontal_supernodal(&m, &sym, &params).expect("factor");

    // Need at least 2 supernodes for a delay to propagate parent-ward.
    // Fundamental-supernode detection in the symbolic phase can still
    // merge adjacent columns that share a row structure, so we don't
    // pin the exact count — only that delays have somewhere to go.
    assert!(
        factors.node_factors.len() >= 2,
        "need at least 2 supernodes for parent-ward delay propagation, got {}",
        factors.node_factors.len()
    );

    // Find the supernode that corresponds to the arrow apex (the unique
    // root — the supernode with no parent in the forest). With AMD on a
    // star-shaped pattern the apex ends up ordered last, and is the last
    // postordered node. Assert it as the root by checking that it is the
    // only node whose `n_delayed_in > 0` — every other node has no
    // delayed children to absorb.
    let parent = factors
        .node_factors
        .iter()
        .find(|n| n.n_delayed_in > 0)
        .expect("expected one parent absorbing a delayed column");

    // Exactly one delayed column propagated up (from the tiny-pivot
    // leaf). The parent's attempted column count (`ncol`) is therefore
    // `own native ncol + 1`, its row_indices list is longer than the
    // old non-delayed layout would have produced, and its fully-summed
    // region includes the delayed global index.
    assert_eq!(parent.n_delayed_in, 1, "one column delayed into parent");
    assert!(
        parent.ncol >= 1 + parent.n_delayed_in,
        "ncol should include native + delayed"
    );
    assert!(
        parent.nrow >= parent.ncol,
        "nrow must be ≥ ncol after absorbing delays"
    );

    // Exactly one supernode must have reported a delay out (nelim < ncol).
    let leaves_that_delayed = factors
        .node_factors
        .iter()
        .filter(|n| n.frontal_factors.n_delayed > 0)
        .count();
    assert_eq!(
        leaves_that_delayed, 1,
        "exactly one leaf should have delayed a column"
    );
}

#[test]
fn factorize_multifrontal_delayed_pivot_succeeds_at_parent() {
    let m = arrow_apex_matrix();

    // Dense oracle: factor the full matrix with the same BK params
    // (may_delay is always false in the dense path). This matrix is
    // small enough that dense LDLᵀ with ForceAccept produces the
    // "true" inertia signature we want the sparse path to match once
    // the delayed pivot has been re-tried at the parent.
    let dense = m.to_dense();
    let params = delay_numeric_params();
    let (_, dense_inertia) = factor(&dense, &params.bk).expect("dense factor");

    let sym = symbolic_factorize(&m, &delay_sparse_params()).expect("symbolic");
    let (factors, sparse_inertia) =
        factorize_multifrontal_supernodal(&m, &sym, &params).expect("sparse factor");

    assert_eq!(
        sparse_inertia, dense_inertia,
        "sparse delayed-pivot path must match dense LDLᵀ inertia"
    );
    assert_eq!(
        sparse_inertia.positive + sparse_inertia.negative + sparse_inertia.zero,
        4,
        "all 4 pivots must be accounted for after delays resolve at the parent"
    );

    // The delay actually fired somewhere in the tree (this is what
    // distinguishes the test from a pass on the non-delay path).
    let delays_in_tree: usize = factors
        .node_factors
        .iter()
        .map(|n| n.frontal_factors.n_delayed)
        .sum();
    assert!(
        delays_in_tree > 0,
        "expected at least one delayed pivot to prove the delay path fired"
    );
}
