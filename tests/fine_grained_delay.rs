//! Track A2 / Fix 1 — fine-grained delayed pivoting (swap-to-boundary).
//!
//! ## The bug
//!
//! Both Bunch-Kaufman driver loops in `src/dense/factor.rs` did
//! `Delayed => break`: the first delayed pivot forfeited the *entire*
//! remaining tail of the supernode (`n_delayed = ncol - nelim`). On
//! `pinene_3200` 3936 scalar delay events became `n_delayed = 133648`
//! (~34 columns forfeited per event) → a 69× fill blowup. See
//! `dev/research/kkt-cascade-amplifier-2026-05-21.md`.
//!
//! ## The fix
//!
//! Swap-to-boundary: when the pivot at column `k` delays, swap it with
//! the last still-eligible column (`ncol_eff - 1`), decrement
//! `ncol_eff`, and keep eliminating. Each stuck pivot forfeits exactly
//! one column instead of the whole tail. Real delayed pivoting — the
//! stuck column is promoted to the parent front — so inertia stays
//! exact by construction.
//!
//! ## Oracle
//!
//! Fine-grained delayed pivoting forfeits exactly the genuinely-stuck
//! columns. The fixtures are built so exactly ONE column is provably
//! stuck — a near-zero diagonal whose only coupling is an out-of-front
//! trailing row, so Bunch-Kaufman can form neither a 1×1 (diagonal
//! below the column-relative threshold) nor a 2×2 (no fully-summed
//! partner) — and every other column is provably pivotable (positive
//! diagonal, no off-diagonal coupling, so `gamma0 = 0`). Break-on-first
//! forfeits the pivotable tail; swap-to-boundary forfeits only the one
//! stuck column. This is external math (Bunch & Kaufman 1977 pivot
//! admissibility + the delayed-pivoting contract), not a property read
//! off the implementation.

use feral::dense::factor::{factor_frontal, factor_frontal_blocked};
use feral::{BunchKaufmanParams, SymmetricMatrix, ZeroPivotAction};

/// `ForceAccept` + a 1 % column-relative threshold, matching the
/// kernel-level delayed-pivot tests in `tests/delayed_pivoting.rs`.
fn delay_params() -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    }
}

/// 5×5 frontal, `ncol = 4`, plain (non-panel) driver:
///
/// ```text
///     [ 2.0     0      0      0      0   ]   col 0 — isolated SPD
///     [ 0      1e-14   0      0      0   ]   col 1 — STUCK
///     [ 0       0     3.0     0      0   ]   col 2 — isolated SPD
///     [ 0       0      0     5.0     0   ]   col 3 — isolated SPD
///     [ 0      100     0      0     1e6  ]   row 4 — trailing (not fully summed)
/// ```
///
/// Column 1 has diagonal `1e-14` and its only off-diagonal is the
/// coupling `100` to row 4 — which is `>= ncol`, so not fully summed.
/// Bunch-Kaufman's argmax lands on the out-of-front row 4: it cannot
/// swap it in for a 2×2, and the last-resort 1×1 fails the
/// column-relative test (`|1e-14| <= 0.01·100`). Column 1 genuinely
/// delays. Columns 0, 2, 3 have `gamma0 = 0` and factor as clean
/// positive 1×1 pivots.
///
/// Break-on-first: `nelim = 1`, columns 1/2/3 all forfeited
/// (`n_delayed = 3`). Swap-to-boundary: column 1 alone is delayed,
/// columns 0/2/3 eliminate → `nelim = 3`, `n_delayed = 1`.
fn one_stuck_column_plain() -> SymmetricMatrix {
    let mut mat = SymmetricMatrix::zeros(5);
    mat.set(0, 0, 2.0);
    mat.set(1, 1, 1e-14);
    mat.set(2, 2, 3.0);
    mat.set(3, 3, 5.0);
    mat.set(4, 1, 100.0); // col 1's only coupling — out of front
    mat.set(4, 4, 1e6);
    mat
}

#[test]
fn plain_driver_swaps_one_stuck_column_to_the_boundary() {
    let mat = one_stuck_column_plain();
    let params = delay_params();

    let ff = factor_frontal(&mat, 4, true, &params).expect("factor_frontal");

    // Swap-to-boundary forfeits exactly the one stuck column.
    assert_eq!(
        ff.nelim, 3,
        "columns 0/2/3 pivotable — only the stuck column is delayed",
    );
    assert_eq!(ff.n_delayed, 1, "exactly one genuinely-stuck column");
    assert_eq!(ff.ncol, 4, "ncol preserved from input");

    // The three eliminated pivots are positive SPD diagonals.
    assert_eq!(
        (ff.inertia.positive, ff.inertia.negative, ff.inertia.zero),
        (3, 0, 0),
        "three clean positive 1×1 pivots; got {:?}",
        ff.inertia,
    );

    // The stuck column (original front index 1) was swapped to the
    // fully-summed boundary: it now sits at front position `nelim` and
    // `perm` tracks its provenance.
    assert_eq!(
        ff.perm[ff.nelim], 1,
        "stuck column (orig index 1) tracked to the delayed boundary",
    );

    // Contribution block: position 0 is the delayed (stuck) column,
    // position 1 is trailing row 4. cdim = nrow - nelim = 2.
    assert_eq!(ff.contrib_dim, 2, "contrib = nrow - nelim = 5 - 3");
    let cdim = ff.contrib_dim;
    let get = |i: usize, j: usize| -> f64 {
        let (ii, jj) = if i >= j { (i, j) } else { (j, i) };
        ff.contrib[jj * cdim + ii]
    };
    assert_eq!(get(0, 0), 1e-14, "delayed column keeps its tiny diagonal");
    assert_eq!(get(1, 0), 100.0, "delayed column's coupling to row 4");
    assert_eq!(get(1, 1), 1e6, "trailing diagonal untouched");
}

/// `ncol`-wide frontal (one trailing row) routed through the *panel*
/// driver (`ncol >= PANEL_MIN_NCOL = 8`): column 1 is the lone stuck
/// column (diagonal `1e-14`, only coupling = out-of-front trailing
/// row), every other fully-summed column is an isolated positive SPD
/// diagonal. Break-on-first forfeits columns `1..ncol`; swap-to-boundary
/// forfeits only column 1.
fn one_stuck_column_panel(ncol: usize) -> SymmetricMatrix {
    let n = ncol + 1; // one trailing (not-fully-summed) row
    let trailing = ncol;
    let mut mat = SymmetricMatrix::zeros(n);
    for c in 0..ncol {
        if c == 1 {
            mat.set(c, c, 1e-14); // stuck: near-zero diagonal
            mat.set(trailing, c, 100.0); // only coupling — out of front
        } else {
            mat.set(c, c, (c as f64) + 2.0); // isolated positive SPD
        }
    }
    mat.set(trailing, trailing, 1e6);
    mat
}

#[test]
fn panel_driver_swaps_one_stuck_column_to_the_boundary() {
    let ncol = 12; // >= PANEL_MIN_NCOL, so factor_frontal_blocked uses the panel
    let mat = one_stuck_column_panel(ncol);
    let params = delay_params();

    let ff = factor_frontal_blocked(&mat, ncol, true, &params).expect("factor_frontal_blocked");

    assert_eq!(
        ff.nelim,
        ncol - 1,
        "panel driver: all pivotable columns eliminated, one delayed",
    );
    assert_eq!(ff.n_delayed, 1, "exactly one genuinely-stuck column");
    assert_eq!(
        (ff.inertia.positive, ff.inertia.negative, ff.inertia.zero),
        (ncol - 1, 0, 0),
        "every eliminated pivot is a clean positive 1×1; got {:?}",
        ff.inertia,
    );
    assert_eq!(
        ff.perm[ff.nelim], 1,
        "stuck column (orig index 1) tracked to the delayed boundary",
    );
}
