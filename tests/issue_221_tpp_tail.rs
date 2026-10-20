//! Issue #221: fronts that cannot delay (cascade-break, root) with many
//! threshold-failing pivots.
//!
//! Before the TPP tail, every such pivot left the blocked panel, paid a
//! rook scan and a full unblocked rank-1 update of the trailing matrix,
//! so a front with thousands of failing columns ran at level-2 speed
//! (qcqp1000-2c, nql180 after pounce's quality escalation). The tail
//! (`dev/plans/issue-221-tpp-tail.md`) keeps the same acceptance rules
//! but defers the contribution-block update. These tests pin the result,
//! not the speed.
//!
//! Oracles are hand calculations on fronts built so that every
//! fully-summed pivot fails the threshold:
//! - the fully-summed block is `diag(δ_i)`, so by Sylvester's law the
//!   eliminated inertia is the signs of the `δ_i`;
//! - the single ghost row couples to every column with `a_i`, so its
//!   Schur complement is `h - Σ a_i² / δ_i`.

use feral::dense::factor::factor_frontal_blocked;
use feral::{BunchKaufmanParams, SymmetricMatrix, ZeroPivotAction};

/// `m` fully-summed columns with diagonal `δ_i = ±(1 + i)·1e-8`
/// (alternating sign) and no coupling among themselves, one ghost row
/// with diagonal `h` and coupling `a_i = 1 + i/m` to column `i`.
/// `|δ_i| <= u·|a_i|` for `u = 0.01`, so every 1×1 fails, and no
/// fully-summed 2×2 partner exists.
fn forced_front(m: usize, h: f64) -> (SymmetricMatrix, Vec<f64>, Vec<f64>) {
    let n = m + 1;
    let mut a = SymmetricMatrix::zeros(n);
    let mut delta = Vec::with_capacity(m);
    let mut coup = Vec::with_capacity(m);
    for i in 0..m {
        let d = if i % 2 == 0 { 1.0 } else { -1.0 } * (1.0 + i as f64) * 1e-8;
        let c = 1.0 + i as f64 / m as f64;
        a.set(i, i, d);
        a.set(m, i, c);
        delta.push(d);
        coup.push(c);
    }
    a.set(m, m, h);
    (a, delta, coup)
}

fn params(on_zero: ZeroPivotAction) -> BunchKaufmanParams {
    BunchKaufmanParams {
        pivot_threshold: 0.01,
        on_zero_pivot: on_zero,
        ..BunchKaufmanParams::default()
    }
}

fn check_forced_front(m: usize, on_zero: ZeroPivotAction) {
    let h = 1e4;
    let (a, delta, coup) = forced_front(m, h);
    let ff = factor_frontal_blocked(&a, m, false, &params(on_zero)).expect("factor");

    assert_eq!(
        ff.nelim, m,
        "a front that cannot delay eliminates every column"
    );
    assert_eq!(ff.n_delayed, 0);

    let want_pos = delta.iter().filter(|d| **d > 0.0).count();
    let want_neg = m - want_pos;
    assert_eq!(
        (ff.inertia.positive, ff.inertia.negative, ff.inertia.zero),
        (want_pos, want_neg, 0),
        "Sylvester: inertia of diag(δ)"
    );

    // Hand oracle for the ghost row's Schur complement.
    let terms: Vec<f64> = delta.iter().zip(&coup).map(|(d, c)| c * c / d).collect();
    let want = h - terms.iter().sum::<f64>();
    let scale = h.abs() + terms.iter().map(|t| t.abs()).sum::<f64>();
    assert_eq!(ff.contrib_dim, 1);
    let got = ff.contrib[0];
    assert!(
        (got - want).abs() <= 1e-12 * scale,
        "ghost Schur complement {got:e}, expected {want:e} (scale {scale:e})"
    );
}

#[test]
fn forced_front_small_force_accept() {
    check_forced_front(8, ZeroPivotAction::ForceAccept);
}

#[test]
fn forced_front_wide_force_accept() {
    // Wider than any panel (block size 64), so the deferred update spans
    // several chunks.
    check_forced_front(300, ZeroPivotAction::ForceAccept);
}

#[test]
fn forced_front_wide_perturb_to_eps() {
    // Cascade-break semantics. Every |δ| >= 1e-8 is above the 1e-10
    // floor, so nothing is floored and the hand oracle still holds.
    check_forced_front(300, ZeroPivotAction::PerturbToEps { abs_floor: 1e-10 });
}

/// Mixed front: `m` pairs `[[0, 1], [1, 0]]` placed `m` columns apart
/// (column `i` couples to column `m + i`), every column also coupled to
/// one ghost row with weight 5 so that BK's argmax is the ghost row.
/// Inertia of the fully-summed block: `(m, m, 0)`, since each pair has
/// eigenvalues ±1 and the pairs are disjoint.
#[test]
fn far_apart_pairs_in_a_front_that_cannot_delay() {
    let m = 150;
    let n_fs = 2 * m;
    let mut a = SymmetricMatrix::zeros(n_fs + 1);
    for i in 0..m {
        a.set(m + i, i, 1.0);
    }
    for j in 0..n_fs {
        a.set(n_fs, j, 5.0);
    }
    a.set(n_fs, n_fs, 1e3);
    let ff = factor_frontal_blocked(&a, n_fs, false, &params(ZeroPivotAction::ForceAccept))
        .expect("factor");
    assert_eq!(ff.nelim, n_fs);
    assert_eq!(
        (ff.inertia.positive, ff.inertia.negative, ff.inertia.zero),
        (m, m, 0)
    );
    // Hand oracle: with P = blockdiag of [[0,1],[1,0]], P⁻¹ = P, and the
    // ghost coupling vector b = 5·1, bᵀ P⁻¹ b = Σ_pairs 2·5·5 = 50 m.
    let want = 1e3 - 50.0 * m as f64;
    assert!(
        (ff.contrib[0] - want).abs() <= 1e-9 * want.abs(),
        "ghost Schur complement {:e}, expected {want:e}",
        ff.contrib[0]
    );
}
