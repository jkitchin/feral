//! Issue #221 review: blocked/scalar bit-parity on fronts that cannot
//! delay (`may_delay == false`) and have columns no pivot can rescue.
//!
//! 676df22 made the blocked kernel run such fronts in two phases (a
//! delay pass, then a forced pass over the stuck columns) while the
//! scalar reference `factor_frontal` stayed single-pass, so the two
//! disagreed in L, D, perm and the contribution block. The parity
//! contract of `tests/blocked_ldlt.rs` (byte-identical factors) must hold
//! here too.
//!
//! Oracle: exact byte identity of `(L, D_diag, D_subdiag, perm, inertia,
//! contrib, nelim, n_delayed, needs_refinement, n_rook_rescues, n_tiny,
//! n_threshold_fail_accepts)` between `factor_frontal` and
//! `factor_frontal_blocked`. The inertia of the forced fronts is also
//! checked against Sylvester's law, so the two kernels cannot agree on a
//! wrong answer.

use feral::dense::factor::{factor_frontal, factor_frontal_blocked, FrontalFactors};
use feral::{BunchKaufmanParams, SymmetricMatrix, ZeroPivotAction};

fn assert_bit_identical(s: &FrontalFactors, b: &FrontalFactors, tag: &str) {
    assert_eq!(s.nelim, b.nelim, "{tag} nelim");
    assert_eq!(s.n_delayed, b.n_delayed, "{tag} n_delayed");
    assert_eq!(s.contrib_dim, b.contrib_dim, "{tag} contrib_dim");
    assert_eq!(s.inertia, b.inertia, "{tag} inertia");
    assert_eq!(s.needs_refinement, b.needs_refinement, "{tag} refine");
    assert_eq!(s.n_rook_rescues, b.n_rook_rescues, "{tag} n_rook_rescues");
    assert_eq!(s.n_tiny, b.n_tiny, "{tag} n_tiny");
    assert_eq!(
        s.n_threshold_fail_accepts, b.n_threshold_fail_accepts,
        "{tag} n_threshold_fail_accepts"
    );
    assert_eq!(s.perm, b.perm, "{tag} perm");
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let first_diff = |x: &[f64], y: &[f64]| {
        x.iter()
            .zip(y)
            .position(|(p, q)| p.to_bits() != q.to_bits())
    };
    assert_eq!(s.l.len(), b.l.len(), "{tag} l.len");
    assert!(
        bits(&s.l) == bits(&b.l),
        "{tag} L differs first at {:?}",
        first_diff(&s.l, &b.l)
    );
    assert!(
        bits(&s.d_diag) == bits(&b.d_diag),
        "{tag} d_diag differs first at {:?}",
        first_diff(&s.d_diag, &b.d_diag)
    );
    assert!(
        bits(&s.d_subdiag) == bits(&b.d_subdiag),
        "{tag} d_subdiag differs first at {:?}",
        first_diff(&s.d_subdiag, &b.d_subdiag)
    );
    assert_eq!(s.contrib.len(), b.contrib.len(), "{tag} contrib.len");
    assert!(
        bits(&s.contrib) == bits(&b.contrib),
        "{tag} contrib differs first at {:?}",
        first_diff(&s.contrib, &b.contrib)
    );
}

fn params(on_zero: ZeroPivotAction) -> BunchKaufmanParams {
    BunchKaufmanParams {
        pivot_threshold: 0.01,
        on_zero_pivot: on_zero,
        ..BunchKaufmanParams::default()
    }
}

/// Same construction as `tests/issue_221_tpp_tail.rs`: `diag(δ_i)` with
/// `δ_i = ±(1 + i)·1e-8`, one ghost row coupled to every column with
/// `1 + i/m`. Every 1×1 fails `u = 0.01` and no fully-summed partner
/// exists, so every column is stuck.
fn forced_front(m: usize, h: f64) -> (SymmetricMatrix, Vec<f64>) {
    let mut a = SymmetricMatrix::zeros(m + 1);
    let mut delta = Vec::with_capacity(m);
    for i in 0..m {
        let d = if i % 2 == 0 { 1.0 } else { -1.0 } * (1.0 + i as f64) * 1e-8;
        a.set(i, i, d);
        a.set(m, i, 1.0 + i as f64 / m as f64);
        delta.push(d);
    }
    a.set(m, m, h);
    (a, delta)
}

fn check_forced(m: usize, on_zero: ZeroPivotAction, tag: &str) -> FrontalFactors {
    let (a, delta) = forced_front(m, 1e4);
    let p = params(on_zero);
    let s = factor_frontal(&a, m, false, &p).expect("scalar");
    let b = factor_frontal_blocked(&a, m, false, &p).expect("blocked");
    assert_bit_identical(&s, &b, tag);
    assert_eq!(
        b.nelim, m,
        "{tag}: a front that cannot delay eliminates all"
    );
    // Sylvester's law on diag(δ): floored pivots keep their sign
    // (`sign(δ)·floor`), so the eliminated inertia is the signs of δ.
    let want_pos = delta.iter().filter(|d| **d > 0.0).count();
    assert_eq!(
        (b.inertia.positive, b.inertia.negative, b.inertia.zero),
        (want_pos, m - want_pos, 0),
        "{tag}: Sylvester inertia of diag(δ)"
    );
    b
}

#[test]
fn forced_front_parity_force_accept() {
    for m in [8usize, 40, 300] {
        check_forced(m, ZeroPivotAction::ForceAccept, &format!("m={m} force"));
    }
}

#[test]
fn forced_front_parity_perturb_no_floor_hit() {
    for m in [8usize, 40, 300] {
        let b = check_forced(
            m,
            ZeroPivotAction::PerturbToEps { abs_floor: 1e-10 },
            &format!("m={m} perturb 1e-10"),
        );
        assert_eq!(b.n_tiny, 0, "every |δ| >= 1e-8 is above the 1e-10 floor");
    }
}

#[test]
fn forced_front_parity_perturb_floor_hit() {
    // |δ_i| = (1+i)·1e-8 < 5.5e-8 for i = 0..=4: five pivots are floored.
    for m in [8usize, 40, 300] {
        let b = check_forced(
            m,
            ZeroPivotAction::PerturbToEps { abs_floor: 5.5e-8 },
            &format!("m={m} perturb 5.5e-8"),
        );
        assert_eq!(b.n_tiny, 5, "m={m}: pivots with |δ| < 5.5e-8 are floored");
    }
}

/// SplitMix64 in (-1, 1).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Mixed front: healthy 1×1 columns, far-apart `[[0,1],[1,0]]` pairs
/// (the partner is fully summed but not adjacent), sparse random weak
/// couplings among the fully-summed columns (so rook, 2×2 partners and
/// threshold bails all fire), and stuck columns coupled mainly to the
/// ghost rows. Interleaved so stuck columns land at many panel offsets.
fn mixed_front(n_fs: usize, n_ghost: usize, seed: u64) -> SymmetricMatrix {
    let n = n_fs + n_ghost;
    let mut rng = Rng(seed);
    let mut a = SymmetricMatrix::zeros(n);
    let half = n_fs / 2;
    for j in 0..n_fs {
        match j % 5 {
            // Healthy 1×1.
            0 => a.set(j, j, 2.0 + rng.next()),
            // Stuck: tiny (or exactly zero) diagonal, strong ghost
            // coupling.
            1 | 3 => {
                let d = if j % 5 == 1 { 1e-9 * rng.next() } else { 0.0 };
                a.set(j, j, d);
                a.set(n_fs + (j % n_ghost), j, 3.0 + rng.next());
            }
            // Zero-diagonal column paired with a far partner.
            _ => {
                let p = (j + half) % n_fs;
                if p != j && a.get(p, j) == 0.0 {
                    a.set(p.max(j), p.min(j), 1.0 + 0.1 * rng.next());
                }
            }
        }
    }
    // Sparse weak random couplings inside the fully-summed block.
    for j in 0..n_fs {
        for i in (j + 1)..n_fs {
            if rng.next() > 0.93 {
                a.set(i, j, 1e-3 * rng.next());
            }
        }
    }
    // Ghost block: diagonal plus light coupling to everything.
    for g in 0..n_ghost {
        let r = n_fs + g;
        a.set(r, r, 50.0 + rng.next());
        for j in 0..n_fs {
            if rng.next() > 0.8 {
                a.set(r, j, a.get(r, j) + 0.5 * rng.next());
            }
        }
    }
    a
}

#[test]
fn mixed_front_parity() {
    let floors = [
        ZeroPivotAction::ForceAccept,
        ZeroPivotAction::PerturbToEps { abs_floor: 1e-10 },
        ZeroPivotAction::PerturbToEps { abs_floor: 1e-6 },
    ];
    for &(n_fs, n_ghost) in &[(10usize, 2usize), (45, 4), (150, 7), (301, 12)] {
        for seed in 1..=4u64 {
            let a = mixed_front(n_fs, n_ghost, seed);
            for on_zero in &floors {
                let p = params(on_zero.clone());
                let tag = format!("n_fs={n_fs} ghosts={n_ghost} seed={seed} {on_zero:?}");
                let s = factor_frontal(&a, n_fs, false, &p).expect("scalar");
                let b = factor_frontal_blocked(&a, n_fs, false, &p).expect("blocked");
                assert_bit_identical(&s, &b, &tag);
                assert_eq!(b.nelim, n_fs, "{tag}: nothing may delay");
            }
        }
    }
}

/// Stuck columns that are weakly coupled to each other: `δ_i` tiny,
/// `a[i+1, i] = ε` and a few random `ε` couplings, plus strong ghost
/// coupling. Every `{i, i+1}` 2×2 fails the growth bound against the
/// ghost rows, so every column is stuck after phase 1 and, in the forced
/// pass, its neighbour is a fully-summed 2×2 partner that the kernels
/// must test and reject identically.
fn coupled_stuck_front(m: usize, n_ghost: usize, eps: f64, seed: u64) -> SymmetricMatrix {
    let n = m + n_ghost;
    let mut rng = Rng(seed);
    let mut a = SymmetricMatrix::zeros(n);
    for j in 0..m {
        a.set(
            j,
            j,
            1e-8 * (1.0 + rng.next().abs()) * if j % 3 == 0 { -1.0 } else { 1.0 },
        );
        if j + 1 < m {
            a.set(j + 1, j, eps * (1.0 + 0.5 * rng.next()));
        }
        for i in (j + 2)..m {
            if rng.next() > 0.97 {
                a.set(i, j, eps * rng.next());
            }
        }
        for g in 0..n_ghost {
            if g == j % n_ghost || rng.next() > 0.7 {
                a.set(m + g, j, 1.0 + rng.next());
            }
        }
    }
    for g in 0..n_ghost {
        a.set(m + g, m + g, 100.0 + rng.next());
    }
    a
}

#[test]
fn coupled_stuck_front_parity() {
    let floors = [
        ZeroPivotAction::ForceAccept,
        ZeroPivotAction::PerturbToEps { abs_floor: 1e-10 },
        ZeroPivotAction::PerturbToEps { abs_floor: 1e-7 },
    ];
    for &m in &[9usize, 40, 130, 300] {
        for &eps in &[1e-6, 1e-4] {
            for seed in 1..=3u64 {
                let a = coupled_stuck_front(m, 3, eps, seed);
                for on_zero in &floors {
                    let p = params(on_zero.clone());
                    let tag = format!("m={m} eps={eps:e} seed={seed} {on_zero:?}");
                    let s = factor_frontal(&a, m, false, &p).expect("scalar");
                    let b = factor_frontal_blocked(&a, m, false, &p).expect("blocked");
                    assert_bit_identical(&s, &b, &tag);
                    assert_eq!(b.nelim, m, "{tag}: nothing may delay");
                }
            }
        }
    }
}
