//! Issue #203: the measured ordering race.
//!
//! `Solver::with_ordering_race` runs every listed ordering once on the
//! first `factor()` for a pattern and adopts the fastest. The property
//! that matters is that it changes *speed only*: whichever arm wins, the
//! inertia and the solution must be what that ordering would have
//! produced on its own.
//!
//! **Oracles are external to the solver.** The test matrix is the
//! augmented system
//!
//! ```text
//!   K = [ H    J^T ]
//!       [ J   -dI  ]
//! ```
//!
//! with `H` strictly diagonally dominant with a positive diagonal (hence
//! SPD by Gershgorin) and `d > 0`. The Schur complement `-dI - J H^-1 J^T`
//! is then negative definite *whatever* `J` is, so by Haynsworth inertia
//! additivity `K` has inertia `(n_vars, m, 0)` with no rank assumption on
//! `J`. That is a hand oracle, independent of anything feral computes.
//! The right-hand side is `b = K x_true` for a chosen `x_true`, so the
//! solution is known by construction too.

use feral::symbolic::OrderingMethod;
use feral::{CscMatrix, FactorStatus, Solver};

/// Chain-structured saddle-point KKT: `t` blocks of `nx` "state"
/// variables, each block coupled to the next, with one constraint row per
/// coupling. Big enough that the orderings genuinely differ.
fn chain_kkt(t: usize, nx: usize) -> (CscMatrix, usize, usize) {
    let n_vars = t * nx;
    let m = (t - 1) * nx;
    let n = n_vars + m;
    // (row, col) in the lower triangle, row >= col.
    let mut ent: Vec<(usize, usize, f64)> = Vec::new();
    let mut push = |r: usize, c: usize, v: f64| {
        if r >= c {
            ent.push((r, c, v));
        } else {
            ent.push((c, r, v));
        }
    };
    // H: dense within each block.
    for b in 0..t {
        for i in 0..nx {
            for j in 0..=i {
                if i != j {
                    push(b * nx + i, b * nx + j, 0.25);
                }
            }
        }
    }
    // J: row (b, i) couples block b and block b+1.
    for b in 0..(t - 1) {
        for i in 0..nx {
            let row = n_vars + b * nx + i;
            for j in 0..nx {
                push(row, b * nx + j, 0.5);
                push(row, (b + 1) * nx + j, -0.5);
            }
        }
    }
    // Diagonal: strictly dominant positive on H, -d on the dual block.
    let mut absrow = vec![0.0f64; n];
    for &(r, c, v) in &ent {
        absrow[r] += v.abs();
        if r != c {
            absrow[c] += v.abs();
        }
    }
    for (k, &row) in absrow.iter().enumerate() {
        let d = if k < n_vars { row + 1.0 } else { -1e-2 };
        ent.push((k, k, d));
    }
    let rows: Vec<usize> = ent.iter().map(|e| e.0).collect();
    let cols: Vec<usize> = ent.iter().map(|e| e.1).collect();
    let vals: Vec<f64> = ent.iter().map(|e| e.2).collect();
    let m_csc = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("from_triplets");
    (m_csc, n_vars, m)
}

/// `y = K x` for the lower-triangle CSC storage.
fn matvec(k: &CscMatrix, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; k.n];
    for j in 0..k.n {
        for p in k.col_ptr[j]..k.col_ptr[j + 1] {
            let i = k.row_idx[p];
            let a = k.values[p];
            y[i] += a * x[j];
            if i != j {
                y[j] += a * x[i];
            }
        }
    }
    y
}

fn arms() -> Vec<OrderingMethod> {
    vec![OrderingMethod::Amd, OrderingMethod::Amf]
}

#[test]
fn race_preserves_the_analytic_inertia_and_the_solution() {
    let (k, n_vars, m) = chain_kkt(40, 6);
    let x_true: Vec<f64> = (0..k.n).map(|i| 1.0 + i as f64 / k.n as f64).collect();
    let b = matvec(&k, &x_true);

    let mut raced = Solver::new().with_ordering_race(arms());
    assert!(matches!(raced.factor(&k, None), FactorStatus::Success));
    let inertia = raced.inertia().expect("inertia").clone();
    assert_eq!(
        (inertia.positive, inertia.negative, inertia.zero),
        (n_vars, m, 0),
        "race must reproduce the analytic saddle-point inertia"
    );
    let x = raced.solve(&b).expect("solve");
    let err: f64 = x
        .iter()
        .zip(&x_true)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-8, "race solution is wrong: max err {err:.3e}");

    // Each arm alone must agree — the race changes speed, not answers.
    for arm in arms() {
        let mut solo = Solver::new().with_ordering(arm.clone());
        assert!(matches!(solo.factor(&k, None), FactorStatus::Success));
        assert_eq!(
            solo.inertia().expect("inertia"),
            &inertia,
            "arm {arm:?} disagrees with the raced inertia"
        );
    }
}

#[test]
fn race_runs_once_per_pattern_and_again_on_a_new_one() {
    let (k, n_vars, m) = chain_kkt(30, 5);
    let mut s = Solver::new().with_ordering_race(arms());

    assert!(matches!(s.factor(&k, None), FactorStatus::Success));
    let first = s.last_race().expect("a race ran").clone();
    assert_eq!(first.arms.len(), 2);
    assert!(first.winner.is_some(), "one arm should have won");
    // The race adopts the winning probe's analysis rather than
    // recomputing it, so the solver's own symbolic counter does not
    // move. That is the saving the race is built around — the analysis
    // is the expensive half — so pin it rather than assume it.
    let after_first = s.symbolic_call_count();

    // Same pattern, different values: no second race, no new symbolic.
    let mut k2 = k.clone();
    for v in k2.values.iter_mut() {
        *v *= 1.000_001;
    }
    assert!(matches!(s.factor(&k2, None), FactorStatus::Success));
    assert_eq!(
        s.symbolic_call_count(),
        after_first,
        "a same-pattern re-factor must not re-race or re-analyse"
    );
    let again = s.last_race().expect("the first race is still recorded");
    assert_eq!(
        again.arms[0].inertia, first.arms[0].inertia,
        "a same-pattern re-factor must not have re-raced"
    );

    // A different pattern re-races. The observable is the recorded
    // inertia: the new matrix has different dimensions, so a stale
    // `RaceResult` would still carry the old one.
    let (k3, n_vars3, m3) = chain_kkt(31, 5);
    assert_ne!((n_vars, m), (n_vars3, m3));
    assert!(matches!(s.factor(&k3, None), FactorStatus::Success));
    let third = s.last_race().expect("a race ran for the new pattern");
    let winner = third.winner.expect("one arm should have won");
    let inr = third.arms[winner]
        .inertia
        .as_ref()
        .expect("the winning arm reported an inertia");
    assert_eq!(
        (inr.positive, inr.negative, inr.zero),
        (n_vars3, m3, 0),
        "the race must have re-run against the new pattern"
    );
}

#[test]
fn fewer_than_two_arms_is_a_no_op() {
    let (k, n_vars, m) = chain_kkt(20, 4);
    for a in [vec![], vec![OrderingMethod::Amd]] {
        let mut s = Solver::new().with_ordering_race(a);
        assert!(matches!(s.factor(&k, None), FactorStatus::Success));
        assert!(s.last_race().is_none(), "no race should have run");
        let inr = s.inertia().expect("inertia");
        assert_eq!((inr.positive, inr.negative, inr.zero), (n_vars, m, 0));
    }
}

#[test]
fn probes_inherit_configuration() {
    // A probe that ignored `with_parallel(false)` would time an arm
    // under a configuration the real solver never uses. The observable
    // here is only that the raced result is still correct under a
    // non-default configuration.
    let (k, n_vars, m) = chain_kkt(25, 5);
    let mut s = Solver::new()
        .with_parallel(false)
        .with_ordering_race(arms());
    assert!(matches!(s.factor(&k, None), FactorStatus::Success));
    let inr = s.inertia().expect("inertia");
    assert_eq!((inr.positive, inr.negative, inr.zero), (n_vars, m, 0));
    let r = s.last_race().expect("a race ran");
    assert!(r.arms.iter().all(|a| a.factor_us.is_some()));
}

#[test]
fn race_result_reports_every_arm_with_one_winner() {
    let (k, _, _) = chain_kkt(30, 5);
    let mut s = Solver::new().with_ordering_race(arms());
    assert!(matches!(s.factor(&k, None), FactorStatus::Success));
    let r = s.last_race().expect("a race ran");
    assert_eq!(r.arms.len(), 2);
    assert!(!r.inertia_disagreement);
    assert_eq!(r.arms.iter().filter(|a| a.winner).count(), 1);
    // Surviving arms must agree on inertia, or the race would have
    // declined; assert that directly.
    let inertias: Vec<_> = r.arms.iter().filter_map(|a| a.inertia.clone()).collect();
    assert!(inertias.windows(2).all(|w| w[0] == w[1]));
}
