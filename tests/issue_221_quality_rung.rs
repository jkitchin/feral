//! Issue #221 (PR #222, pounce side): `Solver::increase_quality` declines a
//! stage-2 rung that cannot help.
//!
//! A rung raises the pivot threshold `u` to `u'`, which asks for a factor
//! whose multipliers satisfy `|L| <= 1/u'`. Two cases are declined
//! (`dev/research/issue-221-quality-rung-admissibility.md`):
//!
//! - the stored factor already meets `u'`: nothing floored or
//!   force-accepted, and realized `max |L| <= 1/u'`;
//! - the previous rung made the factor worse (more pivots escaped the
//!   test). The threshold returns to where the ladder started and the
//!   ladder is exhausted.
//!
//! Oracles are hand calculations:
//! - a diagonal matrix has no multipliers, so `max |L| = 0`;
//! - under `PerturbToEps`, each exactly zero diagonal of a diagonal matrix
//!   is one floored pivot (`n_tiny`), as `tests/issue_55_n_tiny_counter.rs`
//!   pins;
//! - for the tridiagonal `T = tridiag(b, 1, b)` eliminated in natural
//!   order, `d_1 = 1`, `l_k = b / d_k`, `d_{k+1} = 1 - b·l_k`.
//!
//! The scaling is `External(1, ..., 1)`, which leaves the matrix unchanged
//! and is not `Identity`, so `increase_quality` goes straight to stage 2,
//! as it does under pounce's `Auto` scaling.

use feral::dense::factor::ZeroPivotAction;
use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus, QualityLevel, Solver};
use feral::scaling::ScalingStrategy;
use feral::sparse::csc::CscMatrix;
use feral::symbolic::{OrderingMethod, SupernodeParams};
use feral::BunchKaufmanParams;

fn diag_csc(values: &[f64]) -> CscMatrix {
    let n = values.len();
    let idx: Vec<usize> = (0..n).collect();
    CscMatrix::from_triplets(n, &idx, &idx, values).expect("diag csc")
}

fn tridiag_csc(n: usize, b: f64) -> CscMatrix {
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        r.push(i);
        c.push(i);
        v.push(1.0);
        if i + 1 < n {
            r.push(i + 1);
            c.push(i);
            v.push(b);
        }
    }
    CscMatrix::from_triplets(n, &r, &c, &v).expect("tridiag csc")
}

/// Natural order, no amalgamation, unit scaling: the factor is exactly
/// the hand elimination.
fn solver(n: usize, on_zero: ZeroPivotAction) -> Solver {
    let np = NumericParams {
        scaling: ScalingStrategy::External(vec![1.0; n]),
        bk: BunchKaufmanParams {
            on_zero_pivot: on_zero,
            ..NumericParams::default().bk
        },
        ..NumericParams::default()
    };
    let sn = SupernodeParams {
        nemin: 1,
        ..SupernodeParams::default()
    };
    Solver::with_params(np, sn).with_ordering(OrderingMethod::External((0..n).collect()))
}

fn perturb() -> ZeroPivotAction {
    ZeroPivotAction::PerturbToEps { abs_floor: 1e-10 }
}

fn factor_ok(s: &mut Solver, a: &CscMatrix) {
    let st = s.factor(a, None);
    assert!(matches!(st, FactorStatus::Success), "factor: {st:?}");
}

#[test]
fn max_abs_l_is_the_realized_multiplier() {
    let (n, b) = (6, 0.25);
    let mut s = solver(n, NumericParams::default().bk.on_zero_pivot);
    factor_ok(&mut s, &tridiag_csc(n, b));
    // Hand recurrence. T is diagonally dominant, so every l_k < 1 and the
    // natural order is what BK takes (|d_k| >= alpha·b).
    let mut d = 1.0f64;
    let mut want = 0.0f64;
    for _ in 0..n - 1 {
        let l = b / d;
        want = want.max(l);
        d = 1.0 - b * l;
    }
    let got = s.last_factor_stats().expect("stats").max_abs_l;
    assert!(
        (got - want).abs() <= 4.0 * f64::EPSILON * want,
        "max |L| {got:e}, hand {want:e}"
    );

    let mut s = solver(3, NumericParams::default().bk.on_zero_pivot);
    factor_ok(&mut s, &diag_csc(&[2.0, 3.0, 5.0]));
    assert_eq!(s.last_factor_stats().expect("stats").max_abs_l, 0.0);
}

#[test]
fn rung_is_declined_when_the_factor_already_meets_it() {
    let n = 6;
    let mut s = solver(n, NumericParams::default().bk.on_zero_pivot);
    factor_ok(&mut s, &tridiag_csc(n, 0.25));
    let st = s.last_factor_stats().expect("stats");
    assert_eq!((st.n_tiny, st.n_threshold_fail_accepts), (0, 0));

    // max |L| < 0.27, so the factor meets every threshold up to 1/0.27.
    assert!(!s.increase_quality(), "rung 1e-8 -> 1e-6 cannot help");
    assert_eq!(s.pivot_threshold(), 1e-8, "threshold unchanged");
    assert_eq!(s.quality_level(), QualityLevel::Baseline);
}

#[test]
fn rung_proceeds_when_pivots_escaped_the_threshold() {
    let mut s = solver(5, perturb());
    factor_ok(&mut s, &diag_csc(&[1.0, 0.0, 1.0, 0.0, 1.0]));
    assert_eq!(s.last_factor_stats().expect("stats").n_tiny, 2);

    assert!(s.increase_quality());
    assert_eq!(s.pivot_threshold(), 1e-6);
    assert_eq!(s.quality_level(), QualityLevel::PivotRaised);
}

#[test]
fn rung_that_made_the_factor_worse_is_undone() {
    let mut s = solver(5, perturb());
    factor_ok(&mut s, &diag_csc(&[1.0, 0.0, 1.0, 0.0, 1.0]));
    assert!(s.increase_quality());
    assert_eq!(s.pivot_threshold(), 1e-6);

    // At the raised threshold, three pivots escape instead of two.
    factor_ok(&mut s, &diag_csc(&[1.0, 0.0, 0.0, 0.0, 1.0]));
    assert_eq!(s.last_factor_stats().expect("stats").n_tiny, 3);
    assert!(
        !s.increase_quality(),
        "a rung that made it worse is declined"
    );
    assert_eq!(s.pivot_threshold(), 1e-8, "and undone");
    assert_eq!(s.quality_level(), QualityLevel::Exhausted);
    assert!(!s.increase_quality(), "and not retried");
    assert_eq!(s.pivot_threshold(), 1e-8);
}

#[test]
fn undo_returns_to_the_start_of_the_ladder() {
    let mut s = solver(5, perturb());
    factor_ok(&mut s, &diag_csc(&[1.0, 0.0, 1.0, 0.0, 1.0]));
    assert!(s.increase_quality()); // 1e-8 -> 1e-6
    factor_ok(&mut s, &diag_csc(&[1.0, 0.0, 1.0, 0.0, 1.0]));
    assert!(s.increase_quality()); // 1e-6 -> 1e-6^0.75, not worse
    assert!(s.pivot_threshold() > 1e-6);

    factor_ok(&mut s, &diag_csc(&[0.0, 0.0, 1.0, 0.0, 0.0]));
    assert_eq!(s.last_factor_stats().expect("stats").n_tiny, 4);
    assert!(!s.increase_quality());
    assert_eq!(s.pivot_threshold(), 1e-8, "back to where the ladder began");
    assert_eq!(s.quality_level(), QualityLevel::Exhausted);
}

#[test]
fn rung_that_did_not_make_it_worse_is_kept() {
    let mut s = solver(5, perturb());
    factor_ok(&mut s, &diag_csc(&[1.0, 0.0, 1.0, 0.0, 1.0]));
    assert!(s.increase_quality());

    // Same two floored pivots at 1e-6: not worse, so the ladder climbs.
    factor_ok(&mut s, &diag_csc(&[1.0, 0.0, 1.0, 0.0, 1.0]));
    assert!(s.increase_quality());
    let want = 1e-6_f64.powf(0.75);
    assert!((s.pivot_threshold() - want).abs() <= 1e-15 * want);
    assert_eq!(s.quality_level(), QualityLevel::PivotRaised);

    // A clean factor at the raised threshold: kept, and the next rung is
    // declined because the factor already meets it.
    factor_ok(&mut s, &diag_csc(&[1.0, 2.0, 1.0, 2.0, 1.0]));
    assert!(!s.increase_quality());
    assert!((s.pivot_threshold() - want).abs() <= 1e-15 * want);
    assert_eq!(s.quality_level(), QualityLevel::PivotRaised);
}

/// A rung is judged once, by the first call after it, against the factor
/// that prompted it. A later iterate is not compared with that stale
/// factor (pooling_rt2stp: a clean factor declined the next rung, and a
/// later iterate with two floored pivots then read as "worse" than the
/// factor three calls back, which undid the whole ladder).
#[test]
fn a_rung_is_judged_only_by_the_next_call() {
    let mut s = solver(5, perturb());
    factor_ok(&mut s, &diag_csc(&[1.0, 0.0, 1.0, 0.0, 1.0]));
    assert!(s.increase_quality()); // 1e-8 -> 1e-6

    factor_ok(&mut s, &diag_csc(&[1.0, 2.0, 1.0, 2.0, 1.0]));
    assert!(!s.increase_quality(), "clean: next rung already met");
    assert_eq!(s.pivot_threshold(), 1e-6);

    factor_ok(&mut s, &diag_csc(&[1.0, 0.0, 0.0, 0.0, 1.0]));
    assert!(s.increase_quality(), "floored pivots: the ladder climbs");
    let want = 1e-6_f64.powf(0.75);
    assert!((s.pivot_threshold() - want).abs() <= 1e-15 * want);
    assert_eq!(s.quality_level(), QualityLevel::PivotRaised);
}

#[test]
fn without_a_factor_the_ladder_is_unchanged() {
    let mut s = solver(3, NumericParams::default().bk.on_zero_pivot);
    assert!(s.increase_quality());
    assert_eq!(s.pivot_threshold(), 1e-6);
}
