//! Issue #205: report which ordering the solver actually chose.
//!
//! `Auto` routes adaptively and used to report nothing back, so a
//! caller knew the fill it got but not which ordering produced it. The
//! load-bearing part is the `requested` / `used` pair: it distinguishes
//! "I asked for AMD and got AMD" from "I asked for `Auto` and got
//! something that may change between releases".
//!
//! The oracles here are the identity between what a caller asked for and
//! what the analysis reports, checked two ways: asking for a concrete
//! method must echo that method back, and asking for `Auto` must report
//! a *concrete* method that reproduces the same factor when requested
//! directly. The second is the one that matters — it makes
//! `ordering_used` a claim that can be falsified rather than a label.

use feral::symbolic::{OrderingMethod, OrderingPreprocess};
use feral::{CscMatrix, FactorStatus, Solver};

/// Tridiagonal SPD, lower triangle.
fn tridiag(n: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        rows.push(j);
        cols.push(j);
        vals.push(4.0);
        if j + 1 < n {
            rows.push(j + 1);
            cols.push(j);
            vals.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("tridiag")
}

/// A 2-D grid Laplacian — big enough that `Auto` has a real choice and
/// that the orderings differ.
fn grid(k: usize) -> CscMatrix {
    let idx = |r: usize, c: usize| r * k + c;
    let n = k * k;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for r in 0..k {
        for c in 0..k {
            let v = idx(r, c);
            rows.push(v);
            cols.push(v);
            vals.push(8.0);
            if r + 1 < k {
                rows.push(idx(r + 1, c));
                cols.push(v);
                vals.push(-1.0);
            }
            if c + 1 < k {
                rows.push(idx(r, c + 1));
                cols.push(v);
                vals.push(-1.0);
            }
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("grid")
}

#[test]
fn a_concrete_request_is_echoed_back_unchanged() {
    let m = tridiag(500);
    for want in [
        OrderingMethod::Amd,
        OrderingMethod::Amf,
        OrderingMethod::MetisND,
    ] {
        let mut s = Solver::new().with_ordering(want.clone());
        assert!(matches!(s.factor(&m, None), FactorStatus::Success));
        let info = s.last_factor_stats().expect("stats").ordering_info;
        assert_eq!(info.requested, want, "requested must echo the caller");
        assert_eq!(info.used, want, "a concrete request must not be rerouted");
    }
}

#[test]
fn auto_reports_a_concrete_method_that_reproduces_the_same_factor() {
    // The falsifiable form of the claim: whatever `Auto` says it used,
    // requesting that method directly must give the same factor. If
    // `ordering_used` were mislabelled this fails.
    let m = grid(40);
    let mut auto = Solver::new();
    assert!(matches!(auto.factor(&m, None), FactorStatus::Success));
    let stats = auto.last_factor_stats().expect("stats");
    let info = stats.ordering_info.clone();

    assert_eq!(info.requested, OrderingMethod::Auto);
    assert_ne!(
        info.used,
        OrderingMethod::Auto,
        "`used` must be a concrete method, never the sentinel"
    );

    let mut direct = Solver::new().with_ordering(info.used.clone());
    assert!(matches!(direct.factor(&m, None), FactorStatus::Success));
    let direct_stats = direct.last_factor_stats().expect("stats");
    assert_eq!(
        direct_stats.nnz_l, stats.nnz_l,
        "requesting the reported method must reproduce the factor"
    );
    assert_eq!(direct_stats.inertia, stats.inertia);
}

#[test]
fn structural_numbers_match_the_symbolic_analysis() {
    let m = grid(30);
    let mut s = Solver::new();
    assert!(matches!(s.factor(&m, None), FactorStatus::Success));
    let info = s.last_factor_stats().expect("stats").ordering_info;
    let est = s.work_estimate().expect("estimate");

    assert_eq!(info.n_supernodes, est.n_supernodes);
    assert_eq!(info.max_front_rows, est.max_front_rows);
    assert!(info.n_supernodes > 0);
    assert!(info.max_front_rows > 0);
}

#[test]
fn escalation_is_reported_and_is_false_by_default() {
    let m = tridiag(400);
    let mut s = Solver::new();
    assert!(matches!(s.factor(&m, None), FactorStatus::Success));
    let info = s.last_factor_stats().expect("stats").ordering_info;
    assert!(
        !info.escalated,
        "a clean factorization must not report ordering escalation"
    );
}

#[test]
fn preprocess_is_reported_as_a_concrete_choice() {
    let m = grid(35);
    let mut s = Solver::new();
    assert!(matches!(s.factor(&m, None), FactorStatus::Success));
    let info = s.last_factor_stats().expect("stats").ordering_info;
    assert_ne!(
        info.preprocess,
        OrderingPreprocess::Auto,
        "`preprocess` must be resolved, never the sentinel"
    );
}

#[test]
fn the_report_survives_a_symbolic_cache_hit() {
    // Refactoring the same pattern with new values reuses the symbolic
    // analysis. The routing report must still be present and unchanged
    // — a caller logging it per factorization must not see it vanish.
    let m = tridiag(300);
    let mut s = Solver::new();
    assert!(matches!(s.factor(&m, None), FactorStatus::Success));
    let first = s.last_factor_stats().expect("stats").ordering_info;

    let mut m2 = m.clone();
    for v in m2.values.iter_mut() {
        *v *= 1.000_001;
    }
    assert!(matches!(s.factor(&m2, None), FactorStatus::Success));
    let second = s.last_factor_stats().expect("stats").ordering_info;

    assert!(second.pattern_reused, "same pattern, so the cache hit");
    assert_eq!(first.used, second.used);
    assert_eq!(first.n_supernodes, second.n_supernodes);
}
