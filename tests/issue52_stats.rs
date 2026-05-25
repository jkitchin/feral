//! Phase A tests for issue #52: opt-in instrumentation accessors.
//!
//! Targets `Solver::last_factor_stats()` and the `FactorStats`
//! snapshot type. These tests fail to compile against the current
//! Solver surface and pin the API shape spelled out in
//! `dev/plans/issue-52-opt-in-stats.md`.
//!
//! Phase B (`with_profiling`, `profile_report()`) is in a separate
//! test file added in a later commit.
//!
//! Test catalogue:
//! - A1: `last_factor_stats_returns_none_before_factor`
//! - A2: `last_factor_stats_after_success_populates_all_fields`
//! - A3: `pattern_reused_false_first_factor_true_second`
//! - A4: `pattern_reused_false_after_pattern_change`

use feral::scaling::ScalingStrategy;
use feral::{CscMatrix, FactorStats, FactorStatus, Solver};

/// Build a 2×2 SPD diagonal matrix `diag(2, 2)` (lower-triangle CSC).
fn diag2_spd() -> CscMatrix {
    CscMatrix::from_triplets(2, &[0, 1], &[0, 1], &[2.0, 2.0]).expect("from_triplets")
}

/// Build a different-pattern 3×3 SPD matrix to force a fingerprint
/// mismatch in A4. Lower triangle: diag(3, 3, 3) plus a (2,1) entry
/// so `nnz != n` and the pattern is unmistakably distinct from A2's
/// 2×2 pure-diagonal pattern.
fn tri3_spd_with_off_diag() -> CscMatrix {
    // a_00 = 3, a_11 = 3, a_22 = 3, a_21 = 1.
    CscMatrix::from_triplets(3, &[0, 1, 2, 2], &[0, 1, 2, 1], &[3.0, 3.0, 3.0, 1.0])
        .expect("from_triplets")
}

/// A1 — no factor has run, so `last_factor_stats()` must report
/// `None`. Guards the `Option` contract before any state exists.
#[test]
fn a1_last_factor_stats_returns_none_before_factor() {
    let solver = Solver::new();
    let got: Option<FactorStats> = solver.last_factor_stats();
    assert!(
        got.is_none(),
        "expected None before first factor, got {:?}",
        got
    );
}

/// A2 — every `FactorStats` field is populated after one successful
/// `factor()` call. Field values are cross-checked against the
/// already-public per-field accessors so we are not re-asserting
/// numeric ground truth (that lives in dense_ldlt.rs etc.) — only
/// that the snapshot faithfully mirrors the per-field surface.
#[test]
fn a2_last_factor_stats_after_success_populates_all_fields() {
    let csc = diag2_spd();
    let mut solver = Solver::new().with_scaling(ScalingStrategy::Identity);

    let status = solver.factor(&csc, None);
    assert!(
        matches!(status, FactorStatus::Success),
        "factor failed: {:?}",
        status
    );

    let stats = solver
        .last_factor_stats()
        .expect("last_factor_stats() should be Some after success");

    // nnz_a is exactly the CscMatrix nnz (lower triangle stored).
    assert_eq!(stats.nnz_a, csc.nnz(), "nnz_a mirrors CscMatrix::nnz()");

    // nnz_l mirrors SparseFactors::factor_nnz().
    let factors = solver.factors().expect("factors stashed");
    assert_eq!(
        stats.nnz_l,
        factors.factor_nnz(),
        "nnz_l mirrors SparseFactors::factor_nnz()"
    );

    // fill_ratio is definitionally nnz_l / nnz_a.
    let expected_fill = stats.nnz_l as f64 / stats.nnz_a as f64;
    assert!(
        (stats.fill_ratio - expected_fill).abs() < 1e-15,
        "fill_ratio = {} expected {}",
        stats.fill_ratio,
        expected_fill
    );

    // Inertia mirrors the existing accessor. SPD ⇒ (2, 0, 0).
    let inertia = solver.inertia().expect("inertia stashed").clone();
    assert_eq!(stats.inertia, inertia, "inertia mirrors Solver::inertia()");

    // min/max_abs_pivot mirror the existing accessors.
    let min_pivot = solver
        .min_pivot_magnitude()
        .expect("min_pivot_magnitude stashed");
    let max_pivot = solver
        .max_pivot_magnitude()
        .expect("max_pivot_magnitude stashed");
    assert!(
        (stats.min_abs_pivot - min_pivot).abs() < 1e-15,
        "min_abs_pivot = {} expected {}",
        stats.min_abs_pivot,
        min_pivot
    );
    assert!(
        (stats.max_abs_pivot - max_pivot).abs() < 1e-15,
        "max_abs_pivot = {} expected {}",
        stats.max_abs_pivot,
        max_pivot
    );

    // pattern_reused is false on the very first factor of a Solver.
    assert!(
        !stats.pattern_reused,
        "first factor on a fresh Solver is never a cache hit"
    );

    // scaling_info mirrors Solver::scaling_info().
    let scaling = solver.scaling_info().expect("scaling_info stashed").clone();
    assert_eq!(
        stats.scaling_info, scaling,
        "scaling_info mirrors Solver::scaling_info()"
    );
}

/// A3 — bit-identical pattern replay must flip `pattern_reused` to
/// `true` on the second `factor()`. This is the symbolic-cache hit
/// signal pounce will key off when deciding whether the warm path
/// fired as expected.
#[test]
fn a3_pattern_reused_false_first_factor_true_second() {
    let csc = diag2_spd();
    let mut solver = Solver::new().with_scaling(ScalingStrategy::Identity);

    assert!(matches!(solver.factor(&csc, None), FactorStatus::Success));
    let s1 = solver.last_factor_stats().expect("stats after factor 1");
    assert!(!s1.pattern_reused, "factor 1 cannot be a cache hit");

    assert!(matches!(solver.factor(&csc, None), FactorStatus::Success));
    let s2 = solver.last_factor_stats().expect("stats after factor 2");
    assert!(
        s2.pattern_reused,
        "factor 2 on identical pattern must report cache hit"
    );

    // Sanity: symbolic_call_count agrees with pattern_reused.
    assert_eq!(
        solver.symbolic_call_count(),
        1,
        "symbolic should have run exactly once across two same-pattern factors"
    );
}

/// A4 — a structurally distinct matrix between two `factor()` calls
/// must report `pattern_reused = false` on the second call.
/// Complements A3 by exercising the cache-miss branch.
#[test]
fn a4_pattern_reused_false_after_pattern_change() {
    let small = diag2_spd();
    let bigger = tri3_spd_with_off_diag();

    let mut solver = Solver::new().with_scaling(ScalingStrategy::Identity);

    assert!(matches!(solver.factor(&small, None), FactorStatus::Success));
    let s1 = solver.last_factor_stats().expect("stats after factor 1");
    assert!(!s1.pattern_reused, "factor 1 is never a cache hit");

    assert!(matches!(
        solver.factor(&bigger, None),
        FactorStatus::Success
    ));
    let s2 = solver.last_factor_stats().expect("stats after factor 2");
    assert!(
        !s2.pattern_reused,
        "pattern change must invalidate the cache"
    );

    // The fingerprint mismatch should have re-run symbolic.
    assert_eq!(
        solver.symbolic_call_count(),
        2,
        "symbolic must rerun on pattern change"
    );
}
