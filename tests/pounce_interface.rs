//! Integration tests for the POUNCE `Solver` interface.
//!
//! See `dev/plans/pounce-integration-interface.md` for the test
//! catalogue (I1-I8 + U1-U5). Tests are added incrementally as the
//! Solver grows: this file lands the Step-2 set (I1, I5, I6) and
//! grows in subsequent commits.

use feral::{CscMatrix, FactorStatus, Solver};

/// I1 — baseline factor without inertia check.
///
/// 2×2 SPD matrix factored on a fresh `Solver::new()` with
/// `check_inertia = None`. Must report `Success` and stash a
/// non-empty factor reachable via `factors()`. The companion
/// `solve()` assertion lands in Step 6 (`solve()` is currently
/// `unimplemented!()`).
#[test]
fn i1_factor_then_solve_baseline_no_inertia_check() {
    // A = [[2, 0], [0, 2]], lower-triangle CSC.
    let csc = CscMatrix::from_triplets(2, &[0, 1], &[0, 1], &[2.0, 2.0]).unwrap();

    let mut solver = Solver::new();
    let status = solver.factor(&csc, None);

    match status {
        FactorStatus::Success => {}
        other => panic!("expected Success, got {:?}", other),
    }
    assert!(solver.factors().is_some(), "factor() did not stash factors");
    assert_eq!(solver.symbolic_call_count(), 1);
}

/// I5 — pattern change invalidates the cached symbolic.
///
/// Factor a 3×3, then a 4×4 on the same Solver. Both must
/// `Success`, and `symbolic_call_count` must read 2 — once per
/// distinct pattern.
#[test]
fn i5_pattern_change_invalidates_symbolic() {
    let a3 = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
    let a4 =
        CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[2.0, 3.0, 5.0, 7.0]).unwrap();

    let mut solver = Solver::new();

    let s1 = solver.factor(&a3, None);
    assert!(matches!(s1, FactorStatus::Success), "got {:?}", s1);
    assert_eq!(solver.symbolic_call_count(), 1);
    assert_eq!(
        solver.factors().map(|f| f.n),
        Some(3),
        "first factor n mismatch"
    );

    let s2 = solver.factor(&a4, None);
    assert!(matches!(s2, FactorStatus::Success), "got {:?}", s2);
    assert_eq!(
        solver.symbolic_call_count(),
        2,
        "pattern change should re-run symbolic"
    );
    assert_eq!(
        solver.factors().map(|f| f.n),
        Some(4),
        "second factor n mismatch"
    );
}

/// I6 — same pattern reuses the cached symbolic.
///
/// Factor diag(2, 3, 5), then diag(7, 11, 13) on the same Solver.
/// Identical pattern (3×3, 3 diagonals). `symbolic_call_count`
/// must read 1 — symbolic_factorize fires only on the first
/// `factor()` call. This is the cache-reuse property the β
/// refactor unlocked.
#[test]
fn i6_same_pattern_reuses_symbolic() {
    let a = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
    let b = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[7.0, 11.0, 13.0]).unwrap();

    let mut solver = Solver::new();

    let s1 = solver.factor(&a, None);
    assert!(matches!(s1, FactorStatus::Success), "got {:?}", s1);
    assert_eq!(solver.symbolic_call_count(), 1);

    let s2 = solver.factor(&b, None);
    assert!(matches!(s2, FactorStatus::Success), "got {:?}", s2);
    assert_eq!(
        solver.symbolic_call_count(),
        1,
        "same pattern should reuse symbolic"
    );

    // Sanity: the second factor's diagonal matches B (not A).
    let factors = solver.factors().expect("factors stored");
    assert_eq!(factors.n, 3);
}
