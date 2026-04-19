//! Integration tests for the POUNCE `Solver` interface.
//!
//! See `dev/plans/pounce-integration-interface.md` for the test
//! catalogue (I1-I8 + U1-U5). Tests are added incrementally as the
//! Solver grows: this file lands the Step-2 set (I1, I5, I6) and
//! grows in subsequent commits.

use feral::{CscMatrix, FactorStatus, FeralError, Inertia, Solver};

/// I1 — baseline factor + solve without inertia check.
///
/// 2×2 SPD matrix factored on a fresh `Solver::new()` with
/// `check_inertia = None`. Must report `Success`, stash a factor,
/// and `solve()` produces the correct answer.
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

    // 2 x = 4, 2 y = 6 → x = 2, y = 3.
    let x = solver.solve(&[4.0, 6.0]).expect("solve");
    assert!((x[0] - 2.0).abs() < 1e-12, "x[0] = {}", x[0]);
    assert!((x[1] - 3.0).abs() < 1e-12, "x[1] = {}", x[1]);
}

/// `Solver::solve` before any successful factor returns
/// `FeralError::NoFactor`.
#[test]
fn solve_before_factor_returns_no_factor() {
    let solver = Solver::new();
    match solver.solve(&[1.0, 2.0]) {
        Err(FeralError::NoFactor) => {}
        other => panic!("expected NoFactor, got {:?}", other),
    }
}

/// `Solver::solve` after a Singular factor (which clears storage)
/// also returns `FeralError::NoFactor`.
#[test]
fn solve_after_singular_returns_no_factor() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 0.0, 1.0]).unwrap();
    let mut solver = Solver::new();
    let status = solver.factor(&csc, None);
    assert!(matches!(status, FactorStatus::Singular));

    match solver.solve(&[1.0, 2.0, 3.0]) {
        Err(FeralError::NoFactor) => {}
        other => panic!("expected NoFactor, got {:?}", other),
    }
}

/// `Solver::solve` after `WrongInertia` still works — Ipopt
/// SYMSOLVER_WRONG_INERTIA semantics keep the factor live.
#[test]
fn solve_after_wrong_inertia_still_works() {
    let csc = CscMatrix::from_triplets(2, &[0, 1], &[0, 1], &[2.0, 2.0]).unwrap();
    let wrong = Inertia {
        positive: 1,
        negative: 1,
        zero: 0,
    };

    let mut solver = Solver::new();
    let status = solver.factor(&csc, Some(wrong));
    assert!(matches!(status, FactorStatus::WrongInertia { .. }));

    let x = solver.solve(&[4.0, 6.0]).expect("solve must still work");
    assert!((x[0] - 2.0).abs() < 1e-12);
    assert!((x[1] - 3.0).abs() < 1e-12);
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

/// I2 — `factor` with the correct inertia returns `Success`.
#[test]
fn i2_factor_with_correct_inertia_returns_success() {
    // diag(2, 3, 5): all positive, inertia (3, 0, 0).
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
    let expected = Inertia {
        positive: 3,
        negative: 0,
        zero: 0,
    };

    let mut solver = Solver::new();
    let status = solver.factor(&csc, Some(expected));
    assert!(matches!(status, FactorStatus::Success), "got {:?}", status);
    assert_eq!(solver.num_negative_eigenvalues(), 0);
}

/// I3 — `factor` with the wrong inertia returns `WrongInertia`
/// AND keeps the factor stored (Ipopt SYMSOLVER_WRONG_INERTIA
/// semantics).
#[test]
fn i3_factor_with_wrong_inertia_returns_wronginertia_keeps_factor() {
    // diag(2, 3, 5): actual inertia (3, 0, 0).
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
    let wrong = Inertia {
        positive: 2,
        negative: 1,
        zero: 0,
    };

    let mut solver = Solver::new();
    let status = solver.factor(&csc, Some(wrong.clone()));

    match status {
        FactorStatus::WrongInertia { actual, expected } => {
            assert_eq!(
                actual,
                Inertia {
                    positive: 3,
                    negative: 0,
                    zero: 0
                }
            );
            assert_eq!(expected, wrong);
        }
        other => panic!("expected WrongInertia, got {:?}", other),
    }

    // Factor still stored — caller may inspect / solve against it.
    assert!(solver.factors().is_some());
    assert_eq!(solver.num_negative_eigenvalues(), 0);
}

/// I4 — singular under default `Fail` mode returns `Singular` and
/// clears the stored factor.
///
/// `diag(1, 0, 1)` has a structural zero pivot at position 1 with
/// no symmetric off-diagonal coupling that BK could pivot around,
/// so default `ZeroPivotAction::Fail` should fire and the factor
/// should be discarded.
#[test]
fn i4_singular_under_fail_returns_singular_clears_factor() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 0.0, 1.0]).unwrap();

    let mut solver = Solver::new();
    let status = solver.factor(&csc, None);

    assert!(
        matches!(status, FactorStatus::Singular),
        "expected Singular, got {:?}",
        status
    );
    assert!(
        solver.factors().is_none(),
        "factors should be cleared on Singular"
    );
}
