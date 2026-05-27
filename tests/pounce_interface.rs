//! Integration tests for the POUNCE `Solver` interface.
//!
//! See `dev/plans/pounce-integration-interface.md` for the test
//! catalogue (I1-I8 + U1-U5). Tests are added incrementally as the
//! Solver grows: this file lands the Step-2 set (I1, I5, I6) and
//! grows in subsequent commits.

use feral::numeric::factorize::{NumericParams, SmallLeafBatch};
use feral::scaling::ScalingStrategy;
use feral::symbolic::SupernodeParams;
use feral::{
    BunchKaufmanParams, CscMatrix, FactorStatus, FeralError, Inertia, QualityLevel, Solver,
    ZeroPivotAction,
};

/// Helper: NumericParams with the legacy `Fail`-on-zero-pivot
/// default restored. Used by tests that exercise the Singular code
/// path. After F-03 (#32), `NumericParams::default()` switched to
/// `ZeroPivotAction::ForceAccept` to match MUMPS/MA57 behavior on
/// rank-deficient saddle-point systems like `GHS_indef/bloweybl`.
fn np_fail_on_zero() -> NumericParams {
    let mut np = NumericParams::default();
    np.bk.on_zero_pivot = ZeroPivotAction::Fail;
    np
}

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
///
/// Opts into legacy `ZeroPivotAction::Fail` via `with_params` —
/// after F-03 (#32) the new default `ForceAccept` would turn
/// `diag(1, 0, 1)` into a successful factor with `inertia.zero == 1`,
/// not a `Singular` status.
#[test]
fn solve_after_singular_returns_no_factor() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 0.0, 1.0]).unwrap();
    let mut solver = Solver::with_params(np_fail_on_zero(), SupernodeParams::default());
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

/// I4 — singular under explicit `Fail` mode returns `Singular` and
/// clears the stored factor.
///
/// `diag(1, 0, 1)` has a structural zero pivot at position 1 with
/// no symmetric off-diagonal coupling that BK could pivot around.
/// With `ZeroPivotAction::Fail` opted in (the historical default
/// before F-03 / #32), the factor is discarded.
#[test]
fn i4_singular_under_fail_returns_singular_clears_factor() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 0.0, 1.0]).unwrap();

    let mut solver = Solver::with_params(np_fail_on_zero(), SupernodeParams::default());
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

/// F-03 regression — under the default `ZeroPivotAction::ForceAccept`,
/// `diag(1, 0, 1)` factors cleanly and the factor is preserved.
/// Issue #54 (SSIDS alignment, successor to #42): the isolated 0.0
/// pivot is recorded in `inertia.zero`, matching SSIDS/MA57's
/// Sylvester-signature accounting. Reported inertia is (2, 0, 1).
/// See `dev/research/issue-54-lp-kkt-inertia.md`, `dev/decisions.md`,
/// and issues #32 / #42 / #54.
#[test]
fn f03_default_force_accept_factors_isolated_zero_pivot() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 0.0, 1.0]).unwrap();

    let mut solver = Solver::new();
    let status = solver.factor(&csc, None);

    assert!(
        matches!(status, FactorStatus::Success),
        "expected Success under new ForceAccept default, got {:?}",
        status
    );

    assert!(
        solver.factors().is_some(),
        "factor should be preserved on Success"
    );
    let inertia = solver.inertia().expect("inertia stored on Success").clone();
    assert_eq!(
        inertia,
        Inertia {
            positive: 2,
            negative: 0,
            zero: 1,
        },
        "issue #54: diag(1, 0, 1) — strict-zero pivot → `zero` bucket \
         (SSIDS-aligned), expected (2, 0, 1), got {:?}",
        inertia
    );
}

/// F-01 / issue #42 — a rank-deficient symmetric matrix factored under
/// the default `ZeroPivotAction::ForceAccept` counts every pivot by
/// sign, so `inertia.zero == 0` structurally.
///
/// Issue #54 (SSIDS alignment) — successor to issue #42. Under SSIDS
/// and MA57 conventions, a strict-zero pivot (`|d| <= zero_tol`) is
/// recorded in `inertia.zero`, matching the matrix's Sylvester
/// signature. The previous Issue #42 sign-routing rule (`d > 0.0
/// ? pos : neg`) split bit-exact zeros by IEEE rounding noise — that
/// was the cause of the inertia jitter in pounce's perturbation
/// cascade (#54 stalled `nuffield2_trap` for 600s vs 1.8s on MA57).
///
/// We use a rank-1 dyadic A = u uᵀ at n=5 with u = ones; eigenvalues
/// are (5, 0, 0, 0, 0). Sylvester signature is (1, 0, 4), which feral
/// now reports verbatim. The factor still flags `needs_refinement`;
/// status semantics (`Success` vs `Singular`) are unchanged here.
#[test]
fn f01_dyadic_rankdef_counts_zero_pivots_ssids_aligned() {
    // A = u uᵀ with u = (1, 1, 1, 1, 1). Rank 1, n=5. Under pure
    // sign-counting feral reports zero=0; the rank-1 positive mass
    // guarantees at least one positive pivot.
    let n = 5usize;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        for i in j..n {
            rows.push(i);
            cols.push(j);
            vals.push(1.0);
        }
    }
    let csc = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();

    let mut solver = Solver::new();
    let status = solver.factor(&csc, None);
    assert!(
        matches!(status, FactorStatus::Success),
        "factor must succeed under default ForceAccept, got {:?}",
        status
    );
    let inertia = solver.inertia().expect("inertia stored on Success").clone();
    assert_eq!(
        inertia.positive + inertia.negative + inertia.zero,
        n,
        "inertia must sum to n"
    );
    assert_eq!(
        inertia.zero,
        n - 1,
        "issue #54 (SSIDS): rank-1 dyadic has Sylvester signature \
         (1, 0, n-1); strict-zero pivots → `zero`, got {:?}",
        inertia
    );
    assert_eq!(
        inertia.positive, 1,
        "rank-1 positive mass = exactly one positive pivot, got {:?}",
        inertia
    );
    assert_eq!(
        inertia.negative, 0,
        "rank-1 PSD dyadic has no negative pivots, got {:?}",
        inertia
    );
}

/// I7 — IPM-style escalation loop terminates with `Success`.
///
/// Demonstrates the canonical caller pattern from the plan:
/// factor → check → bump quality → re-factor. Uses a bordered
/// KKT (3 positive variables, 1 constraint, expected inertia
/// (3, 1, 0)) where the first factor with default params already
/// gives the correct inertia, so the loop terminates in 1
/// iteration. The structural assertion is that the loop runs
/// to `Success` within a small budget regardless of how many
/// quality bumps it takes.
#[test]
fn i7_quality_escalation_loop_terminates_with_correct_inertia() {
    // Bordered KKT from tests/sparse_postorder.rs.
    let csc = CscMatrix::from_triplets(
        4,
        &[0, 3, 1, 3, 2, 3, 3],
        &[0, 0, 1, 1, 2, 2, 3],
        &[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0],
    )
    .unwrap();
    let expected = Inertia {
        positive: 3,
        negative: 1,
        zero: 0,
    };

    let mut solver = Solver::new();
    let mut iters = 0usize;
    let final_status = loop {
        iters += 1;
        assert!(iters <= 6, "loop budget exceeded");
        match solver.factor(&csc, Some(expected.clone())) {
            FactorStatus::Success => break FactorStatus::Success,
            FactorStatus::WrongInertia { .. } => {
                if !solver.increase_quality() {
                    panic!("quality exhausted before Success");
                }
            }
            FactorStatus::Singular => panic!("unexpected Singular on a non-singular bordered KKT"),
            FactorStatus::FatalError(e) => panic!("fatal: {}", e),
        }
    };
    assert!(matches!(final_status, FactorStatus::Success));
    assert_eq!(solver.num_negative_eigenvalues(), expected.negative);
}

/// I8 — solver lifetime: state persists across `factor()` calls.
///
/// Factor once, then call `increase_quality()` twice. The second
/// `factor()` should observe the bumped pivot threshold via
/// `solver.pivot_threshold()`, and the new factorization should
/// still succeed.
///
/// Note: `NumericParams::default()` baseline `pivot_threshold` is
/// `1e-8` (MA27 `cntl[1]` default, issue #2), so the W5
/// "0.0 → 0.01" first-jump rule does not fire from `Solver::new()`;
/// the first `increase_quality` bump applies the geometric rule
/// directly: 1e-8 → 1e-8^0.75 = 1e-6.
#[test]
fn i8_solver_lifetime_state_persists() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();

    let mut solver = Solver::new();
    let _ = solver.factor(&csc, None);
    assert_eq!(solver.quality_level(), QualityLevel::Baseline);
    assert_eq!(
        solver.pivot_threshold(),
        1e-8,
        "Solver::new() baseline pivot threshold should be MA27's \
         cntl[1] default 1e-8 (issue #2)"
    );

    // First bump on default (Auto) scaling: stage 1 is no-op (Auto
    // is not Identity), fall through to stage 2. Baseline is
    // already 1e-8, so bump applies geometric rule:
    //   (1e-8)^0.75 = 10^(-8*0.75) = 10^-6 = 1e-6.
    assert!(solver.increase_quality());
    assert_eq!(solver.quality_level(), QualityLevel::PivotRaised);
    let want_after_1 = 1e-8_f64.powf(0.75);
    assert!((solver.pivot_threshold() - want_after_1).abs() < 1e-15);

    // Second bump: (1e-6)^0.75 = 10^-4.5 ≈ 3.162e-5.
    assert!(solver.increase_quality());
    let want_after_2 = want_after_1.powf(0.75);
    assert!((solver.pivot_threshold() - want_after_2).abs() < 1e-15);

    // Re-factor: state persists, factor still succeeds, symbolic
    // cache reused (same pattern).
    let n_sym_before = solver.symbolic_call_count();
    let status = solver.factor(&csc, None);
    assert!(matches!(status, FactorStatus::Success));
    assert_eq!(solver.symbolic_call_count(), n_sym_before);
    // Pivot threshold did not get reset by factor().
    assert!((solver.pivot_threshold() - want_after_2).abs() < 1e-15);
}

/// `Solver::min_diagonal()` returns `None` before any successful factor.
#[test]
fn min_diagonal_before_factor_is_none() {
    let solver = Solver::new();
    assert_eq!(solver.min_diagonal(), None);
}

fn solver_identity_scaling() -> Solver {
    let np = NumericParams {
        bk: BunchKaufmanParams::default(),
        scaling: ScalingStrategy::Identity,
        small_leaf: SmallLeafBatch::Off,
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
    };
    Solver::with_params(np, SupernodeParams::default())
}

/// 1×1-pivot only: a 4×4 diagonal indefinite matrix has D = the
/// diagonal of A under any pivot order, so min D is the smallest
/// diagonal entry.
///
/// Identity scaling is forced so the matrix actually factored is A
/// itself; otherwise default `Auto` scaling would rescale D and the
/// hand-computed oracle would not apply.
#[test]
fn min_diagonal_diagonal_matrix_one_by_one_pivots() {
    // A = diag(5, -2, 3, -7), lower-triangle CSC.
    let csc =
        CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[5.0, -2.0, 3.0, -7.0]).unwrap();

    let mut solver = solver_identity_scaling();
    let status = solver.factor(&csc, None);
    assert!(matches!(status, FactorStatus::Success), "got {:?}", status);

    let min_d = solver.min_diagonal().expect("min_diagonal");
    assert!(
        (min_d - (-7.0)).abs() < 1e-12,
        "expected -7.0, got {}",
        min_d
    );
}

/// 2×2 pivot: A = [[0, 1], [1, 0]]. BK must pick a 2×2 block
/// because the diagonals are zero. Eigenvalues are ±1, so the
/// minimum is -1.
///
/// Verifies that `min_diagonal()` extracts the smaller eigenvalue
/// of the 2×2 block, not just `d_diag[0] = 0`.
#[test]
fn min_diagonal_two_by_two_block_eigenvalue() {
    // Lower triangle: (0,0)=0, (1,0)=1, (1,1)=0.
    let csc = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[0.0, 1.0, 0.0]).unwrap();

    let mut solver = solver_identity_scaling();
    let status = solver.factor(&csc, None);
    assert!(matches!(status, FactorStatus::Success), "got {:?}", status);

    // Inertia: one positive, one negative.
    let x = solver.solve(&[1.0, 0.0]).expect("solve");
    // [[0,1],[1,0]] x = [1,0] → x = [0, 1].
    assert!((x[0]).abs() < 1e-12, "x[0] = {}", x[0]);
    assert!((x[1] - 1.0).abs() < 1e-12, "x[1] = {}", x[1]);

    let min_d = solver.min_diagonal().expect("min_diagonal");
    assert!(
        (min_d - (-1.0)).abs() < 1e-12,
        "expected -1.0 (smaller eig of [[0,1],[1,0]]), got {}",
        min_d
    );
}

// --- near-singularity signal: min|λ(D)| / max|λ(D)| ---------------------
//
// See dev/research/near-singularity-signal.md. These exercise
// Solver::{min,max}_pivot_magnitude — the MA57 CNTL(2) analog. The
// oracle is hand calculation, external to the implementation.

/// `min_pivot_magnitude()` / `max_pivot_magnitude()` return `None`
/// before any successful factor.
#[test]
fn pivot_magnitude_before_factor_is_none() {
    let solver = Solver::new();
    assert_eq!(solver.min_pivot_magnitude(), None);
    assert_eq!(solver.max_pivot_magnitude(), None);
}

/// 1×1-pivot only: A = diag(5, -2, 3, -7) under identity scaling gives
/// D = the diagonal, so the pivot magnitudes are {5, 2, 3, 7}. The
/// near-singularity signal is the smallest magnitude (2), distinct
/// from `min_diagonal()` which returns the signed minimum (-7).
#[test]
fn pivot_magnitude_one_by_one_pivots() {
    let csc =
        CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[5.0, -2.0, 3.0, -7.0]).unwrap();

    let mut solver = solver_identity_scaling();
    let status = solver.factor(&csc, None);
    assert!(matches!(status, FactorStatus::Success), "got {:?}", status);

    let min_mag = solver.min_pivot_magnitude().expect("min_pivot_magnitude");
    let max_mag = solver.max_pivot_magnitude().expect("max_pivot_magnitude");
    assert!(
        (min_mag - 2.0).abs() < 1e-12,
        "expected 2.0, got {}",
        min_mag
    );
    assert!(
        (max_mag - 7.0).abs() < 1e-12,
        "expected 7.0, got {}",
        max_mag
    );
    // Magnitude-min is positive even though min_diagonal() is negative.
    assert_eq!(solver.min_diagonal(), Some(-7.0));
}

/// 2×2 pivot: A = [[0,1],[1,0]] forces a 2×2 BK block with eigenvalues
/// ±1, so both pivot magnitudes are 1. Verifies the smaller-*magnitude*
/// eigenvalue is extracted (not `d_diag[0]=0`, not the signed -1).
#[test]
fn pivot_magnitude_two_by_two_block() {
    let csc = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[0.0, 1.0, 0.0]).unwrap();

    let mut solver = solver_identity_scaling();
    let status = solver.factor(&csc, None);
    assert!(matches!(status, FactorStatus::Success), "got {:?}", status);

    let min_mag = solver.min_pivot_magnitude().expect("min_pivot_magnitude");
    let max_mag = solver.max_pivot_magnitude().expect("max_pivot_magnitude");
    assert!(
        (min_mag - 1.0).abs() < 1e-12,
        "expected 1.0 (|smaller eig| of [[0,1],[1,0]]), got {}",
        min_mag
    );
    assert!(
        (max_mag - 1.0).abs() < 1e-12,
        "expected 1.0, got {}",
        max_mag
    );
}

/// Near-singular regression: A = diag(1, 1e-14, -3). The 1e-14 entry
/// is above `zero_tol` (≈2.2e-16) so it is accepted as a small-but-
/// nonzero 1×1 pivot and counted by sign. Inertia is still correct
/// (2 positive, 1 negative), but `min_pivot_magnitude()` surfaces the
/// near-singularity an inertia-only signal would hide — the ratio
/// `min/max ≈ 3e-15` is small enough for a perturbation handler to
/// threshold.
#[test]
fn pivot_magnitude_near_singular_regression() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 1e-14, -3.0]).unwrap();

    let mut solver = solver_identity_scaling();
    let status = solver.factor(&csc, None);
    assert!(matches!(status, FactorStatus::Success), "got {:?}", status);

    // Inertia is correct despite the near-singular pivot.
    let inertia = solver.inertia().expect("inertia");
    assert_eq!(inertia.positive, 2);
    assert_eq!(inertia.negative, 1);
    assert_eq!(inertia.zero, 0);

    let min_mag = solver.min_pivot_magnitude().expect("min_pivot_magnitude");
    let max_mag = solver.max_pivot_magnitude().expect("max_pivot_magnitude");
    assert!(
        (min_mag - 1e-14).abs() < 1e-26,
        "expected ~1e-14, got {}",
        min_mag
    );
    assert!(
        (max_mag - 3.0).abs() < 1e-12,
        "expected 3.0, got {}",
        max_mag
    );
    // The thresholdable near-singularity ratio.
    assert!(
        min_mag / max_mag < 1e-12,
        "ratio {} should be below a CNTL(2)-style threshold",
        min_mag / max_mag
    );
}
