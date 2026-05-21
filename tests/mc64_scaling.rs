//! Unit tests for MC64 matching-based scaling.
//!
//! Exercises the public `compute_scaling` API with
//! `ScalingStrategy::Mc64Symmetric`. These tests are written before
//! the real implementation lands (Phase 2.2.1 Step 2 of
//! `dev/plans/mc64-scaling.md`). The stub in `src/scaling/mc64.rs`
//! returns identity scaling and `ScalingInfo::NotApplied` for any
//! input, so the tests that assert on non-trivial scaling
//! behavior or on `ScalingInfo::Applied` MUST fail until Step 4
//! lands the real wrapper.
//!
//! Test design: any hand-computable cases that exercise the MC64
//! output invariants without needing to know the exact dual
//! values. The key invariants we check:
//!
//!   * On a positive-definite diagonal matrix, symmetric scaling
//!     produces `s_i = 1/sqrt(A_{ii})` so that `s_i^2 * A_{ii} = 1`
//!     (unit diagonal on `D·A·D`).
//!   * On a matrix with wide dynamic range, the maximum-magnitude
//!     entry of the scaled matrix is bounded by 1.
//!   * The returned `ScalingInfo` accurately reflects whether the
//!     matching ran (`Applied`) vs was skipped (`NotApplied`) vs
//!     was partial (`PartialSingular { n_unmatched }`).

#![allow(
    clippy::manual_range_contains,
    clippy::approx_constant,
    clippy::needless_range_loop
)]
use feral::scaling::{compute_scaling, ScalingInfo, ScalingStrategy};
use feral::CscMatrix;

/// Helper: scaled matrix entry value `s[i] * A_{ij} * s[j]`.
fn scaled_entry(csc: &CscMatrix, scaling: &[f64], i: usize, j: usize) -> f64 {
    // Look up A_{ij} in the lower-triangle-only CSC. For (i,j)
    // with i < j, swap to (j,i).
    let (row, col) = if i >= j { (i, j) } else { (j, i) };
    let mut val = 0.0;
    for k in csc.col_ptr[col]..csc.col_ptr[col + 1] {
        if csc.row_idx[k] == row {
            val = csc.values[k];
            break;
        }
    }
    scaling[i] * val * scaling[j]
}

/// On an SPD diagonal matrix, MC64 scaling should produce
/// `s[i] = 1/sqrt(A[i,i])`, making the scaled diagonal exactly 1.
/// The stub returns identity scaling `s[i] = 1.0`, which on a
/// non-unit diagonal like `diag(2, 3, 5)` leaves the diagonal at
/// 2, 3, 5 — so this test fails on the stub and passes after
/// Step 4.
#[test]
fn mc64_diagonal_matrix_unit_scaled_diagonal() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();

    let (scaling, info) = compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric).unwrap();
    assert_eq!(
        info,
        ScalingInfo::Applied,
        "matching should run on non-singular input"
    );
    assert_eq!(scaling.len(), 3);

    for i in 0..3 {
        let scaled = scaled_entry(&csc, &scaling, i, i);
        assert!(
            (scaled - 1.0).abs() < 1e-12,
            "scaled diagonal at position {} should be 1.0, got {}",
            i,
            scaled
        );
    }
}

/// The identity matrix maps to identity scaling. Both the stub
/// and the real implementation should pass this test, because
/// `sqrt(1) = 1`.
#[test]
fn mc64_identity_matrix_identity_scaling() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0]).unwrap();

    let (scaling, _info) = compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric).unwrap();
    for i in 0..3 {
        assert!(
            (scaling[i] - 1.0).abs() < 1e-12,
            "scaling[{}] = {} should be 1.0 on identity matrix",
            i,
            scaling[i]
        );
    }
}

/// A matrix with wide dynamic range: `diag(1e-8, 1, 1e8)`.
/// After MC64 symmetric scaling, the largest scaled diagonal
/// magnitude should be ≤ 1 + ε (the unit-diagonal property).
/// Post-scaling: `s_0 = 1e4, s_1 = 1, s_2 = 1e-4`, scaled
/// diagonal is `[1, 1, 1]`. The stub gives identity scaling,
/// leaving the diagonal at `[1e-8, 1, 1e8]` — max entry 1e8,
/// fails the ≤ 1 bound by 8 orders of magnitude.
#[test]
fn mc64_wide_dynamic_range_unit_bound() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1e-8, 1.0, 1e8]).unwrap();

    let (scaling, _info) = compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric).unwrap();

    // Every scaled diagonal entry is within an order of magnitude
    // of 1 (since `s_i = 1/sqrt(A_{ii})` exactly gives unit-scaled
    // diagonal here, but we allow slack in case the wrapper does
    // something slightly different).
    for i in 0..3 {
        let scaled = scaled_entry(&csc, &scaling, i, i).abs();
        assert!(
            scaled >= 0.1 && scaled <= 10.0,
            "scaled diagonal at position {} = {}, should be within [0.1, 10]",
            i,
            scaled
        );
    }
}

/// The `Identity` strategy returns a `[1.0; n]` vector with
/// `ScalingInfo::NotApplied`. This is true for both the stub and
/// the real implementation — a "don't scale" request is a
/// don't-scale result.
#[test]
fn identity_strategy_returns_ones() {
    let csc =
        CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[2.0, 5.0, 7.0, 11.0]).unwrap();
    let (scaling, info) = compute_scaling(&csc, &ScalingStrategy::Identity).unwrap();
    assert_eq!(scaling, vec![1.0; 4]);
    assert_eq!(info, ScalingInfo::NotApplied);
}

/// `External(vec)` passes through the user-supplied vector
/// verbatim and reports `Applied`: the factor scales the matrix by
/// `D = diag(s)` unconditionally, so the solve must undo it.
/// `NotApplied` is reserved for the all-ones (genuine no-op) case;
/// pairing it with a real `s` would factor `D·A·D` but solve it as
/// `A` (the solve keys `needs_scaling` off `scaling_info`).
#[test]
fn external_strategy_passes_through() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0]).unwrap();
    let user = vec![0.5, 2.0, 3.14];
    let (scaling, info) = compute_scaling(&csc, &ScalingStrategy::External(user.clone())).unwrap();
    assert_eq!(scaling, user);
    assert_eq!(info, ScalingInfo::Applied);
}

/// `External(vec)` with wrong length returns an error rather
/// than silently accepting.
#[test]
fn external_strategy_wrong_length_errors() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0]).unwrap();
    let wrong_user = vec![0.5, 2.0]; // length 2, matrix is 3
    assert!(compute_scaling(&csc, &ScalingStrategy::External(wrong_user)).is_err());
}

/// A 2×2 SPD matrix with off-diagonal coupling: `A = [[4, 2], [2, 4]]`.
/// By hand: the column maxes are both `log 4`. The cost graph is
/// `col 0: (0, 0), (1, log(4)-log(2))`, `col 1: (0, log(4)-log(2)), (1, 0)`.
/// Minimum-cost matching pairs 0↔0, 1↔1 with total cost 0; the
/// matching on the off-diagonal would have cost 2*log(2) > 0.
/// Dual variables can be `u = v = [0, 0]` (feasibility on all edges
/// because `c_{ii} = 0` and `c_{01} = c_{10} = log 2 > 0`).
/// Unwound: `u' = -u = [0, 0]`, `v' = C - v = [log 4, log 4]`.
/// Symmetric average: `s_i = exp((0 + log 4) / 2) = exp(log 2) = 2`.
/// Wait — that gives `s_i = 2`, so scaled `A[0,0] = 2 * 4 * 2 = 16`.
/// That can't be right. Let me re-derive.
///
/// Actually I had the sign wrong in the research note. Re-deriving
/// from SPRAL scaling.f90:681-682:
///     rscaling[i] = dualu[i]
///     cscaling[j] = dualv[j] - cmax[j]
/// and then scaling.f90:169:
///     scaling[i] = exp((rscaling[i] + cscaling[i]) / 2)
///             = exp((dualu[i] + dualv[i] - cmax[i]) / 2)
/// For the diagonal case above with u=v=[0,0] and cmax = [log 4, log 4]:
///     scaling[i] = exp((0 + 0 - log 4) / 2) = exp(-log 2) = 1/2
/// Scaled A[0,0] = (1/2) * 4 * (1/2) = 1 ✓
/// Scaled A[1,0] = (1/2) * 2 * (1/2) = 1/2, bounded by 1 ✓
/// Good. Now the test:
#[test]
fn mc64_2x2_spd_off_diagonal_bounded() {
    // A = [[4, 2], [2, 4]], stored lower triangle
    let csc = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[4.0, 2.0, 4.0]).unwrap();

    let (scaling, info) = compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric).unwrap();
    assert_eq!(info, ScalingInfo::Applied);

    // After scaling, the diagonal should be exactly 1.
    for i in 0..2 {
        let scaled = scaled_entry(&csc, &scaling, i, i);
        assert!(
            (scaled - 1.0).abs() < 1e-12,
            "A[{},{}] scaled = {}, expected 1",
            i,
            i,
            scaled
        );
    }

    // Off-diagonals should be bounded by 1 in absolute value.
    let scaled_offdiag = scaled_entry(&csc, &scaling, 1, 0);
    assert!(
        scaled_offdiag.abs() <= 1.0 + 1e-12,
        "scaled off-diagonal {} exceeds 1",
        scaled_offdiag
    );
}

// ---------------------------------------------------------------------------
// Symbolic-side integration tests for the scaling cache used to live
// here. Scaling now lives on the numeric path — see the parity panel
// in tests/parity.rs and the structural test
// `factorize_multifrontal_with_two_strategies_on_one_symbolic` in
// src/numeric/factorize.rs for end-to-end coverage. The
// `compute_scaling` API tests above remain the unit tests for the
// scaling primitives themselves.
// ---------------------------------------------------------------------------
