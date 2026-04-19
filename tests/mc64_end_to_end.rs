//! Phase 2.2.1 Step 7 end-to-end correctness tests.
//!
//! These tests exercise the full `symbolic_factorize` →
//! `factorize_multifrontal` → `solve_sparse` path with MC64 scaling
//! enabled (the default strategy) and assert that the returned
//! solution reproduces the hand-derived answer.
//!
//! The assembly-time multiply in Step 6 factors `M = D · A · D`
//! rather than `A`. Without Step 7's pre/post-scale around the core
//! solver the residuals blow up (see the 9 failing tests recovered
//! by this commit). The tests below verify that the same `D` is
//! applied on both ends — not its inverse — by constructing cases
//! where the expected answer can be written down exactly.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::solve_sparse;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    })
}

/// `A = diag(2, 3, 5)`, `b = [2, 3, 5]` ⇒ `x = [1, 1, 1]`.
///
/// MC64 produces `scaling = [1/sqrt(2), 1/sqrt(3), 1/sqrt(5)]`, so
/// `D·A·D = I`. The sanity check walks through the algebra:
///   * `b_scaled = [sqrt(2), sqrt(3), sqrt(5)]`
///   * `core_solve(I, b_scaled) = [sqrt(2), sqrt(3), sqrt(5)]`
///   * `x = D · y = [1, 1, 1]`
///
/// Getting `x = [1, 1, 1]` (and not `[1/2, 1/3, 1/5]` or similar)
/// confirms the same-direction application of `D`.
#[test]
fn mc64_end_to_end_diagonal() {
    let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
    let rhs = vec![2.0, 3.0, 5.0];

    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (factors, _inertia) = factorize_multifrontal(&csc, &sym, &ldlt_params()).expect("factor");
    let x = solve_sparse(&factors, &rhs).expect("solve");

    assert!((x[0] - 1.0).abs() < 1e-12, "x[0] = {}", x[0]);
    assert!((x[1] - 1.0).abs() < 1e-12, "x[1] = {}", x[1]);
    assert!((x[2] - 1.0).abs() < 1e-12, "x[2] = {}", x[2]);
}

/// `A = [[4, 2], [2, 4]]`, `b = [6, 6]` ⇒ `x = [1, 1]`.
///
/// By hand: `det(A) = 12`, `A^-1 = (1/12) [[4, -2], [-2, 4]]`,
/// `A^-1 · [6, 6] = (1/12) [24 - 12, -12 + 24] = [1, 1]`.
///
/// This exercises the full symmetric scaling path through an
/// off-diagonal entry: MC64 scales as `s = [1/2, 1/2]` (see the
/// derivation in `tests/mc64_scaling.rs::mc64_2x2_spd_off_diagonal_bounded`)
/// and `D · A · D = [[1, 1/2], [1/2, 1]]`, an indefinite-pivot-free
/// SPD system whose factorization and solve round-trip back to
/// `x = [1, 1]` only if pre- and post-scale use the same direction.
#[test]
fn mc64_end_to_end_2x2_spd_off_diagonal() {
    // Store the lower triangle of A = [[4, 2], [2, 4]].
    let csc = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[4.0, 2.0, 4.0]).unwrap();
    let rhs = vec![6.0, 6.0];

    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (factors, _inertia) = factorize_multifrontal(&csc, &sym, &ldlt_params()).expect("factor");
    let x = solve_sparse(&factors, &rhs).expect("solve");

    assert!((x[0] - 1.0).abs() < 1e-12, "x[0] = {}", x[0]);
    assert!((x[1] - 1.0).abs() < 1e-12, "x[1] = {}", x[1]);
}
