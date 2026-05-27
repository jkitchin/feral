//! Regression coverage for Phase A1 of the issue #55 plan
//! (`/Users/jkitchin/.claude/plans/feral-is-a-cached-raccoon.md`).
//!
//! Locks the MUMPS `INFO(25)` / NBTINYW equivalent counter
//! (`FactorStats::n_tiny`, surfaced via `Solver::last_factor_stats`)
//! by direct positive and negative tests:
//!
//!   * negative: factoring a well-conditioned synthetic KKT under the
//!     default solver (cascade-break off, `on_zero_pivot = Fail`) must
//!     not trigger any static-perturbation event — `n_tiny == 0`.
//!   * positive: factoring a deliberately rank-deficient diagonal
//!     matrix under `ZeroPivotAction::PerturbToEps` must trigger
//!     exactly one perturbation per intentionally-zero diagonal — and
//!     the perturbed inertia must match the MUMPS-aligned convention
//!     `sign(d)·tau` with `sign(0) = +1`.
//!
//! These two assertions together verify the full plumbing from the
//! dense kernel's `perturb_to_floor` site through `FrontalFactors`,
//! `SparseFactors::n_tiny()`, and `FactorStats::n_tiny`.

use feral::dense::factor::ZeroPivotAction;
use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus, Solver};
use feral::sparse::csc::CscMatrix;
use feral::symbolic::SupernodeParams;
use feral::{BunchKaufmanParams, Inertia};

/// Build A = diag(values) as a symmetric CSC matrix (lower triangle).
fn diag_csc(values: &[f64]) -> CscMatrix {
    let n = values.len();
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    CscMatrix::from_triplets(n, &rows, &cols, values).expect("diag csc")
}

#[test]
fn n_tiny_is_zero_on_well_conditioned_default_factor() {
    // diag(2, 3, 5, 7) — non-singular, well-spread spectrum, no
    // perturbation should fire under any default-path branch.
    let csc = diag_csc(&[2.0, 3.0, 5.0, 7.0]);
    let expected = Inertia {
        positive: 4,
        negative: 0,
        zero: 0,
    };
    let mut solver = Solver::new();
    let status = solver.factor(&csc, Some(expected.clone()));
    assert!(
        matches!(status, FactorStatus::Success),
        "well-conditioned diag factor must succeed: {status:?}"
    );
    let stats = solver
        .last_factor_stats()
        .expect("FactorStats present after Success");
    assert_eq!(
        stats.n_tiny, 0,
        "default factor on a well-conditioned matrix must not perturb"
    );
}

#[test]
fn n_tiny_counts_perturbed_pivots_under_perturb_to_eps() {
    // diag(1, 0, 1, 0, 1): three nonsingular diagonals and two strict
    // zero diagonals. Under PerturbToEps each zero is replaced by
    // sign(0)·eps == +eps, so the perturbed inertia is (5, 0, 0) and
    // n_tiny == 2 (one increment per zero diagonal eliminated through
    // the PerturbToEps branch of count_1x1_inertia / do_1x1_pivot).
    let csc = diag_csc(&[1.0, 0.0, 1.0, 0.0, 1.0]);

    let np = NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-10 },
            ..BunchKaufmanParams::default()
        },
        ..NumericParams::default()
    };
    let mut solver = Solver::with_params(np, SupernodeParams::default());

    // Don't gate factor() on a specific inertia — we are validating the
    // counter, not the sign convention here. Inertia is asserted from
    // the post-factor accessor below.
    let status = solver.factor(&csc, None);
    assert!(
        matches!(status, FactorStatus::Success),
        "PerturbToEps factor on rank-deficient diag must succeed: {status:?}"
    );

    let stats = solver
        .last_factor_stats()
        .expect("FactorStats present after Success");
    assert_eq!(
        stats.n_tiny, 2,
        "two strict-zero diagonals must produce two perturbation events; \
         got n_tiny = {}",
        stats.n_tiny,
    );

    // MUMPS-aligned sign convention: `sign(0) == +1`, so the zeros
    // become +eps and the perturbed inertia is fully positive.
    let got = solver.inertia().expect("inertia recorded on Success");
    assert_eq!(
        (got.positive, got.negative, got.zero),
        (5, 0, 0),
        "perturbed inertia must follow MUMPS sign(0)=+1 convention"
    );
}
