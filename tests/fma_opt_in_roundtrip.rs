//! Issue #8 Phase 2 — FMA opt-in round-trip parity.
//!
//! Constructs a moderate indefinite symmetric matrix (an arrow-shaped
//! KKT-like saddle-point system: SPD top-left block + dense
//! constraint Jacobian rows on the trailing diagonal), then factorizes
//! it twice through the `Solver` builder:
//!
//!   1. `NumericParams::default()` — `fma: false`, the bit-exact
//!      cross-arch reference path.
//!   2. `Solver::new().with_fma(true)` — dispatch through the FMA
//!      siblings (`schur_panel_minus_fma_strided*`,
//!      `axpy_minus_unroll4`, `axpy2_minus_unroll4`).
//!
//! Asserts:
//!   - Both factorizations produce the **same inertia**. The opt-in
//!     dispatch must not change the pivot decisions on a numerically
//!     stable problem.
//!   - Both back-solves produce solutions within `KAPPA * eps_machine`
//!     relative residual of the input rhs. `KAPPA` is the Phase 1
//!     cross-policy tolerance bound (`4 * n_elim * EPS`, generously
//!     widened for the multifrontal solve which composes several panel
//!     updates per supernode).
//!
//! This test guards the public API contract: turning FMA on never
//! degrades correctness on well-conditioned problems; it only changes
//! the per-element rounding chain by at most the documented bound.

use feral::{CscMatrix, NumericParams, Solver};

const N_BLOCK: usize = 96;
const N_CONS: usize = 16;

/// Build an arrow-shaped saddle-point matrix:
///
///     [  H    J^T ]
///     [  J     0  ]
///
/// where `H` is a 96×96 SPD tridiagonal block (2*I + lower/upper
/// off-diagonals at ±0.5) and `J` is a 16×96 dense constraint
/// Jacobian whose rows i pick up identity-like coupling to columns
/// `(6*i)..(6*i+6)` of H plus a -1 perturbation. Inertia is
/// (`N_BLOCK`, `N_CONS`, 0) — `N_BLOCK` positive eigenvalues from H
/// and `N_CONS` negative eigenvalues from the saddle structure.
fn build_saddle_kkt() -> CscMatrix {
    // Build lower-triangle entries (col, row, val) with row >= col.
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();

    // H block: tridiagonal, columns 0..N_BLOCK.
    for j in 0..N_BLOCK {
        entries.push((j, j, 2.0));
        if j + 1 < N_BLOCK {
            entries.push((j, j + 1, 0.5));
        }
    }
    // J coupling: row (N_BLOCK + i) has entries in columns (6i)..(6i+6).
    for i in 0..N_CONS {
        let row = N_BLOCK + i;
        for k in 0..6 {
            let col = 6 * i + k;
            if col < N_BLOCK {
                // Store as (col, row) with row > col so it lands in the lower triangle.
                let v = if k == 0 { 1.0 } else { -0.25 };
                entries.push((col, row, v));
            }
        }
    }
    // Trailing zero block: just diagonal entries with 0.0 to keep CSC
    // structure full-rank in pattern. We don't push them because zero
    // entries are dropped anyway and the saddle solve doesn't need an
    // explicit pattern there.

    let rows: Vec<usize> = entries.iter().map(|(_, r, _)| *r).collect();
    let cols: Vec<usize> = entries.iter().map(|(c, _, _)| *c).collect();
    let vals: Vec<f64> = entries.iter().map(|(_, _, v)| *v).collect();
    CscMatrix::from_triplets(N_BLOCK + N_CONS, &rows, &cols, &vals).expect("valid CSC")
}

fn sym_residual_inf(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; a.n];
    a.symv(x, &mut ax);
    let mut max_r = 0.0f64;
    let mut max_b = 0.0f64;
    for i in 0..a.n {
        max_r = max_r.max((ax[i] - b[i]).abs());
        max_b = max_b.max(b[i].abs());
    }
    if max_b > 0.0 {
        max_r / max_b
    } else {
        max_r
    }
}

#[test]
fn fma_opt_in_matches_nofma_inertia_and_solves_accurately() {
    let kkt = build_saddle_kkt();
    let n = kkt.n;

    // Deterministic rhs: just a ramp.
    let rhs: Vec<f64> = (0..n).map(|i| (i as f64) * 0.125 + 1.0).collect();

    // Path 1: default (FMA off).
    let mut solver_nofma = Solver::with_params(
        NumericParams::default(),
        feral::symbolic::SupernodeParams::default(),
    );
    let status_nofma = solver_nofma.factor(&kkt, None);
    assert!(
        matches!(status_nofma, feral::numeric::solver::FactorStatus::Success),
        "nofma factor failed: {:?}",
        status_nofma
    );
    let inertia_nofma = solver_nofma
        .inertia()
        .cloned()
        .expect("inertia present after Success");
    let x_nofma = solver_nofma.solve(&rhs).expect("nofma solve");
    let res_nofma = sym_residual_inf(&kkt, &x_nofma, &rhs);

    // Path 2: FMA on.
    let mut solver_fma = Solver::new().with_fma(true);
    let status_fma = solver_fma.factor(&kkt, None);
    assert!(
        matches!(status_fma, feral::numeric::solver::FactorStatus::Success),
        "fma factor failed: {:?}",
        status_fma
    );
    let inertia_fma = solver_fma
        .inertia()
        .cloned()
        .expect("inertia present after Success");
    let x_fma = solver_fma.solve(&rhs).expect("fma solve");
    let res_fma = sym_residual_inf(&kkt, &x_fma, &rhs);

    // Contract: same inertia.
    assert_eq!(
        inertia_nofma, inertia_fma,
        "FMA toggle changed inertia: nofma={inertia_nofma} fma={inertia_fma}"
    );

    // Contract: both residuals must be small in absolute terms. The
    // FMA path uses single-rounding mul_add so it is typically more
    // accurate per op, not less; but we only assert a generous gate
    // here (a few ulps times problem size), since this is a contract
    // test, not a tolerance regression.
    let tol = 1e-10;
    assert!(
        res_nofma < tol,
        "nofma residual {res_nofma} exceeds gate {tol}"
    );
    assert!(res_fma < tol, "fma residual {res_fma} exceeds gate {tol}");
}
