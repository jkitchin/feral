//! Issue #107: `OrderingMethod::External(Vec<usize>)` lets a caller inject a
//! precomputed fill-reducing permutation (block-triangular Schur reuse,
//! tearing orderings) instead of selecting an internal ordering algorithm.
//! This mirrors `ScalingStrategy::External(Vec<f64>)`.
//!
//! Oracles here are external to the solver:
//!   - the saddle-point KKT `[[H, Aᵀ],[A, 0]]` with H SPD (n×n) and A (m×n)
//!     full row rank has inertia `(n, m, 0)` by the standard saddle-point
//!     inertia theorem (Nocedal & Wright, *Numerical Optimization*, Lemma
//!     16.1 / Gould 1985) — a hand oracle, not the solver's own output;
//!   - the right-hand sides are `b = K · x_true` for a chosen `x_true`, so
//!     the recovered solution is known by construction;
//!   - the tridiagonal `diag=2, offdiag=-1` matrix is SPD (irreducibly
//!     diagonally dominant with positive diagonal), inertia `(n, 0, 0)`.
//!
//! A *valid but different* permutation must change fill/timing only, never
//! the answer or the inertia — the property these tests pin.

use feral::numeric::factorize::factorize_multifrontal;
use feral::symbolic::{
    symbolic_factorize_with_method, OrderingMethod, OrderingPreprocess, SupernodeParams,
};
use feral::{CscMatrix, FactorStatus, NumericParams, Solver};

/// Saddle-point KKT `[[I₂, Aᵀ],[A, 0]]` with `A = [1, 1]` (m=1), stored as
/// lower triangle CSC:
///
/// ```text
///   [ 1  0  1 ]
///   [ 0  1  1 ]
///   [ 1  1  0 ]
/// ```
///
/// H = I₂ is SPD, A = [1,1] has full row rank, so the saddle-point inertia
/// theorem gives inertia `(2, 1, 0)`.
fn saddle_kkt_3x3() -> CscMatrix {
    // lower triangle (row >= col): (0,0)=1, (1,1)=1, (2,0)=1, (2,1)=1, (2,2)=0
    // The (2,2) zero pivot is the saddle block; store it explicitly so the
    // pattern includes the diagonal.
    let rows = vec![0usize, 2, 1, 2, 2];
    let cols = vec![0usize, 0, 1, 1, 2];
    let vals = vec![1.0, 1.0, 1.0, 1.0, 0.0];
    CscMatrix::from_triplets(3, &rows, &cols, &vals).unwrap()
}

/// Tridiagonal SPD of order `n` (`diag=2, offdiag=-1`), lower triangle CSC.
/// Inertia `(n, 0, 0)`.
fn tridiag_spd(n: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        rows.push(j);
        cols.push(j);
        vals.push(2.0);
        if j + 1 < n {
            rows.push(j + 1);
            cols.push(j);
            vals.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

/// `b = K · x` for the symmetric matrix `K` given as its full (symmetrized)
/// dense action; here we hardcode the small saddle case.
fn saddle_matvec(x: &[f64]) -> Vec<f64> {
    // K = [[1,0,1],[0,1,1],[1,1,0]]
    vec![x[0] + x[2], x[1] + x[2], x[0] + x[1]]
}

#[test]
fn external_identity_ordering_solves_and_reports_inertia() {
    let csc = saddle_kkt_3x3();
    let params = SupernodeParams::default();

    let perm = vec![0usize, 1, 2]; // identity
    let sym =
        symbolic_factorize_with_method(&csc, &params, OrderingMethod::External(perm)).unwrap();

    // The supplied permutation IS the fill-reducing ordering; it is never
    // routed through the compression preprocessor.
    assert_eq!(sym.resolved_preprocess, OrderingPreprocess::None);
    assert!(
        matches!(sym.resolved_method, OrderingMethod::External(ref p) if p == &[0, 1, 2]),
        "resolved_method should faithfully report External, got {:?}",
        sym.resolved_method
    );
    // perm is a bijection of 0..3.
    let mut sorted = sym.perm.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2]);

    let np = NumericParams::default();
    let (_factors, inertia) = factorize_multifrontal(&csc, &sym, &np).unwrap();
    // Saddle-point inertia theorem: (n_H, m_A, 0) = (2, 1, 0).
    assert_eq!(
        (inertia.positive, inertia.negative, inertia.zero),
        (2, 1, 0)
    );
}

#[test]
fn external_nontrivial_ordering_gives_same_answer() {
    // A valid but non-identity permutation must not change the inertia or the
    // solution — only the fill. Compare identity vs a reversed ordering.
    let csc = saddle_kkt_3x3();

    let x_true = vec![1.0f64, -2.0, 3.0];
    let b = saddle_matvec(&x_true);

    let solve_with = |perm: Vec<usize>| -> (feral::Inertia, Vec<f64>) {
        let mut solver = Solver::new().with_ordering(OrderingMethod::External(perm));
        assert!(matches!(solver.factor(&csc, None), FactorStatus::Success));
        let inertia = solver
            .inertia()
            .expect("inertia after successful factor")
            .clone();
        let x = solver.solve(&b).expect("solve after successful factor");
        (inertia, x)
    };

    let (in_id, x_id) = solve_with(vec![0, 1, 2]);
    let (in_rev, x_rev) = solve_with(vec![2, 1, 0]);

    assert_eq!(
        (in_id.positive, in_id.negative, in_id.zero),
        (2, 1, 0),
        "identity-ordering inertia"
    );
    assert_eq!(
        (in_rev.positive, in_rev.negative, in_rev.zero),
        (2, 1, 0),
        "reversed-ordering inertia must match — ordering never changes inertia"
    );
    for i in 0..3 {
        assert!(
            (x_id[i] - x_true[i]).abs() < 1e-10,
            "identity solve wrong at {i}: {} vs {}",
            x_id[i],
            x_true[i]
        );
        assert!(
            (x_rev[i] - x_true[i]).abs() < 1e-10,
            "reversed-ordering solve wrong at {i}: {} vs {}",
            x_rev[i],
            x_true[i]
        );
    }
}

#[test]
fn external_ordering_matches_default_on_spd() {
    // End-to-end parity: an External identity ordering must solve the SPD
    // tridiagonal system as accurately as the default (Auto) ordering, and
    // report the same inertia (n, 0, 0).
    let n = 60;
    let csc = tridiag_spd(n);

    let x_true = vec![1.0; n];
    let mut b = vec![0.0; n];
    for j in 0..n {
        b[j] += 2.0 * x_true[j];
        if j + 1 < n {
            b[j + 1] -= x_true[j];
            b[j] -= x_true[j + 1];
        }
    }

    let mut solver_default = Solver::new();
    // A hand-built valid permutation (reverse) — exercises postorder on a
    // non-identity external ordering.
    let rev: Vec<usize> = (0..n).rev().collect();
    let mut solver_ext = Solver::new().with_ordering(OrderingMethod::External(rev));

    assert!(matches!(
        solver_default.factor(&csc, None),
        FactorStatus::Success
    ));
    assert!(matches!(
        solver_ext.factor(&csc, None),
        FactorStatus::Success
    ));

    assert_eq!(
        solver_default.inertia().unwrap(),
        solver_ext.inertia().unwrap(),
        "SPD inertia must be (n,0,0) under any valid ordering"
    );
    let inertia = solver_ext.inertia().unwrap();
    assert_eq!(
        (inertia.positive, inertia.negative, inertia.zero),
        (n, 0, 0)
    );

    let x_default = solver_default.solve(&b).unwrap();
    let x_ext = solver_ext.solve(&b).unwrap();
    for i in 0..n {
        assert!(
            (x_ext[i] - 1.0).abs() < 1e-10,
            "external-ordering solve diverged at i={i}: {}",
            x_ext[i]
        );
        // Both feral paths should agree closely on an SPD system.
        assert!(
            (x_ext[i] - x_default[i]).abs() < 1e-9,
            "external vs default mismatch at i={i}: {} vs {}",
            x_ext[i],
            x_default[i]
        );
    }
}

#[test]
fn external_ordering_rejects_wrong_length() {
    let csc = saddle_kkt_3x3();
    let params = SupernodeParams::default();
    // length 2 != n=3
    let err = symbolic_factorize_with_method(&csc, &params, OrderingMethod::External(vec![0, 1]));
    assert!(err.is_err(), "wrong-length permutation must be rejected");
}

#[test]
fn external_ordering_rejects_out_of_range_index() {
    let csc = saddle_kkt_3x3();
    let params = SupernodeParams::default();
    // 3 is out of range for n=3 (valid indices 0..=2)
    let err =
        symbolic_factorize_with_method(&csc, &params, OrderingMethod::External(vec![0, 1, 3]));
    assert!(err.is_err(), "out-of-range index must be rejected");
}

#[test]
fn external_ordering_rejects_duplicate_index() {
    let csc = saddle_kkt_3x3();
    let params = SupernodeParams::default();
    // 1 repeated, 2 missing — not a bijection
    let err =
        symbolic_factorize_with_method(&csc, &params, OrderingMethod::External(vec![0, 1, 1]));
    assert!(err.is_err(), "non-bijective permutation must be rejected");
}
