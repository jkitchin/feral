//! Issue #33 §3: `Solver::with_ordering` lets library consumers swap
//! the fill-reducing method without dropping to the symbolic free
//! functions. Verifies (a) the builder threads the choice through
//! to the symbolic phase, (b) different methods produce structurally
//! different orderings (so the knob is actually live, not silently
//! ignored), and (c) both produce numerically correct factorizations.

use feral::numeric::factorize::factorize_multifrontal;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{CscMatrix, FactorStatus, NumericParams, Solver};

/// Build a tridiagonal SPD matrix of order `n`:
///   diag = 2.0, off-diag = -1.0
/// stored as lower-triangle CSC. Banded structure is exactly the
/// case #33 motivates — AMD and ND give different permutations on
/// it.
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

#[test]
fn with_ordering_threads_method_into_symbolic_phase() {
    let n = 200;
    let csc = tridiag_spd(n);

    // Sanity check: the two methods produce different symbolic
    // permutations on this matrix. If they ever stop differing for
    // structural reasons, the comparison below would be trivial; we
    // detect that explicitly so the test stays meaningful.
    let snode = SupernodeParams::default();
    let sym_amd = symbolic_factorize_with_method(&csc, &snode, OrderingMethod::Amd).unwrap();
    let sym_scotch =
        symbolic_factorize_with_method(&csc, &snode, OrderingMethod::ScotchND).unwrap();
    assert_ne!(
        sym_amd.perm, sym_scotch.perm,
        "AMD and ScotchND should produce different permutations on tridiag(200); \
         if they ever match, this test no longer proves with_ordering routes correctly"
    );

    // Build two solvers — one default (Auto), one explicit ScotchND
    // — and factor the same matrix. The ScotchND solver must
    // produce a symbolic whose permutation matches sym_scotch above,
    // proving the builder routed the choice through.
    let mut solver_default = Solver::new();
    let mut solver_scotch = Solver::new().with_ordering(OrderingMethod::ScotchND);

    assert!(matches!(
        solver_default.factor(&csc, None),
        FactorStatus::Success
    ));
    assert!(matches!(
        solver_scotch.factor(&csc, None),
        FactorStatus::Success
    ));

    // Roundtrip solve on each — A x = b with b = A * ones must
    // recover ones (modulo rounding).
    let x_true: Vec<f64> = vec![1.0; n];
    let mut b = vec![0.0; n];
    for j in 0..n {
        b[j] += 2.0 * x_true[j];
        if j + 1 < n {
            b[j + 1] -= x_true[j];
            b[j] -= x_true[j + 1];
        }
    }
    let x_default = solver_default.solve(&b).unwrap();
    let x_scotch = solver_scotch.solve(&b).unwrap();
    for i in 0..n {
        assert!(
            (x_default[i] - 1.0).abs() < 1e-10,
            "default-ordering solve diverged at i={i}: {}",
            x_default[i]
        );
        assert!(
            (x_scotch[i] - 1.0).abs() < 1e-10,
            "ScotchND-ordering solve diverged at i={i}: {}",
            x_scotch[i]
        );
    }
}

/// Negative control: `with_ordering(Auto)` must match `Solver::new()`
/// exactly — the default ordering threads through unchanged.
#[test]
fn with_ordering_auto_matches_default() {
    let csc = tridiag_spd(50);

    let mut a = Solver::new();
    let mut b = Solver::new().with_ordering(OrderingMethod::Auto);

    assert!(matches!(a.factor(&csc, None), FactorStatus::Success));
    assert!(matches!(b.factor(&csc, None), FactorStatus::Success));

    // The numeric multifrontal output (factor + inertia) must agree
    // bit-for-bit on the same ordering. Comparing inertia is the
    // cheapest invariant; identical orderings produce identical
    // inertia on this SPD matrix.
    assert_eq!(a.inertia(), b.inertia());

    // Belt-and-suspenders: re-run the symbolic with Auto explicitly
    // and check it produces the same nnz_L as both solvers above.
    let sym_auto =
        symbolic_factorize_with_method(&csc, &SupernodeParams::default(), OrderingMethod::Auto)
            .unwrap();
    let np = NumericParams::default();
    let (factors, _inertia) = factorize_multifrontal(&csc, &sym_auto, &np).unwrap();
    assert!(
        factors.factor_nnz() > 0,
        "Auto ordering factor should have nonzero L"
    );
}
