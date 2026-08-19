//! Issue #178 item 2 — in-place solve entry points on `Solver`.
//!
//! A host that already owns its right-hand-side buffer had to accept an
//! owned `Vec` from every solve entry point and copy it back: `dim ×
//! nrhs` doubles per back-solve. The `*_into` variants write the solution
//! into a caller slice instead.
//!
//! Verification per issue #178:
//!   - each `_into` is bit-identical to its allocating twin,
//!   - a dimension mismatch returns `DimensionMismatch`, never a panic,
//!   - aliasing `rhs` / `x_out` is addressed explicitly (see the module
//!     note below).
//!
//! **Aliasing.** Issue #178 asks that aliasing `rhs` and `x_out` be
//! "supported or rejected explicitly rather than silently wrong". In safe
//! Rust it is neither — it is unrepresentable: `&[f64]` and `&mut [f64]`
//! borrowing one allocation cannot coexist, so no caller of these
//! signatures can construct the aliased case. That is checked by the
//! compiler on every build rather than by a test, which is why there is
//! no aliasing test here. `tests/issue178_alias_rejected.rs` is not a
//! file; the guarantee lives in the type signature.

use feral::numeric::factorize::NumericParams;
use feral::numeric::solve::RefineOptions;
use feral::symbolic::SupernodeParams;
use feral::{BunchKaufmanParams, CscMatrix, FeralError, Solver, ZeroPivotAction};

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    })
}

/// Small indefinite bordered KKT — exercises 2×2 pivots, so the `_into`
/// paths are compared on a factor whose solve is not trivially exact.
fn kkt() -> CscMatrix {
    CscMatrix::from_triplets(
        6,
        &[0, 4, 1, 5, 2, 4, 3, 5, 4, 5],
        &[0, 0, 1, 1, 2, 2, 3, 3, 4, 5],
        &[100.0, -1.0, 100.0, -1.0, 1e-6, 1.0, 1e-6, 1.0, -1e-4, -1e-4],
    )
    .expect("kkt")
}

fn solver_for(m: &CscMatrix) -> Solver {
    let mut s = Solver::with_params(ldlt_params(), SupernodeParams::default());
    let status = s.factor(m, None);
    assert!(
        s.factors().is_some(),
        "factor failed with status {status:?}"
    );
    s
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

const RHS: [f64; 6] = [1.0, 2.0, 0.5, -0.25, -50.0, -75.0];

#[test]
fn solve_into_is_bit_identical_to_solve() {
    let m = kkt();
    let s = solver_for(&m);
    let want = s.solve(&RHS).expect("solve");
    let mut got = vec![f64::NAN; m.n];
    s.solve_into(&RHS, &mut got).expect("solve_into");
    assert_eq!(bits(&got), bits(&want));
}

#[test]
fn solve_refined_into_is_bit_identical_to_solve_refined() {
    let m = kkt();
    let s = solver_for(&m);
    let want = s.solve_refined(&m, &RHS).expect("solve_refined");
    let mut got = vec![f64::NAN; m.n];
    s.solve_refined_into(&m, &RHS, &mut got, RefineOptions::default())
        .expect("solve_refined_into");
    assert_eq!(bits(&got), bits(&want));
}

#[test]
fn solve_refined_into_honours_the_cap() {
    let m = kkt();
    let s = solver_for(&m);
    let opts = RefineOptions::with_max_steps(0);
    let want = s.solve(&RHS).expect("solve");
    let mut got = vec![f64::NAN; m.n];
    s.solve_refined_into(&m, &RHS, &mut got, opts)
        .expect("solve_refined_into");
    assert_eq!(bits(&got), bits(&want));

    // ... and the allocating capped twin agrees.
    let alloc = s
        .solve_refined_opts(&m, &RHS, opts)
        .expect("solve_refined_opts");
    assert_eq!(bits(&alloc), bits(&want));
}

#[test]
fn solve_many_into_is_bit_identical_to_solve_many() {
    let m = kkt();
    let s = solver_for(&m);
    for nrhs in [1usize, 2, 20] {
        let wide: Vec<f64> = (0..m.n * nrhs)
            .map(|k| RHS[k % m.n] * (1.0 + (k / m.n) as f64))
            .collect();
        let want = s.solve_many(&wide, nrhs).expect("solve_many");
        let mut got = vec![f64::NAN; m.n * nrhs];
        s.solve_many_into(&wide, nrhs, &mut got)
            .expect("solve_many_into");
        assert_eq!(bits(&got), bits(&want), "nrhs = {nrhs}");
    }
}

#[test]
fn solve_many_refined_into_is_bit_identical_to_solve_many_refined() {
    let m = kkt();
    let s = solver_for(&m);
    // Both sides of the BLAS3_REFINE_THRESHOLD (16) dispatch.
    for nrhs in [1usize, 2, 20] {
        let wide: Vec<f64> = (0..m.n * nrhs)
            .map(|k| RHS[k % m.n] * (1.0 + (k / m.n) as f64))
            .collect();
        let want = s
            .solve_many_refined(&m, &wide, nrhs)
            .expect("solve_many_refined");
        let mut got = vec![f64::NAN; m.n * nrhs];
        s.solve_many_refined_into(&m, &wide, nrhs, &mut got, RefineOptions::default())
            .expect("solve_many_refined_into");
        assert_eq!(bits(&got), bits(&want), "nrhs = {nrhs}");
    }
}

#[test]
fn into_variants_reject_a_wrong_length_output_slice() {
    let m = kkt();
    let s = solver_for(&m);
    let n = m.n;

    let mut short = vec![0.0; n - 1];
    assert!(matches!(
        s.solve_into(&RHS, &mut short),
        Err(FeralError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        s.solve_refined_into(&m, &RHS, &mut short, RefineOptions::default()),
        Err(FeralError::DimensionMismatch { .. })
    ));

    let wide: Vec<f64> = (0..n * 2).map(|k| RHS[k % n]).collect();
    let mut wrong = vec![0.0; n * 3];
    assert!(matches!(
        s.solve_many_into(&wide, 2, &mut wrong),
        Err(FeralError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        s.solve_many_refined_into(&m, &wide, 2, &mut wrong, RefineOptions::default()),
        Err(FeralError::DimensionMismatch { .. })
    ));
}

#[test]
fn into_variants_reject_a_wrong_length_rhs() {
    let m = kkt();
    let s = solver_for(&m);
    let n = m.n;
    let mut out = vec![0.0; n];
    let short_rhs = vec![1.0; n - 1];
    assert!(matches!(
        s.solve_into(&short_rhs, &mut out),
        Err(FeralError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        s.solve_refined_into(&m, &short_rhs, &mut out, RefineOptions::default()),
        Err(FeralError::DimensionMismatch { .. })
    ));
}

#[test]
fn into_variants_report_no_factor_before_checking_dimensions() {
    // A solver with no factor returns NoFactor even when the output slice
    // is also wrong — same precedence as the allocating entry points.
    let s = Solver::new();
    let m = kkt();
    let mut out = vec![0.0; 3];
    assert!(matches!(
        s.solve_into(&RHS, &mut out),
        Err(FeralError::NoFactor)
    ));
    assert!(matches!(
        s.solve_refined_into(&m, &RHS, &mut out, RefineOptions::default()),
        Err(FeralError::NoFactor)
    ));
    assert!(matches!(
        s.solve_many_into(&RHS, 1, &mut out),
        Err(FeralError::NoFactor)
    ));
    assert!(matches!(
        s.solve_many_refined_into(&m, &RHS, 1, &mut out, RefineOptions::default()),
        Err(FeralError::NoFactor)
    ));
}

#[test]
fn solve_many_into_with_zero_columns_is_a_no_op() {
    let m = kkt();
    let s = solver_for(&m);
    let empty: [f64; 0] = [];
    let mut out: [f64; 0] = [];
    s.solve_many_into(&empty, 0, &mut out).expect("nrhs = 0");
    s.solve_many_refined_into(&m, &empty, 0, &mut out, RefineOptions::default())
        .expect("nrhs = 0");
}
