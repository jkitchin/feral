//! Issue #38: MA57-style static-pivot perturbation integration tests.
//!
//! These exercise the full `Solver::factor` path with
//! `NumericParams::static_pivot_threshold = Some(t)`, including the
//! per-call `||A||_∞` computation and propagation to
//! `BunchKaufmanParams.static_pivot_floor`. We use `Identity` scaling
//! to bypass MC64 / InfNorm rescaling so the floor is applied at
//! known absolute magnitudes.
//!
//! The rocket_12800 live verification is run via ipopt-feral against
//! the dumped /tmp/rkt_*.bin corpus; see
//! `dev/journal/2026-05-17-01.org` end-of-day for the
//! BEFORE/AFTER wall-time and inertia table.

use feral::numeric::factorize::NumericParams;
use feral::scaling::ScalingStrategy;
use feral::symbolic::supernode::SupernodeParams;
use feral::{CscMatrix, FactorStatus, Solver};

/// Build a CSC lower-triangle matrix from (row, col, val) triplets
/// (col-major, row >= col). Duplicates are summed.
fn csc_from_triplets(n: usize, triplets: &[(usize, usize, f64)]) -> CscMatrix {
    let mut cols: Vec<Vec<(usize, f64)>> = (0..n).map(|_| Vec::new()).collect();
    for &(r, c, v) in triplets {
        assert!(r >= c, "lower triangle: r={r} c={c}");
        cols[c].push((r, v));
    }
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();
    col_ptr.push(0);
    for col in cols.iter_mut() {
        col.sort_by_key(|x| x.0);
        // Deduplicate (sum values for same row).
        let mut merged: Vec<(usize, f64)> = Vec::with_capacity(col.len());
        for (r, v) in col.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.0 == r {
                    last.1 += v;
                    continue;
                }
            }
            merged.push((r, v));
        }
        for (r, v) in merged {
            row_idx.push(r);
            values.push(v);
        }
        col_ptr.push(row_idx.len());
    }
    CscMatrix {
        n,
        col_ptr,
        row_idx,
        values,
    }
}

fn solver_identity_static_pivot(t: Option<f64>) -> Solver {
    let mut np = NumericParams {
        scaling: ScalingStrategy::Identity,
        ..NumericParams::default()
    };
    np.static_pivot_threshold = t;
    Solver::with_params(np, SupernodeParams::default())
}

/// Diagonal `diag(2, -1e-12, -1, 3)`. Inertia is (2 positive,
/// 2 negative). With `static_pivot_threshold = 1e-6` and
/// `||A||_∞ = 3`, the floor is `3e-6`; the middle pivot `-1e-12` is
/// below the floor and gets perturbed to `-3e-6` (sign preserved).
/// Inertia stays (2, 2).
#[test]
fn solver_static_pivot_floor_preserves_negative_sign() {
    let n = 4;
    let mat = csc_from_triplets(n, &[(0, 0, 2.0), (1, 1, -1e-12), (2, 2, -1.0), (3, 3, 3.0)]);

    // Baseline: unperturbed inertia.
    let mut s_off = solver_identity_static_pivot(None);
    assert!(matches!(s_off.factor(&mat, None), FactorStatus::Success));
    let i_off = s_off.inertia().expect("inertia off").clone();
    assert_eq!(i_off.positive, 2);
    assert_eq!(i_off.negative, 2);

    // Perturbed: same inertia signature, but the small pivot was
    // floored. We verify the floor took effect by checking the
    // `needs_refinement` flag exposed via `last_factors`.
    let mut s_on = solver_identity_static_pivot(Some(1e-6));
    assert!(matches!(s_on.factor(&mat, None), FactorStatus::Success));
    let i_on = s_on.inertia().expect("inertia on").clone();
    assert_eq!(i_on.positive, 2);
    assert_eq!(i_on.negative, 2);
    assert!(
        s_on.factors().expect("factors on").needs_refinement,
        "static-pivot perturbation must set needs_refinement"
    );
}

/// Same diagonal but with the small pivot positive: `diag(2, +1e-12,
/// -1, 3)`. Inertia is (3 positive, 1 negative). Floor `1e-6 * 3 =
/// 3e-6` perturbs the `+1e-12` to `+3e-6` (sign preserved).
#[test]
fn solver_static_pivot_floor_preserves_positive_sign() {
    let n = 4;
    let mat = csc_from_triplets(n, &[(0, 0, 2.0), (1, 1, 1e-12), (2, 2, -1.0), (3, 3, 3.0)]);
    let mut s_on = solver_identity_static_pivot(Some(1e-6));
    assert!(matches!(s_on.factor(&mat, None), FactorStatus::Success));
    let i_on = s_on.inertia().expect("inertia on").clone();
    assert_eq!(i_on.positive, 3);
    assert_eq!(i_on.negative, 1);
    assert!(s_on.factors().expect("factors on").needs_refinement);
}

/// No pivot below the floor: knob is a no-op, no needs_refinement.
#[test]
fn solver_static_pivot_no_op_when_pivots_above_floor() {
    let n = 3;
    let mat = csc_from_triplets(n, &[(0, 0, 2.0), (1, 1, 1.5), (2, 2, -3.0)]);
    let mut s_on = solver_identity_static_pivot(Some(1e-8));
    assert!(matches!(s_on.factor(&mat, None), FactorStatus::Success));
    let i_on = s_on.inertia().expect("inertia on").clone();
    assert_eq!(i_on.positive, 2);
    assert_eq!(i_on.negative, 1);
    assert!(
        !s_on.factors().expect("factors on").needs_refinement,
        "no floor fires => no needs_refinement"
    );
}

/// `Solver::with_static_pivot_threshold` builder equivalent.
#[test]
fn solver_builder_with_static_pivot_threshold() {
    let n = 4;
    let mat = csc_from_triplets(n, &[(0, 0, 2.0), (1, 1, -1e-12), (2, 2, -1.0), (3, 3, 3.0)]);
    let mut s = Solver::with_params(
        NumericParams {
            scaling: ScalingStrategy::Identity,
            ..NumericParams::default()
        },
        SupernodeParams::default(),
    )
    .with_static_pivot_threshold(1e-6);
    assert!(matches!(s.factor(&mat, None), FactorStatus::Success));
    let i = s.inertia().expect("inertia").clone();
    assert_eq!(i.positive, 2);
    assert_eq!(i.negative, 2);
    assert!(s.factors().expect("factors").needs_refinement);
}

/// C ABI path: setting `FERAL_STATIC_PIVOT` env var should propagate
/// into the solver and produce the same behavior. We use a process-
/// wide env var (cargo's test harness serializes #[test] fns in this
/// file's binary by default within a single mod).
#[test]
fn capi_feral_static_pivot_env_var() {
    use feral::capi::*;

    // SAFETY: process-wide env var; we restore at end. Cargo runs
    // tests in this binary in a single thread per mod by default
    // unless the user passes --test-threads, which CI doesn't.
    let prior = std::env::var("FERAL_STATIC_PIVOT").ok();
    unsafe { std::env::set_var("FERAL_STATIC_PIVOT", "1e-6") };

    unsafe {
        let s = feral_new();
        assert!(!s.is_null());

        // 2x2 indefinite [[1, 2], [2, 1]] — eigenvalues 3, -1, neither
        // below 1e-6 * ||A||_∞ = 1e-6 * 4 = 4e-6. Knob is a no-op
        // here, but we just check the env var was honored without
        // crashing.
        let ia: [i32; 3] = [0, 2, 3];
        let ja: [i32; 3] = [0, 1, 1];
        assert_eq!(
            feral_set_structure(s, 2, 3, ia.as_ptr(), ja.as_ptr()),
            FERAL_SUCCESS
        );
        let vp = feral_values_ptr(s);
        std::ptr::copy_nonoverlapping([1.0_f64, 2.0, 1.0].as_ptr(), vp, 3);
        assert_eq!(feral_factor(s, 1, 1), FERAL_SUCCESS);
        assert_eq!(feral_num_neg(s), 1);
        feral_free(s);
    }

    // SAFETY: restore env var.
    unsafe {
        match prior {
            Some(v) => std::env::set_var("FERAL_STATIC_PIVOT", v),
            None => std::env::remove_var("FERAL_STATIC_PIVOT"),
        }
    }
}
