//! diag_mc64_reuse_threat — does reusing a stale MC64 *scaling vector*
//! actually corrupt inertia?
//!
//! The value-bound gate (`src/scaling/value_bound.rs`) exists to stop
//! issue #38. But the two artifacts issue #38 names are not the same
//! object:
//!
//!   * `symbolic.cached_mc64` — the MC64 *matching*, which feeds the
//!     `LdltCompress` permutation and therefore changes the
//!     elimination structure. `mc64_cache_invalidated_after_factor_issue_38`
//!     guards this one.
//!   * `Solver::mc64_scaling_cache` — the *scaling vector* `D`. Values
//!     only; the pattern and the ordering are untouched.
//!
//! The value-bound gate guards the second. The in-tree guard test
//! `mc64_cache_rejected_on_value_drift_issue_38_guard` asserts only
//! that the gate *rejects* on drift — it never establishes that
//! reusing would have been wrong. This probe asks that question
//! directly: force `ScalingStrategy::External(D0)` onto a drifted
//! matrix and check the inertia against an oracle computed
//! independently of feral's factorization.
//!
//! Oracle: for a symmetric tridiagonal matrix the LDL^T scalar
//! recurrence `d_1 = a_1`, `d_k = a_k - b_{k-1}^2 / d_{k-1}` is a
//! Sturm sequence — the signs of `d_k` give the inertia directly
//! (Golub & Van Loan, Matrix Computations, 4th ed., §8.4.1). This is
//! arithmetic on the matrix entries, not a call into the solver.

use feral::scaling::ScalingStrategy;
use feral::{CscMatrix, FactorStatus, NumericParams, Solver};

/// Symmetric tridiagonal `n x n`, lower triangle, in CSC.
fn tridiag(n: usize, diag: f64, off: f64) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        rows.push(j);
        cols.push(j);
        vals.push(diag);
        if j + 1 < n {
            rows.push(j + 1);
            cols.push(j);
            vals.push(off);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("valid CSC")
}

/// Inertia of a symmetric tridiagonal via the Sturm/LDL^T recurrence.
/// Independent of feral's factorization. Returns (pos, neg, zero).
fn sturm_inertia(n: usize, diag: f64, off: f64) -> (usize, usize, usize) {
    let (mut pos, mut neg, mut zero) = (0usize, 0usize, 0usize);
    let mut d = diag;
    for k in 0..n {
        if k > 0 {
            d = diag - off * off / d;
        }
        // Standard Sturm convention: an exactly-zero pivot is perturbed
        // to a tiny POSITIVE value and counted positive, then the
        // recurrence continues. A zero pivot is not a zero eigenvalue —
        // tridiag(4, 10, 10) = 10·tridiag(4, 1, 1) has eigenvalues
        // 10(1 + 2cos(kπ/5)) = 26.18, 16.18, 3.82, -6.18, i.e. (3,1,0),
        // even though d_2 = 10 - 100/10 = 0 exactly.
        if d == 0.0 {
            d = f64::MIN_POSITIVE;
        }
        if d > 0.0 {
            pos += 1;
        } else if d < 0.0 {
            neg += 1;
        } else {
            zero += 1;
        }
    }
    (pos, neg, zero)
}

fn main() {
    let n = 4;
    let base_diag = 10.0;
    let base_off = 1.0;

    // Iterate 0: install the cache from the undrifted matrix.
    let a0 = tridiag(n, base_diag, base_off);

    println!(
        "{:<10}{:>16}{:>16}{:>16}{:>10}",
        "off", "sturm(p,n,z)", "fresh(p,n,z)", "reuseD0(p,n,z)", "verdict"
    );

    // Sweep the off-diagonal upward: this is the axis the in-tree
    // guard test drifts along (1.0 -> 50.0), extended far past it.
    for &off in &[
        1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 1e2, 1e3, 1e4, 1e6, 1e9, 1e12,
    ] {
        let a1 = tridiag(n, base_diag, off);
        let (sp, sn_, sz) = sturm_inertia(n, base_diag, off);

        // Arm A: fresh MC64 on the drifted matrix.
        let mut solver_fresh = Solver::new().with_scaling(ScalingStrategy::Mc64Symmetric);
        let fresh = match solver_fresh.factor(&a1, None) {
            FactorStatus::Success | FactorStatus::WrongInertia { .. } => solver_fresh.inertia(),
            _ => None,
        };

        // Arm B: iterate-0's scaling forced onto the drifted matrix.
        let mut solver_d0 = Solver::new().with_scaling(ScalingStrategy::Mc64Symmetric);
        if !matches!(solver_d0.factor(&a0, None), FactorStatus::Success) {
            println!("{off:<10.0e}  baseline factor failed");
            continue;
        }
        let d0 = match solver_d0.factors() {
            Some(f) => f.scaling.clone(),
            None => {
                println!("{off:<10.0e}  no baseline scaling");
                continue;
            }
        };
        let np = NumericParams {
            scaling: ScalingStrategy::External(d0),
            ..NumericParams::default()
        };
        let mut solver_reuse = Solver::with_params(np, Default::default());
        let reuse = match solver_reuse.factor(&a1, None) {
            FactorStatus::Success | FactorStatus::WrongInertia { .. } => solver_reuse.inertia(),
            _ => None,
        };

        let fmt = |i: Option<&feral::Inertia>| match i {
            Some(i) => format!("({},{},{})", i.positive, i.negative, i.zero),
            None => "FAIL".to_string(),
        };
        let oracle = format!("({sp},{sn_},{sz})");
        let f = fmt(fresh);
        let r = fmt(reuse);
        let verdict = if r != oracle && f == oracle {
            "REUSE WRONG"
        } else if f != oracle {
            "fresh wrong"
        } else {
            "ok"
        };
        println!("{off:<10.0e}{oracle:>16}{f:>16}{r:>16}{verdict:>10}");
    }
}
