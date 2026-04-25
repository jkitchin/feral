//! Triage the bratu3d plain-solve numerical-bug found via
//! `tests/large_matrix_smoke.rs`. Documented in
//! `dev/journal/2026-04-25-02.org`.
//!
//! Symptom: under default `Solver` parameters, plain `solve()` on
//! `tests/data/large/bratu3d.mtx` (n=27,792, GHS_indef) returns
//! `||A x - b|| / ||b|| = 4.66e6` while `solve_refined` recovers to
//! 2.46e-9. No FactorStatus::Singular, no WrongInertia.
//!
//! This example tests the four hypotheses from the journal:
//!   1. Is the factor itself wrong (huge L entries / tiny D / NaN/Inf)?
//!   2. Does `ScalingStrategy::InfNorm` fix the plain solve?
//!   3. Does `ScalingStrategy::Mc64Symmetric` fix the plain solve?
//!   4. Does raising `bk.pivot_threshold` from 0.0 fix it?
//!
//! For each configuration we report:
//!   - factor wall (s)
//!   - inertia
//!   - scaling_info
//!   - max |L|, min |D|, max |D| (over all supernodes), nan/inf counts
//!   - plain   ||A x - b|| / ||b||
//!   - refined ||A x - b|| / ||b||
//!
//! Run with: cargo run --release --example triage_bratu3d

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::NumericParams;
use feral::scaling::ScalingStrategy;
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, FactorStatus, Solver, ZeroPivotAction};

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        let r = ax[i] - b[i];
        rs += r * r;
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

fn factor_stats(solver: &Solver) -> (f64, f64, f64, usize, usize) {
    let factors = solver.factors().expect("factor present");
    let mut max_l: f64 = 0.0;
    let mut min_d_abs: f64 = f64::INFINITY;
    let mut max_d_abs: f64 = 0.0;
    let mut nan_count: usize = 0;
    let mut inf_count: usize = 0;

    for nf in &factors.node_factors {
        let ff = &nf.frontal_factors;
        for &v in &ff.l {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else {
                let a = v.abs();
                if a > max_l {
                    max_l = a;
                }
            }
        }
        for &v in &ff.d_diag[..ff.nelim] {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else {
                let a = v.abs();
                if a > max_d_abs {
                    max_d_abs = a;
                }
                if a > 0.0 && a < min_d_abs {
                    min_d_abs = a;
                }
            }
        }
        // d_subdiag entries that are nonzero indicate 2x2 pivots; for
        // those the magnitude bound is on the 2x2 block. We just track
        // the |off-diag| as another contribution.
        for &v in &ff.d_subdiag[..ff.nelim] {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else {
                let a = v.abs();
                if a > max_d_abs {
                    max_d_abs = a;
                }
            }
        }
    }
    if !min_d_abs.is_finite() {
        min_d_abs = 0.0;
    }
    (max_l, min_d_abs, max_d_abs, nan_count, inf_count)
}

fn run(name: &str, np: NumericParams, csc: &CscMatrix, rhs: &[f64]) {
    println!("\n=== {} ===", name);
    println!(
        "  scaling      = {:?}",
        match &np.scaling {
            ScalingStrategy::Identity => "Identity".to_string(),
            ScalingStrategy::InfNorm => "InfNorm".to_string(),
            ScalingStrategy::Mc64Symmetric => "Mc64Symmetric".to_string(),
            ScalingStrategy::Auto => "Auto".to_string(),
            ScalingStrategy::External(_) => "External".to_string(),
        }
    );
    println!("  pivot_thresh = {:.3e}", np.bk.pivot_threshold);
    println!("  on_zero      = {:?}", np.bk.on_zero_pivot);

    let mut solver = Solver::with_params(np, Default::default());

    let t = Instant::now();
    let status = solver.factor(csc, None);
    let factor_s = t.elapsed().as_secs_f64();

    print!("  factor       = {:.2}s, status = ", factor_s);
    match &status {
        FactorStatus::Success => println!("Success"),
        FactorStatus::WrongInertia { actual, expected } => {
            println!("WrongInertia(act={:?} exp={:?})", actual, expected)
        }
        FactorStatus::Singular => {
            println!("Singular (skip rest)");
            return;
        }
        FactorStatus::FatalError(e) => {
            println!("FatalError: {}", e);
            return;
        }
    }

    if let Some(f) = solver.factors() {
        println!("  scaling_info = {:?}", f.scaling_info);
        println!("  needs_refine = {}", f.needs_refinement);
    }

    let (max_l, min_d, max_d, nan_count, inf_count) = factor_stats(&solver);
    println!(
        "  max|L| = {:.3e}  min|D| = {:.3e}  max|D| = {:.3e}",
        max_l, min_d, max_d
    );
    println!("  NaN entries = {}, Inf entries = {}", nan_count, inf_count);

    let t = Instant::now();
    match solver.solve(rhs) {
        Ok(x) => {
            let plain_t = t.elapsed().as_secs_f64();
            let rel = rel_residual(csc, &x, rhs);
            println!("  plain solve  = {:.3}s, rel_res = {:.3e}", plain_t, rel);
        }
        Err(e) => println!("  plain solve failed: {}", e),
    }

    let t = Instant::now();
    match solver.solve_refined(csc, rhs) {
        Ok(x) => {
            let ref_t = t.elapsed().as_secs_f64();
            let rel = rel_residual(csc, &x, rhs);
            println!("  refined      = {:.3}s, rel_res = {:.3e}", ref_t, rel);
        }
        Err(e) => println!("  refined failed: {}", e),
    }
}

fn main() {
    let path = Path::new("tests/data/large/bratu3d.mtx");
    if !path.exists() {
        eprintln!("SKIP: {} not found.", path.display());
        return;
    }

    let mtx = read_mtx(path).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let n = csc.n;
    let nnz = csc.row_idx.len();
    println!("bratu3d: n={}, nnz={}", n, nnz);

    let ones = vec![1.0f64; n];
    let mut rhs = vec![0.0f64; n];
    csc.symv(&ones, &mut rhs);
    let b_norm = (rhs.iter().map(|x| x * x).sum::<f64>()).sqrt();
    println!("||b||_2 = ||A * 1||_2 = {:.6e}", b_norm);

    // (A) Solver default — what plain `Solver::new()` does.
    run(
        "A: Solver default (Auto scaling, threshold=0.0, on_zero=Fail)",
        NumericParams::default(),
        &csc,
        &rhs,
    );

    // (B) InfNorm scaling, default pivot threshold.
    run(
        "B: ScalingStrategy::InfNorm, threshold=0.0",
        NumericParams {
            scaling: ScalingStrategy::InfNorm,
            ..Default::default()
        },
        &csc,
        &rhs,
    );

    // (C) Mc64Symmetric scaling, default pivot threshold.
    run(
        "C: ScalingStrategy::Mc64Symmetric, threshold=0.0",
        NumericParams {
            scaling: ScalingStrategy::Mc64Symmetric,
            ..Default::default()
        },
        &csc,
        &rhs,
    );

    // (D) Default scaling, raised pivot threshold (FERAL Stage 2 first jump).
    run(
        "D: Auto scaling, threshold=0.01 (FERAL Stage 2 first jump)",
        NumericParams {
            bk: BunchKaufmanParams {
                pivot_threshold: 0.01,
                on_zero_pivot: ZeroPivotAction::ForceAccept,
                ..BunchKaufmanParams::default()
            },
            ..Default::default()
        },
        &csc,
        &rhs,
    );

    // (E) Identity scaling, default threshold — to confirm what the
    //     user's "default" actually was if they got a 4.66e6 plain
    //     residual under Solver::new().
    run(
        "E: ScalingStrategy::Identity, threshold=0.0",
        NumericParams {
            scaling: ScalingStrategy::Identity,
            ..Default::default()
        },
        &csc,
        &rhs,
    );
}
