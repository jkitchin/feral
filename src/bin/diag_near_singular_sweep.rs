//! Issue #31 sweep: factor the parametric `near_singular_eps_<p>`
//! matrices for p ∈ {6..14} and report the detection boundary of
//! feral's default Bunch-Kaufman threshold.
//!
//! For each matrix:
//!   1. Reads `external_benchmarks/stress/matrices/synth/near_singular_eps_<p>.mtx`.
//!   2. Factors with default `Solver::new()` settings (no inertia check).
//!   3. Constructs RHS b = A * x_true with x_true = (1, 1, ..., 1)^T,
//!      solves with iterative refinement, and computes
//!      rel_res = ||A x - b||_2 / ||b||_2.
//!   4. Reports the factor status, inertia (pos/neg/zero), pivot
//!      threshold, and rel_res.
//!
//! The "boundary" is the smallest p at which `inertia.zero == 0`
//! (feral fails to flag the one expected null pivot).
//!
//! Usage: `cargo run --release --bin diag_near_singular_sweep`

use std::path::PathBuf;

use feral::{read_mtx, CscMatrix, FactorStatus, Solver};

fn rel_res_2norm(csc: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = csc.n;
    let mut r = b.iter().map(|v| -v).collect::<Vec<f64>>();
    for j in 0..n {
        for p in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[p];
            let a = csc.values[p];
            r[i] += a * x[j];
            if i != j {
                r[j] += a * x[i];
            }
        }
    }
    let rn: f64 = r.iter().map(|v| v * v).sum();
    let bn: f64 = b.iter().map(|v| v * v).sum();
    if bn == 0.0 {
        0.0
    } else {
        (rn / bn).sqrt()
    }
}

fn spmv(csc: &CscMatrix, x: &[f64]) -> Vec<f64> {
    let n = csc.n;
    let mut y = vec![0.0; n];
    for j in 0..n {
        for p in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[p];
            let a = csc.values[p];
            y[i] += a * x[j];
            if i != j {
                y[j] += a * x[i];
            }
        }
    }
    y
}

fn min_abs_diag(csc: &CscMatrix) -> f64 {
    let n = csc.n;
    let mut m = f64::INFINITY;
    for j in 0..n {
        for p in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            if csc.row_idx[p] == j {
                let v = csc.values[p].abs();
                if v < m {
                    m = v;
                }
            }
        }
    }
    m
}

fn main() {
    let stem = "external_benchmarks/stress/matrices/synth";
    println!(
        "{:>4}  {:<14}  {:>5}  {:>5}  {:>5}  {:>10}  {:>12}  {:>10}",
        "p", "status", "pos", "neg", "zero", "min|D_ii|", "rel_res", "pivtol"
    );
    println!("{}", "-".repeat(80));

    let mut boundary: Option<u32> = None;
    for p in 6..=14u32 {
        let path = PathBuf::from(format!("{}/near_singular_eps_{}.mtx", stem, p));
        let mtx = match read_mtx(&path) {
            Ok(m) => m,
            Err(e) => {
                println!("p={:>2}: read_mtx({}) failed: {:?}", p, path.display(), e);
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                println!("p={:>2}: to_csc failed: {:?}", p, e);
                continue;
            }
        };
        drop(mtx);

        let x_true = vec![1.0_f64; csc.n];
        let b = spmv(&csc, &x_true);

        let mut solver = Solver::new();
        let pivtol = solver.pivot_threshold();
        let status = solver.factor(&csc, None);
        let status_label = match &status {
            FactorStatus::Success => "Success",
            FactorStatus::Singular => "Singular",
            FactorStatus::WrongInertia { .. } => "WrongInertia",
            FactorStatus::FatalError(_) => "FatalError",
        };

        let inertia = solver.inertia().cloned().unwrap_or(feral::Inertia {
            positive: 0,
            negative: 0,
            zero: 0,
        });

        let mind = solver.min_diagonal().unwrap_or(f64::NAN);

        let rel = match status {
            FactorStatus::Success | FactorStatus::WrongInertia { .. } => {
                match solver.solve_refined(&csc, &b) {
                    Ok(x) => rel_res_2norm(&csc, &x, &b),
                    Err(_) => f64::NAN,
                }
            }
            _ => f64::NAN,
        };

        println!(
            "{:>4}  {:<14}  {:>5}  {:>5}  {:>5}  {:>10.2e}  {:>12.3e}  {:>10.2e}",
            p, status_label, inertia.positive, inertia.negative, inertia.zero, mind, rel, pivtol,
        );

        // Boundary = first p where feral stops reporting the null pivot
        // (zero == 0) despite the matrix containing one eigenvalue at 10^-p.
        if boundary.is_none() && inertia.zero == 0 {
            boundary = Some(p);
        }
    }
    println!();
    match boundary {
        Some(p) => println!(
            "detection boundary: p = {} (first p with inertia.zero == 0)",
            p
        ),
        None => println!("detection boundary: not reached in p ∈ [6, 14]"),
    }

    // Sanity reporting of min-|diagonal| from the raw matrix (pre-factor),
    // to show that the small eigenvalue is not necessarily a small diagonal.
    println!();
    println!("raw matrix min |A_ii| (before factor):");
    for p in 6..=14u32 {
        let path = PathBuf::from(format!("{}/near_singular_eps_{}.mtx", stem, p));
        if let Ok(mtx) = read_mtx(&path) {
            if let Ok(csc) = mtx.to_csc() {
                println!("  p={:>2}: min|A_ii| = {:.3e}", p, min_abs_diag(&csc));
            }
        }
    }

    // Cross-check on the canonical stress matrices already in the manifest
    // (`near_singular_eps9`, seed=5; `near_singular_eps12`, seed=6). These
    // use a different RNG seed than the sweep above, so they probe an
    // independent random basis Q at the same eps_pow.
    println!();
    println!("manifest stress matrices (different RNG seeds):");
    println!(
        "{:<22}  {:<14}  {:>5}  {:>5}  {:>5}  {:>10}  {:>12}",
        "matrix", "status", "pos", "neg", "zero", "min|D_ii|", "rel_res"
    );
    for name in ["near_singular_eps9", "near_singular_eps12"] {
        let path = PathBuf::from(format!("{}/{}.mtx", stem, name));
        let mtx = match read_mtx(&path) {
            Ok(m) => m,
            Err(e) => {
                println!("  {} read failed: {:?}", name, e);
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                println!("  {} to_csc failed: {:?}", name, e);
                continue;
            }
        };
        drop(mtx);
        let x_true = vec![1.0_f64; csc.n];
        let b = spmv(&csc, &x_true);
        let mut solver = Solver::new();
        let status = solver.factor(&csc, None);
        let label = match &status {
            FactorStatus::Success => "Success",
            FactorStatus::Singular => "Singular",
            FactorStatus::WrongInertia { .. } => "WrongInertia",
            FactorStatus::FatalError(_) => "FatalError",
        };
        let inertia = solver.inertia().cloned().unwrap_or(feral::Inertia {
            positive: 0,
            negative: 0,
            zero: 0,
        });
        let mind = solver.min_diagonal().unwrap_or(f64::NAN);
        let rel = match status {
            FactorStatus::Success | FactorStatus::WrongInertia { .. } => {
                match solver.solve_refined(&csc, &b) {
                    Ok(x) => rel_res_2norm(&csc, &x, &b),
                    Err(_) => f64::NAN,
                }
            }
            _ => f64::NAN,
        };
        println!(
            "{:<22}  {:<14}  {:>5}  {:>5}  {:>5}  {:>10.2e}  {:>12.3e}",
            name, label, inertia.positive, inertia.negative, inertia.zero, mind, rel
        );
    }
}
