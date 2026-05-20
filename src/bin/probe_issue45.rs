//! Standalone reproducer for issue #45 — CHO `parmest` KKT back-solve
//! returns garbage despite a successful factor with correct inertia.
//!
//! On the CHO `parmest` KKT system (n=43332, symmetric nnz≈400212,
//! cond₂≈1.4e15 — ill-conditioned but NOT singular) FERAL factors
//! successfully and reports the correct inertia (21660 negatives), yet
//! the back-solve produces ‖A·x−b‖∞ ≈ 1e10–1e18 for a system whose
//! true solution has ‖x‖∞ ≈ 3.24e5. MA57 and SciPy LU solve the same
//! matrix to residual ~1e-5.
//!
//! This binary loads the committed reproducer pair, runs it through a
//! default `Solver`, and prints status / inertia / pivot magnitudes /
//! residual so the bug can be confirmed and traced.
//!
//! Usage:
//!     cargo run --release --bin probe_issue45 -- <kkt.mtx> <rhs.txt>
//! Defaults to the pounce CHO reproducer paths when no args given.

use std::path::Path;
use std::time::Instant;

use feral::numeric::solver::{FactorStatus, Solver};
use feral::read_mtx;

const DEFAULT_MTX: &str =
    "/Users/jkitchin/projects/pounce/benchmarks/cho/feral_repro/cho_iter0_kkt.mtx";
const DEFAULT_RHS: &str =
    "/Users/jkitchin/projects/pounce/benchmarks/cho/feral_repro/cho_iter0_rhs.txt";

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mtx_path = args.next().unwrap_or_else(|| DEFAULT_MTX.to_string());
    let rhs_path = args.next().unwrap_or_else(|| DEFAULT_RHS.to_string());

    if !Path::new(&mtx_path).exists() {
        eprintln!("SKIP: {mtx_path} not present");
        std::process::exit(2);
    }

    let mtx = match read_mtx(Path::new(&mtx_path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("read_mtx failed: {e:?}");
            std::process::exit(1);
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("to_csc failed: {e:?}");
            std::process::exit(1);
        }
    };

    let rhs_text = match std::fs::read_to_string(&rhs_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read rhs {rhs_path}: {e}");
            std::process::exit(1);
        }
    };
    let rhs: Vec<f64> = rhs_text
        .split_whitespace()
        .filter_map(|tok| tok.parse::<f64>().ok())
        .collect();

    println!("matrix : {mtx_path}");
    println!("  n={}, nnz={}", csc.n, csc.col_ptr[csc.n]);
    println!("rhs    : {rhs_path}");
    println!("  len={}, ||b||_inf={:.6e}", rhs.len(), norm_inf(&rhs));
    if rhs.len() != csc.n {
        eprintln!("MISMATCH: rhs len {} != n {}", rhs.len(), csc.n);
        std::process::exit(1);
    }

    let mut solver = Solver::new();
    let t0 = Instant::now();
    let status = solver.factor(&csc, None);
    let fac_s = t0.elapsed().as_secs_f64();

    println!("\n--- factor ---");
    println!("  status        : {status:?}");
    println!("  factor time   : {fac_s:.3} s");
    if let Some(i) = solver.inertia() {
        println!(
            "  inertia       : pos={}, neg={}, zero={} (sum={})",
            i.positive,
            i.negative,
            i.zero,
            i.positive + i.negative + i.zero
        );
    }
    println!("  num_negative  : {}", solver.num_negative_eigenvalues());
    println!("  min_pivot_mag : {:?}", solver.min_pivot_magnitude());
    println!("  max_pivot_mag : {:?}", solver.max_pivot_magnitude());
    println!("  min_diagonal  : {:?}", solver.min_diagonal());
    if let (Some(lo), Some(hi)) = (solver.min_pivot_magnitude(), solver.max_pivot_magnitude()) {
        if lo > 0.0 {
            println!("  pivot spread  : {:.3e}", hi / lo);
        }
    }

    if !matches!(
        status,
        FactorStatus::Success | FactorStatus::WrongInertia { .. }
    ) {
        eprintln!("\nfactor did not produce usable factors; stopping");
        std::process::exit(0);
    }

    // Plain back-solve.
    let x = match solver.solve(&rhs) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("solve failed: {e:?}");
            std::process::exit(1);
        }
    };
    let mut ax = vec![0.0; csc.n];
    csc.symv(&x, &mut ax);
    let resid: Vec<f64> = ax.iter().zip(&rhs).map(|(&a, &b)| a - b).collect();
    let abs_res = norm_inf(&resid);
    let rel_res = abs_res / norm_inf(&rhs).max(1.0);

    println!("\n--- solve (plain) ---");
    println!("  ||x||_inf       : {:.6e}", norm_inf(&x));
    println!("  ||A x - b||_inf : {:.6e}", abs_res);
    println!("  rel residual    : {rel_res:.6e}");
    println!("  (true ||x||_inf ~ 3.24e5; MA57/SciPy rel res ~ 1e-5)");

    // Refined back-solve.
    match solver.solve_refined(&csc, &rhs) {
        Ok(xr) => {
            let mut axr = vec![0.0; csc.n];
            csc.symv(&xr, &mut axr);
            let rr: Vec<f64> = axr.iter().zip(&rhs).map(|(&a, &b)| a - b).collect();
            println!("\n--- solve_refined ---");
            println!("  ||x||_inf       : {:.6e}", norm_inf(&xr));
            println!("  ||A x - b||_inf : {:.6e}", norm_inf(&rr));
            println!(
                "  rel residual    : {:.6e}",
                norm_inf(&rr) / norm_inf(&rhs).max(1.0)
            );
        }
        Err(e) => println!("\nsolve_refined failed: {e:?}"),
    }

    let confirmed = abs_res > 1.0 || norm_inf(&x) > 1e9;
    println!(
        "\nissue #45 {}",
        if confirmed {
            "CONFIRMED — back-solve residual is catastrophically large"
        } else {
            "NOT reproduced — back-solve looks healthy"
        }
    );
}
