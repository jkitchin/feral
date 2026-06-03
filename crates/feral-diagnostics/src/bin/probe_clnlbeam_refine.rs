//! Probe: what relative residual does iterative refinement actually
//! reach on clnlbeam_{0000,0001}? Used to validate the proposed
//! refinement-gate tightening (task #65,
//! `dev/journal/2026-05-17-01.org` §07:01).
//!
//! Builds a synthetic RHS `b = A·x_exact` where `x_exact` is a
//! deterministic pseudo-random vector, runs the diagnostic refinement
//! path, and prints per-step relative residual + the κ̂·rr forward
//! error bound. The current gate is `ε·√n` (~7e-14 for clnlbeam,
//! n=99999); MA57's de-facto target is ~1e-15. If refinement easily
//! reaches ~ε we know tightening the gate is safe; if it stagnates
//! at 1e-14 the floor is real and tightening just wastes steps.
//!
//! Usage:
//!     cargo run --release --bin probe_clnlbeam_refine -- [problem] [iter]

use std::env;
use std::path::Path;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::solve_sparse_refined_with_diagnostics;
use feral::read_mtx;
use feral::symbolic::{symbolic_factorize, SupernodeParams};

fn main() {
    let problem = env::args().nth(1).unwrap_or_else(|| "clnlbeam".to_string());
    let iter: usize = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mtx_path = format!("data/matrices/kkt-mittelmann/{problem}/{problem}_{iter:04}.mtx");
    if !Path::new(&mtx_path).exists() {
        eprintln!("SKIP: {mtx_path} not present");
        std::process::exit(2);
    }

    let mtx = match read_mtx(Path::new(&mtx_path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("read_mtx error: {e:?}");
            std::process::exit(1);
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("to_csc error: {e:?}");
            std::process::exit(1);
        }
    };
    let n = csc.n;
    println!("# {problem} iter {iter}: n={n}, nnz={}", csc.row_idx.len());

    // Deterministic pseudo-random x_exact and RHS b = A·x_exact.
    let mut x_exact = vec![0.0_f64; n];
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    for v in x_exact.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
    }
    let mut b = vec![0.0_f64; n];
    csc.symv(&x_exact, &mut b);

    let sym = match symbolic_factorize(&csc, &SupernodeParams::default()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("symbolic error: {e:?}");
            std::process::exit(1);
        }
    };
    let np = NumericParams::default();
    let (factors, _) = match factorize_multifrontal(&csc, &sym, &np) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("factor error: {e:?}");
            std::process::exit(1);
        }
    };

    let (_x, diag) = match solve_sparse_refined_with_diagnostics(&csc, &factors, &b) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("refine error: {e:?}");
            std::process::exit(1);
        }
    };
    println!(
        "# κ̂_1 ≈ {:.3e}, ||A||_1 = {:.3e}",
        diag.kappa_1_est, diag.anorm_1
    );
    println!(
        "# current gate = ε·√n = {:.3e}",
        f64::EPSILON * (n as f64).sqrt()
    );
    println!("# proposed gate = ε     = {:.3e}", f64::EPSILON);
    println!();
    println!(
        "{:>5}  {:>14}  {:>14}  {:>14}  {:>8}",
        "step", "||r||_2", "rel_res", "kappa*rr", "improved"
    );
    for s in &diag.steps {
        println!(
            "{:>5}  {:>14.3e}  {:>14.3e}  {:>14.3e}  {:>8}",
            s.step, s.residual_2norm, s.relative_residual, s.forward_error_bound, s.improved
        );
    }
}
