//! Measure the MC64 vs InfNorm scaling-vector spread across the parity
//! corpus, and the resulting solve residual under each scaling.
//!
//! Purpose: find a safe threshold for a "catastrophic MC64 scaling"
//! guard. The CHO KKT produces an MC64 scaling spanning ~1e83 and a
//! garbage solve; this probe shows where legitimate matrices sit so
//! the guard can be set without regressing them.
//!
//! Usage: cargo run --release --bin probe_mc64_spread

use std::path::Path;

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse;
use feral::scaling::{compute_scaling, ScalingStrategy};
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, CscMatrix, NumericParams};

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

fn spread(s: &[f64]) -> f64 {
    let mut mn = f64::INFINITY;
    let mut mx = 0.0_f64;
    for &x in s {
        let a = x.abs();
        if a > 0.0 && a < mn {
            mn = a;
        }
        if a > mx {
            mx = a;
        }
    }
    if mn.is_finite() && mn > 0.0 {
        mx / mn
    } else {
        f64::INFINITY
    }
}

/// Factor with the given explicit scaling and return the relative
/// residual for a unit RHS (NaN on solve failure).
fn relres(m: &CscMatrix, scaling: ScalingStrategy) -> f64 {
    let snode = SupernodeParams::default();
    let np = NumericParams {
        scaling,
        ..NumericParams::default()
    };
    let sym = match symbolic_factorize_with_method(m, &snode, OrderingMethod::Auto) {
        Ok(s) => s,
        Err(_) => return f64::NAN,
    };
    let (factors, _) = match factorize_multifrontal(m, &sym, &np) {
        Ok(fi) => fi,
        Err(_) => return f64::NAN,
    };
    let rhs = vec![1.0_f64; m.n];
    match solve_sparse(&factors, &rhs) {
        Ok(x) => {
            let mut ax = vec![0.0; m.n];
            m.symv(&x, &mut ax);
            let r: Vec<f64> = ax.iter().zip(&rhs).map(|(&a, &b)| a - b).collect();
            norm_inf(&r) / norm_inf(&rhs).max(1.0)
        }
        Err(_) => f64::NAN,
    }
}

fn main() {
    let root = Path::new("tests/data/parity");
    if !root.exists() {
        eprintln!("SKIP: {} not present", root.display());
        std::process::exit(2);
    }
    let mut dirs: Vec<_> = std::fs::read_dir(root)
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    dirs.sort();

    println!(
        "{:<22} {:>8} {:>11} {:>11} {:>11} {:>11}",
        "matrix", "n", "in_spread", "mc_spread", "in_relres", "mc_relres"
    );
    for d in dirs {
        if !d.is_dir() {
            continue;
        }
        // One representative .mtx per family.
        let mut mtx: Option<std::path::PathBuf> = None;
        if let Ok(rd) = std::fs::read_dir(&d) {
            let mut files: Vec<_> = rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().map(|x| x == "mtx").unwrap_or(false))
                .collect();
            files.sort();
            mtx = files.into_iter().next();
        }
        let Some(mtx) = mtx else { continue };
        let m = match read_mtx(&mtx).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let name = d.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        let in_s = compute_scaling(&m, &ScalingStrategy::InfNorm)
            .map(|(v, _)| spread(&v))
            .unwrap_or(f64::NAN);
        let mc_s = compute_scaling(&m, &ScalingStrategy::Mc64Symmetric)
            .map(|(v, _)| spread(&v))
            .unwrap_or(f64::NAN);
        let in_r = relres(&m, ScalingStrategy::InfNorm);
        let mc_r = relres(&m, ScalingStrategy::Mc64Symmetric);
        let flag = if mc_r.is_nan() || mc_r > 1.0 {
            "  <-- MC64 BAD"
        } else {
            ""
        };
        println!(
            "{name:<22} {:>8} {in_s:>11.2e} {mc_s:>11.2e} {in_r:>11.2e} {mc_r:>11.2e}{flag}",
            m.n
        );
    }
}
