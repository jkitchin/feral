//! Find the ACOPP30 matrix where `factor_single_front` and
//! `factorize_multifrontal` disagree on inertia.
//!
//! Bench 2026-04-14-01 reported 68 dense ACOPP30 failures vs 67 sparse,
//! meaning exactly 1 matrix is dense-only. This script identifies it and
//! dumps both inertias plus residuals for comparison.

use std::path::Path;

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    factor_single_front, read_mtx, read_sidecar, solve_refined, BunchKaufmanParams, Inertia,
    ZeroPivotAction,
};

fn rel_residual(a: &feral::CscMatrix, x: &[f64], b: &[f64]) -> f64 {
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

fn main() {
    let dir = Path::new("data/matrices/kkt/ACOPP30");
    let mut entries: Vec<String> = std::fs::read_dir(dir)
        .expect("read ACOPP30 dir")
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("mtx") {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
        .collect();
    entries.sort();

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };

    let mut diverge = Vec::new();
    let mut dense_only_fail = Vec::new();
    let mut sparse_only_fail = Vec::new();

    println!("scanning {} ACOPP30 matrices...", entries.len());
    for name in &entries {
        let mtx_path = dir.join(format!("{}.mtx", name));
        let json_path = dir.join(format!("{}.json", name));
        let mtx = match read_mtx(&mtx_path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(_) => continue,
        };
        let dense_mat = mtx.to_dense();
        let sc = match read_sidecar(&json_path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let rhs = match sc.finite_rhs() {
            Some(r) => r,
            None => continue,
        };

        let expected = Inertia {
            positive: sc.inertia.positive,
            negative: sc.inertia.negative,
            zero: sc.inertia.zero,
        };

        let dense = factor_single_front(&dense_mat, &params);
        let sym = symbolic_factorize(&csc, &SupernodeParams::default());
        let np = feral::numeric::factorize::NumericParams::with_bk(params.clone());
        let sparse = sym
            .as_ref()
            .ok()
            .and_then(|s| factorize_multifrontal(&csc, s, &np).ok());

        match (dense, sparse) {
            (Ok((df, di)), Some((sf, si))) => {
                let dmatch = di == expected;
                let smatch = si == expected;
                if di != si {
                    let dx = solve_refined(&dense_mat, &df, &rhs).unwrap_or_default();
                    let sx = solve_sparse_refined(&csc, &sf, &rhs).unwrap_or_default();
                    let dres = if dx.is_empty() {
                        f64::NAN
                    } else {
                        rel_residual(&csc, &dx, &rhs)
                    };
                    let sres = if sx.is_empty() {
                        f64::NAN
                    } else {
                        rel_residual(&csc, &sx, &rhs)
                    };
                    diverge.push((name.clone(), di, si, expected, dres, sres));
                }
                if dmatch != smatch {
                    if !dmatch {
                        dense_only_fail.push(name.clone());
                    } else {
                        sparse_only_fail.push(name.clone());
                    }
                }
            }
            _ => {}
        }
    }

    println!("\n=== Inertia divergences ({}) ===", diverge.len());
    for (name, di, si, exp, dres, sres) in &diverge {
        println!(
            "{:20} dense={} sparse={} expected={} dres={:.2e} sres={:.2e}",
            name, di, si, exp, dres, sres
        );
    }

    println!("\n=== Dense-only failures ({}) ===", dense_only_fail.len());
    for name in &dense_only_fail {
        println!("  {}", name);
    }

    println!(
        "\n=== Sparse-only failures ({}) ===",
        sparse_only_fail.len()
    );
    for name in &sparse_only_fail {
        println!("  {}", name);
    }
}
