//! Issue #44 — per-factor cost on the NARX_CFy KKT corpus.
//!
//! #44 reports ipopt-feral timing out at 600 s on `NARX_CFy.nl`: the
//! per-factor cost is ~2.5× MA57 and the IPM does ~485 factorizations.
//! This probe factors the captured KKT snapshots
//! `NARX_CFy_{0000,0001,0002}` with a default `Solver` — the exact
//! path the IPM uses — on the current HEAD (Fix 1 fine-grained delayed
//! pivoting + Fix 2 cancellation-free 2×2 inertia) and reports factor
//! wall time, factor nnz, inertia, and the refined-solve residual, so
//! we can see whether the recent cascade fixes moved per-factor cost.
//!
//! Usage: cargo run --release --bin probe_narx_factor

use std::path::Path;
use std::time::Instant;

use feral::{read_mtx, CscMatrix, Solver};

const DIR: &str = "data/matrices/kkt-mittelmann/NARX_CFy";

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn rel_residual(csc: &CscMatrix, x: &[f64], rhs: &[f64]) -> f64 {
    let mut ax = vec![0.0; csc.n];
    csc.symv(x, &mut ax);
    let r: Vec<f64> = ax.iter().zip(rhs).map(|(&a, &b)| a - b).collect();
    norm2(&r) / norm2(rhs).max(1.0)
}

fn run(iter: usize) {
    let mtx_path = format!("{DIR}/NARX_CFy_{iter:04}.mtx");
    let json_path = format!("{DIR}/NARX_CFy_{iter:04}.json");
    if !Path::new(&mtx_path).exists() {
        eprintln!("SKIP iter {iter}: {mtx_path} not present (corpus gitignored)");
        return;
    }
    let csc = read_mtx(Path::new(&mtx_path))
        .and_then(|m| m.to_csc())
        .expect("load NARX_CFy mtx");
    let oracle: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json_path).expect("read oracle json"))
            .expect("parse oracle json");
    let rhs: Vec<f64> = oracle["rhs"]
        .as_array()
        .expect("rhs array")
        .iter()
        .map(|v| v.as_f64().expect("rhs f64"))
        .collect();

    let mut s = Solver::new();
    let t = Instant::now();
    let status = s.factor(&csc, None);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let fnnz = s.factors().map(|f| f.factor_nnz()).unwrap_or(0);
    let inertia = s
        .inertia()
        .map(|i| format!("({},{},{})", i.positive, i.negative, i.zero))
        .unwrap_or_else(|| "-".to_string());
    let res = s
        .solve_refined(&csc, &rhs)
        .map(|x| rel_residual(&csc, &x, &rhs))
        .unwrap_or(f64::NAN);
    println!(
        "NARX_CFy_{iter:04}  n={:<7} in_nnz={:<9} factor_ms={ms:>9.1}  factor_nnz={fnnz:<11} \
         inertia={inertia:<18} refined_res={res:.2e}  {status:?}",
        csc.n, csc.col_ptr[csc.n]
    );
}

fn main() {
    println!("Issue #44 — NARX_CFy per-factor cost on current HEAD (Fix 1 + Fix 2)\n");
    for iter in 0..3 {
        run(iter);
    }
}
