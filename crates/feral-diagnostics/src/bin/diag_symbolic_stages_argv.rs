//! Per-stage symbolic profiler over matrices named on the command line.
//!
//! `diag_symbolic_stages` hardcodes two small matrices. This variant
//! takes paths as argv so the same breakdown can be pulled for an
//! arbitrary matrix -- written to attribute the 60x spread in
//! analyse-cost-per-nonzero across the chain KKT corpus
//! (clnlbeam_0000 19.2 us/nnz vs dtoc1nd_0001 0.20 us/nnz).
//!
//! Usage: diag_symbolic_stages_argv <a.mtx> [b.mtx ...]

use std::path::Path;
use std::sync::{Arc, Mutex};

use feral::read_mtx;
use feral::symbolic::{
    symbolic_factorize_with_method, OrderingMethod, SupernodeParams, SymbolicProfiler,
};

const N_REPEAT: usize = 3;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: diag_symbolic_stages_argv <a.mtx> [b.mtx ...]");
        std::process::exit(2);
    }
    for path in &args {
        let name = Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());
        if !Path::new(path).exists() {
            eprintln!("SKIP missing: {}", path);
            continue;
        }
        let Ok(mtx) = read_mtx(Path::new(path)) else {
            eprintln!("SKIP unreadable: {}", path);
            continue;
        };
        let Ok(csc) = mtx.to_csc() else {
            eprintln!("SKIP not convertible: {}", path);
            continue;
        };

        let mut runs = Vec::new();
        for _ in 0..N_REPEAT {
            let prof = Arc::new(Mutex::new(SymbolicProfiler::new()));
            let params = SupernodeParams {
                symbolic_profiler: Some(prof.clone()),
                ..Default::default()
            };
            if symbolic_factorize_with_method(&csc, &params, OrderingMethod::Amd).is_err() {
                eprintln!("SKIP symbolic failed: {}", path);
                break;
            }
            let Ok(guard) = prof.lock() else { break };
            runs.push(guard.report());
        }
        if runs.is_empty() {
            continue;
        }
        // Median run by total.
        runs.sort_by_key(|r| r.total_us);
        let rep = &runs[runs.len() / 2];

        println!("\n=== {}  (n={}, nnz={}) ===", name, csc.n, csc.nnz());
        println!("{:<28}{:>12}{:>9}", "stage", "us", "%");
        let total = rep.total_us.max(1) as f64;
        let mut stages: Vec<_> = rep.stages.iter().collect();
        stages.sort_by_key(|s| std::cmp::Reverse(s.us));
        for s in stages {
            println!(
                "{:<28}{:>12}{:>8.1}%",
                s.name,
                s.us,
                100.0 * s.us as f64 / total
            );
        }
        println!("{:<28}{:>12}{:>8.1}%", "TOTAL", rep.total_us, 100.0);
    }
}
