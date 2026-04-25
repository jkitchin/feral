//! Phase 2.13b — per-stage symbolic profiler probe.
//!
//! Runs `symbolic_factorize_with_method` 5 times on KIRBY2_0007 +
//! MUONSINE_0000 (the new sparse Top-10 worst tail under
//! `AmalgamationStrategy::Renumber` default) and prints the
//! per-stage breakdown using the median run.
//!
//! Diagnostic question: which symbolic stage(s) carry the 924 µs
//! cost on KIRBY2_0007 (n=458)?
//!
//! See `dev/research/phase-2.13b-symbolic-profiler.md`.

use std::path::Path;
use std::sync::{Arc, Mutex};

use feral::symbolic::{
    symbolic_factorize_with_method, OrderingMethod, SupernodeParams, SymbolicProfileReport,
    SymbolicProfiler,
};
use feral::{read_mtx, CscMatrix};

const N_REPEAT: usize = 5;

const MATRICES: &[(&str, &str)] = &[
    ("KIRBY2_0007", "data/matrices/kkt/KIRBY2/KIRBY2_0007.mtx"),
    (
        "MUONSINE_0000",
        "data/matrices/kkt/MUONSINE/MUONSINE_0000.mtx",
    ),
];

fn load_csc(path: &str) -> Option<CscMatrix> {
    if !Path::new(path).exists() {
        eprintln!("SKIP missing: {}", path);
        return None;
    }
    let mtx = read_mtx(Path::new(path)).ok()?;
    mtx.to_csc().ok()
}

fn run_one(csc: &CscMatrix) -> Option<SymbolicProfileReport> {
    let prof = Arc::new(Mutex::new(SymbolicProfiler::new()));
    let params = SupernodeParams {
        symbolic_profiler: Some(Arc::clone(&prof)),
        ..SupernodeParams::default()
    };
    let _ = symbolic_factorize_with_method(csc, &params, OrderingMethod::Amd).ok()?;
    let guard = prof.lock().ok()?;
    Some(guard.report())
}

fn median_report(mut reports: Vec<SymbolicProfileReport>) -> SymbolicProfileReport {
    reports.sort_by_key(|r| r.total_us);
    let mid = reports.len() / 2;
    reports.swap_remove(mid)
}

fn print_report(label: &str, n: usize, r: &SymbolicProfileReport) {
    println!("=== {} (n={}) ===", label, n);
    println!(
        "total = {} µs, accounted = {} µs, overhead = {:.1}%",
        r.total_us, r.accounted_us, r.overhead_pct
    );
    println!("{:<22}  {:>8}  {:>6}", "stage", "us", "%");
    let mut sorted: Vec<_> = r.stages.iter().collect();
    sorted.sort_by(|a, b| b.us.cmp(&a.us));
    for s in sorted {
        println!("{:<22}  {:>8}  {:>5.1}%", s.name, s.us, s.pct_of_total);
    }
    if !r.validation_warnings.is_empty() {
        println!("WARNINGS: {:?}", r.validation_warnings);
    }
    println!();
}

fn main() {
    for &(label, path) in MATRICES {
        let Some(csc) = load_csc(path) else { continue };
        let n = csc.n;
        let mut reports = Vec::with_capacity(N_REPEAT);
        for _ in 0..N_REPEAT {
            if let Some(r) = run_one(&csc) {
                reports.push(r);
            }
        }
        if reports.is_empty() {
            eprintln!("FAILED to profile {}", label);
            continue;
        }
        let med = median_report(reports);
        print_report(label, n, &med);
    }
}
