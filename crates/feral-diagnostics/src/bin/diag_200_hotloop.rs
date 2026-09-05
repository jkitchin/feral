//! Issue #200 — a plain, *uninstrumented* factorization loop for an
//! external sampling profiler (`samply record`).
//!
//! `phase_timing` costs ~350–430 ns per front (see `diag_200_probe_tax`),
//! which on a KKT matrix with a 3×3 median front inflates total factor
//! time by up to 1.77×. Any attribution read off an instrumented run is
//! therefore measuring the apparatus. This binary factors a warm matrix
//! in a loop with all counters off so a sampler can attribute the real
//! thing.
use feral::{read_mtx, Solver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut iters = 200usize;
    let mut paths: Vec<String> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--iters" {
            if let Some(v) = it.next() {
                iters = v.parse()?;
            }
        } else {
            paths.push(a.clone());
        }
    }
    for p in &paths {
        let csc = read_mtx(std::path::Path::new(p)).and_then(|m| m.to_csc())?;
        let mut solver = Solver::new();
        for _ in 0..3 {
            let _ = solver.factor(&csc, None);
        }
        for _ in 0..iters {
            let _ = solver.factor(&csc, None);
        }
    }
    Ok(())
}
