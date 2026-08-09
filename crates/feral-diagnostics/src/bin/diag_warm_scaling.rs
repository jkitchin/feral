//! diag_warm_scaling — is scaling still paid on warm `Solver::factor`?
//!
//! `diag_factor_phases` drives the numeric driver directly, so its
//! `scaling%` column reflects a driver with no `Solver`-level MC64
//! cache behind it. This probe runs the same breakdown through
//! `Solver` — which owns `mc64_scaling_cache` — and prints
//! `mc64_cache_hit_count` alongside, so the scaling share can be read
//! against proof the cache was actually live.
//!
//! Usage: diag_warm_scaling <a.mtx> [b.mtx ...]

use feral::scaling::ScalingStrategy;
use feral::symbolic::SupernodeParams;
use feral::{read_mtx, FactorStatus, NumericParams, Solver};
use std::path::Path;
use std::time::Instant;

const N_REPS: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: diag_warm_scaling <a.mtx> [b.mtx ...]");
        std::process::exit(2);
    }
    println!(
        "{:<22}{:>9}{:>8}{:>9}{:>9}{:>10}{:>10}",
        "matrix", "drv_us", "prol%", "scaling%", "permute%", "mc64_hits", "scaling_info"
    );
    for a in &args {
        let csc = match read_mtx(Path::new(a)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {a}: {e:?}");
                continue;
            }
        };
        let mut np = NumericParams::default();
        if std::env::var("FORCE_INFNORM").is_ok() {
            np.scaling = ScalingStrategy::InfNorm;
        }
        if std::env::var("FORCE_MC64").is_ok() {
            np.scaling = ScalingStrategy::Mc64Symmetric;
        }
        let mut solver = Solver::with_params(np, SupernodeParams::default())
            .with_profiling(true)
            .with_parallel(false);
        // Warm-up: primes symbolic cache, MC64 cache, workspace.
        match solver.factor(&csc, None) {
            FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
            other => {
                eprintln!("{a}: warm-up failed: {other:?}");
                continue;
            }
        }
        let baseline_hits = solver.mc64_cache_hit_count();

        let mut best = u64::MAX;
        let mut keep = [0u64; 3];
        for _ in 0..N_REPS {
            let t0 = Instant::now();
            let status = solver.factor(&csc, None);
            let wall_us = t0.elapsed().as_micros() as u64;
            if !matches!(
                status,
                FactorStatus::Success | FactorStatus::WrongInertia { .. }
            ) {
                eprintln!("{a}: factor failed mid-loop: {status:?}");
                break;
            }
            if let Some(report) = solver.profile_report() {
                if report.total_us > 0 && wall_us < best {
                    best = wall_us;
                    keep = [
                        report.prologue_us,
                        report.prologue_breakdown.scaling_us,
                        report.prologue_breakdown.permute_us,
                    ];
                }
            }
        }
        if best == u64::MAX {
            eprintln!("{a}: no profiled rep");
            continue;
        }
        let hits = solver.mc64_cache_hit_count() - baseline_hits;
        let sinfo = format!("{:?}", solver.scaling_info());
        let pct = |v: u64| 100.0 * v as f64 / best.max(1) as f64;
        let name = Path::new(a)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(a);
        println!(
            "{:<22}{:>9}{:>7.1}%{:>8.1}%{:>8.1}%{:>10}  {}",
            name,
            best,
            pct(keep[0]),
            pct(keep[1]),
            pct(keep[2]),
            hits,
            sinfo.chars().take(46).collect::<String>()
        );
    }
    Ok(())
}
