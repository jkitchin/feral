//! diag_trajectory_reuse — the end-to-end economics of reusing iter-0's
//! MC64 scaling across a whole IPM trajectory.
//!
//! Every earlier probe here measured half the question.
//! `diag_trajectory_scaling` drove ONE solver (as ripopt does) but only
//! with fresh scaling. `diag_scaling_reuse_correctness` compared fresh
//! against reuse but built a FRESH solver per iterate, which hides the
//! MC64 cost in `analyze()` -- `Solver::factor` clears
//! `symbolic.cached_mc64` after every call (issue #38,
//! `src/numeric/solver.rs:1374`), so only a warm solver pays the
//! Hungarian inside the timed driver.
//!
//! This drives TWO warm solvers over the same trajectory in the same
//! process: one on `Auto` (today's behaviour), one pinned to
//! `External(d0)` where `d0` is iter-0's own MC64 vector (what a
//! perfectly permissive cache would do). It reports per-iterate
//! wallclock and inertia for both.
//!
//! Issue #38's claim is that stale scaling "silently corrupts inertia
//! and (eventually) explodes factor cost". This measures both halves
//! against the arm that actually pays for the alternative.
//!
//! Usage: diag_trajectory_reuse <iter0.mtx> <iter1.mtx> ...

use feral::scaling::ScalingStrategy;
use feral::symbolic::SupernodeParams;
use feral::{read_mtx, CscMatrix, FactorStatus, NumericParams, Solver};
use std::path::Path;

fn build(strategy: Option<ScalingStrategy>) -> Solver {
    let mut np = NumericParams::default();
    if let Some(s) = strategy {
        np.scaling = s;
    }
    Solver::with_params(np, SupernodeParams::default())
        .with_parallel(false)
        .with_profiling(true)
}

/// Returns (total_us, scaling_us, "pos/neg/zero").
fn step(solver: &mut Solver, csc: &CscMatrix) -> Option<(u64, u64, String)> {
    match solver.factor(csc, None) {
        FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
        other => {
            eprintln!("  factor failed: {other:?}");
            return None;
        }
    }
    let inertia = solver
        .inertia()
        .map(|i| format!("{}/{}/{}", i.positive, i.negative, i.zero))?;
    let (total_us, scal_us) = match solver.profile_report() {
        Some(r) => (r.total_us, r.prologue_breakdown.scaling_us),
        None => (0, 0),
    };
    Some((total_us, scal_us, inertia))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: diag_trajectory_reuse <iter0.mtx> <iter1.mtx> ...");
        std::process::exit(2);
    }
    // iter-0's own MC64 vector, from a throwaway solver.
    let first = read_mtx(Path::new(&args[0])).and_then(|m| m.to_csc())?;
    let mut seed = build(None);
    if !matches!(
        seed.factor(&first, None),
        FactorStatus::Success | FactorStatus::WrongInertia { .. }
    ) {
        eprintln!("seed factorization failed");
        std::process::exit(1);
    }
    let d0 = match seed.factors() {
        Some(f) => f.scaling.clone(),
        None => {
            eprintln!("seed produced no factors");
            std::process::exit(1);
        }
    };
    drop(seed);
    println!("baseline scaling from {} (n={})", args[0], d0.len());

    let mut fresh = build(None);
    let mut reuse = build(Some(ScalingStrategy::External(d0.clone())));
    println!(
        "{:<24}{:>9}{:>11}{:>11}{:>11}{:>11}{:>8}{:>18}",
        "iterate", "nnz", "fresh_tot", "fresh_scl", "reuse_tot", "reuse_scl", "speedup", "inertia"
    );
    let (mut sum_fresh, mut sum_reuse) = (0u64, 0u64);
    let mut differs = 0usize;
    for a in &args {
        let csc = match read_mtx(Path::new(a)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {a}: {e:?}");
                continue;
            }
        };
        let name = Path::new(a)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(a);
        if csc.n != d0.len() {
            println!("{name:<24} n changed -- cache invalid by fingerprint");
            continue;
        }
        let f = step(&mut fresh, &csc);
        let r = step(&mut reuse, &csc);
        match (f, r) {
            (Some((ft, fs, fi)), Some((rt, rs, ri))) => {
                sum_fresh += ft;
                sum_reuse += rt;
                let same = fi == ri;
                if !same {
                    differs += 1;
                }
                println!(
                    "{:<24}{:>9}{:>11}{:>11}{:>11}{:>11}{:>7.2}x{:>18}",
                    name,
                    csc.nnz(),
                    ft,
                    fs,
                    rt,
                    rs,
                    ft as f64 / rt.max(1) as f64,
                    if same {
                        fi
                    } else {
                        format!("{fi} vs {ri} DIFFERS")
                    }
                );
            }
            _ => println!("{name:<24} factorization failed"),
        }
    }
    println!();
    println!(
        "trajectory: fresh {} us, reuse {} us, speedup {:.2}x, inertia differs on {} iterate(s)",
        sum_fresh,
        sum_reuse,
        sum_fresh as f64 / sum_reuse.max(1) as f64,
        differs
    );
    Ok(())
}
