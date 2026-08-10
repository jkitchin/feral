//! diag_delayed_fronts — how often does issue #125's static-layout fast
//! path miss?
//!
//! Step 1 of #125 (PR #139) reuses the symbolic static frontal layout
//! only when `n_delayed_in == 0`. Step 2 (still open) would extend it to
//! fronts that receive delayed columns. This reports what fraction of
//! fronts, and what fraction of frontal *rows*, take the dynamic
//! `build_row_indices` path today — i.e. the ceiling on what step 2 can
//! recover.
//!
//! Usage: diag_delayed_fronts <a.mtx> [b.mtx ...]

use feral::dense::factor::{phase_timing, PHASE_TIMING_ENABLED};
use feral::read_mtx;
use feral::symbolic::SupernodeParams;
use feral::{NumericParams, Solver};
use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: diag_delayed_fronts <a.mtx> [b.mtx ...]");
        std::process::exit(2);
    }
    println!(
        "{:<24}{:>9}{:>10}{:>9}{:>11}{:>12}{:>11}",
        "matrix", "fronts", "delayed", "pct", "rows", "rows_dyn", "pct_rows"
    );
    let mut buildrow_rows = Vec::new();
    for a in &args {
        let csc = match read_mtx(Path::new(a)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {a}: {e:?}");
                continue;
            }
        };
        let mut solver = Solver::with_params(NumericParams::default(), SupernodeParams::default())
            .with_parallel(false);
        let status = solver.factor(&csc, None);
        if !matches!(status, feral::FactorStatus::Success) {
            eprintln!("{a}: factor status {status:?}");
        }
        let f = match solver.factors() {
            Some(f) => f,
            None => {
                eprintln!("{a}: no factors");
                continue;
            }
        };
        let (mut fronts, mut delayed, mut rows, mut rows_dyn) = (0usize, 0usize, 0usize, 0usize);
        for nf in &f.node_factors {
            fronts += 1;
            rows += nf.nrow;
            if nf.n_delayed_in > 0 {
                delayed += 1;
                rows_dyn += nf.nrow;
            }
        }
        // Warm refactor with phase timing on: `BUILDROW_NS` wraps both
        // arms of the #125 branch (static `.to_vec()` copy and dynamic
        // `build_row_indices`), so buildrow/factor is a hard ceiling on
        // what step 2 could ever recover.
        PHASE_TIMING_ENABLED.store(true, Relaxed);
        phase_timing::reset();
        let t0 = Instant::now();
        let _ = solver.factor(&csc, None);
        let factor_ns = t0.elapsed().as_nanos() as u64;
        PHASE_TIMING_ENABLED.store(false, Relaxed);
        let (buildrow_ns, ..) = phase_timing::snapshot_detail();

        let name = Path::new(a)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(a);
        println!(
            "{:<24}{:>9}{:>10}{:>8.2}%{:>11}{:>12}{:>10.2}%",
            name,
            fronts,
            delayed,
            100.0 * delayed as f64 / fronts.max(1) as f64,
            rows,
            rows_dyn,
            100.0 * rows_dyn as f64 / rows.max(1) as f64
        );
        buildrow_rows.push((
            name.to_string(),
            factor_ns,
            buildrow_ns,
            100.0 * rows_dyn as f64 / rows.max(1) as f64,
        ));
    }

    println!();
    println!(
        "{:<24}{:>13}{:>13}{:>12}{:>12}",
        "matrix", "factor_us", "buildrow_us", "buildrow%", "dyn_rows%"
    );
    for (name, factor_ns, buildrow_ns, pct_rows) in &buildrow_rows {
        println!(
            "{:<24}{:>13}{:>13}{:>11.2}%{:>11.2}%",
            name,
            factor_ns / 1000,
            buildrow_ns / 1000,
            100.0 * *buildrow_ns as f64 / (*factor_ns).max(1) as f64,
            pct_rows
        );
    }
    Ok(())
}
