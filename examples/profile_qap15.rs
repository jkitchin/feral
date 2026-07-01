//! Per-supernode profile of the (fixed) qap15 conic-KKT factor — issue #91
//! follow-up (dense-kernel throughput). Answers: where does the ~0.7 s go,
//! by front size, and how much does FMA move the large-front buckets?
//!
//! Usage: cargo run --release --example profile_qap15 -- <path.mtx>

use feral::numeric::factorize::{Profiler, SupernodeTiming};
use feral::{read_mtx, CscMatrix, NumericParams, Solver};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn profile_once(csc: &CscMatrix, fma: bool) {
    let prof = Arc::new(Mutex::new(Profiler::new()));
    let np = NumericParams {
        profiler: Some(Arc::clone(&prof)),
        fma,
        ..NumericParams::default()
    };
    // Sequential driver: the per-supernode profiler is only populated on
    // the sequential path.
    let mut solver =
        Solver::with_params(np, feral::symbolic::SupernodeParams::default()).with_parallel(false);

    let t = Instant::now();
    let st = solver.factor(csc, None);
    let wall_ms = t.elapsed().as_secs_f64() * 1e3;

    let p = prof.lock().expect("profiler lock");
    let report = p.report();
    let stats = solver.last_factor_stats();

    println!(
        "\n=== fma={fma}  wall={wall_ms:.1}ms  status={st:?}  nnz_L={}  n_supernodes={} ===",
        stats.as_ref().map(|s| s.nnz_l).unwrap_or(0),
        report.n_supernodes
    );
    println!(
        "  loop={:.1}ms  prologue={:.1}ms  epilogue={:.1}ms  overhead={:.1}%",
        report.loop_us as f64 / 1e3,
        report.prologue_us as f64 / 1e3,
        report.epilogue_us as f64 / 1e3,
        report.overhead_pct
    );
    println!("  front-size buckets (by nrow):");
    println!(
        "    {:>8}  {:>7}  {:>10}  {:>7}  {:>10}",
        "range", "count", "sum_ms", "%loop", "avg_us"
    );
    for b in &report.buckets {
        println!(
            "    {:>8}  {:>7}  {:>10.1}  {:>6.1}%  {:>10.1}",
            b.range,
            b.count,
            b.sum_us as f64 / 1e3,
            b.pct_of_total,
            b.avg_us
        );
    }

    // Top fronts by time — the concrete targets for a blocked kernel.
    let mut timings: Vec<SupernodeTiming> = p.timings().to_vec();
    timings.sort_by_key(|b| std::cmp::Reverse(b.us));
    println!("  top fronts by time (nrow x ncol -> ms):");
    let loop_us = report.loop_us.max(1) as f64;
    for t in timings.iter().take(10) {
        println!(
            "    {:>6} x {:<6} -> {:>8.2} ms  ({:>4.1}% of loop)",
            t.nrow,
            t.ncol,
            t.us as f64 / 1e3,
            t.us as f64 * 100.0 / loop_us
        );
    }
    if !report.validation_warnings.is_empty() {
        println!("  warnings: {:?}", report.validation_warnings);
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| panic!("usage: profile_qap15 <path.mtx>"));
    let csc = read_mtx(Path::new(&path))
        .and_then(|m| m.to_csc())
        .expect("read mtx");
    println!("matrix: n={} nnz={}", csc.n, csc.row_idx.len());
    profile_once(&csc, false); // default nofma
    profile_once(&csc, true); // fma
}
