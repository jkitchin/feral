//! Per-supernode profile of the (fixed) qap15 conic-KKT factor — issue #91
//! follow-up (dense-kernel throughput). Answers: where does the ~0.7 s go,
//! by front size, and how much does FMA move the large-front buckets?
//!
//! Usage: cargo run --release --example profile_qap15 -- <path.mtx>

use feral::dense::factor::{phase_timing, PHASE_TIMING_ENABLED};
use feral::numeric::factorize::{Profiler, SupernodeTiming};
use feral::{read_mtx, CscMatrix, NumericParams, Solver};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Aggregate phase split (Lever A step 1): what fraction of a *sequential*
/// factor is the serial panel factorization (`panelfactor` + `scalartail`)
/// vs the parallelizable trailing update (`schur`)? The serial fraction is
/// the Amdahl ceiling look-ahead must attack.
fn phase_split(csc: &CscMatrix) {
    PHASE_TIMING_ENABLED.store(true, Ordering::Relaxed);
    phase_timing::reset();
    // Sequential: intrafront parallelism off, so panel and schur are both
    // wall-serial and their ratio is the true per-front work split.
    let mut solver = Solver::new().with_parallel(false);
    let t = Instant::now();
    let st = solver.factor(csc, None);
    let wall = t.elapsed().as_secs_f64() * 1e3;
    let (assembly, densefactor, panel, schur, scalartail) = phase_timing::snapshot();
    PHASE_TIMING_ENABLED.store(false, Ordering::Relaxed);

    let ns = |x: u64| x as f64 / 1e6; // ns → ms
    let panel_ms = ns(panel);
    let schur_ms = ns(schur);
    let tail_ms = ns(scalartail);
    let asm_ms = ns(assembly);
    let df_ms = ns(densefactor);
    // "serial" under the parallelize-trailing-only model = everything except
    // the trailing Schur update.
    let serial_ms = panel_ms + tail_ms + asm_ms + (df_ms - panel_ms - schur_ms).max(0.0);
    let par_ms = schur_ms;
    let frac = serial_ms / (serial_ms + par_ms).max(1e-9);
    println!("--- phase split (sequential, {st:?}, wall={wall:.0}ms) ---");
    println!("  panelfactor : {panel_ms:>8.1} ms   (serial — the look-ahead target)");
    println!("  scalartail  : {tail_ms:>8.1} ms   (serial — ramp-down)");
    println!("  schur       : {schur_ms:>8.1} ms   (parallelizable trailing update)");
    println!("  assembly    : {asm_ms:>8.1} ms   (densefactor total {df_ms:.1} ms)");
    println!(
        "  => serial (non-schur) {serial_ms:.0} ms vs parallel (schur) {par_ms:.0} ms  \
         ⇒ serial fraction ≈ {:.2}",
        frac
    );
    println!(
        "  Amdahl ceiling on 10 cores if only schur parallelizes: {:.1}×",
        1.0 / (frac + (1.0 - frac) / 10.0)
    );
}

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
        report.loop_ns as f64 / 1e6,
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
            b.sum_ns as f64 / 1e6,
            b.pct_of_total,
            b.avg_ns / 1000.0
        );
    }

    // Top fronts by time — the concrete targets for a blocked kernel.
    let mut timings: Vec<SupernodeTiming> = p.timings().to_vec();
    timings.sort_by_key(|b| std::cmp::Reverse(b.ns));
    println!("  top fronts by time (nrow x ncol -> ms):");
    let loop_us = report.loop_ns.max(1) as f64 / 1000.0;
    for t in timings.iter().take(10) {
        println!(
            "    {:>6} x {:<6} -> {:>8.2} ms  ({:>4.1}% of loop)",
            t.nrow,
            t.ncol,
            t.ns as f64 / 1e6,
            t.ns as f64 / 1000.0 * 100.0 / loop_us
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
    phase_split(&csc); // Lever A step 1: measure the serial split
    profile_once(&csc, false); // default nofma
    profile_once(&csc, true); // fma
}
