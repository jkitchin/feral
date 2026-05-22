//! Issue #44 — achieved GFLOP/s of feral's supernode loop on NARX_CFy.
//!
//! `probe_rocket_profile NARX_CFy` already localized the cost: 94.6%
//! of the iter-0 numeric loop is in 449 supernodes with `nrow > 128`
//! (tall fronts, e.g. nrow=1877 ncol=77). What that probe does not
//! answer is *how efficient* that work is: is the kernel compute-bound
//! near peak, or memory-bound far below it?
//!
//! This diag factors each captured NARX_CFy KKT with the `Profiler`
//! attached, then for every supernode estimates the dense LDLᵀ flop
//! count from `(nrow, ncol)` and divides by the measured per-supernode
//! wall time. The headline number is the loop-wide achieved GFLOP/s
//! and the same figure restricted to the `nrow > 128` fronts that
//! dominate. Compare against a blocked f64 GEMM ceiling (~50-90
//! GFLOP/s single-core on this class of CPU) to size the headroom.
//!
//! Usage: cargo run --release --bin diag_narx_kernel_gflops

use std::path::Path;
use std::sync::{Arc, Mutex};

use feral::numeric::factorize::Profiler;
use feral::symbolic::supernode::SupernodeParams;
use feral::{read_mtx, NumericParams, Solver};

const DIR: &str = "data/matrices/kkt-mittelmann/NARX_CFy";

/// Dense LDLᵀ flop estimate for one supernode: a front of dimension
/// `nrow` from which `ncol` columns are eliminated.
///
///  * diagonal block factor   ~ ncol³/3
///  * off-diagonal panel solve ~ t·ncol²       (t = nrow-ncol)
///  * Schur / trailing update  ~ t²·ncol       (symmetric mul-adds ×2)
fn snode_flops(nrow: usize, ncol: usize) -> f64 {
    let n = ncol as f64;
    let t = nrow.saturating_sub(ncol) as f64;
    n * n * n / 3.0 + t * n * n + t * t * n
}

fn run(iter: usize) {
    let mtx_path = format!("{DIR}/NARX_CFy_{iter:04}.mtx");
    if !Path::new(&mtx_path).exists() {
        eprintln!("SKIP iter {iter}: {mtx_path} not present (corpus gitignored)");
        return;
    }
    let csc = read_mtx(Path::new(&mtx_path))
        .and_then(|m| m.to_csc())
        .expect("load NARX_CFy mtx");

    let prof = Arc::new(Mutex::new(Profiler::new()));
    let np = NumericParams {
        profiler: Some(prof.clone()),
        ..NumericParams::default()
    };
    let mut solver = Solver::with_params(np, SupernodeParams::default()).with_parallel(false);

    // Cold call builds the symbolic factorization; re-arm and time a
    // warm call so the profiler reflects only the numeric loop.
    let _ = solver.factor(&csc, None);
    if let Ok(mut p) = prof.lock() {
        *p = Profiler::new();
    }
    let _ = solver.factor(&csc, None);

    let prof = match prof.lock() {
        Ok(p) => p.clone(),
        Err(_) => return,
    };
    let report = prof.report();

    // Aggregate flops / time, whole loop and the nrow>128 tail.
    let (mut flop_all, mut us_all) = (0.0f64, 0.0f64);
    let (mut flop_big, mut us_big, mut n_big) = (0.0f64, 0.0f64, 0usize);
    let (mut worst_gflops, mut worst): (f64, Option<&feral::numeric::factorize::SupernodeTiming>) =
        (f64::INFINITY, None);
    for t in prof.timings() {
        let f = snode_flops(t.nrow, t.ncol);
        let us = t.us as f64;
        flop_all += f;
        us_all += us;
        if t.nrow > 128 {
            flop_big += f;
            us_big += us;
            n_big += 1;
            if us > 50.0 {
                let g = f / (us * 1e3);
                if g < worst_gflops {
                    worst_gflops = g;
                    worst = Some(t);
                }
            }
        }
    }
    let gflops = |flop: f64, us: f64| if us > 0.0 { flop / (us * 1e3) } else { 0.0 };
    println!(
        "NARX_CFy_{iter:04}  n={}  loop={:.1} ms  flops={:.3} GFLOP",
        csc.n,
        report.loop_us as f64 / 1e3,
        flop_all / 1e9,
    );
    println!(
        "  whole loop : {:>8.3} GFLOP  in {:>8.1} ms  -> {:>6.2} GFLOP/s",
        flop_all / 1e9,
        us_all / 1e3,
        gflops(flop_all, us_all),
    );
    println!(
        "  nrow>128   : {:>8.3} GFLOP  in {:>8.1} ms  -> {:>6.2} GFLOP/s   ({n_big} fronts)",
        flop_big / 1e9,
        us_big / 1e3,
        gflops(flop_big, us_big),
    );
    if let Some(w) = worst {
        println!(
            "  slowest big front: snode {} nrow={} ncol={}  {:.3} GFLOP in {:.2} ms -> {:.2} GFLOP/s",
            w.snode_idx,
            w.nrow,
            w.ncol,
            snode_flops(w.nrow, w.ncol) / 1e9,
            w.us as f64 / 1e3,
            worst_gflops,
        );
    }
}

fn main() {
    println!("Issue #44 — NARX_CFy supernode-loop achieved GFLOP/s\n");
    println!("(reference: blocked f64 GEMM single-core ~50-90 GFLOP/s)\n");
    for iter in 0..3 {
        run(iter);
    }
}
