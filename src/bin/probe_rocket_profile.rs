//! Profile feral's per-iter factor cost on rocket_12800 vs MA57.
//!
//! Context: on rocket_12800 ipopt-feral takes ~270 ms/iter vs MA57
//! ~55 ms/iter (5x). Goal: localize where the time goes — coarse
//! phase split (symbolic / scaling / numeric / overhead) plus
//! per-supernode bucket distribution using the existing Profiler.
//!
//! Loads the dumped rocket_12800 KKTs from
//! data/matrices/kkt-mittelmann/rocket_12800/ and factors each one
//! both fresh (one Solver per call) and warm (one Solver across the
//! sequence). Uses sequential mode so per-supernode timings line up
//! 1:1 with the bucket report.

use feral::numeric::factorize::Profiler;
use feral::symbolic::supernode::SupernodeParams;
use feral::{read_mtx, NumericParams, Solver};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn factor_with_profile(label: &str, csc: &feral::CscMatrix) {
    let prof = Arc::new(Mutex::new(Profiler::new()));
    let np = NumericParams {
        profiler: Some(prof.clone()),
        ..NumericParams::default()
    };
    let mut solver = Solver::with_params(np, SupernodeParams::default()).with_parallel(false);

    // First call (cold: includes symbolic).
    let t0 = Instant::now();
    let st = solver.factor(csc, None);
    let cold_ms = t0.elapsed().as_secs_f64() * 1e3;

    // Re-arm profiler for a warm call so the report reflects just
    // the numeric phase.
    if let Ok(mut p) = prof.lock() {
        *p = Profiler::new();
    }
    let t0 = Instant::now();
    let st_warm = solver.factor(csc, None);
    let warm_ms = t0.elapsed().as_secs_f64() * 1e3;

    let prof = match prof.lock() {
        Ok(p) => p.clone(),
        Err(_) => return,
    };
    let report = prof.report();
    println!(
        "{}: cold={:.1} ms ({:?})  warm={:.1} ms ({:?})  n_snodes={}",
        label, cold_ms, st, warm_ms, st_warm, report.n_supernodes,
    );
    println!(
        "  prologue={:.1} ms  epilogue={:.1} ms  loop={:.1} ms  total={:.1} ms  overhead={:.1}%",
        report.prologue_us as f64 / 1e3,
        report.epilogue_us as f64 / 1e3,
        report.loop_us as f64 / 1e3,
        report.total_us as f64 / 1e3,
        report.overhead_pct,
    );
    println!("  bucket            count    sum_ms     pct      avg_us");
    for b in &report.buckets {
        if b.count == 0 {
            continue;
        }
        println!(
            "    nrow {:>7}  {:>6}  {:>8.1}  {:>5.1}%  {:>8.1}",
            b.range,
            b.count,
            b.sum_us as f64 / 1e3,
            b.pct_of_total,
            b.avg_us,
        );
    }
    let top: Vec<_> = {
        let mut v: Vec<_> = prof.timings().iter().collect();
        v.sort_by_key(|t| std::cmp::Reverse(t.us));
        v.into_iter().take(5).collect()
    };
    println!("  top 5 supernodes by time:");
    for t in top {
        println!(
            "    snode {:>5}  nrow={:>5} ncol={:>5}  {:>8.2} ms",
            t.snode_idx,
            t.nrow,
            t.ncol,
            t.us as f64 / 1e3,
        );
    }
}

fn warm_sequence(csc_list: &[(usize, feral::CscMatrix)], use_parallel: bool) {
    println!(
        "\n=== WARM sequence (one Solver, parallel={}, no profiler) ===",
        use_parallel
    );
    let mut solver = Solver::new().with_parallel(use_parallel);
    let mut total = 0.0;
    for (i, csc) in csc_list {
        let t = Instant::now();
        let st = solver.factor(csc, None);
        let dt = t.elapsed().as_secs_f64() * 1e3;
        total += dt;
        println!("  iter {}: {:.1} ms ({:?})", i, dt, st);
    }
    println!("  TOTAL: {:.1} ms ({} iters)", total, csc_list.len());
}

fn main() {
    let csc_list: Vec<(usize, feral::CscMatrix)> = (0..18)
        .filter_map(|i| {
            let p = format!(
                "data/matrices/kkt-mittelmann/rocket_12800/rocket_12800_{:04}.mtx",
                i
            );
            let path = Path::new(&p);
            if !path.exists() {
                return None;
            }
            let csc = read_mtx(path).ok()?.to_csc().ok()?;
            Some((i, csc))
        })
        .collect();
    if csc_list.is_empty() {
        eprintln!("no rocket_12800 KKTs found");
        return;
    }
    let (i0, csc0) = &csc_list[0];
    println!(
        "rocket_12800 iter {} (n={}, nnz={})",
        i0,
        csc0.n,
        csc0.row_idx.len()
    );
    println!("\n=== PER-FACTOR PROFILE (sequential, fresh Solver) ===");
    for (i, csc) in &csc_list[..csc_list.len().min(3)] {
        factor_with_profile(&format!("iter {}", i), csc);
    }
    warm_sequence(&csc_list, true);
    warm_sequence(&csc_list, false);
}
