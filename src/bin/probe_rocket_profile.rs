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
//!
//! Track B1: takes an optional problem name argument
//! (`probe_rocket_profile NARX_CFy`) and prints the per-sub-phase
//! attribution of the numeric prologue.

use feral::numeric::factorize::Profiler;
use feral::symbolic::supernode::SupernodeParams;
use feral::{read_mtx, NumericParams, Solver};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Track B1: print the per-sub-phase attribution of the prologue.
fn print_prologue_breakdown(report: &feral::numeric::factorize::ProfileReport) {
    let bd = &report.prologue_breakdown;
    let ms = |us: u64| us as f64 / 1e3;
    let pct = |us: u64| {
        if report.prologue_us > 0 {
            us as f64 * 100.0 / report.prologue_us as f64
        } else {
            0.0
        }
    };
    let rows: [(&str, u64); 7] = [
        ("row_map", bd.row_map_us),
        ("scaling", bd.scaling_us),
        ("scaling_pivot_order", bd.scaling_pivot_order_us),
        ("permute_csc_values", bd.permute_us),
        ("symmetric_pattern", bd.symmetric_pattern_us),
        ("infnorm + null_pivot_tol", bd.infnorm_tol_us),
        ("setup (is_root/alloc)", bd.setup_us),
    ];
    println!(
        "  prologue sub-phases (of {:.1} ms):",
        ms(report.prologue_us)
    );
    for (name, us) in rows {
        println!("    {:<26} {:>9.1} ms  {:>5.1}%", name, ms(us), pct(us));
    }
    println!(
        "      └─ from_triplets (subset of permute) {:>6.1} ms  {:>5.1}%",
        ms(bd.permute_from_triplets_us),
        pct(bd.permute_from_triplets_us),
    );
}

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
    print_prologue_breakdown(&report);
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

/// Track B1 follow-up: the prologue breakdown localizes the cost to
/// the `scaling` sub-phase (`compute_scaling_with_cache`). This drills
/// one level deeper — which strategy `Auto` resolves to, and the
/// standalone wall of InfNorm (Knight-Ruiz) vs MC64 (Hungarian) — so
/// B2 knows whether the lever is the matching or the equilibration.
fn diagnose_scaling(csc: &feral::CscMatrix) {
    use feral::scaling::{compute_scaling, pick_scaling_strategy, ScalingStrategy};
    println!("\n=== SCALING DIAGNOSTIC (iter 0) ===");
    println!(
        "  pick_scaling_strategy -> {:?}",
        pick_scaling_strategy(csc)
    );
    for (label, strat) in [
        ("InfNorm", ScalingStrategy::InfNorm),
        ("Mc64Symmetric", ScalingStrategy::Mc64Symmetric),
    ] {
        let t = Instant::now();
        let r = compute_scaling(csc, &strat);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        match r {
            Ok((_, info)) => {
                println!(
                    "  compute_scaling({:<14}) {:>9.1} ms  ({:?})",
                    label, ms, info
                )
            }
            Err(e) => println!("  compute_scaling({:<14}) ERROR {:?}", label, e),
        }
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
    // Track B1: defaults to rocket_12800 but takes an optional problem
    // name so the prologue breakdown can be captured on a second
    // large-n problem (e.g. `probe_rocket_profile NARX_CFy`).
    let problem = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "rocket_12800".into());
    let csc_list: Vec<(usize, feral::CscMatrix)> = (0..200)
        .filter_map(|i| {
            let p = format!(
                "data/matrices/kkt-mittelmann/{}/{}_{:04}.mtx",
                problem, problem, i
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
        eprintln!("no {} KKTs found", problem);
        return;
    }
    let (i0, csc0) = &csc_list[0];
    println!(
        "{} iter {} (n={}, nnz={})",
        problem,
        i0,
        csc0.n,
        csc0.row_idx.len()
    );
    println!("\n=== PER-FACTOR PROFILE (sequential, fresh Solver) ===");
    for (i, csc) in &csc_list[..csc_list.len().min(3)] {
        factor_with_profile(&format!("iter {}", i), csc);
    }
    diagnose_scaling(csc0);
    warm_sequence(&csc_list, true);
    warm_sequence(&csc_list, false);
}
