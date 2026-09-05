//! Issue #200 — where does the supernode loop actually spend its time,
//! bucketed by front size?
//!
//! The 2026-09-05 comment on #200 reports, on a 351k-row AC-OPF KKT,
//! that fronts of `nrow <= 8` are 68% of the supernodes but only 10.7%
//! of the loop, and that 85% of loop time sits in fronts of 17 rows or
//! more. That is the opposite conclusion from the one this branch drew
//! from a least-squares `t = a*fronts + b*flops` fit over five corpus
//! matrices, whose fitted per-front term came out at 96% of optmass.
//!
//! Those two claims are made with different instruments on different
//! matrices, so neither refutes the other as stated. This binary runs
//! *their* instrument — feral's own `Solver::with_profiling(true)`,
//! whose `ProfileReport` buckets supernode wallclock by `nrow` in
//! exactly the ranges they report — on *our* five matrices, so the two
//! datasets can be compared column for column.
//!
//! `with_profiling` costs one `Instant::now()` pair per supernode, not
//! the ~10 per supernode that `PHASE_TIMING_ENABLED` costs (see
//! `diag_200_probe_tax`), and the loop share it reports is a ratio of
//! like-instrumented quantities. The uninstrumented factor time is
//! printed beside it so the probe tax on each matrix is visible rather
//! than assumed.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin diag_200_front_size_buckets \
//!     -- [--reps N] <matrix.mtx>...
use feral::{read_mtx, NumericParams, Solver};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut reps = 9usize;
    let mut paths: Vec<String> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--reps" {
            if let Some(v) = it.next() {
                reps = v.parse()?;
            }
        } else {
            paths.push(a.clone());
        }
    }

    for p in &paths {
        let csc = read_mtx(std::path::Path::new(p)).and_then(|m| m.to_csc())?;
        let label = std::path::Path::new(p)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();

        // Uninstrumented reference: warm solver, min of `reps`.
        let mut plain =
            Solver::with_params(NumericParams::default(), Default::default()).with_parallel(false);
        for _ in 0..3 {
            let _ = plain.factor(&csc, None);
        }
        let mut best_plain = u128::MAX;
        for _ in 0..reps {
            let t0 = Instant::now();
            let _ = plain.factor(&csc, None);
            best_plain = best_plain.min(t0.elapsed().as_micros());
        }

        // Profiled: same warm-solver protocol, `with_profiling(true)`.
        let mut prof = Solver::with_params(NumericParams::default(), Default::default())
            .with_parallel(false)
            .with_profiling(true);
        for _ in 0..3 {
            let _ = prof.factor(&csc, None);
        }
        let mut best_prof = u128::MAX;
        for _ in 0..reps {
            let t0 = Instant::now();
            let _ = prof.factor(&csc, None);
            best_prof = best_prof.min(t0.elapsed().as_micros());
        }
        let report = match prof.profile_report() {
            Some(r) => r,
            None => {
                println!("{label}: no profile report (fast path?)");
                continue;
            }
        };

        println!(
            "\n{label}  n={} nnz={} snodes={} plain_us={} profiled_us={} tax={:.2}x",
            csc.n,
            csc.row_idx.len(),
            report.n_supernodes,
            best_plain,
            best_prof,
            best_prof as f64 / best_plain.max(1) as f64
        );
        println!(
            "  loop_ns={} prologue_us={} epilogue_us={} total_us={}",
            report.loop_ns, report.prologue_us, report.epilogue_us, report.total_us
        );
        println!(
            "  {:<8}{:>10}{:>9}{:>12}{:>10}{:>12}",
            "nrow", "snodes", "% nodes", "sum_ns", "% loop", "avg_ns"
        );
        for b in &report.buckets {
            if b.count == 0 {
                continue;
            }
            println!(
                "  {:<8}{:>10}{:>8.1}%{:>12}{:>9.1}%{:>12.0}",
                b.range,
                b.count,
                100.0 * b.count as f64 / report.n_supernodes.max(1) as f64,
                b.sum_ns,
                100.0 * b.sum_ns as f64 / report.loop_ns.max(1) as f64,
                b.avg_ns,
            );
        }
        // The claim under test: is the loop dominated by the many tiny
        // fronts, or by the few large ones?
        let small: u64 = report
            .buckets
            .iter()
            .filter(|b| b.range == "<=8" || b.range == "9-16")
            .map(|b| b.sum_ns)
            .sum();
        let small_n: usize = report
            .buckets
            .iter()
            .filter(|b| b.range == "<=8" || b.range == "9-16")
            .map(|b| b.count)
            .sum();
        println!(
            "  nrow<=16: {:.1}% of nodes, {:.1}% of loop",
            100.0 * small_n as f64 / report.n_supernodes.max(1) as f64,
            100.0 * small as f64 / report.loop_ns.max(1) as f64
        );
    }
    Ok(())
}
