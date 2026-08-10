//! diag_mc64_work_trajectory — does the MC64 Hungarian get more
//! expensive along an IPM trajectory, on a pattern that never changes?
//!
//! `diag_mc64_warm_vs_cold` showed pinene_3200's `scaling_us` climbing
//! 8,052 -> 214,213 us on a warm solver while a cold solver stayed
//! flat. That is not a slowdown: `Solver::factor` clears
//! `symbolic.cached_mc64` after every call (issue #38,
//! `src/numeric/solver.rs:1374`), so the cold arm's Hungarian ran
//! inside `analyze()` and is billed outside `ProfileReport::total_us`
//! ("wallclock for the entire driver call"). The two arms bill the
//! same work to different phases.
//!
//! So the question the warm/cold split cannot answer: is the Hungarian
//! *itself* value-dependent? `diagnose_mc64_matching` runs the matching
//! standalone -- no solver, no cache, no scaling post-processing -- so
//! the per-iterate work counters are a property of the matrix alone.
//!
//! Usage: diag_mc64_work_trajectory <iter0.mtx> <iter1.mtx> ...

use feral::read_mtx;
use feral::scaling::diagnose_mc64_matching;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: diag_mc64_work_trajectory <iter0.mtx> ...");
        std::process::exit(2);
    }
    println!(
        "{:<24}{:>9}{:>10}{:>12}{:>14}{:>14}{:>14}{:>14}",
        "iterate",
        "nnz",
        "match_us",
        "augment",
        "touched",
        "heap_init",
        "phase3_inner",
        "edge_scans"
    );
    for a in &args {
        let csc = match read_mtx(Path::new(a)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {a}: {e:?}");
                continue;
            }
        };
        let t = Instant::now();
        let s = match diagnose_mc64_matching(&csc) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{a}: {e:?}");
                continue;
            }
        };
        let us = t.elapsed().as_micros() as u64;
        let name = Path::new(a)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(a);
        println!(
            "{:<24}{:>9}{:>10}{:>12}{:>14}{:>14}{:>14}{:>14}",
            name,
            csc.nnz(),
            us,
            s.augment_searches,
            s.touched_total,
            s.heap_init_slots,
            s.phase3_inner_iters,
            s.main_loop_edge_scans
        );
    }
    Ok(())
}
