//! diag_mc64_warm_vs_cold — is MC64 scaling cost a property of the
//! matrix, or of the solver that has already factored something else?
//!
//! `diag_trajectory_scaling` (ONE solver over the iterates, as ripopt
//! drives it) reported pinene_3200's per-iterate `scaling_us` climbing
//! 8,500 -> 207,611 on a pattern that never changes.
//! `diag_scaling_reuse_correctness` (a FRESH solver per iterate)
//! reported the same matrices flat at ~9,000-10,300. Same matrices,
//! same `with_parallel(false)`, same profiling. The only difference is
//! solver reuse.
//!
//! This runs both arms side by side on the same iterate so the
//! comparison is not across two binaries: a warm solver carried over
//! the whole sequence, and a cold solver constructed for that one
//! matrix.
//!
//! Usage: diag_mc64_warm_vs_cold <iter0.mtx> <iter1.mtx> ...

use feral::{read_mtx, FactorStatus, Solver};
use std::path::Path;

fn scal_of(solver: &Solver) -> (u64, u64, u64) {
    match solver.profile_report() {
        Some(r) => (
            r.total_us,
            r.prologue_breakdown.scaling_us,
            r.prologue_breakdown.scaling_pivot_order_us,
        ),
        None => (0, 0, 0),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: diag_mc64_warm_vs_cold <iter0.mtx> ...");
        std::process::exit(2);
    }
    let mut warm = Solver::new().with_profiling(true).with_parallel(false);
    println!(
        "{:<24}{:>9}{:>11}{:>11}{:>10}{:>11}{:>11}{:>8}",
        "iterate", "nnz", "warm_scal", "warm_pivord", "warm_tot", "cold_scal", "cold_tot", "hit"
    );
    let mut prev_hits = 0usize;
    for a in &args {
        let csc = match read_mtx(Path::new(a)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {a}: {e:?}");
                continue;
            }
        };
        let st = warm.factor(&csc, None);
        if !matches!(
            st,
            FactorStatus::Success | FactorStatus::WrongInertia { .. }
        ) {
            eprintln!("{a} warm: {st:?}");
            continue;
        }
        let (wtot, wscal, wpiv) = scal_of(&warm);
        let hits_now = warm.mc64_cache_hit_count();
        let hit = hits_now > prev_hits;
        prev_hits = hits_now;

        let mut cold = Solver::new().with_profiling(true).with_parallel(false);
        let st = cold.factor(&csc, None);
        if !matches!(
            st,
            FactorStatus::Success | FactorStatus::WrongInertia { .. }
        ) {
            eprintln!("{a} cold: {st:?}");
            continue;
        }
        let (ctot, cscal, _) = scal_of(&cold);

        let name = Path::new(a)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(a);
        println!(
            "{:<24}{:>9}{:>11}{:>11}{:>10}{:>11}{:>11}{:>8}",
            name,
            csc.nnz(),
            wscal,
            wpiv,
            wtot,
            cscal,
            ctot,
            if hit { "yes" } else { "NO" }
        );
    }
    Ok(())
}
