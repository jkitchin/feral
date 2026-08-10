//! diag_trajectory_scaling — MC64 scaling cost across a full IPM
//! iterate sequence, not a single factorization.
//!
//! The 2026-05-21 rejection of the value-bounded MC64 cache (Track B2)
//! recorded this lesson: "A per-factor profile of the *named target's
//! full iteration sequence*, not a single iteration, should precede
//! the plan." Repeatedly refactoring one iterate measures a workload
//! nobody runs — real IPM callers feed a new value set each time, and
//! the previous lever died because a single-iterate profile said MC64
//! dominated when the full trajectory said the delayed-pivot blowup
//! did.
//!
//! This drives ONE `Solver` over the iterates in order, as ripopt
//! would, and reports per-iterate whether the scaling cache hit, what
//! scaling cost, and whether the pattern changed underneath it (a
//! pattern change invalidates the cache by fingerprint and is not a
//! gate failure).
//!
//! Usage: diag_trajectory_scaling <iter0.mtx> <iter1.mtx> ...

use feral::{read_mtx, FactorStatus, Solver};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: diag_trajectory_scaling <iter0.mtx> <iter1.mtx> ...");
        std::process::exit(2);
    }
    let mut solver = Solver::new().with_profiling(true).with_parallel(false);
    println!(
        "{:<26}{:>9}{:>10}{:>9}{:>9}{:>8}{:>16}",
        "iterate", "nnz", "factor_us", "scal_us", "scaling%", "hit", "inertia"
    );
    let mut prev_nnz: Option<usize> = None;
    let mut prev_hits = 0usize;
    let mut cum_scal: u64 = 0;
    let mut cum_total: u64 = 0;
    for a in &args {
        let csc = match read_mtx(Path::new(a)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {a}: {e:?}");
                continue;
            }
        };
        let nnz = csc.nnz();
        let status = solver.factor(&csc, None);
        if !matches!(
            status,
            FactorStatus::Success | FactorStatus::WrongInertia { .. }
        ) {
            eprintln!("{a}: {status:?}");
            continue;
        }
        let inertia = match solver.inertia() {
            Some(i) => format!("{}/{}/{}", i.positive, i.negative, i.zero),
            None => "-".to_string(),
        };
        let hits_now = solver.mc64_cache_hit_count();
        let hit = hits_now > prev_hits;
        prev_hits = hits_now;

        let (total_us, scal_us) = match solver.profile_report() {
            Some(r) if r.total_us > 0 => (r.total_us, r.prologue_breakdown.scaling_us),
            _ => (0, 0),
        };
        cum_scal += scal_us;
        cum_total += total_us;
        let pat = match prev_nnz {
            Some(p) if p != nnz => " (pattern changed)",
            _ => "",
        };
        prev_nnz = Some(nnz);
        let name = Path::new(a)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(a);
        println!(
            "{:<26}{:>9}{:>10}{:>9}{:>8.1}%{:>8}{:>16}{}",
            name,
            nnz,
            total_us,
            scal_us,
            100.0 * scal_us as f64 / total_us.max(1) as f64,
            if hit { "yes" } else { "NO" },
            inertia,
            pat
        );
    }
    println!();
    println!(
        "trajectory total: factor {} us, scaling {} us ({:.1}% of all factorizations)",
        cum_total,
        cum_scal,
        100.0 * cum_scal as f64 / cum_total.max(1) as f64
    );
    Ok(())
}
