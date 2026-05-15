//! Issue #19 reality check: does the work-aware flop gate buy
//! anything on robot_1600 *after* the persistent rayon ThreadPool
//! reuse landed in `91e028a`?
//!
//! Reads a robot_1600 KKT dump, reports the symbolic shape (n,
//! nnz, n_supernodes, est_flops, what the default gate decides),
//! then times `Solver::factor()` under three configurations:
//!
//!   1. `parallel=false` (sequential, the workaround the issue
//!      reporter used).
//!   2. `parallel=true` with `min_parallel_flops=None` — default,
//!      the gate decides.
//!   3. `parallel=true` with `min_parallel_flops=Some(0)` — gate
//!      disabled, parallel forced. The pre-#19 default behaviour.
//!
//! If (2) ≈ (1) and (3) >> (1), the gate is doing useful work on
//! this workload. If (1) ≈ (2) ≈ (3), the gate is irrelevant once
//! ThreadPool reuse is in place (and lowering `PAR_MIN_FLOPS`
//! based on Poisson-KKT calibration would be a red herring).
//!
//! Usage:
//!     cargo run --release --bin probe_issue_19 -- <mtx> [reps]
//!
//! Default reps = 30 (enough to median through OS noise).

use feral::numeric::factorize::{
    estimate_assembly_flops, should_parallelize_assembly, NumericParams, PAR_MIN_FLOPS,
};
use feral::numeric::solver::Solver;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, CscMatrix};
use std::path::Path;
use std::time::Instant;

fn time_factor(label: &str, csc: &CscMatrix, mut solver: Solver, reps: usize) {
    // Warm-up to let the persistent ThreadPool build and caches warm.
    let _ = solver.factor(csc, None);

    let mut wall = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let status = solver.factor(csc, None);
        wall.push(t.elapsed().as_secs_f64() * 1e3);
        if !matches!(status, feral::numeric::solver::FactorStatus::Success) {
            println!("  [{label}] factor returned non-Success: {status:?}");
        }
    }
    wall.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = wall[wall.len() / 2];
    let min = wall[0];
    let p90 = wall[(wall.len() * 9) / 10];
    println!(
        "  [{label:<28}] min={min:6.2} med={med:6.2} p90={p90:6.2} ms  (n={})",
        wall.len()
    );
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: probe_issue_19 <mtx> [reps]");
        std::process::exit(2);
    }
    let path = &args[0];
    let reps: usize = args.get(1).map(|s| s.parse().unwrap_or(30)).unwrap_or(30);

    let mtx = read_mtx(Path::new(path)).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let flops = estimate_assembly_flops(&sym.supernodes);
    let gate_default = should_parallelize_assembly(&sym);
    let mc = sym
        .supernodes
        .iter()
        .filter(|s| s.children.len() >= 2)
        .count();

    println!("[{path}]");
    println!(
        "  n={}  nnz={}  n_snodes={}  multi_child={}  est_flops={}  ({:.2e})",
        csc.n,
        csc.row_idx.len(),
        sym.supernodes.len(),
        mc,
        flops,
        flops as f64,
    );
    println!(
        "  default gate (PAR_MIN_FLOPS={}={:.0e}): {}",
        PAR_MIN_FLOPS,
        PAR_MIN_FLOPS as f64,
        if gate_default {
            "PARALLEL"
        } else {
            "sequential"
        },
    );
    println!("  reps={reps}");

    // (1) parallel=false. Sequential driver, no ThreadPool, no gate.
    let s_seq = Solver::new().with_parallel(false);
    time_factor("parallel=false (sequential)", &csc, s_seq, reps);

    // (2) parallel=true, gate default (=current PAR_MIN_FLOPS).
    let s_gated = Solver::new().with_parallel(true);
    time_factor("parallel=true gate=default", &csc, s_gated, reps);

    // (3) parallel=true, gate forced off via min_parallel_flops=0.
    let np = NumericParams {
        min_parallel_flops: Some(0),
        ..NumericParams::default()
    };
    let s_forced = Solver::with_params(np, SupernodeParams::default()).with_parallel(true);
    time_factor("parallel=true gate=OFF", &csc, s_forced, reps);

    Ok(())
}
