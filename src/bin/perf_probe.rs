//! Repeatable per-factor performance probe for the perf-issue round
//! (#124–#134). Factors a matrix repeatedly with a **warm** `Solver`
//! (symbolic cache hot — the IPM-host workload: one pattern, drifting
//! values) and reports the numeric-phase wall and the profiled prologue
//! breakdown, so each perf fix can show a before/after number.
//!
//! Usage:
//!   cargo run --release --bin perf_probe -- <matrix.mtx> [iters]
//!
//! Env:
//!   FERAL_PROBE_SEQUENTIAL=1   use the sequential driver (default parallel)
//!
//! Output is a single line of `key=value` pairs plus a prologue breakdown,
//! stable enough to diff across commits.

use std::path::PathBuf;
use std::time::Instant;

use feral::{read_mtx, Inertia, NumericParams, Solver};

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v.get(v.len() / 2).copied().unwrap_or(0)
}

fn make_solver(sequential: bool, profiling: bool) -> Solver {
    Solver::with_params(NumericParams::default(), Default::default())
        .with_parallel(!sequential)
        .with_profiling(profiling)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: perf_probe <matrix.mtx> [iters]");
            std::process::exit(2);
        }
    };
    let iters: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50)
        .max(1);

    let csc = match read_mtx(&path).and_then(|m| m.to_csc()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("load failed: {e}");
            std::process::exit(1);
        }
    };
    let n = csc.n;
    let nnz = csc.row_idx.len();
    let sequential = std::env::var("FERAL_PROBE_SEQUENTIAL").is_ok();

    // Warm-up (profiling off for the clean timing number): one factor so the
    // symbolic cache, scaling cache, and pooled workspaces are all hot.
    let mut solver = make_solver(sequential, false);
    let _ = solver.factor(&csc, None);
    let baseline_inertia: Option<Inertia> = solver.inertia().cloned();
    let calls_before = solver.symbolic_call_count();

    // Timed warm re-factors — the measured path.
    let mut walls = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = solver.factor(&csc, None);
        walls.push(t0.elapsed().as_micros());
    }
    let symbolic_reran = solver.symbolic_call_count() != calls_before;
    let inertia_stable = solver.inertia().cloned() == baseline_inertia;

    println!(
        "matrix={} n={} nnz={} driver={} iters={} min_us={} median_us={} \
         symbolic_reran={} inertia_stable={} inertia={}",
        path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
        n,
        nnz,
        if sequential { "sequential" } else { "parallel" },
        iters,
        walls.iter().min().copied().unwrap_or(0),
        median(walls),
        symbolic_reran,
        inertia_stable,
        baseline_inertia
            .map(|i| i.to_string())
            .unwrap_or_else(|| "none".into()),
    );

    // Solve timing: repeated single-RHS solves on the warm factor (the
    // refinement/condition-estimation inner loop). Measures issue #126.
    let rhs = vec![1.0f64; n];
    let _ = solver.solve(&rhs); // warm
    let mut solve_walls = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = solver.solve(&rhs);
        solve_walls.push(t0.elapsed().as_nanos());
    }
    let solve_min_ns = solve_walls.iter().min().copied().unwrap_or(0);
    let mut sv: Vec<u128> = solve_walls;
    sv.sort_unstable();
    let solve_med_ns = sv.get(sv.len() / 2).copied().unwrap_or(0);
    println!("  solve_min_ns={solve_min_ns} solve_median_ns={solve_med_ns}");

    // A separate profiled solver to attribute the prologue (profiling adds
    // overhead, so it is kept out of the timing number above).
    let mut prof = make_solver(sequential, true);
    // Warm 3×: factor 1 is cold (hint off, no cache built); factor 2 builds
    // the permute cache (hint on, miss); factor 3+ hit it. Snapshot the 3rd so
    // the breakdown reflects the steady-state warm path.
    let _ = prof.factor(&csc, None);
    let _ = prof.factor(&csc, None);
    let _ = prof.factor(&csc, None); // measured snapshot = last (cache hit)
    if let Some(r) = prof.profile_report() {
        let b = &r.prologue_breakdown;
        println!(
            "  prologue_us={} loop_us={} total_us={} | permute_us={} \
             permute_from_triplets_us={} scaling_us={} symmetric_pattern_us={} \
             setup_us={} row_map_us={} infnorm_tol_us={}",
            r.prologue_us,
            r.loop_us,
            r.total_us,
            b.permute_us,
            b.permute_from_triplets_us,
            b.scaling_us,
            b.symmetric_pattern_us,
            b.setup_us,
            b.row_map_us,
            b.infnorm_tol_us,
        );
    } else {
        println!("  (no profile report — tiny/dense fast path?)");
    }
}
