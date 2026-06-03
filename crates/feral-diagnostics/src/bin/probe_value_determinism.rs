//! Probe: is feral's *parallel* factorization value-deterministic?
//!
//! The suspect is thread scheduling: parallel floating-point reduction
//! is order-associative-dependent, so a parallel factorization could
//! produce ULP-different L/D — hence a ULP-different solution — run to
//! run, while taking roughly the same time every time. `probe_hang_loop`
//! only measured wall time and so could never have caught this.
//!
//! This probe factors ONE fixed `.mtx` `REPEAT` times on feral's
//! PARALLEL path, solves a fixed RHS (all ones), and records the exact
//! bit pattern of the solution. Identical input must give an identical
//! bit pattern if the parallel path is value-deterministic.
//!
//!   distinct_solutions == 1  -> value-deterministic
//!   distinct_solutions  > 1  -> value-NONdeterministic  <<< the bug
//!
//! Per-run wall time is reported (min/median/max) so thread engagement
//! is visible: with N>1 rayon workers the median must drop well below
//! the `RAYON_NUM_THREADS=1` baseline, otherwise the "parallel" run was
//! silently serial and proves nothing. `cold` builds a fresh `Solver`
//! per call; `warm` reuses one (the analyze-once/refactor-many path
//! POUNCE drives, carrying the B2 cache and reused scratch).
//!
//! Serial is intentionally NOT tested — a serial path cannot exhibit a
//! thread-scheduling race, so it carries no signal here.
//!
//! This is a diagnostic probe; the relaxed probe-bin convention
//! (unwrap/expect permitted) applies.
//!
//! Usage: `cargo run --release --bin probe_value_determinism -- KKT.mtx [REPEAT]`
//!   set `RAYON_NUM_THREADS` to control / stress the worker count.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::env;
use std::hash::Hasher;
use std::path::Path;
use std::time::Instant;

use feral::{read_mtx, Solver};

/// Deterministic hash of a solution vector's exact bit patterns.
/// `DefaultHasher::new()` uses fixed keys, so this is stable across the
/// process — any difference in the hash means a difference in the bits.
fn hash_bits(v: &[f64]) -> u64 {
    let mut h = DefaultHasher::new();
    for &x in v {
        h.write_u64(x.to_bits());
    }
    h.finish()
}

/// `warm == true` reuses ONE `Solver` across every repetition.
fn run_mode(csc: &feral::CscMatrix, rhs: &[f64], repeat: usize, warm: bool) {
    let label = if warm {
        "parallel warm"
    } else {
        "parallel cold"
    };
    let mut hashes: Vec<u64> = Vec::with_capacity(repeat);
    let mut inertias: Vec<String> = Vec::with_capacity(repeat);
    let mut times_ms: Vec<f64> = Vec::with_capacity(repeat);
    let mut reference: Option<Vec<f64>> = None;
    let mut worst_diff = 0.0f64;
    let mut worst_diff_call = 0usize;
    let mut failures = 0usize;

    let mut warm_solver = Solver::new().with_parallel(true);

    for call in 0..repeat {
        let mut fresh;
        let s: &mut Solver = if warm {
            &mut warm_solver
        } else {
            fresh = Solver::new().with_parallel(true);
            &mut fresh
        };
        let t = Instant::now();
        let status = s.factor(csc, None);
        let sol = s.solve(rhs);
        times_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let inertia = s
            .inertia()
            .map(|i| format!("{i:?}"))
            .unwrap_or_else(|| "none".to_string());
        match sol {
            Ok(sol) => {
                hashes.push(hash_bits(&sol));
                inertias.push(inertia);
                match &reference {
                    None => reference = Some(sol),
                    Some(r) => {
                        let d = r
                            .iter()
                            .zip(&sol)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f64, f64::max);
                        if d > worst_diff {
                            worst_diff = d;
                            worst_diff_call = call;
                        }
                    }
                }
            }
            Err(_) => {
                failures += 1;
                hashes.push(0);
                inertias.push(format!("SOLVE_FAIL({status:?})"));
            }
        }
    }

    let distinct_hashes: HashSet<u64> = hashes.iter().copied().collect();
    let distinct_inertia: HashSet<&String> = inertias.iter().collect();
    let mut sorted = times_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (tmin, tmed, tmax) = (
        sorted[0],
        sorted[sorted.len() / 2],
        sorted[sorted.len() - 1],
    );
    let verdict = if distinct_hashes.len() == 1 && failures == 0 {
        "VALUE-DETERMINISTIC"
    } else {
        "VALUE-NONDETERMINISTIC <<<"
    };
    println!(
        "  {label}  runs={repeat}  distinct_solutions={}  distinct_inertia={}  \
         max_dev={worst_diff:.3e}@call{worst_diff_call}  solve_fail={failures}",
        distinct_hashes.len(),
        distinct_inertia.len(),
    );
    println!(
        "                 wall_ms  min={tmin:.1}  median={tmed:.1}  max={tmax:.1}  \
         max/median={:.1}x   -> {verdict}",
        tmax / tmed
    );
    if distinct_hashes.len() > 1 {
        for h in &distinct_hashes {
            let cnt = hashes.iter().filter(|&&x| x == *h).count();
            println!("      solution hash {h:016x}  seen {cnt}x");
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} KKT.mtx [REPEAT]", args[0]);
        std::process::exit(2);
    }
    let repeat: usize = args
        .get(2)
        .map(|s| s.parse().expect("REPEAT"))
        .unwrap_or(300);

    let mtx = read_mtx(Path::new(&args[1])).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let rhs = vec![1.0f64; csc.n];
    println!(
        "matrix {}  n={}  stored_nnz={}  repeat={}  rayon_threads={}",
        args[1],
        csc.n,
        csc.row_idx.len(),
        repeat,
        rayon::current_num_threads(),
    );

    run_mode(&csc, &rhs, repeat, false);
    run_mode(&csc, &rhs, repeat, true);
}
