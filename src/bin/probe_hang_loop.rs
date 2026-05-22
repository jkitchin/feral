//! Probe for issue #49 — the intermittent parallel-factorization hang.
//!
//! Factors one KKT `.mtx` `REPEAT` times on a single warm `Solver`
//! (the analyze-once / refactor-many path POUNCE drives) and prints
//! per-call wall time, flushing after every line. A genuine hang shows
//! up as a stalled line; a soft outlier shows up as a time spike. The
//! goal is to turn the ~1/200-per-factorization race into something
//! reproducible by sheer repetition.
//!
//! Parallel by default; pass `serial` as arg 3 to drive the unblocked
//! path for an A/B against the parallel scheduler.
//!
//! This is a diagnostic probe; the relaxed probe-bin convention
//! (unwrap/expect permitted) applies.
//!
//! Usage: `cargo run --release --bin probe_hang_loop -- KKT.mtx REPEAT [serial]`

use std::env;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use feral::{read_mtx, Solver};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} KKT.mtx REPEAT [serial]", args[0]);
        std::process::exit(2);
    }
    let repeat: usize = args[2].parse().expect("REPEAT parse");
    let parallel = args.get(3).map(|s| s != "serial").unwrap_or(true);

    let mtx = read_mtx(Path::new(&args[1])).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    println!(
        "matrix {}  n={} stored_nnz={}  mode={}",
        args[1],
        csc.n,
        csc.row_idx.len(),
        if parallel { "parallel" } else { "serial" }
    );
    let _ = std::io::stdout().flush();

    let mut s = Solver::new().with_parallel(parallel);
    let mut times = Vec::with_capacity(repeat);
    let mut worst = 0.0f64;
    let mut worst_call = 0usize;

    for call in 0..repeat {
        let t = Instant::now();
        let status = s.factor(&csc, None);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        times.push(ms);
        if ms > worst {
            worst = ms;
            worst_call = call;
        }
        // Flag a spike: > 5x the warm median seen so far.
        let mut sorted: Vec<f64> = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = sorted[sorted.len() / 2];
        let spike = times.len() > 4 && ms > 5.0 * med;
        println!(
            "  call {call:>5}  factor_ms={ms:>10.1}  med={med:>8.1}  {status:?}{}",
            if spike { "   <<< SPIKE" } else { "" }
        );
        let _ = std::io::stdout().flush();
    }

    let mut sorted: Vec<f64> = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2];
    let p10 = sorted[sorted.len() / 10];
    let total: f64 = times.iter().sum();
    println!(
        "summary  calls={repeat}  median={med:.1}ms  fast(p10)={p10:.1}ms  \
         worst={worst:.1}ms@call{worst_call}  worst/median={:.1}x  total={:.1}s",
        worst / med,
        total / 1e3
    );
}
