//! Is the bench harness's large-n solve reading reproducible?
//!
//!   cargo run -p feral-diagnostics --bin probe_bench_solve_noise --release [-- <name>...]
//!
//! `src/bin/bench.rs:1677-1684` times one solve, once:
//!
//! ```ignore
//! let t1 = Instant::now();
//! let x = solve_refined(&matrix, &factors, &rhs)?;
//! let solve_us = t1.elapsed().as_micros();
//! ```
//!
//! and only replicates it when the MUMPS oracle says the *factor* took
//! under 200 us (`should_resample`, `:1042`) — a small-matrix denoise.
//! Large matrices, the ones feral#189 wants to gate, get a single sample.
//!
//! This measures what that single sample is worth. For each matrix it
//! repeats the bench's exact solve call `REPS` times and reports the
//! spread, then reports what three candidate reductions (min, median,
//! median-of-5-medians) would have yielded. The question it answers is
//! Step 1 of `dev/plans/large-n-solve-gate.md`: how noisy is the reading,
//! and which reduction makes it gateable.
//!
//! Deliberately uses `solve_sparse_refined` — the same serial
//! shared-vector entry the bench reaches (`src/numeric/solve.rs:2077`
//! passes `SolveCore::SharedVector`) — so this characterises the harness
//! as it exists, not as it might be.

use feral::numeric::solve::{solve_sparse_refined, RefineOptions};
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::Path;
use std::time::Instant;

/// Enough repeats to see a tail without turning this into a corpus run.
const REPS: usize = 15;
/// Group size for the "median of medians" reduction, matching
/// `RESAMPLE_COLD_REPS` in the bench.
const GROUP: usize = 5;

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

fn run(path: &Path) {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let mtx = match read_mtx(path) {
        Ok(m) => m,
        Err(e) => {
            println!("SKIP {name}: read_mtx: {e}");
            return;
        }
    };
    let csc: CscMatrix = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            println!("SKIP {name}: to_csc: {e}");
            return;
        }
    };
    let n = csc.n;

    let mut solver = Solver::new();
    match solver.factor(&csc, None) {
        FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
        other => {
            println!("SKIP {name}: factor {other:?}");
            return;
        }
    }
    let factors = match solver.factors() {
        Some(f) => f,
        None => {
            println!("SKIP {name}: no factors");
            return;
        }
    };

    // Well-scaled deterministic RHS with a known O(1) solution, so the
    // refinement loop does a representative amount of work rather than
    // converging instantly on a trivial vector.
    let mut rhs = vec![0.0f64; n];
    let mut v = vec![0.0f64; n];
    for (i, s) in v.iter_mut().enumerate() {
        *s = 1.0 + (i % 7) as f64 / 8.0;
    }
    csc.symv(&v, &mut rhs);

    let mut samples: Vec<f64> = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        match solve_sparse_refined(&csc, factors, &rhs) {
            Ok(x) => {
                std::hint::black_box(&x);
            }
            Err(e) => {
                println!("SKIP {name}: solve: {e}");
                return;
            }
        }
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }

    let raw = samples.clone();
    let mut sorted = samples;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = sorted[0];
    let hi = sorted[sorted.len() - 1];
    let med = pct(&sorted, 0.50);

    // What each candidate reduction would report, and how much the
    // *reported value* still moves between independent groups of GROUP.
    let mut group_meds: Vec<f64> = Vec::new();
    let mut group_mins: Vec<f64> = Vec::new();
    for g in raw.chunks(GROUP) {
        let mut s = g.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        group_meds.push(s[s.len() / 2]);
        group_mins.push(s[0]);
    }
    let spread = |v: &[f64]| -> f64 {
        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if mn > 0.0 {
            mx / mn
        } else {
            f64::NAN
        }
    };

    println!(
        "{name:<20} n={n:<8} serial   {lo:9.0}..{hi:9.0} us  med {med:9.0}  \
         spread {:.2}x | med-of-{GROUP} {:.2}x | min-of-{GROUP} {:.2}x",
        hi / lo,
        spread(&group_meds),
        spread(&group_mins),
    );

    // Same measurement, same reps, on the PARALLEL schedule. The earlier
    // levers probe saw up to 2.20x drift on this path across runs, while
    // the serial numbers above are tight -- but those were different
    // probes on different days. Measuring both here, back to back, is the
    // only way to attribute the noise to the schedule rather than to the
    // machine or the method.
    let mut psolver = Solver::new().with_parallel(true);
    match psolver.factor(&csc, None) {
        FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
        other => {
            println!("{:<20} parallel: factor {other:?}", "");
            return;
        }
    }
    let opts = RefineOptions::default();
    let mut px = vec![0.0f64; n];
    let mut psamples: Vec<f64> = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        if let Err(e) = psolver.solve_refined_into(&csc, &rhs, &mut px, opts) {
            println!("{:<20} parallel: solve: {e}", "");
            return;
        }
        std::hint::black_box(&px);
        psamples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let praw = psamples.clone();
    let mut psorted = psamples;
    psorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (plo, phi) = (psorted[0], psorted[psorted.len() - 1]);
    let pmed = pct(&psorted, 0.50);
    let mut pgm: Vec<f64> = Vec::new();
    let mut pgn: Vec<f64> = Vec::new();
    for g in praw.chunks(GROUP) {
        let mut s = g.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        pgm.push(s[s.len() / 2]);
        pgn.push(s[0]);
    }
    println!(
        "{:<20} {:<10} parallel {plo:9.0}..{phi:9.0} us  med {pmed:9.0}  \
         spread {:.2}x | med-of-{GROUP} {:.2}x | min-of-{GROUP} {:.2}x   (par/ser med {:.2}x)",
        "",
        "",
        phi / plo,
        spread(&pgm),
        spread(&pgn),
        med / pmed,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names: Vec<String> = if args.is_empty() {
        vec![
            "bcsstk38".into(),
            "r05_kkt".into(),
            "bratu3d".into(),
            "qap15_kkt".into(),
            "dirichlet120_kkt".into(),
            "cont-201".into(),
            "cont5_late_kkt".into(),
        ]
    } else {
        args
    };
    println!(
        "bench single-shot solve reproducibility ({REPS} reps, serial shared-vector, 1 RHS)\n"
    );
    for nm in &names {
        run(&Path::new("tests/data/large").join(format!("{nm}.mtx")));
    }
}
