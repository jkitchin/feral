//! Where is the multi-RHS BLAS-3 crossover, actually?
//!
//!   cargo run -p feral-diagnostics --features kernel-probe \
//!     --bin probe_blas3_crossover --release [-- <name>...]
//!
//! `BLAS3_NRHS_THRESHOLD = 32` (`src/numeric/solve.rs:17`) has never been
//! measured. Its doc comment cites "the `k ~ 16` crossover from
//! `dev/research/multi-rhs.md` D3", but D3 is a design note written before
//! the kernels existed and simply asserts that BLAS-3 shapes pay off above
//! `k ~ 16`. `issue-57-blas3-panel.md:121` then picks 32 as "a conservative
//! crossover" relative to that assertion, and measures only
//! `nrhs` in {31, 32, 37, 64, 256}, on 2-D Laplacians. The band 2..31 —
//! where callers live — has no measurement behind it, and neither does any
//! KKT matrix.
//!
//! ## Method
//!
//! For each matrix and `nrhs`, both kernels are timed at the **same `nrhs`,
//! interleaved within each repetition**, in one process, via the
//! `kernel-probe` threshold override.
//!
//! This replaces a two-binary design that failed. Building stock (threshold
//! 32) and patched (threshold 2) binaries and alternating the runs sounds
//! interleaved but is not: a full sweep takes ~25 minutes, so "adjacent"
//! runs are 25 minutes apart. Two controls caught it — the looped
//! single-RHS path (byte-identical in both builds) drifted by up to 2.0x,
//! and the `nrhs >= 32` rows, where both builds run the *same* kernel and
//! the ratio must be exactly 1.00, read 0.86-0.90. That is the same blocked
//! measurement error that produced two retractions on 2026-08-20, one level
//! down. Discarded rather than reported.
//!
//! ## Accuracy
//!
//! The two kernels reassociate differently, so they do not agree bit-for-bit.
//! `tests/multi_rhs.rs:227,256` assert `max_diff == 0.0` between batched-
//! refined and per-column-refined at `nrhs` 24 and 20 — a contract that holds
//! only because `solve_sparse_many` stays on the rank-1 kernels below 32, and
//! that pounce's Schur path consumes
//! (`../pounce/crates/pounce-feral/src/schur.rs:303`). So this probe also
//! reports `max |rank1 - blas3|` per `nrhs`: the size of what a threshold cut
//! would actually trade away.

use feral::numeric::solve::{set_blas3_nrhs_threshold, solve_sparse_many_into, SolveManyWorkspace};
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::Path;
use std::time::Instant;

/// Odd, so the median is a measured sample rather than an average.
const REPS: usize = 9;

/// Dense around the two candidate crossovers (the asserted 16, the shipped
/// 32). 2 is the IPM predictor-corrector width; 48 is past the shipped
/// threshold, where BLAS-3 is already known to win.
const NRHS_SWEEP: &[usize] = &[2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 31, 32, 33, 40, 48];

/// `nrhs >= this` is false for any real `nrhs`, forcing the rank-1 kernel.
/// (`usize::MAX` itself is the sentinel for "no override".)
const FORCE_RANK1: usize = usize::MAX - 1;
/// `nrhs >= 0` is always true, forcing the BLAS-3 kernel.
const FORCE_BLAS3: usize = 0;

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[s.len() / 2]
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
    let Some(factors) = solver.factors() else {
        println!("SKIP {name}: no factors");
        return;
    };

    for &nrhs in NRHS_SWEEP {
        // Deterministic, well-scaled RHS block; column `c` is a shifted ramp
        // so no two columns are identical.
        let mut rhs = vec![0.0f64; n * nrhs];
        for c in 0..nrhs {
            for i in 0..n {
                rhs[c * n + i] = 1.0 + ((i + 3 * c) % 7) as f64 / 8.0;
            }
        }

        // Every buffer is allocated once, outside every timed region. A
        // per-replicate allocation inside a timed solve is what caused the
        // bench regression on 2026-08-20.
        let mut x_r = vec![0.0f64; n * nrhs];
        let mut x_b = vec![0.0f64; n * nrhs];
        let mut ws = SolveManyWorkspace::for_factors(factors, nrhs);

        let mut t_r = Vec::with_capacity(REPS);
        let mut t_b = Vec::with_capacity(REPS);

        // Warm both paths so neither pays the first-touch cost alone.
        set_blas3_nrhs_threshold(FORCE_RANK1);
        let _ = solve_sparse_many_into(factors, &rhs, nrhs, &mut x_r, &mut ws);
        set_blas3_nrhs_threshold(FORCE_BLAS3);
        let _ = solve_sparse_many_into(factors, &rhs, nrhs, &mut x_b, &mut ws);

        for _ in 0..REPS {
            set_blas3_nrhs_threshold(FORCE_RANK1);
            let t = Instant::now();
            if solve_sparse_many_into(factors, &rhs, nrhs, &mut x_r, &mut ws).is_err() {
                println!("SKIP {name}: rank1 nrhs={nrhs} failed");
                return;
            }
            t_r.push(t.elapsed().as_secs_f64() * 1e6);

            set_blas3_nrhs_threshold(FORCE_BLAS3);
            let t = Instant::now();
            if solve_sparse_many_into(factors, &rhs, nrhs, &mut x_b, &mut ws).is_err() {
                println!("SKIP {name}: blas3 nrhs={nrhs} failed");
                return;
            }
            t_b.push(t.elapsed().as_secs_f64() * 1e6);
        }

        // How far apart the two kernels' answers are, absolute and relative
        // to the solution scale. This is what a threshold cut trades away.
        let mut max_abs = 0.0f64;
        let mut max_x = 0.0f64;
        for i in 0..n * nrhs {
            max_abs = max_abs.max((x_r[i] - x_b[i]).abs());
            max_x = max_x.max(x_r[i].abs());
        }
        let rel = if max_x > 0.0 { max_abs / max_x } else { 0.0 };

        let r = median(&t_r);
        let b = median(&t_b);
        println!(
            "DATA {name} {nrhs} {r:.1} {b:.1} {:.4} {max_abs:.3e} {rel:.3e}",
            r / b
        );
    }
    // Leave the process on the shipped dispatch.
    set_blas3_nrhs_threshold(usize::MAX);
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
    println!("# both kernels timed at the same nrhs, interleaved, one process");
    println!(
        "# DATA <name> <nrhs> <rank1_us> <blas3_us> <rank1/blas3> <max_abs_diff> <max_rel_diff>"
    );
    for nm in &names {
        run(&Path::new("tests/data/large").join(format!("{nm}.mtx")));
    }
}
