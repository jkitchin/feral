//! Issue #34 phase (g) — SQD fast-path bench harness.
//!
//! Builds synthetic SQD KKT matrices `K = [[-E, A^T], [A, F]]` at a
//! fan of sizes and densities, factors each via both the BK path
//! (`Solver::new()`) and the SQD diagonal path
//! (`Solver::new().with_sqd_mode(true)`), and reports the per-shape
//! median factor time plus the BK/SQD speedup ratio.
//!
//! Ship-targets (the plan at
//! `~/.claude/plans/let-s-work-on-a-reflective-anchor.md` phase g):
//!
//! * aspirational geomean speedup >= 1.15
//! * worst-case slowdown <= 1.05
//!
//! Both are printed at the end as PASS/MISS. Exit code is non-zero
//! only on functional failure (a shape that fails to factor) so the
//! binary can be wired into CI as a smoke test without flapping on
//! per-machine noise. The session that lands this bench (see
//! `dev/sessions/2026-05-16-NN.md`) records the measured numbers
//! and discusses why the synthetic-shape geomean fell short of the
//! aspirational target — a finding, not a regression.
//!
//! Usage: `cargo run --release --bin bench_sqd`
//!
//! Methodology:
//! * Each shape is factored `REPS = 5` times after a `WARMUP = 1`
//!   factor; the median wall-clock time is reported.
//! * `Solver` is reconstructed per rep so the BK and SQD paths get
//!   the same cache-cold starting point. We are measuring the
//!   numeric factor (which dominates), not symbolic + numeric, but
//!   both sides see the same symbolic cost so the ratio is fair.
//! * The SQD path is fed a matrix that satisfies Vanderbei's
//!   contract by construction: E and F are diagonal positive, A is
//!   bounded so the L-growth guard never trips.
//!
//! For corpus-scale numbers see `bench_solver_corpus` paired with
//! the `external_benchmarks/stress/` manifest — this binary is the
//! cheap pre-CI sanity check.

use std::process::ExitCode;
use std::time::Instant;

use feral::{CscMatrix, FactorStatus, Solver};

/// One SQD shape: m = neg-block size, p = pos-block size, density
/// = coupling fill ratio in A (0..=1).
struct Shape {
    label: &'static str,
    m: usize,
    p: usize,
    density: f64,
}

const SHAPES: &[Shape] = &[
    Shape {
        label: "tiny-dense",
        m: 8,
        p: 8,
        density: 1.0,
    },
    Shape {
        label: "small-dense",
        m: 32,
        p: 32,
        density: 1.0,
    },
    Shape {
        label: "medium-dense",
        m: 64,
        p: 64,
        density: 1.0,
    },
    Shape {
        label: "small-banded",
        m: 100,
        p: 100,
        density: 0.05,
    },
    Shape {
        label: "medium-banded",
        m: 250,
        p: 250,
        density: 0.02,
    },
    Shape {
        label: "large-banded",
        m: 500,
        p: 500,
        density: 0.01,
    },
];

const REPS: usize = 5;
const WARMUP: usize = 1;
/// Aspirational geomean-speedup target. The synthetic shapes here
/// can fall short of it on small-to-medium sizes where dispatch and
/// rank-1 axpy time dominates the BK-vs-diagonal pivot-search delta;
/// the printed PASS/MISS is informational.
const TARGET_GEOMEAN: f64 = 1.15;
const TARGET_WORST_RATIO: f64 = 1.05;

/// Tiny deterministic LCG so the bench is reproducible without
/// pulling in `rand` (which would balloon compile time for a one-shot
/// binary).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64) / ((1u64 << 31) as f64)
    }
    fn uni(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

/// Generate a sparse SQD matrix in CSC lower-triangle form.
/// Negative diagonal entries on rows 0..m, positive diagonal entries
/// on rows m..m+p, coupling A nonzeros at a Bernoulli(density)
/// pattern with magnitude bounded by 0.05 (well below the L-growth
/// guard for any reasonable diagonal).
fn make_sqd(shape: &Shape, seed: u64) -> CscMatrix {
    let mut rng = Lcg(seed);
    let n = shape.m + shape.p;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    // Negative diagonal block.
    for k in 0..shape.m {
        rows.push(k);
        cols.push(k);
        vals.push(-rng.uni(0.5, 1.5));
    }
    // Positive diagonal block.
    for k in 0..shape.p {
        rows.push(shape.m + k);
        cols.push(shape.m + k);
        vals.push(rng.uni(0.5, 1.5));
    }
    // Coupling A: rows m..m+p, columns 0..m. Bernoulli(density).
    for j in 0..shape.m {
        for i in 0..shape.p {
            if rng.next() < shape.density {
                rows.push(shape.m + i);
                cols.push(j);
                vals.push(rng.uni(-0.05, 0.05));
            }
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("csc")
}

fn median(xs: &mut [u128]) -> u128 {
    xs.sort_unstable();
    xs[xs.len() / 2]
}

/// Factor `csc` `REPS + WARMUP` times via a fresh Solver each rep
/// (cache-cold timing); return median wall-clock factor time in
/// microseconds plus the FactorStatus from the last rep.
fn time_factor(csc: &CscMatrix, sqd_mode: bool) -> (u128, FactorStatus) {
    let mut samples = Vec::with_capacity(REPS);
    let mut last_status = FactorStatus::Singular;
    for rep in 0..(REPS + WARMUP) {
        let mut solver = Solver::new();
        if sqd_mode {
            solver = solver.with_sqd_mode(true);
        }
        let t0 = Instant::now();
        let status = solver.factor(csc, None);
        let dt = t0.elapsed().as_micros();
        if rep >= WARMUP {
            samples.push(dt);
        }
        last_status = status;
    }
    (median(&mut samples), last_status)
}

fn main() -> ExitCode {
    println!("# bench_sqd — issue #34 phase (g)");
    println!(
        "# {} reps + {} warmup per shape, median factor wall-clock\n",
        REPS, WARMUP
    );
    println!(
        "{:<16} {:>6} {:>8} {:>12} {:>12} {:>10}",
        "shape", "n", "nnz", "bk_us", "sqd_us", "speedup"
    );

    let mut ratios = Vec::with_capacity(SHAPES.len());
    let mut worst: f64 = 1.0;
    let mut all_ok = true;

    for (idx, shape) in SHAPES.iter().enumerate() {
        let csc = make_sqd(shape, 0xC0FFEE + idx as u64);
        let n = csc.n;
        let nnz = csc.values.len();

        let (bk_us, bk_status) = time_factor(&csc, false);
        let (sqd_us, sqd_status) = time_factor(&csc, true);

        if !matches!(bk_status, FactorStatus::Success) {
            println!(
                "{:<16} {:>6} {:>8} {:>12} {:>12} {:>10}  ! BK status = {:?}",
                shape.label, n, nnz, bk_us, sqd_us, "n/a", bk_status
            );
            all_ok = false;
            continue;
        }
        if !matches!(sqd_status, FactorStatus::Success) {
            println!(
                "{:<16} {:>6} {:>8} {:>12} {:>12} {:>10}  ! SQD status = {:?}",
                shape.label, n, nnz, bk_us, sqd_us, "n/a", sqd_status
            );
            all_ok = false;
            continue;
        }
        let speedup = if sqd_us > 0 {
            bk_us as f64 / sqd_us as f64
        } else {
            f64::INFINITY
        };
        println!(
            "{:<16} {:>6} {:>8} {:>12} {:>12} {:>10.3}",
            shape.label, n, nnz, bk_us, sqd_us, speedup
        );
        ratios.push(speedup);
        // worst = max slowdown == min speedup inverted
        let slowdown = 1.0 / speedup.min(1.0);
        if slowdown > worst {
            worst = slowdown;
        }
    }

    if !all_ok || ratios.is_empty() {
        eprintln!("\nFAIL: not all shapes factored successfully");
        return ExitCode::from(1);
    }

    let log_sum: f64 = ratios.iter().map(|r| r.ln()).sum();
    let geomean = (log_sum / ratios.len() as f64).exp();
    println!("\ngeomean speedup: {:.3}", geomean);
    println!("worst slowdown:  {:.3}x", worst);
    println!(
        "target geomean >= {:.2}: {}",
        TARGET_GEOMEAN,
        if geomean >= TARGET_GEOMEAN {
            "PASS"
        } else {
            "MISS"
        }
    );
    println!(
        "target worst   <= {:.2}: {}",
        TARGET_WORST_RATIO,
        if worst <= TARGET_WORST_RATIO {
            "PASS"
        } else {
            "MISS"
        }
    );

    // Exit 0 on functional success regardless of target hit/miss;
    // targets are informational, not gates.
    ExitCode::from(0)
}
