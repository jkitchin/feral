//! Issue #237 step 1: warm-path cost of the value permutation P·A·Pᵀ on
//! an IPM-style sequence of same-pattern refactorizations.
//!
//! Builds a chain-structured (block-banded) saddle-point KKT shaped like
//! the issue's direct-collocation system — `t` dense blocks of `nx`
//! variables, each coupled to the next by `nx` constraint rows — and
//! factors it repeatedly with one warm `Solver`, perturbing the values
//! between calls the way an interior-point iteration does (pattern fixed).
//!
//! Reports:
//! - sequential driver: median profiled `permute_us` and factor wall;
//! - parallel driver: median factor wall;
//! - a bitwise digest of one solve after the last factorization, so a
//!   change that must not move a bit can be checked against the previous
//!   build's output.
//!
//! The inertia has an analytic oracle: `t·nx` positive (the H block is
//! strictly diagonally dominant and positive) and `(t-1)·nx` negative.
//!
//! Usage:
//!   cargo run --release --example issue237_permute_warm -- [t] [nx] [iters]
//!
//! The last line runs the profiled sequential driver on a rayon worker,
//! where the permute's parallel gather engages.

use std::time::Instant;

use feral::{CscMatrix, FactorStatus, Solver};

fn chain_kkt(t: usize, nx: usize) -> CscMatrix {
    let n_vars = t * nx;
    let m = (t - 1) * nx;
    let n = n_vars + m;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut absrow = vec![0.0f64; n];
    let mut push = |r: usize, c: usize, v: f64, absrow: &mut Vec<f64>| {
        let (r, c) = if r >= c { (r, c) } else { (c, r) };
        rows.push(r);
        cols.push(c);
        vals.push(v);
        absrow[r] += v.abs();
        if r != c {
            absrow[c] += v.abs();
        }
    };
    for b in 0..t {
        for i in 0..nx {
            for j in 0..i {
                let v = 0.25 + 0.01 * (((b * 31 + i * 7 + j) % 13) as f64);
                push(b * nx + i, b * nx + j, v, &mut absrow);
            }
        }
    }
    for b in 0..(t - 1) {
        for i in 0..nx {
            let row = n_vars + b * nx + i;
            for j in 0..nx {
                let v = 0.5 + 0.01 * (((b + i * 3 + j) % 11) as f64);
                push(row, b * nx + j, v, &mut absrow);
                push(row, (b + 1) * nx + j, -v, &mut absrow);
            }
        }
    }
    for (k, &a) in absrow.iter().enumerate() {
        let d = if k < n_vars { a + 1.0 } else { -1e-2 };
        rows.push(k);
        cols.push(k);
        vals.push(d);
    }
    match CscMatrix::from_triplets(n, &rows, &cols, &vals) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("from_triplets failed: {e}");
            std::process::exit(1);
        }
    }
}

/// Deterministic multiplicative drift, standing in for one IPM step.
/// Keeps the sign of every entry, so the analytic inertia is unchanged
/// (diagonal dominance has a margin of 1.0 that a ±1 % drift of the
/// off-diagonals cannot erase at these sizes... the run checks it).
fn drift(m: &mut CscMatrix, step: usize) {
    for (k, v) in m.values.iter_mut().enumerate() {
        let u = ((k.wrapping_mul(2654435761) ^ step.wrapping_mul(40503)) % 1000) as f64;
        *v *= 1.0 + 0.002 * (u / 1000.0 - 0.5);
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn run(base: &CscMatrix, parallel: bool, profiling: bool, iters: usize) -> (f64, f64, u64) {
    let mut m = base.clone();
    let mut solver = Solver::new()
        .with_parallel(parallel)
        .with_profiling(profiling);
    // ISSUE237_METIS=1: MetisND (POUNCE's #237 configuration).
    // ISSUE237_REUSE=1: opt-in InfNorm reuse with the default policy.
    if std::env::var("ISSUE237_METIS").is_ok() {
        solver = solver.with_ordering(feral::symbolic::OrderingMethod::MetisND);
    }
    if std::env::var("ISSUE237_REUSE").is_ok() {
        solver = solver.with_infnorm_reuse(Some(feral::InfNormReuse::default()));
    }
    let mut walls = Vec::with_capacity(iters);
    let mut permutes = Vec::with_capacity(iters);
    let mut scalings = Vec::with_capacity(iters);
    // Calls 0 and 1 are cold (analysis; cache build). Time 2.. .
    for step in 0..iters + 2 {
        drift(&mut m, step);
        let t0 = Instant::now();
        let status = solver.factor(&m, None);
        if !matches!(status, FactorStatus::Success) {
            eprintln!("factor failed at step {step}: {status:?}");
            std::process::exit(1);
        }
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if step >= 2 {
            walls.push(dt);
            if let Some(r) = solver.profile_report() {
                permutes.push(r.prologue_breakdown.permute_us as f64 / 1e3);
                scalings.push(r.prologue_breakdown.scaling_us as f64 / 1e3);
            }
        }
    }
    let n = base.n;
    let n_vars_neg = solver.inertia().map(|i| (i.positive, i.negative, i.zero));
    let rhs: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
    let x = match solver.solve(&rhs) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("solve failed: {e}");
            std::process::exit(1);
        }
    };
    // Refined solve of the last system: relative residual, as a check
    // that reuse does not cost accuracy after refinement.
    let refined_rel_res = match solver.solve_refined(&m, &rhs) {
        Ok(xr) => {
            let mut r = rhs.clone();
            for j in 0..n {
                for k in m.col_ptr[j]..m.col_ptr[j + 1] {
                    let i = m.row_idx[k];
                    r[i] -= m.values[k] * xr[j];
                    if i != j {
                        r[j] -= m.values[k] * xr[i];
                    }
                }
            }
            let nr = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            let nb = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
            nr / nb
        }
        Err(_) => f64::NAN,
    };
    let mut h: u64 = 0xcbf29ce484222325;
    for v in &x {
        h ^= v.to_bits();
        h = h.wrapping_mul(0x100000001b3);
    }
    println!(
        "  driver={} profiling={} inertia(+,-,0)={:?} factor_median_ms={:.2} \
         permute_median_ms={:.3} scaling_median_ms={:.3} reuse_hits={} \
         refined_rel_res={:.2e} solve_digest={:016x}",
        if parallel { "parallel" } else { "sequential" },
        profiling,
        n_vars_neg,
        median(walls.clone()),
        median(permutes),
        median(scalings),
        solver.infnorm_reuse_hit_count(),
        refined_rel_res,
        h
    );
    (median(walls), 0.0, h)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let t: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2600);
    let nx: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(15);
    let a = chain_kkt(t, nx);
    println!(
        "chain_kkt t={t} nx={nx}: n={} nnz={} expected inertia (+,-,0)=({}, {}, 0) threads={}",
        a.n,
        a.values.len(),
        t * nx,
        (t - 1) * nx,
        rayon::current_num_threads()
    );
    run(&a, false, true, iters);
    run(&a, false, false, iters);
    run(&a, true, false, iters);
    // Profiled sequential driver run on a rayon worker: the permute's
    // parallel gather engages there (it does inside a parallel Solver's
    // pool too, where the profiler cannot attribute it).
    match rayon::ThreadPoolBuilder::new().build() {
        Ok(pool) => {
            println!("  (next line: sequential driver called on a rayon worker)");
            pool.install(|| run(&a, false, true, iters));
        }
        Err(e) => eprintln!("pool build failed: {e}"),
    }
}
