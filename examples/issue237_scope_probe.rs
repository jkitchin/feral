//! Issue #237 follow-up: per-phase telemetry of the parallel driver on
//! warm, same-pattern refactorizations (scaling, permute, tree scope,
//! factor wall), for an A/B of the parallel factorization phase
//! ("scope") between builds.
//!
//! Uses only public API that 0.23.0 also has, so the same file can be
//! built against an older checkout.
//!
//! Usage (thread count from RAYON_NUM_THREADS, as the Solver's pool):
//!   cargo run --release --example issue237_scope_probe -- [t] [nx] [iters]

use std::sync::Arc;
use std::time::Instant;

use feral::numeric::factorize::AtomicLockStats;
use feral::{CscMatrix, FactorStatus, NumericParams, Solver};

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

fn main() {
    let mut args = std::env::args().skip(1);
    let t: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2600);
    let nx: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);
    let base = chain_kkt(t, nx);
    let stats = Arc::new(AtomicLockStats::default());
    let np = NumericParams {
        parallel_telemetry: Some(Arc::clone(&stats)),
        ..NumericParams::default()
    };
    let mut solver = Solver::with_params(np, Default::default()).with_parallel(true);
    // ISSUE237_METIS=1 forces METIS nested dissection (what POUNCE
    // configures in #237); otherwise FERAL's default ordering.
    if std::env::var("ISSUE237_METIS").is_ok() {
        solver = solver.with_ordering(feral::symbolic::OrderingMethod::MetisND);
    }
    let mut m = base.clone();
    let (mut wall, mut scal, mut perm, mut scope) = (vec![], vec![], vec![], vec![]);
    for step in 0..iters + 2 {
        drift(&mut m, step);
        let s0 = stats.snapshot();
        let t0 = Instant::now();
        let st = solver.factor(&m, None);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        let s1 = stats.snapshot();
        if !matches!(st, FactorStatus::Success) {
            eprintln!("factor failed: {st:?}");
            std::process::exit(1);
        }
        if step >= 2 {
            wall.push(dt);
            scal.push((s1.phase_scaling_ns - s0.phase_scaling_ns) as f64 / 1e6);
            perm.push((s1.phase_permute_ns - s0.phase_permute_ns) as f64 / 1e6);
            scope.push((s1.phase_scope_ns - s0.phase_scope_ns) as f64 / 1e6);
        }
    }
    println!(
        "n={} nnz={} threads={} ordering={} snodes={} inertia={:?} median_ms: wall={:.1} scaling={:.1} permute={:.1} scope={:.1} | scope_all={:?}",
        base.n,
        base.values.len(),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into()),
        solver
            .factors()
            .map(|f| format!("{:?}", f.resolved_method))
            .unwrap_or_default(),
        solver.symbolic().map(|s| s.supernodes.len()).unwrap_or(0),
        solver.inertia().map(|i| (i.positive, i.negative, i.zero)),
        median(wall),
        median(scal),
        median(perm),
        median(scope.clone()),
        scope
            .iter()
            .map(|v| (v * 10.0).round() / 10.0)
            .collect::<Vec<_>>(),
    );
}
