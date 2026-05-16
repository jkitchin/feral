//! Probe the cascade-break L-perturbation behaviour.
//!
//! Reads a KKT mtx dump, factors it with three configurations:
//!
//!   1. cb=off       — `cascade_break_ratio = None, cascade_break_eps = None`
//!      (the current default after 2026-05-15)
//!   2. cb=default   — `Some(0.5), Some(1e-10)` (the previous auto-armed
//!      defaults; now requires explicit opt-in)
//!   3. cb=fa        — `Some(0.5), None` (ForceAccept variant)
//!
//! Picks a deterministic RHS, solves under each config (un-refined), and
//! reports `||Ax - b||_inf / ||b||_inf` plus the factor wall.
//!
//! Used by `dev/research/cascade-break-l-perturbation-2026-05-15.md`
//! to compare residuals: cb=off (clean), cb=default (small bounded
//! perturbation through Schur update), cb=fa (large residual unless
//! iterative refinement is applied).
//!
//! Usage:
//!     cargo run --release --bin probe_cascade_perturb -- <mtx> [reps]

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus, Solver};
use feral::symbolic::SupernodeParams;
use feral::{read_mtx, CscMatrix};
use std::path::Path;
use std::time::Instant;

fn cb_params(ratio: Option<f64>, eps: Option<f64>) -> NumericParams {
    NumericParams {
        cascade_break_ratio: ratio,
        cascade_break_eps: eps,
        ..NumericParams::default()
    }
}

fn factor_and_solve(
    label: &str,
    csc: &CscMatrix,
    np: NumericParams,
    rhs: &[f64],
    reps: usize,
) -> Option<Vec<f64>> {
    let mut solver = Solver::with_params(np, SupernodeParams::default());
    // Warm-up
    let _ = solver.factor(csc, None);

    let mut walls = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let status = solver.factor(csc, None);
        walls.push(t.elapsed().as_secs_f64() * 1e3);
        if !matches!(status, FactorStatus::Success) {
            println!("  [{label}] non-Success: {status:?}");
            return None;
        }
    }
    walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = walls[walls.len() / 2];
    let min = walls[0];
    let p90 = walls[(walls.len() * 9) / 10];

    let inertia = solver.inertia().cloned();
    let needs_ref = solver
        .factors()
        .map(|f| f.needs_refinement)
        .unwrap_or(false);

    let x = match solver.solve(rhs) {
        Ok(x) => x,
        Err(e) => {
            println!("  [{label}] solve error: {e:?}");
            return None;
        }
    };

    println!(
        "  [{label:<14}] factor min={min:6.2} med={med:6.2} p90={p90:6.2} ms  \
         inertia={:?} needs_refinement={needs_ref}",
        inertia,
    );
    Some(x)
}

fn relative_inf_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut max_diff: f64 = 0.0;
    let mut max_abs: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        max_diff = max_diff.max((x - y).abs());
        max_abs = max_abs.max(y.abs());
    }
    if max_abs > 0.0 {
        max_diff / max_abs
    } else {
        max_diff
    }
}

fn residual(csc: &CscMatrix, x: &[f64], rhs: &[f64]) -> f64 {
    let mut ax = vec![0.0f64; csc.n];
    csc.symv(x, &mut ax);
    let mut max_r: f64 = 0.0;
    let mut max_b: f64 = 0.0;
    for i in 0..csc.n {
        max_r = max_r.max((ax[i] - rhs[i]).abs());
        max_b = max_b.max(rhs[i].abs());
    }
    if max_b > 0.0 {
        max_r / max_b
    } else {
        max_r
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: probe_cascade_perturb <mtx> [reps]");
        std::process::exit(2);
    }
    let path = &args[0];
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);

    let mtx = read_mtx(Path::new(path)).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let n = csc.n;

    // Deterministic RHS: sin(i)*sqrt(i+1).
    let rhs: Vec<f64> = (0..n)
        .map(|i| (i as f64).sin() * ((i + 1) as f64).sqrt())
        .collect();

    println!("[{path}]  n={n}  nnz={}  reps={reps}", csc.row_idx.len());

    let x_off = factor_and_solve("cb=off", &csc, cb_params(None, None), &rhs, reps);
    let x_def = factor_and_solve(
        "cb=default",
        &csc,
        cb_params(Some(0.5), Some(1e-10)),
        &rhs,
        reps,
    );
    let x_fa = factor_and_solve("cb=fa", &csc, cb_params(Some(0.5), None), &rhs, reps);

    if let Some(x) = x_off.as_ref() {
        println!("  residual cb=off       = {:.3e}", residual(&csc, x, &rhs));
    }
    if let Some(x) = x_def.as_ref() {
        println!("  residual cb=default   = {:.3e}", residual(&csc, x, &rhs));
    }
    if let Some(x) = x_fa.as_ref() {
        println!("  residual cb=fa        = {:.3e}", residual(&csc, x, &rhs));
    }

    if let (Some(a), Some(b)) = (x_off.as_ref(), x_def.as_ref()) {
        let d = relative_inf_diff(b, a);
        println!("  ||x_cb=default - x_cb=off||_inf / ||x_cb=off||_inf = {d:.3e}");
    }
    if let (Some(a), Some(b)) = (x_off.as_ref(), x_fa.as_ref()) {
        let d = relative_inf_diff(b, a);
        println!("  ||x_cb=fa      - x_cb=off||_inf / ||x_cb=off||_inf = {d:.3e}");
    }

    Ok(())
}
