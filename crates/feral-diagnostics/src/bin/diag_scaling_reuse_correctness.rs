//! diag_scaling_reuse_correctness — would reusing an earlier iterate's
//! MC64 scaling actually be CORRECT on later iterates?
//!
//! The value-bound gate (`src/scaling/value_bound.rs`) exists to stop
//! issue #38: a stale MC64 scaling silently corrupting inertia on warm
//! IPM replays. The 2026-05-21 Track-B2 rejection recorded that no
//! cheap value proxy separates "safe to reuse" from "matching
//! changed". So before touching the gate, the question is not whether
//! it rejects too often — it is whether the rejected reuses were
//! actually safe.
//!
//! This factors each iterate twice: once with the solver's own
//! strategy (fresh MC64 — the oracle), and once with
//! `ScalingStrategy::External` carrying the FIRST iterate's scaling
//! vector (what a perfectly permissive cache would do). It compares
//! inertia and residual. The oracle is a fresh independent
//! factorization, not this probe's own arithmetic.
//!
//! Usage: diag_scaling_reuse_correctness <iter0.mtx> <iter1.mtx> ...

use feral::scaling::ScalingStrategy;
use feral::symbolic::SupernodeParams;
use feral::{read_mtx, CscMatrix, FactorStatus, Inertia, NumericParams, Solver};
use std::path::Path;

/// ‖Ax − b‖∞ / (1 + ‖b‖∞) for the all-ones RHS.
fn rel_res(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    // Lower-triangle CSC of a symmetric matrix: each stored (i,j)
    // contributes to both ax[i] and ax[j] unless i == j.
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let v = a.values[k];
            ax[i] += v * x[j];
            if i != j {
                ax[j] += v * x[i];
            }
        }
    }
    let num = ax
        .iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f64, f64::max);
    let den = 1.0 + b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    num / den
}

fn factor_once(
    csc: &CscMatrix,
    strategy: Option<ScalingStrategy>,
) -> Option<(Inertia, f64, f64, Vec<f64>)> {
    let mut np = NumericParams::default();
    if let Some(s) = strategy {
        np.scaling = s;
    }
    let mut solver = Solver::with_params(np, SupernodeParams::default()).with_parallel(false);
    match solver.factor(csc, None) {
        FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
        other => {
            eprintln!("  factor failed: {other:?}");
            return None;
        }
    }
    let inertia = solver.inertia()?.clone();
    let scaling = solver.factors()?.scaling.clone();
    let b = vec![1.0; csc.n];
    let res = match solver.solve(&b) {
        Ok(x) => rel_res(csc, &x, &b),
        Err(_) => f64::NAN,
    };
    let res_ref = match solver.solve_refined(csc, &b) {
        Ok(x) => rel_res(csc, &x, &b),
        Err(_) => f64::NAN,
    };
    Some((inertia, res, res_ref, scaling))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: diag_scaling_reuse_correctness <iter0.mtx> <iter1.mtx> ...");
        std::process::exit(2);
    }
    // Baseline scaling: the first iterate's own MC64 vector.
    let first = read_mtx(Path::new(&args[0])).and_then(|m| m.to_csc())?;
    let (_, _, _, d0) = match factor_once(&first, None) {
        Some(t) => t,
        None => {
            eprintln!("baseline factorization failed");
            std::process::exit(1);
        }
    };
    println!("baseline scaling from {} (n={})", args[0], d0.len());
    println!(
        "{:<26}{:>7}{:>12}{:>10}{:>10}{:>12}{:>10}{:>10}{:>8}",
        "iterate", "nnz", "fresh(n,z)", "res", "res_ref", "reuse(n,z)", "res", "res_ref", "inertia"
    );
    for a in &args {
        let csc = match read_mtx(Path::new(a)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {a}: {e:?}");
                continue;
            }
        };
        let name = Path::new(a)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(a);
        if csc.n != d0.len() {
            println!(
                "{name:<26}{:>7}  n changed -- cache invalid by fingerprint",
                csc.nnz()
            );
            continue;
        }
        let fresh = factor_once(&csc, None);
        let reuse = factor_once(&csc, Some(ScalingStrategy::External(d0.clone())));
        match (fresh, reuse) {
            (Some((fi, fr, frr, _)), Some((ri, rr, rrr, _))) => {
                let same = fi.negative == ri.negative && fi.zero == ri.zero;
                println!(
                    "{:<26}{:>7}{:>12}{:>10.1e}{:>10.1e}{:>12}{:>10.1e}{:>10.1e}{:>8}",
                    name,
                    csc.nnz(),
                    format!("({},{})", fi.negative, fi.zero),
                    fr,
                    frr,
                    format!("({},{})", ri.negative, ri.zero),
                    rr,
                    rrr,
                    if same { "same" } else { "DIFFERS" }
                );
            }
            _ => println!("{name:<26} factorization failed"),
        }
    }
    Ok(())
}
