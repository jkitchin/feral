//! Measure what issue #190's stopping criteria buy on IPM-scale KKT
//! matrices.
//!
//!   cargo run -p feral-diagnostics --bin probe_refine_stop_criterion --release [-- <name>...]
//!
//! #190's claim is that the hardwired `eps*sqrt(n)` target is unreachable
//! at IPM scale, so every refined solve runs the full `max_steps` budget
//! and pays for corrections the caller did not need. This probe tests that
//! claim directly: for each matrix it factors once, then runs the same
//! solve under several `StopCriterion` settings, reporting steps taken,
//! which exit fired, wall time, and the *quality actually delivered*
//! (relative residual and componentwise backward error `omega`).
//!
//! The comparison that matters is not "fewer steps" — that is trivially
//! true of any early exit — but "fewer steps at equivalent backward
//! quality". So every row carries its own omega, and the default row is
//! the reference.
//!
//! Everything goes through `Solver::solve_refined_into`, the entry point a
//! host actually calls, so the numbers are the ones pounce would see.

use feral::numeric::solve::{RefineOptions, RefineStop};
use feral::{read_mtx, CscMatrix, FactorStatus, Solver, DEFAULT_REFINE_MAX_STEPS};
use std::path::Path;
use std::time::Instant;

/// Best-of-`REPS` wall clock: these are timings on a shared machine and
/// the minimum is the least noisy estimator of the achievable cost.
const REPS: usize = 5;

fn residual_and_omega(a: &CscMatrix, x: &[f64], b: &[f64]) -> (f64, f64) {
    let n = a.n;
    let mut ax = vec![0.0f64; n];
    a.symv(x, &mut ax);
    let r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
    let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rel = if bn > 0.0 { rn / bn } else { rn };

    let mut d = vec![0.0f64; n];
    a.abs_symv(x, &mut d);
    let safe1 = ((n + 1) as f64) * f64::MIN_POSITIVE;
    let safe2 = safe1 / f64::EPSILON;
    let mut omega = 0.0f64;
    for i in 0..n {
        let den = d[i] + b[i].abs();
        if den == 0.0 {
            continue;
        }
        let v = if den > safe2 {
            r[i].abs() / den
        } else {
            (r[i].abs() + safe1) / (den + safe1)
        };
        if v > omega {
            omega = v;
        }
    }
    (rel, omega)
}

/// Deterministic well-scaled RHS with a known O(1) solution. This is the
/// benign case: every entry of `b` is the same order of magnitude.
fn build_rhs_easy(a: &CscMatrix) -> Vec<f64> {
    let n = a.n;
    let mut v = vec![0.0f64; n];
    for (i, s) in v.iter_mut().enumerate() {
        *s = 1.0 + (i % 7) as f64 / 8.0;
    }
    let mut b = vec![0.0f64; n];
    a.symv(&v, &mut b);
    b
}

/// Badly-scaled RHS: entries span twelve orders of magnitude and
/// alternate sign. This is the case the componentwise criterion exists
/// for -- `||r||_2/||b||_2` is dominated by the few large rows and says
/// nothing about the small ones, while `omega` weighs every row against
/// its own magnitude.
fn build_rhs_hard(a: &CscMatrix) -> Vec<f64> {
    let n = a.n;
    let mut v = vec![0.0f64; n];
    for (i, s) in v.iter_mut().enumerate() {
        let mag = 10f64.powi(((i % 13) as i32) - 6);
        *s = if i % 2 == 0 { mag } else { -mag };
    }
    let mut b = vec![0.0f64; n];
    a.symv(&v, &mut b);
    b
}

fn run(path: &Path) {
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let mtx = match read_mtx(path) {
        Ok(m) => m,
        Err(e) => {
            println!("SKIP {name}: {e}");
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            println!("SKIP {name}: to_csc: {e}");
            return;
        }
    };
    let n = csc.n;
    let mut solver = Solver::new();
    let t = Instant::now();
    let status = solver.factor(&csc, None);
    let factor_s = t.elapsed().as_secs_f64();
    match &status {
        FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
        other => {
            println!("SKIP {name}: factor {other:?}");
            return;
        }
    }
    if solver.factors().is_none() {
        println!("SKIP {name}: no factors");
        return;
    }
    let eps_sqrt_n = f64::EPSILON * (n as f64).sqrt();
    println!();
    println!(
        "=== {name}  n={n}  nnz={}  factor={factor_s:.3}s  eps*sqrt(n)={eps_sqrt_n:.3e} ===",
        csc.row_idx.len()
    );
    for (rhs_kind, b) in [
        ("easy (uniform scale)", build_rhs_easy(&csc)),
        ("hard (1e-6..1e6, alternating sign)", build_rhs_hard(&csc)),
    ] {
        println!("-- RHS: {rhs_kind}");
        println!(
            "{:<28} {:>5} {:>10} {:>10} {:>11} {:>11} {:>7}",
            "criterion", "steps", "stop", "wall(s)", "rel", "omega", "vs def"
        );
        let cases: Vec<(String, RefineOptions)> = vec![
            (
                format!("EpsSqrtN (default, k={DEFAULT_REFINE_MAX_STEPS})"),
                RefineOptions::default(),
            ),
            (
                "BackwardError(1e-14)".into(),
                RefineOptions::with_backward_error(1e-14),
            ),
            (
                "BackwardError(1e-12)".into(),
                RefineOptions::with_backward_error(1e-12),
            ),
            (
                "BackwardError(1e-10)".into(),
                RefineOptions::with_backward_error(1e-10),
            ),
            (
                "RelativeResidual(1e-12)".into(),
                RefineOptions::with_target(1e-12),
            ),
            (
                "max_steps=0 (unrefined)".into(),
                RefineOptions::with_max_steps(0),
            ),
        ];

        let mut baseline = f64::NAN;
        for (label, opts) in cases {
            let mut x = vec![0.0f64; n];
            let mut outcome = None;
            let mut best = f64::INFINITY;
            for _ in 0..REPS {
                let t = Instant::now();
                let o = match solver.solve_refined_into(&csc, &b, &mut x, opts) {
                    Ok(o) => o,
                    Err(e) => {
                        println!("{label:<28} error: {e:?}");
                        break;
                    }
                };
                let s = t.elapsed().as_secs_f64();
                if s < best {
                    best = s;
                }
                outcome = Some(o);
            }
            let Some(o) = outcome else { continue };
            let (rel, omega) = residual_and_omega(&csc, &x, &b);
            if baseline.is_nan() {
                baseline = best;
            }
            let stop = match o.stop {
                RefineStop::Converged => "Converged",
                RefineStop::MaxSteps => "MaxSteps",
                RefineStop::Stagnated => "Stagnated",
                RefineStop::Diverged => "Diverged",
            };
            println!(
                "{label:<28} {:>5} {stop:>10} {best:>10.4} {rel:>11.3e} {omega:>11.3e} {:>6.2}x",
                o.steps,
                baseline / best
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names: Vec<String> = if args.is_empty() {
        vec![
            "r05_kkt".into(),
            "qap15_kkt".into(),
            "dirichlet120_kkt".into(),
            "cont-201".into(),
            "cont5_late_kkt".into(),
            "bratu3d".into(),
        ]
    } else {
        args
    };
    println!("best-of-{REPS} wall clock");
    for nm in names {
        let p = format!("tests/data/large/{nm}.mtx");
        run(Path::new(&p));
    }
}
