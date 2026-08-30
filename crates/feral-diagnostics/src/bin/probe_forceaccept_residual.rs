//! Is the unrefined residual actually caused by `ZeroPivotAction::ForceAccept`?
//!
//!   cargo run -p feral-diagnostics --bin probe_forceaccept_residual --release [-- <name>...]
//!
//! Issue #190 asserts that POUNCE cannot use `max_steps = 0` because
//! "`ZeroPivotAction::ForceAccept` is the default, so feral's raw solve can
//! leave a non-trivial residual against the very system it factorized."
//! That is a claim about *mechanism*, and it has a testable consequence:
//! a matrix whose unrefined solve is inaccurate should show force-accepted
//! pivots, and a matrix with none should solve cleanly without refinement.
//!
//! ForceAccept fires in exactly one place (`try_reject_1x1_frontal`, the
//! `may_delay == false` arm): a pivot that failed the column-relative
//! threshold, that rook rescue could not turn into a 2x2, and that has no
//! parent supernode left to be delayed into. It zeroes the L column and
//! counts a zero eigenvalue -- so `inertia.zero > 0` is its fingerprint.
//! `n_tiny` counts the *other* perturbation family (static-pivot floor /
//! PerturbToEps), which is off by default and should read 0 throughout.
//!
//! So: print the fingerprint next to the unrefined error. If large
//! unrefined error shows up with `zero = 0` and `n_tiny = 0`, ForceAccept
//! is not the mechanism and the workaround in #190 is aimed at the wrong
//! target.

use feral::numeric::solve::RefineOptions;
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::Path;

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
    let status = solver.factor(&csc, None);
    match &status {
        FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
        other => {
            println!("SKIP {name}: factor {other:?}");
            return;
        }
    }
    let (zero, n_tiny, min_piv, needs_ref) = match solver.factors() {
        Some(f) => (
            solver.inertia().map(|i| i.zero).unwrap_or(usize::MAX),
            f.n_tiny(),
            f.min_pivot_magnitude().unwrap_or(f64::NAN),
            f.needs_refinement,
        ),
        None => {
            println!("SKIP {name}: no factors");
            return;
        }
    };

    // `max_steps = 0` is the unrefined solve: the raw forward/back
    // substitution against the factor, which is what MA57/MUMPS hand back.
    let unref = RefineOptions::default().and_max_steps(0);
    let mut x = vec![0.0f64; n];
    let mut out = Vec::new();
    for (tag, b) in [
        ("easy", build_rhs_easy(&csc)),
        ("hard", build_rhs_hard(&csc)),
    ] {
        if solver.solve_refined_into(&csc, &b, &mut x, unref).is_err() {
            out.push(format!("{tag}: solve error"));
            continue;
        }
        let (rel, om) = residual_and_omega(&csc, &x, &b);
        out.push(format!("{tag} rel={rel:.3e} omega={om:.3e}"));
    }

    println!(
        "{name:<18} n={n:<7} zero={zero:<5} n_tiny={n_tiny:<4} \
         min|piv|={min_piv:.3e} needs_ref={needs_ref:<5} | {}",
        out.join("  |  ")
    );
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
            "bcsstk38".into(),
        ]
    } else {
        args
    };
    println!("unrefined (max_steps=0) error vs ForceAccept fingerprint");
    println!(
        "zero = force-accepted zero pivots (inertia.zero); n_tiny = static/PerturbToEps events"
    );
    for nm in names {
        let p = format!("tests/data/large/{nm}.mtx");
        run(Path::new(&p));
    }
}
