//! Head-to-head against the canonical MUMPS 5.8.2 oracle on identical systems.
//!
//!   cargo run -p feral-diagnostics --bin probe_vs_mumps_residual --release [-- <name>...]
//!
//! `probe_pivot_threshold_residual` showed that on `qap15_kkt` feral's
//! unrefined relative residual sits at ~3e-6 and does not move as the pivot
//! threshold is swept from 1e-8 to 0.5. That ruled the pivot threshold out as
//! the cause but left open whether ~3e-6 is simply what the matrix admits.
//!
//! The MUMPS oracle (`external_benchmarks/mumps_oracle`, CNTL(1)=0.01,
//! ICNTL(10)=2 refinement steps) answers that: on the same matrix with the
//! same RHS it reports relative residual 3.61e-15 and omega1 2.57e-14.
//!
//! **That is not a nine-order gap, and an earlier version of this comment was
//! wrong to call it one.** It compares feral *unrefined* against MUMPS *with*
//! its refinement enabled. Refined, feral matches or beats MUMPS on all seven
//! large matrices under this RHS, and the inertia agrees exactly on all seven.
//! What this probe went on to find is a different defect, visible only under
//! `HARD_RHS=1`: see the addendum in `dev/research/issue-190-refine-target.md`.
//!
//! Environment knobs, both used to produce the tables in that addendum:
//!
//! - `HARD_RHS=1` -- build `b = A*v` from `v[i] = +/-10^((i%13)-6)` instead of
//!   the well-scaled default, i.e. a RHS spanning twelve orders of magnitude,
//!   the shape an interior-point method produces near convergence. This is the
//!   regime where the old normwise-only default stopped at step 0 with a
//!   componentwise error up to 9.5e-5.
//! - `OMEGA_EPS=1` -- make the probe's omega-only criterion column target
//!   `f64::EPSILON` (LAPACK `dgerfs`) rather than `sqrt(eps)` (MUMPS
//!   `CNTL(2)`). Measured and rejected as a default: it stagnates or exhausts
//!   the step budget on five of the seven large matrices under `HARD_RHS=1`.
//!
//! Default RHS is `b = A * v`, `v[i] = 1 + (i % 7) / 8`, byte-identical to the
//! vector `mkrhs.py` fed the Fortran oracle.

use feral::numeric::solve::{RefineOptions, StopCriterion};
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::Path;
use std::time::Instant;

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

/// `sqrt(eps)` is MUMPS's `CNTL(2)` (`ref/mumps/src/dini_defaults.F:1094`);
/// `eps` is LAPACK `dgerfs`'s componentwise target. `OMEGA_EPS=1` selects it.
fn omega_target() -> f64 {
    if std::env::var("OMEGA_EPS").is_ok() {
        f64::EPSILON
    } else {
        f64::EPSILON.sqrt()
    }
}

fn build_rhs_easy(a: &CscMatrix) -> Vec<f64> {
    let n = a.n;
    let mut v = vec![0.0f64; n];
    let hard = std::env::var("HARD_RHS").is_ok();
    for (i, s) in v.iter_mut().enumerate() {
        *s = if hard {
            let mag = 10f64.powi(((i % 13) as i32) - 6);
            if i % 2 == 0 {
                mag
            } else {
                -mag
            }
        } else {
            1.0 + (i % 7) as f64 / 8.0
        };
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
    let Ok(mtx) = read_mtx(path) else {
        println!("SKIP {name}: read_mtx");
        return;
    };
    let Ok(csc) = mtx.to_csc() else {
        println!("SKIP {name}: to_csc");
        return;
    };
    let n = csc.n;
    let b = build_rhs_easy(&csc);

    let mut s = Solver::new();
    let t = Instant::now();
    let st = s.factor(&csc, None);
    let fac = t.elapsed().as_secs_f64();
    if !matches!(
        st,
        FactorStatus::Success | FactorStatus::WrongInertia { .. }
    ) {
        println!("{name:18} factor {st:?}");
        return;
    }
    let (nnzl, min_piv, n_tiny) = match s.factors() {
        Some(f) => (
            f.factor_nnz(),
            f.min_pivot_magnitude().unwrap_or(f64::NAN),
            f.n_tiny(),
        ),
        None => {
            println!("{name:18} no factors");
            return;
        }
    };
    let inertia = s.inertia();
    let (p, ng, z) = match inertia {
        Some(i) => (i.positive, i.negative, i.zero),
        None => (usize::MAX, usize::MAX, usize::MAX),
    };

    let mut xu = vec![0.0f64; n];
    if s.solve_refined_into(&csc, &b, &mut xu, RefineOptions::default().and_max_steps(0))
        .is_err()
    {
        println!("{name:18} unrefined solve error");
        return;
    }
    let (rel_u, om_u) = residual_and_omega(&csc, &xu, &b);

    let mut xr = vec![0.0f64; n];
    let t = Instant::now();
    let out = s.solve_refined_into(&csc, &b, &mut xr, RefineOptions::default());
    let solve = t.elapsed().as_secs_f64();
    let (steps, stop) = match &out {
        Ok(o) => (o.steps, format!("{:?}", o.stop)),
        Err(_) => {
            println!("{name:18} refined solve error");
            return;
        }
    };
    let (rel_r, om_r) = residual_and_omega(&csc, &xr, &b);

    // MUMPS refines to omega <= CNTL(2) = sqrt(eps) (`ref/mumps/src/
    // dini_defaults.F:1094`). This is the same ladder with feral's
    // componentwise criterion selected instead of the normwise default.
    let mut xw = vec![0.0f64; n];
    let t = Instant::now();
    let outw = s.solve_refined_into(
        &csc,
        &b,
        &mut xw,
        RefineOptions::default().and_stop(StopCriterion::BackwardError(omega_target())),
    );
    let solve_w = t.elapsed().as_secs_f64();
    let (steps_w, stop_w) = match &outw {
        Ok(o) => (o.steps, format!("{:?}", o.stop)),
        Err(_) => {
            println!("{name:18} omega-stop solve error");
            return;
        }
    };
    let (rel_w, om_w) = residual_and_omega(&csc, &xw, &b);

    println!(
        "{name:18} {n:>7} {fac:>8.3} {solve:>8.3} {nnzl:>10} {min_piv:>10.2e} {n_tiny:>6} \
         {p}/{ng}/{z:<6} {rel_u:>10.3e} {om_u:>10.3e} {steps:>3} {rel_r:>10.3e} {om_r:>10.3e}  {steps_w:>3} {rel_w:>10.3e} {om_w:>10.3e} {solve_w:>7.3}  {stop}/{stop_w}"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names: Vec<String> = if args.is_empty() {
        vec![
            "qap15_kkt".into(),
            "cont5_late_kkt".into(),
            "r05_kkt".into(),
            "cont-201".into(),
            "bratu3d".into(),
            "dirichlet120_kkt".into(),
            "bcsstk38".into(),
        ]
    } else {
        args
    };
    println!("feral defaults (Solver::new), RHS b = A*v, v[i] = 1 + (i%7)/8");
    println!(
        "{:18} {:>7} {:>8} {:>8} {:>10} {:>10} {:>6} {:>14} {:>10} {:>10} {:>3} {:>10} {:>10}  stop",
        "matrix",
        "n",
        "fac(s)",
        "sol(s)",
        "nnzL",
        "min|piv|",
        "n_tiny",
        "inertia p/n/z",
        "unref rel",
        "unref om",
        "it",
        "ref rel",
        "ref om"
    );
    for nm in names {
        let p = format!("tests/data/large/{nm}.mtx");
        run(Path::new(&p));
    }
}
