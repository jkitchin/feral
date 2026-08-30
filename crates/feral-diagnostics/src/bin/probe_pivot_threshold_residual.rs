//! Does raising the pivot threshold fix the unrefined residual?
//!
//!   cargo run -p feral-diagnostics --bin probe_pivot_threshold_residual --release [-- <name>...]
//!
//! `probe_forceaccept_residual` established that `ZeroPivotAction::ForceAccept`
//! never fires on these matrices (`inertia.zero == 0`, `n_tiny == 0`), so the
//! residual issue #190 attributes to it comes from somewhere else. The
//! remaining suspect is the pivot threshold itself: feral's multifrontal
//! default is `pivot_threshold = 1e-8` (`numeric/factorize.rs:645`), where
//! MA57's `CNTL(1)` and MUMPS's `CNTL(1)` both default to `0.01` -- six
//! orders of magnitude tighter. A loose threshold accepts pivots that are
//! small relative to their column, and threshold partial pivoting then only
//! promises `|L_ij| <= 1/u`, i.e. multipliers up to 1e8.
//!
//! This probe sweeps `u` and reports what it costs and what it buys: the
//! unrefined (`max_steps = 0`) relative residual and componentwise backward
//! error, against factor wall time, `nnz(L)`, and how many pivots got
//! delayed up the tree as a result.
//!
//! If the residual falls as `u` rises, the "unrefined solve is untrustworthy"
//! problem is a *tuning* problem with a real fix, not something a caller has
//! to work around with a refinement knob.

use feral::numeric::factorize::NumericParams;
use feral::numeric::solve::RefineOptions;
use feral::symbolic::supernode::SupernodeParams;
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::Path;
use std::time::Instant;

const REPS: usize = 3;

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
    let b = build_rhs_easy(&csc);
    println!();
    println!("=== {name}  n={n}  nnz={} ===", csc.nnz());
    println!(
        "{:>10}  {:>9}  {:>10}  {:>11}  {:>10}  {:>5}  {:>6}  {:>7}",
        "u", "factor(s)", "nnzL", "min|piv|", "unref rel", "zero", "n_tiny", "omega"
    );

    // 1e-8 is feral's shipped multifrontal default; 0.01 is MA57's and
    // MUMPS's CNTL(1); 0.5 is `pivtol_max`, the MA27-style ceiling that
    // `Solver::increase_quality` walks toward.
    for u in [1e-8f64, 1e-6, 1e-4, 1e-2, 1e-1, 0.5] {
        let mut np = NumericParams::default();
        np.bk.pivot_threshold = u;
        let mut best = f64::INFINITY;
        let mut solver = Solver::with_params(np.clone(), SupernodeParams::default());
        for _ in 0..REPS {
            let mut s = Solver::with_params(np.clone(), SupernodeParams::default());
            let t = Instant::now();
            let st = s.factor(&csc, None);
            let el = t.elapsed().as_secs_f64();
            if !matches!(
                st,
                FactorStatus::Success | FactorStatus::WrongInertia { .. }
            ) {
                println!("{u:>10.1e}  factor {st:?}");
                return;
            }
            if el < best {
                best = el;
            }
            solver = s;
        }
        let (nnzl, min_piv, n_tiny) = match solver.factors() {
            Some(f) => (
                f.factor_nnz(),
                f.min_pivot_magnitude().unwrap_or(f64::NAN),
                f.n_tiny(),
            ),
            None => {
                println!("{u:>10.1e}  no factors");
                continue;
            }
        };
        let zero = solver.inertia().map(|i| i.zero).unwrap_or(usize::MAX);
        let mut x = vec![0.0f64; n];
        let unref = RefineOptions::default().and_max_steps(0);
        if solver.solve_refined_into(&csc, &b, &mut x, unref).is_err() {
            println!("{u:>10.1e}  solve error");
            continue;
        }
        let (rel, om) = residual_and_omega(&csc, &x, &b);
        println!(
            "{u:>10.1e}  {best:>9.3}  {nnzl:>10}  {min_piv:>11.3e}  {rel:>10.3e}  \
             {zero:>5}  {n_tiny:>6}  {om:>7.3e}"
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names: Vec<String> = if args.is_empty() {
        vec![
            "qap15_kkt".into(),
            "cont5_late_kkt".into(),
            "r05_kkt".into(),
            "cont-201".into(),
        ]
    } else {
        args
    };
    println!("best-of-{REPS} factor wall; unrefined = max_steps 0, well-scaled RHS");
    for nm in names {
        let p = format!("tests/data/large/{nm}.mtx");
        run(Path::new(&p));
    }
}
