//! Issue #63 probe: near-singular KKT backward error by ordering.
//!
//! Built to test the issue's hypothesis that on scrs8-2c-8's iter-26 KKT
//! (μ→0, regularization just turned on), FERAL's solve backward error
//! ‖Ax−b‖/‖b‖ is ordering-dependent (AMD/AMF ~2e-8 vs MetisND/ScotchND
//! ~1e-8). **It is not.** FERAL solves the regularized stepped system to
//! ~1e-22 under every ordering and both scaling modes (Auto, Identity),
//! even at max_piv ~6e16. What is ordering-dependent is the *inertia*
//! reported on the pre-regularization singular matrix, which steers
//! pounce's δ_w path. Static pivoting (`static_pivot=1e-8`) on the
//! singular matrix backfires: it perturbs ~half the pivots and drives the
//! backward error to 1.0. See
//! `dev/research/issue-63-nearsingular-ordering-diagnosis.md`.
//!
//! Run: `cargo run --release --bin probe_issue63_nearsingular -- <kkt.mtx> <rhs.txt>`
//! (mtx/rhs produced by pounce `--dump kkt:` + jsonl_to_mtx.py).
//!
//! Throwaway diagnostic, not a test.

use feral::scaling::ScalingStrategy;
use feral::symbolic::OrderingMethod;
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::Path;

fn read_vec(path: &Path) -> Vec<f64> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect()
}

/// y = A*x for symmetric A stored as lower-triangle CSC (the mtx is
/// `symmetric`, so to_csc() keeps the lower triangle).
fn symv(a: &CscMatrix, x: &[f64]) -> Vec<f64> {
    let n = a.n;
    let mut y = vec![0.0f64; n];
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let v = a.values[k];
            y[i] += v * x[j];
            if i != j {
                y[j] += v * x[i];
            }
        }
    }
    y
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

fn run(
    label: &str,
    a: &CscMatrix,
    b: &[f64],
    method: OrderingMethod,
    scaling: ScalingStrategy,
    static_pivot: Option<f64>,
) {
    let mut s = Solver::new()
        .with_ordering(method)
        .with_scaling(scaling.clone());
    if let Some(t) = static_pivot {
        s = s.with_static_pivot_threshold(t);
    }
    let status = s.factor(a, None);
    let stats = s.last_factor_stats();
    let bnorm = norm2(b);

    let be_plain = match s.solve(b) {
        Ok(x) => {
            let r: Vec<f64> = symv(a, &x).iter().zip(b).map(|(ax, bi)| ax - bi).collect();
            norm2(&r) / bnorm
        }
        Err(e) => {
            eprintln!("  {label}: solve failed: {e:?}");
            f64::NAN
        }
    };
    let be_ref = match s.solve_refined(a, b) {
        Ok(x) => {
            let r: Vec<f64> = symv(a, &x).iter().zip(b).map(|(ax, bi)| ax - bi).collect();
            norm2(&r) / bnorm
        }
        Err(e) => {
            eprintln!("  {label}: refined solve failed: {e:?}");
            f64::NAN
        }
    };

    let summary = match &stats {
        Some(st) => format!(
            "nnz_L={} min_piv={:.3e} max_piv={:.3e} n_tiny={} inertia=({},{},{})",
            st.nnz_l,
            st.min_abs_pivot,
            st.max_abs_pivot,
            st.n_tiny,
            st.inertia.positive,
            st.inertia.negative,
            st.inertia.zero,
        ),
        None => "no stats".to_string(),
    };
    println!(
        "  {label:<10} status={:<8}\n             {summary}\n             backward_err plain={be_plain:.4e} refined={be_ref:.4e}",
        status_short(&status),
    );
}

fn status_short(s: &FactorStatus) -> &'static str {
    match s {
        FactorStatus::Success => "Success",
        FactorStatus::Singular => "Singular",
        FactorStatus::WrongInertia { .. } => "WrongInertia",
        FactorStatus::FatalError(_) => "FatalError",
        FactorStatus::Interrupted => "Interrupted",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: probe_issue63_nearsingular <kkt.mtx> <rhs.txt>");
        return;
    }
    let mtx = Path::new(&args[1]);
    let rhs = Path::new(&args[2]);
    let a = match read_mtx(mtx).and_then(|m| m.to_csc()) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("read {:?} failed: {e:?}", mtx);
            return;
        }
    };
    let b = read_vec(rhs);
    println!(
        "=== {} (n={}, nnz={}, |b|={:.3e}) ===",
        mtx.display(),
        a.n,
        a.row_idx.len(),
        norm2(&b)
    );
    if b.len() != a.n {
        eprintln!("rhs length {} != n {}", b.len(), a.n);
        return;
    }
    for (cname, scaling, sp) in [
        ("Auto-scale, no static-pivot", ScalingStrategy::Auto, None),
        (
            "Auto-scale, static_pivot=1e-8",
            ScalingStrategy::Auto,
            Some(1e-8),
        ),
    ] {
        println!("  --- {cname} ---");
        for m in [
            OrderingMethod::Amd,
            OrderingMethod::Amf,
            OrderingMethod::MetisND,
            OrderingMethod::ScotchND,
        ] {
            run(&format!("{m:?}"), &a, &b, m, scaling.clone(), sp);
        }
    }
}
