//! Quick A/B diagnostic for the W-2 2×2 inline regression on SWOPF/HIMMELBJ.
//!
//! Runs each matrix with FORCE_SCALAR_FRONTAL = false (default; W-2 2×2 active)
//! and FORCE_SCALAR_FRONTAL = true (scalar oracle), reporting inertia and
//! solve residual for both. Confirms (or refutes) the hypothesis that the
//! regression source is in the panel path.

use std::path::Path;
use std::sync::atomic::Ordering;

use feral::dense::factor::{DISABLE_PANEL_INLINE_2X2, FORCE_SCALAR_FRONTAL};
use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        rs += (ax[i] - b[i]).powi(2);
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

fn run_one(label: &str, fam: &str, stem: &str) {
    let base = format!("tests/data/parity/{}/{}", fam, stem);
    let mtx = read_mtx(Path::new(&format!("{}.mtx", base))).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let sidecar = read_sidecar(Path::new(&format!("{}.json", base))).expect("sidecar");
    let rhs = sidecar.finite_rhs().expect("finite rhs");

    let params = NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    });
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");

    for &(force_scalar, disable_2x2, kind) in &[
        (false, false, "BLOCKED+2x2 "),
        (false, true, "BLOCKED-2x2 "),
        (true, false, "SCALAR      "),
    ] {
        FORCE_SCALAR_FRONTAL.store(force_scalar, Ordering::Relaxed);
        DISABLE_PANEL_INLINE_2X2.store(disable_2x2, Ordering::Relaxed);
        let (fac, inertia) = factorize_multifrontal(&csc, &sym, &params).expect("factor");
        let x = solve_sparse_refined(&csc, &fac, &rhs).expect("solve");
        let res = rel_residual(&csc, &x, &rhs);
        println!(
            "  [{}] {} inertia={} residual={:.6e} needs_refine={}",
            kind, label, inertia, res, fac.needs_refinement
        );
    }
    FORCE_SCALAR_FRONTAL.store(false, Ordering::Relaxed);
    DISABLE_PANEL_INLINE_2X2.store(false, Ordering::Relaxed);
}

fn main() {
    println!("=== W-2 2×2 inline diagnostic ===\n");
    run_one("SWOPF_0000", "swopf", "SWOPF_0000");
    run_one("HIMMELBJ_0032", "himmelbj", "HIMMELBJ_0032");
}
