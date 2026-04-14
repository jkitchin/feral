//! Triage the dense ACOPP30 residual gap.
//!
//! The full KKT bench reports 67 ACOPP30 variants that fail "shared"
//! (both dense and sparse) with the pattern:
//!   dense  inertia = (72, 137, 0) [matches sidecar], residual ~ 2.7e-2
//!   sparse inertia = (71, 137, 1) [off by one zero], residual ~ 1e-14
//!
//! This script runs dense on ACOPP30_0026 under four parameter
//! configurations to isolate the root cause:
//!   A) default (pivot_threshold = 0.0) — matches current bench dense
//!   B) pivot_threshold = 0.01 — matches current bench sparse
//!   C) pivot_threshold = 0.01 + refined solve (10-step)
//!   D) Knight-Ruiz equilibration + (C)
//!
//! Expected outcome: if (B) or (C) closes the gap, we have a cheap
//! port of the sparse fix to dense for Phase 2.4. If only (D) works,
//! the issue is scaling-sensitivity and Phase 2.4 needs a dense
//! equilibration wiring.

use std::path::Path;

use feral::{
    factor, read_mtx, read_sidecar, solve, solve_refined, BunchKaufmanParams, SymmetricMatrix,
    ZeroPivotAction,
};

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn rel_residual(a: &SymmetricMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let rs: f64 = (0..n).map(|i| (ax[i] - b[i]).powi(2)).sum();
    let bs: f64 = b.iter().map(|x| x * x).sum();
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

fn try_config(
    matrix: &SymmetricMatrix,
    rhs: &[f64],
    params: &BunchKaufmanParams,
    label: &str,
    refined: bool,
) {
    match factor(matrix, params) {
        Ok((fac, inertia)) => {
            let x = if refined {
                solve_refined(matrix, &fac, rhs).expect("solve_refined")
            } else {
                solve(&fac, rhs).expect("solve")
            };
            let rel = rel_residual(matrix, &x, rhs);
            println!(
                "  {:<45} inertia={}  rel_res={:.3e}  (||b||={:.3e})",
                label,
                inertia,
                rel,
                norm2(rhs)
            );
        }
        Err(e) => println!("  {:<45} factor failed: {}", label, e),
    }
}

fn main() {
    let cases = ["ACOPP30_0026", "ACOPP30_0018", "ACOPP30_0000"];
    for stem in cases.iter() {
        println!("\n=== {} ===", stem);
        let base = format!("data/matrices/kkt/acopp30/{}", stem);
        let mtx = read_mtx(Path::new(&format!("{}.mtx", base))).expect("mtx");
        let sym = mtx.to_dense();
        let sc = read_sidecar(Path::new(&format!("{}.json", base))).expect("sidecar");
        let rhs = sc.finite_rhs().expect("rhs");
        println!("  n={}", sym.n);

        let mumps: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&format!("{}.mumps.json", base)).unwrap_or_default(),
        )
        .unwrap_or(serde_json::Value::Null);
        if let Some(r) = mumps["residual_2norm_relative"].as_f64() {
            println!("  MUMPS residual oracle: {:.3e}", r);
        }

        // A) default (matches params_kkt_dense in bench)
        let a = BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            ..BunchKaufmanParams::default()
        };
        try_config(
            &sym,
            &rhs,
            &a,
            "A) default (threshold=0)  plain solve",
            false,
        );
        try_config(&sym, &rhs, &a, "A') default (threshold=0)  refined", true);

        // B) pivot_threshold = 0.01, plain solve
        let b = BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        };
        try_config(
            &sym,
            &rhs,
            &b,
            "B) threshold=0.01           plain solve",
            false,
        );
        try_config(&sym, &rhs, &b, "C) threshold=0.01           refined", true);
    }
}
