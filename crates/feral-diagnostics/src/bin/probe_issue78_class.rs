//! Issue #78: is InfNorm generally the better scaling for the
//! high-dynamic-range non-arrow KKT class, with discs as the exception?
//!
//! For each matrix, sweeps InfNorm / MC64 / Identity (and reports what
//! `Auto` picks) and prints BOTH the backward residual `||Ax-b||/||b||`
//! and the FORWARD error `||x - x_true||/||x_true||`. The issue is a
//! forward-error failure (backward-stable solve of a forward-ill-
//! conditioned system), so backward residual alone is blind to it.
//!
//! rhs is synthesized as `A * x_true` with a fixed deterministic x_true.

use std::path::Path;

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus, Solver};
use feral::scaling::ScalingStrategy;
use feral::{read_mtx, CscMatrix};

fn entry_drng(csc: &CscMatrix) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = 0.0f64;
    for &v in &csc.values {
        let a = v.abs();
        if a > 0.0 {
            lo = lo.min(a);
            hi = hi.max(a);
        }
    }
    if lo.is_finite() && lo > 0.0 {
        hi / lo
    } else {
        f64::INFINITY
    }
}

fn max_col_nnz(csc: &CscMatrix) -> usize {
    let mut m = 0;
    for j in 0..csc.n {
        let mut c = 0;
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            if csc.values[k] != 0.0 {
                c += 1;
            }
        }
        m = m.max(c);
    }
    m
}

fn max_off_diag_ratio(csc: &CscMatrix, s: &[f64]) -> f64 {
    let mut diag_abs = vec![0.0f64; csc.n];
    let mut max_off = vec![0.0f64; csc.n];
    for j in 0..csc.n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = (csc.values[k] * s[i] * s[j]).abs();
            if i == j {
                diag_abs[j] = v;
            } else {
                max_off[i] = max_off[i].max(v);
                max_off[j] = max_off[j].max(v);
            }
        }
    }
    let mut w = 0.0f64;
    for j in 0..csc.n {
        let r = if diag_abs[j] > 0.0 {
            max_off[j] / diag_abs[j]
        } else if max_off[j] > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        w = w.max(r);
    }
    w
}

fn matvec_lower_sym(csc: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for j in 0..csc.n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = csc.values[k];
            out[i] += v * x[j];
            if i != j {
                out[j] += v * x[i];
            }
        }
    }
}

fn back_res(csc: &CscMatrix, x: &[f64], rhs: &[f64]) -> f64 {
    let mut ax = vec![0.0; csc.n];
    matvec_lower_sym(csc, x, &mut ax);
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..csc.n {
        let d = ax[i] - rhs[i];
        num += d * d;
        den += rhs[i] * rhs[i];
    }
    num.sqrt() / den.sqrt().max(1.0)
}

fn fwd_err(x: &[f64], x_true: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..x.len() {
        let d = x[i] - x_true[i];
        num += d * d;
        den += x_true[i] * x_true[i];
    }
    num.sqrt() / den.sqrt().max(1e-300)
}

fn run(label: &str, csc: &CscMatrix, rhs: &[f64], x_true: &[f64], strat: ScalingStrategy) {
    let params = NumericParams {
        scaling: strat,
        ..NumericParams::default()
    };
    let mut solver = Solver::with_params(params, feral::symbolic::SupernodeParams::default());
    let status = solver.factor(csc, None);
    if !matches!(status, FactorStatus::Success) {
        println!("    {label:<9} FACTOR FAILED: {status:?}");
        return;
    }
    let inertia = solver.inertia();
    match solver.solve_refined(csc, rhs) {
        Ok(x) => {
            println!(
                "    {label:<9} back={:.2e}  fwd={:.2e}  inertia={:?}",
                back_res(csc, &x, rhs),
                fwd_err(&x, x_true),
                inertia
            );
        }
        Err(e) => println!("    {label:<9} SOLVE FAILED: {e:?}  inertia={inertia:?}"),
    }
}

fn process(path: &str) {
    if !Path::new(path).exists() {
        return;
    }
    let mtx = match read_mtx(Path::new(path)) {
        Ok(m) => m,
        Err(e) => {
            println!("\n[{path}] read error: {e:?}");
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            println!("\n[{path}] csc error: {e:?}");
            return;
        }
    };
    let picked = feral::scaling::pick_scaling_strategy(&csc);
    println!(
        "\n[{}] n={} entry_drng={:.1e} max_col_nnz={} -> Auto picks {:?}",
        Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path),
        csc.n,
        entry_drng(&csc),
        max_col_nnz(&csc),
        picked
    );
    // Scaling-quality probe: how well does each scaling EQUILIBRATE the
    // matrix (max scaled |off|/|diag| over columns)? Computable a-priori
    // from the scaling vector alone, no solve. A scaling that fails to
    // equilibrate leaves a large ratio.
    if let (Ok((mc, _)), Ok((inv, _))) = (
        feral::scaling::compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric),
        feral::scaling::compute_scaling(&csc, &ScalingStrategy::InfNorm),
    ) {
        println!(
            "    scaled max|off|/|diag|:  InfNorm={:.2e}   MC64={:.2e}",
            max_off_diag_ratio(&csc, &inv),
            max_off_diag_ratio(&csc, &mc)
        );
    }
    let mut x_true = vec![0.0; csc.n];
    for (i, v) in x_true.iter_mut().enumerate() {
        *v = (i as f64 * 0.7).sin() + 0.5;
    }
    let mut rhs = vec![0.0; csc.n];
    matvec_lower_sym(&csc, &x_true, &mut rhs);
    run("InfNorm", &csc, &rhs, &x_true, ScalingStrategy::InfNorm);
    run("MC64", &csc, &rhs, &x_true, ScalingStrategy::Mc64Symmetric);
    run("Identity", &csc, &rhs, &x_true, ScalingStrategy::Identity);
}

fn main() {
    // The actual #78 external oracle: discs.nl iter-35 KKT (cond 1.97e20),
    // attached by the issue author. Not in the corpus.
    println!("=== #78 EXTERNAL ORACLE: discs.nl iter-35 KKT (cond 1.97e20) ===");
    process("/Users/jkitchin/feral-repro-issue78/discs-kkt-iter35.mtx");

    // High-dynamic-range non-arrow class: the DISCS family (closest
    // in-repo analog to the issue's discs.nl iter-35 KKT).
    let discs_dir = "data/matrices/kkt/DISCS";
    let mut discs: Vec<String> = std::fs::read_dir(discs_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path().to_string_lossy().into_owned())
                .filter(|p| p.ends_with(".mtx"))
                .collect()
        })
        .unwrap_or_default();
    discs.sort();
    println!("=== DISCS family (high-drng, the #78 class) ===");
    for p in &discs {
        process(p);
    }
    // Controls.
    println!("\n=== controls: clnlbeam (high-drng banded, InfNorm SHOULD win) ===");
    process("data/matrices/kkt-mittelmann/clnlbeam/clnlbeam_0000.mtx");
    println!("\n=== controls: true arrow KKTs (MC64 SHOULD win) ===");
    process("data/matrices/kkt/VESUVIO/VESUVIO_0000.mtx");
    process("data/matrices/kkt/MUONSINE/MUONSINE_0000.mtx");
    process("data/matrices/kkt/MSS1/MSS1_0009.mtx");
    process("data/matrices/kkt/ACOPP30/ACOPP30_0064.mtx");
}
