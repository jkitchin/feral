//! Probe whether dropping the `raw_diag_range >= RAW_GUARD` early-return
//! in Policy 4 (scaling/mod.rs:284) would regress the matrices it was
//! designed to protect: MEYER3NE_0220, MSS1_0009, VESUVIA, VESUVIO,
//! VESUVIOU, and one ACOPP30 plateau (0064).
//!
//! For each matrix, reports raw_diag_range, MC64 mc_off, InfNorm in_off,
//! plus rel_ref under each scaling strategy with a synthesized rhs.

use std::path::Path;

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::Solver;
use feral::{read_mtx, CscMatrix};

fn raw_diag_range(csc: &CscMatrix) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = 0.0f64;
    for j in 0..csc.n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            if csc.row_idx[k] == j {
                let a = csc.values[k].abs();
                if a > 0.0 {
                    lo = lo.min(a);
                    hi = hi.max(a);
                }
            }
        }
    }
    if lo.is_finite() && lo > 0.0 {
        hi / lo
    } else {
        f64::INFINITY
    }
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

fn rel_res(csc: &CscMatrix, x: &[f64], rhs: &[f64]) -> f64 {
    let mut ax = vec![0.0; csc.n];
    matvec_lower_sym(csc, x, &mut ax);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..csc.n {
        let d = ax[i] - rhs[i];
        num += d * d;
        den += rhs[i] * rhs[i];
    }
    num.sqrt() / den.sqrt().max(1.0)
}

fn try_scaling(
    label: &str,
    csc: &CscMatrix,
    rhs: &[f64],
    strategy: feral::scaling::ScalingStrategy,
) -> f64 {
    let params = NumericParams {
        scaling: strategy,
        ..NumericParams::default()
    };
    let mut solver = Solver::with_params(params, feral::symbolic::SupernodeParams::default());
    let status = solver.factor(csc, None);
    if !matches!(status, feral::numeric::solver::FactorStatus::Success) {
        println!("    [{label:<10}] factor failed: {status:?}");
        return f64::NAN;
    }
    let x = solver.solve_refined(csc, rhs).map(|x| x.to_vec()).unwrap();
    let r = rel_res(csc, &x, rhs);
    println!("    [{label:<10}] rel_ref = {r:.2e}");
    r
}

fn process(path: &str) {
    if !Path::new(path).exists() {
        println!("\n[{path}] MISSING");
        return;
    }
    let mtx = read_mtx(Path::new(path)).unwrap();
    let csc = mtx.to_csc().unwrap();
    let drng = raw_diag_range(&csc);
    let (mc, _) =
        feral::scaling::compute_scaling(&csc, &feral::scaling::ScalingStrategy::Mc64Symmetric)
            .unwrap();
    let (inv, _) =
        feral::scaling::compute_scaling(&csc, &feral::scaling::ScalingStrategy::InfNorm).unwrap();
    let mc_off = max_off_diag_ratio(&csc, &mc);
    let in_off = max_off_diag_ratio(&csc, &inv);
    let picked = feral::scaling::pick_scaling_strategy(&csc);

    let mc_range = mc.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), v| {
        let a = v.abs();
        if a > 0.0 {
            (lo.min(a), hi.max(a))
        } else {
            (lo, hi)
        }
    });
    let in_range = inv.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), v| {
        let a = v.abs();
        if a > 0.0 {
            (lo.min(a), hi.max(a))
        } else {
            (lo, hi)
        }
    });
    let mc_spread = mc_range.1 / mc_range.0;
    let in_spread = in_range.1 / in_range.0;

    // Frobenius norm of scaled matrix (cheap conditioning proxy).
    let frob = |s: &[f64]| -> f64 {
        let mut sum = 0.0;
        for j in 0..csc.n {
            for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
                let i = csc.row_idx[k];
                let v = (csc.values[k] * s[i] * s[j]).abs();
                sum += v * v;
                if i != j {
                    sum += v * v;
                }
            }
        }
        sum.sqrt()
    };
    let mc_frob = frob(&mc);
    let in_frob = frob(&inv);

    println!("\n[{path}] n={} nnz={}", csc.n, csc.row_idx.len());
    println!("  raw_drng={:.2e}  picked={:?}", drng, picked);
    println!(
        "  mc: off={:.2e}  s_spread={:.2e}  ||DAD||_F={:.2e}",
        mc_off, mc_spread, mc_frob,
    );
    println!(
        "  in: off={:.2e}  s_spread={:.2e}  ||DAD||_F={:.2e}",
        in_off, in_spread, in_frob,
    );
    // Synthesize rhs from random unit-norm x: rhs = A * x_true.
    let mut x_true = vec![0.0; csc.n];
    for (i, v) in x_true.iter_mut().enumerate() {
        *v = ((i as f64 * 0.7).sin() + 0.5).cos();
    }
    let mut rhs = vec![0.0; csc.n];
    matvec_lower_sym(&csc, &x_true, &mut rhs);
    try_scaling(
        "Auto    ",
        &csc,
        &rhs,
        feral::scaling::ScalingStrategy::default(),
    );
    try_scaling(
        "MC64    ",
        &csc,
        &rhs,
        feral::scaling::ScalingStrategy::Mc64Symmetric,
    );
    try_scaling(
        "InfNorm ",
        &csc,
        &rhs,
        feral::scaling::ScalingStrategy::InfNorm,
    );
    try_scaling(
        "Identity",
        &csc,
        &rhs,
        feral::scaling::ScalingStrategy::Identity,
    );
}

fn main() {
    let candidates = [
        "data/matrices/kkt/ACOPP30/ACOPP30_0064.mtx",
        "data/matrices/kkt/MEYER3NE/MEYER3NE_0220.mtx",
        "data/matrices/kkt/MSS1/MSS1_0009.mtx",
        "data/matrices/kkt/VESUVIA/VESUVIA_0000.mtx",
        "data/matrices/kkt/VESUVIO/VESUVIO_0000.mtx",
        "data/matrices/kkt/VESUVIOU/VESUVIOU_0000.mtx",
        "data/matrices/kkt/HS75/HS75_0000.mtx",
        "data/matrices/kkt/CRESC132/CRESC132_0000.mtx",
        "data/matrices/kkt/MUONSINE/MUONSINE_0000.mtx",
    ];
    for p in &candidates {
        process(p);
    }
}
