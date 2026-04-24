//! Triage: why does MC64 break inertia on POLAK6_0021?
//!
//! Session 2026-04-19 corpus bench (lever C, Policies 1+2+3) found
//! that switching the scaling default from InfNorm to MC64 — directly,
//! or via the adaptive `diag_only/n >= 0.3` heuristic — flips
//! POLAK6_0021's inertia from the correct (5, 4, 0) to (3, 4, 2)
//! (two zero pivots) and causes the back-solve residual to blow up
//! from 9.21e-17 to 1.31e13. See
//! `dev/research/lever-c-corpus-bench-2026-04-19.md`.
//!
//! This is a "matched but bad" failure mode: MC64 matching succeeds
//! (no `PartialSingular`), but the resulting scaling is numerically
//! worse than InfNorm. Policy 4 (try-MC64-fallback-to-InfNorm) needs
//! a post-scaling diagnostic to *detect* this case.
//!
//! This binary characterizes the failure on the actual matrix and
//! tests several candidate detection heuristics:
//!   1. Min absolute scaled-diagonal magnitude
//!   2. Range of scaled diagonals (max / min)
//!   3. Per-column max-off-diagonal-vs-diagonal ratio after scaling
//!   4. Inertia mismatch detection (would require trial factorization)
//!
//! Usage: `cargo run --release --bin polak6_diag`.
//!
//! No production code change. Output is read by hand for the
//! research note.

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse;
use feral::scaling::{compute_scaling, ScalingStrategy};
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, ZeroPivotAction};
use std::path::Path;

fn matrix_diagonal(csc: &CscMatrix) -> Vec<f64> {
    let n = csc.n;
    let mut d = vec![0.0; n];
    for (j, dj) in d.iter_mut().enumerate().take(n) {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            if csc.row_idx[k] == j {
                *dj = csc.values[k];
            }
        }
    }
    d
}

/// Apply symmetric scaling D · A · D in place; return scaled
/// diagonal and per-column "max |off-diagonal| / |diagonal|" ratio
/// for diagnostic purposes.
fn scaled_diag_and_offratio(csc: &CscMatrix, scaling: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = csc.n;
    let mut diag = vec![0.0_f64; n];
    let mut max_off = vec![0.0_f64; n];
    for j in 0..n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = csc.values[k] * scaling[i] * scaling[j];
            if i == j {
                diag[j] = v;
            } else {
                let av = v.abs();
                if av > max_off[i] {
                    max_off[i] = av;
                }
                if av > max_off[j] {
                    max_off[j] = av;
                }
            }
        }
    }
    let ratio: Vec<f64> = diag
        .iter()
        .zip(max_off.iter())
        .map(|(&d, &m)| {
            if d.abs() > 0.0 {
                m / d.abs()
            } else {
                f64::INFINITY
            }
        })
        .collect();
    (diag, ratio)
}

fn try_factor_and_solve(
    label: &str,
    csc: &CscMatrix,
    scaling: ScalingStrategy,
    expected_inertia: (usize, usize, usize),
) {
    println!("--- factor under {} ---", label);
    let snode_params = SupernodeParams::default();
    let factor_params = feral::numeric::factorize::NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        },
        scaling: scaling.clone(),
        small_leaf: Default::default(),
    };
    let sym = match symbolic_factorize_with_method(csc, &snode_params, OrderingMethod::Amd) {
        Ok(s) => s,
        Err(e) => {
            println!("  symbolic FAILED: {:?}", e);
            return;
        }
    };
    let (factors, inertia) = match factorize_multifrontal(csc, &sym, &factor_params) {
        Ok(r) => r,
        Err(e) => {
            println!("  numeric FAILED: {:?}", e);
            return;
        }
    };
    let inertia_ok = inertia.positive == expected_inertia.0
        && inertia.negative == expected_inertia.1
        && inertia.zero == expected_inertia.2;
    println!(
        "  inertia=({}, {}, {})  expected=({}, {}, {})  {}",
        inertia.positive,
        inertia.negative,
        inertia.zero,
        expected_inertia.0,
        expected_inertia.1,
        expected_inertia.2,
        if inertia_ok { "MATCH" } else { "MISMATCH" },
    );
    // Build a deterministic RHS and check residual.
    let n = csc.n;
    let rhs: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 0.1).collect();
    let x = match solve_sparse(&factors, &rhs) {
        Ok(x) => x,
        Err(e) => {
            println!("  solve FAILED: {:?}", e);
            return;
        }
    };
    // residual = b - A x (use lower triangle * 2 - diag)
    let mut ax = vec![0.0_f64; n];
    for j in 0..n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            ax[i] += csc.values[k] * x[j];
            if i != j {
                ax[j] += csc.values[k] * x[i];
            }
        }
    }
    let mut res2 = 0.0_f64;
    let mut rhs2 = 0.0_f64;
    for i in 0..n {
        let r = rhs[i] - ax[i];
        res2 += r * r;
        rhs2 += rhs[i] * rhs[i];
    }
    let rel = res2.sqrt() / rhs2.sqrt().max(1.0);
    println!("  ||b - A x|| / ||b|| = {:.3e}", rel);
}

fn print_scaling_diagnostic(label: &str, csc: &CscMatrix, strategy: &ScalingStrategy) {
    let n = csc.n;
    let (s, info) = match compute_scaling(csc, strategy) {
        Ok(t) => t,
        Err(e) => {
            println!("--- {} compute_scaling FAILED: {:?} ---", label, e);
            return;
        }
    };
    println!("--- {} ---", label);
    println!("  ScalingInfo: {:?}", info);
    let mut s_min = f64::INFINITY;
    let mut s_max: f64 = 0.0;
    for &v in &s {
        let a = v.abs();
        if a > 0.0 {
            s_min = s_min.min(a);
            s_max = s_max.max(a);
        }
    }
    println!(
        "  scaling vector: min={:.3e}  max={:.3e}  range(max/min)={:.3e}",
        s_min,
        s_max,
        s_max / s_min.max(1e-300),
    );
    println!("  scaling: {:?}", s);
    let (diag, ratio) = scaled_diag_and_offratio(csc, &s);
    let d_min = diag.iter().map(|x| x.abs()).fold(f64::INFINITY, f64::min);
    let d_max = diag.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let r_max = ratio.iter().cloned().fold(0.0_f64, f64::max);
    println!(
        "  scaled |diag|: min={:.3e}  max={:.3e}  range={:.3e}",
        d_min,
        d_max,
        d_max / d_min.max(1e-300),
    );
    println!("  scaled max(|off|/|diag|) per column: max={:.3e}", r_max);
    println!("  scaled diag: {:?}", diag);
    println!("  off/diag ratio per col: {:?}", ratio);
    let _ = n;
}

fn main() {
    println!("POLAK6_0021 triage — MC64 inertia regression");
    println!("{}", "=".repeat(72));

    let path = Path::new("data/matrices/kkt/POLAK6/POLAK6_0021.mtx");
    let mtx = match read_mtx(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("FAIL: read {}: {}", path.display(), e);
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL: csc: {:?}", e);
            return;
        }
    };
    let n = csc.n;
    let nnz = csc.row_idx.len();
    println!("shape: n={} stored_nnz={}", n, nnz);

    // Raw diagonal stats.
    let raw_diag = matrix_diagonal(&csc);
    let mut d_min = f64::INFINITY;
    let mut d_max: f64 = 0.0;
    for v in &raw_diag {
        let a = v.abs();
        if a > 0.0 {
            d_min = d_min.min(a);
            d_max = d_max.max(a);
        }
    }
    println!(
        "raw |diag|: min={:.3e}  max={:.3e}  range={:.3e}",
        d_min,
        d_max,
        d_max / d_min.max(1e-300),
    );
    println!("raw diag: {:?}", raw_diag);

    // Per-column degree (which columns drive the diag_only count).
    let mut diag_only = 0usize;
    for j in 0..n {
        let len = csc.col_ptr[j + 1] - csc.col_ptr[j];
        if len == 1 && csc.row_idx[csc.col_ptr[j]] == j {
            diag_only += 1;
        }
    }
    println!(
        "diag_only={} / n={} = {:.3} ({})",
        diag_only,
        n,
        diag_only as f64 / n as f64,
        if diag_only as f64 / n as f64 >= 0.30 {
            ">= 0.30 → adaptive routes to MC64"
        } else {
            "< 0.30 → adaptive keeps InfNorm"
        },
    );
    println!();

    print_scaling_diagnostic("InfNorm", &csc, &ScalingStrategy::InfNorm);
    println!();
    print_scaling_diagnostic("Mc64Symmetric", &csc, &ScalingStrategy::Mc64Symmetric);
    println!();

    // Read expected inertia from the sidecar JSON if present.
    // Fallback to (5, 4, 0) per the corpus-bench note.
    let expected = (5usize, 4usize, 0usize);
    println!(
        "Expected inertia (per session-04-19 bench): ({}, {}, {})",
        expected.0, expected.1, expected.2
    );
    println!();

    try_factor_and_solve("InfNorm", &csc, ScalingStrategy::InfNorm, expected);
    println!();
    try_factor_and_solve(
        "Mc64Symmetric",
        &csc,
        ScalingStrategy::Mc64Symmetric,
        expected,
    );
    println!();
    try_factor_and_solve("Auto (adaptive)", &csc, ScalingStrategy::Auto, expected);

    println!();
    println!("{}", "=".repeat(72));
    println!("Heuristic candidates for Policy 4:");
    println!("  H1  scaled |diag|.min < 1e-12              → reject MC64");
    println!("  H2  scaled |diag|.range > 1e8              → reject MC64");
    println!("  H3  scaled max(|off|/|diag|) > 1e3         → reject MC64");
    println!("Compare the printed values above against these thresholds.");
    println!("A robust heuristic should fire on Mc64Symmetric for POLAK6 *and*");
    println!("not fire on the seven VESUVIO/CRESC matrices (where MC64 wins).");
}
