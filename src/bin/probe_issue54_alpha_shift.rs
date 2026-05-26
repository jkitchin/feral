//! Issue #54 — pivot-stability probe.
//!
//! Per pounce gh#52 comment (2026-05-26): the SSIDS routing change
//! tightened first-shot accuracy (off by +4 instead of −54) but the
//! cascade is still non-monotone. The proposed diagnostic:
//!
//!   "factor the iter-0 KKT plus α·I (or α·diag(1_x, 0_s, 0_c, 0_d)
//!    for the x-block only) for α ∈ {0, 1e-4, 1, 1e4, 1e8, 1e12, 1e16,
//!    1e20} and assert the inertia signature is monotone (neg
//!    non-increasing, pos non-decreasing). The non-monotone trajectory
//!    above is a smoking gun against any implementation that's purely
//!    algebraic."
//!
//! Plus the recommended A/B:
//!
//!   "Compare with `PerturbToEps { abs_floor: 1e-12 }`: this variant
//!    bypasses the strict-zero branch entirely. If `PerturbToEps` is
//!    stable on the same α-sweep, it confirms the instability is in
//!    the zero-pivot branch's BK ordering."
//!
//! Output format:
//!   * Two shift modes: uniform (`α·I`) and x-block-only
//!     (`α·diag(1_x, 0, 0, 0)` where the x-block is approximated by
//!     rows-with-positive-diag).
//!   * Two zero-pivot configs: default `ForceAccept` and
//!     `PerturbToEps { abs_floor: 1e-12 }`.
//!   * For each combo: inertia at each α, and a Weyl-monotonicity
//!     verdict (uniform `α·I`: `neg` must be non-increasing in α;
//!     x-block-only: `neg` must be non-increasing because the x-block
//!     additive is PSD, so Weyl's inequality still applies one-sided).
//!
//! Usage:
//!   cargo run --release --bin probe_issue54_alpha_shift [path/to/.mtx]

#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use std::path::PathBuf;

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::Solver;
use feral::read_mtx;
use feral::scaling::ScalingStrategy;
use feral::symbolic::SupernodeParams;
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};

/// `α · I` shift: bump every diagonal by `α`.
fn shifted_uniform(a: &CscMatrix, alpha: f64) -> CscMatrix {
    let mask = vec![true; a.n];
    shifted_mask(a, alpha, &mask)
}

/// `α · diag(mask)` shift: bump only diagonals where `mask[col]`.
fn shifted_mask(a: &CscMatrix, alpha: f64, mask: &[bool]) -> CscMatrix {
    let n = a.n;
    let col_ptr = a.col_ptr.clone();
    let row_idx = a.row_idx.clone();
    let mut values = a.values.clone();
    let mut diag_present = vec![false; n];
    for col in 0..n {
        let s = col_ptr[col];
        let e = col_ptr[col + 1];
        for k in s..e {
            if row_idx[k] == col {
                if mask[col] {
                    values[k] += alpha;
                }
                diag_present[col] = true;
                break;
            }
        }
    }
    let mut rows = Vec::with_capacity(row_idx.len() + n);
    let mut cols = Vec::with_capacity(row_idx.len() + n);
    let mut vals = Vec::with_capacity(row_idx.len() + n);
    for col in 0..n {
        let s = col_ptr[col];
        let e = col_ptr[col + 1];
        let mut inserted = diag_present[col];
        for k in s..e {
            let r = row_idx[k];
            let v = values[k];
            if !inserted && r > col {
                if mask[col] && alpha != 0.0 {
                    rows.push(col);
                    cols.push(col);
                    vals.push(alpha);
                }
                inserted = true;
            }
            rows.push(r);
            cols.push(col);
            vals.push(v);
        }
        if !inserted && mask[col] && alpha != 0.0 {
            rows.push(col);
            cols.push(col);
            vals.push(alpha);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("rebuild csc")
}

fn classify_pos_diag(a: &CscMatrix) -> Vec<bool> {
    let n = a.n;
    let mut mask = vec![false; n];
    for col in 0..n {
        let s = a.col_ptr[col];
        let e = a.col_ptr[col + 1];
        for k in s..e {
            if a.row_idx[k] == col {
                if a.values[k] > 0.0 {
                    mask[col] = true;
                }
                break;
            }
        }
    }
    mask
}

fn spmv_sym_lower(a: &CscMatrix, x: &[f64]) -> Vec<f64> {
    let n = a.n;
    let mut y = vec![0.0; n];
    for j in 0..n {
        let s = a.col_ptr[j];
        let e = a.col_ptr[j + 1];
        for k in s..e {
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

fn run_sweep(
    label: &str,
    a0: &CscMatrix,
    shift_kind: &str,
    mask: Option<&[bool]>,
    alphas: &[f64],
    zero_action: ZeroPivotAction,
    pivot_threshold: f64,
    scaling: ScalingStrategy,
) {
    let mut np = NumericParams::default();
    np.bk = BunchKaufmanParams {
        on_zero_pivot: zero_action,
        pivot_threshold,
        ..np.bk
    };
    println!(
        "\n=== {} | shift={} | scaling={:?} ===",
        label, shift_kind, scaling
    );
    np.scaling = scaling;
    println!(
        " {:<10} {:<8} {:<8} {:<8} {:<10} {:<10} {:<10} status        Δneg",
        "alpha", "neg", "zero", "pos", "neg+zero", "rel_resid", "rel_resid_ir",
    );
    let mut last_neg: Option<isize> = None;
    let mut weyl_violations = 0;
    let mut nz_violations = 0;
    let mut last_nz: Option<isize> = None;
    for &alpha in alphas {
        let a_shift = if let Some(m) = mask {
            shifted_mask(a0, alpha, m)
        } else {
            shifted_uniform(a0, alpha)
        };
        let mut solver = Solver::with_params(np.clone(), SupernodeParams::default());
        let status = solver.factor(&a_shift, None);
        let inertia = solver.inertia().cloned();
        // Residual check: factor correct iff ||A x - b|| / ||b|| small.
        let n = a_shift.n;
        let mut b = vec![0.0f64; n];
        for i in 0..n {
            b[i] = (((i as u64).wrapping_mul(2654435761) % 9999) as f64) / 9999.0 - 0.5;
        }
        let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rel_resid = match solver.solve(&b) {
            Ok(x) => {
                let ax = spmv_sym_lower(&a_shift, &x);
                let r2: f64 = ax.iter().zip(&b).map(|(a, b)| (a - b).powi(2)).sum();
                r2.sqrt() / b_norm.max(1e-300)
            }
            Err(_) => f64::NAN,
        };
        let rel_resid_ir = match solver.solve_refined(&a_shift, &b) {
            Ok(x) => {
                let ax = spmv_sym_lower(&a_shift, &x);
                let r2: f64 = ax.iter().zip(&b).map(|(a, b)| (a - b).powi(2)).sum();
                r2.sqrt() / b_norm.max(1e-300)
            }
            Err(_) => f64::NAN,
        };
        match (status, inertia) {
            (s, Some(inertia)) => {
                let neg = inertia.negative as isize;
                let nz = (inertia.negative + inertia.zero) as isize;
                let neg_str = match last_neg {
                    None => "          ".to_string(),
                    Some(prev) => {
                        let d = neg - prev;
                        if d > 0 {
                            weyl_violations += 1;
                            format!(" Δ={:+5} W!", d)
                        } else {
                            format!(" Δ={:+5}   ", d)
                        }
                    }
                };
                let nz_str = match last_nz {
                    None => "".to_string(),
                    Some(prev) => {
                        let d = nz - prev;
                        if d > 0 {
                            nz_violations += 1;
                            format!(" Δnz={:+5} W!", d)
                        } else {
                            format!(" Δnz={:+5}   ", d)
                        }
                    }
                };
                println!(
                    " {:<10.1e} {:<8} {:<8} {:<8} {:<10} {:<10.3e} {:<10.3e} {:?}{}{}",
                    alpha,
                    inertia.negative,
                    inertia.zero,
                    inertia.positive,
                    inertia.negative + inertia.zero,
                    rel_resid,
                    rel_resid_ir,
                    s,
                    neg_str,
                    nz_str,
                );
                last_neg = Some(neg);
                last_nz = Some(nz);
            }
            (s, None) => {
                println!(" {:<10.1e} status={:?}", alpha, s);
            }
        }
    }
    println!(
        " summary: Weyl(neg-monotone) violations = {}, (neg+zero) violations = {}",
        weyl_violations, nz_violations
    );
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("dev/repros/issue-54/nuffield2_trap_iter1.mtx"));

    let mtx = read_mtx(&path).expect("read mtx");
    let a0 = mtx.to_csc().expect("to_csc");
    let n = a0.n;
    let pos_mask = classify_pos_diag(&a0);
    let n_pos: usize = pos_mask.iter().filter(|p| **p).count();
    println!("matrix       = {:?}", path);
    println!("n            = {}", n);
    println!("|pos-diag|   = {}", n_pos);
    println!("|zero-diag|  = {}", n - n_pos);

    let alphas = [
        0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e8, 1e12, 1e16, 1e20,
    ];

    let default_scaling = NumericParams::default().scaling;

    // (a) Uniform α·I sweep, ForceAccept (default), pivot_threshold=0.
    run_sweep(
        "ForceAccept pivtol=0 (default)",
        &a0,
        "uniform α·I",
        None,
        &alphas,
        ZeroPivotAction::ForceAccept,
        0.0,
        default_scaling.clone(),
    );

    // (a') x-block-only α·diag(1_x, 0) sweep, ForceAccept.
    run_sweep(
        "ForceAccept pivtol=0 (default)",
        &a0,
        "x-block α·diag(pos-diag rows)",
        Some(&pos_mask),
        &alphas,
        ZeroPivotAction::ForceAccept,
        0.0,
        default_scaling.clone(),
    );

    // (b) PerturbToEps A/B — uniform shift.
    run_sweep(
        "PerturbToEps pivtol=0 abs_floor=1e-12",
        &a0,
        "uniform α·I",
        None,
        &alphas,
        ZeroPivotAction::PerturbToEps { abs_floor: 1e-12 },
        0.0,
        default_scaling.clone(),
    );

    // (b') PerturbToEps A/B — x-block-only shift.
    run_sweep(
        "PerturbToEps pivtol=0 abs_floor=1e-12",
        &a0,
        "x-block α·diag(pos-diag rows)",
        Some(&pos_mask),
        &alphas,
        ZeroPivotAction::PerturbToEps { abs_floor: 1e-12 },
        0.0,
        default_scaling.clone(),
    );

    // (c) MA57-style threshold pivoting (pivot_threshold=0.01).
    // Hypothesis: stable BK 1×1 vs 2×2 selection under shifts.
    run_sweep(
        "ForceAccept pivtol=0.01 (MA57 default-ish)",
        &a0,
        "x-block α·diag(pos-diag rows)",
        Some(&pos_mask),
        &alphas,
        ZeroPivotAction::ForceAccept,
        0.01,
        default_scaling.clone(),
    );

    // (c') uniform shift cross-check at pivtol=0.01.
    run_sweep(
        "ForceAccept pivtol=0.01 (MA57 default-ish)",
        &a0,
        "uniform α·I",
        None,
        &alphas,
        ZeroPivotAction::ForceAccept,
        0.01,
        default_scaling.clone(),
    );

    // (c2) Real NumericParams::default() pivot_threshold = 1e-8.
    // This is the configuration pounce actually runs (issue #2 fix).
    run_sweep(
        "ForceAccept pivtol=1e-8 (true default)",
        &a0,
        "x-block α·diag(pos-diag rows)",
        Some(&pos_mask),
        &alphas,
        ZeroPivotAction::ForceAccept,
        1e-8,
        default_scaling.clone(),
    );

    // (c3) Mid-strength pivot_threshold = 1e-4.
    run_sweep(
        "ForceAccept pivtol=1e-4 (mid)",
        &a0,
        "x-block α·diag(pos-diag rows)",
        Some(&pos_mask),
        &alphas,
        ZeroPivotAction::ForceAccept,
        1e-4,
        default_scaling.clone(),
    );

    // (d) Identity-scaling rule-out: x-block shift, ForceAccept.
    // If scaling silently changes the matrix across the sweep, this
    // sweep should be more monotone than (a').
    run_sweep(
        "ForceAccept pivtol=0 scaling=Identity",
        &a0,
        "x-block α·diag(pos-diag rows)",
        Some(&pos_mask),
        &alphas,
        ZeroPivotAction::ForceAccept,
        0.0,
        ScalingStrategy::Identity,
    );

    // (d') Identity-scaling rule-out: uniform shift, ForceAccept.
    run_sweep(
        "ForceAccept pivtol=0 scaling=Identity",
        &a0,
        "uniform α·I",
        None,
        &alphas,
        ZeroPivotAction::ForceAccept,
        0.0,
        ScalingStrategy::Identity,
    );

    println!(
        "\nVerdict legend:\n\
         - 'W!' on Δneg = Weyl violation (neg increased as α grew).\n\
         - 'W!' on Δnz  = (neg+zero) violation.\n\
         - For uniform α·I: λ_i(A+αI) = λ_i(A)+α, so neg(α) must be\n\
           non-increasing. Any Weyl violation = pivot-selection jitter\n\
           that misclassifies an eigenvalue across the zero line.\n\
         - For x-block-only α·diag(1_x): the additive perturbation is\n\
           PSD on the x-block, so eigenvalues only move 'up' — neg\n\
           must still be non-increasing as α → ∞.\n\
         - rel_resid = ||A_shifted x - b|| / ||b|| with random b. If\n\
           small (≲ 1e-6) at a Weyl-violation α, the L·D·L^T factor is\n\
           correct and only the inertia counter is buggy. If large,\n\
           the factor itself is wrong."
    );
}
