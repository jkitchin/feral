//! Deep triage of ACOPP30_0004: dense `factor_single_front` produces
//! (71, 137, 1) vs sparse `factorize_multifrontal` produces (72, 137, 0)
//! which matches the sidecar. Residuals are both ~1e-14 so both are
//! numerically valid — the disagreement is on where the zero line falls.
//!
//! Hypotheses to distinguish:
//!   (A) Knight-Ruiz equilibration in `factor_single_front` scales some
//!       small positive pivot to below `zero_tol`, causing it to be
//!       force-zeroed.
//!   (B) Natural-order processing (no AMD) in `factor_single_front` picks
//!       a different pivot sequence than the sparse AMD-ordered path,
//!       and one of the natural-order pivots lands at the edge of the
//!       zero_tol threshold.

use std::path::Path;

use feral::dense::factor::factor_frontal;
use feral::numeric::factorize::factorize_multifrontal;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    factor_single_front, read_mtx, read_sidecar, BunchKaufmanParams, Inertia, ZeroPivotAction,
};

fn main() {
    let mtx_path = Path::new("data/matrices/kkt/ACOPP30/ACOPP30_0004.mtx");
    let json_path = Path::new("data/matrices/kkt/ACOPP30/ACOPP30_0004.json");

    let mtx = read_mtx(mtx_path).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let dense_mat = mtx.to_dense();
    let sc = read_sidecar(json_path).expect("read sidecar");
    let n = csc.n;

    let expected = Inertia {
        positive: sc.inertia.positive,
        negative: sc.inertia.negative,
        zero: sc.inertia.zero,
    };

    println!("=== ACOPP30_0004 deep triage ===");
    println!("n = {}", n);
    println!("expected inertia = {}", expected);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };

    // --- Path 1: factor_single_front (equilibration + natural order)
    println!("\n--- factor_single_front (eq + natural order) ---");
    let (f1, i1) = factor_single_front(&dense_mat, &params).expect("f_sf");
    let d1_small: Vec<(usize, f64)> = f1
        .d_diag
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.abs() < 1e-6)
        .map(|(i, &v)| (i, v))
        .collect();
    println!(
        "inertia = {} ({})",
        i1,
        if i1 == expected { "MATCH" } else { "MISMATCH" }
    );
    println!("needs_refinement = {}", f1.needs_refinement);
    println!("zero_tol = {:.3e}", f1.zero_tol);
    println!("|d| < 1e-6 count = {}", d1_small.len());
    for (i, v) in d1_small.iter().take(10) {
        println!("  d[{:3}] = {:+.3e}", i, v);
    }

    // --- Path 2: factor_frontal with no equilibration, natural order
    println!("\n--- factor_frontal (no eq, natural order) ---");
    let f2 = factor_frontal(&dense_mat, n, false, &params).expect("f_frontal_raw");
    println!(
        "inertia = {} ({})",
        f2.inertia,
        if f2.inertia == expected {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    println!("needs_refinement = {}", f2.needs_refinement);
    let d2_small: Vec<(usize, f64)> = f2
        .d_diag
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.abs() < 1e-6)
        .map(|(i, &v)| (i, v))
        .collect();
    println!("|d| < 1e-6 count = {}", d2_small.len());
    for (i, v) in d2_small.iter().take(10) {
        println!("  d[{:3}] = {:+.3e}", i, v);
    }

    // --- Path 3: factorize_multifrontal (AMD order via sparse)
    println!("\n--- factorize_multifrontal (AMD via sparse) ---");
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let np = feral::numeric::factorize::NumericParams::with_bk(params.clone());
    let (f3, i3) = factorize_multifrontal(&csc, &sym, &np).expect("factorize");
    println!(
        "inertia = {} ({})",
        i3,
        if i3 == expected { "MATCH" } else { "MISMATCH" }
    );
    println!("needs_refinement = {}", f3.needs_refinement);
    println!("n_supernodes = {}", sym.supernodes.len());

    // --- Path 4: factor_frontal with zero_tol loosened
    println!("\n--- factor_frontal (no eq, natural, zero_tol = 0) ---");
    let params_no_zt = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        zero_tol: 0.0,
        zero_tol_2x2: 0.0,
        ..BunchKaufmanParams::default()
    };
    let f4 = factor_frontal(&dense_mat, n, false, &params_no_zt).expect("f_frontal_zt0");
    println!(
        "inertia = {} ({})",
        f4.inertia,
        if f4.inertia == expected {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    println!("needs_refinement = {}", f4.needs_refinement);

    // --- Walk factor_frontal's D structure block-by-block
    // For each position k in d_diag, determine if it's a 1x1 or 2x2 block
    // (based on d_subdiag[k] != 0), print the block, and compute its
    // manual inertia contribution.
    println!("\n--- D block walk (factor_frontal no-eq natural) ---");
    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut zer = 0usize;
    let mut k = 0;
    while k < n {
        let d11 = f2.d_diag[k];
        if k + 1 < n && f2.d_subdiag[k] != 0.0 {
            // 2x2 block
            let d21 = f2.d_subdiag[k];
            let d22 = f2.d_diag[k + 1];
            let trace = d11 + d22;
            let det = d11 * d22 - d21 * d21;
            // inertia of the 2x2: if det < 0, one positive one negative;
            // if det > 0, two same sign; if det == 0, one zero.
            let (p_add, n_add, z_add) = if det.abs() < 1e-20 {
                // singular 2x2 — rank 1, one non-zero eigenvalue (trace)
                if trace > 0.0 {
                    (1, 0, 1)
                } else if trace < 0.0 {
                    (0, 1, 1)
                } else {
                    (0, 0, 2)
                }
            } else if det < 0.0 {
                (1, 1, 0)
            } else if trace > 0.0 {
                (2, 0, 0)
            } else {
                (0, 2, 0)
            };
            if d11.abs() < 1e-6 || d21.abs() < 1e-6 || d22.abs() < 1e-6 || det.abs() < 1e-20 {
                println!(
                    "  k={:3} 2x2: [[{:+.2e}, {:+.2e}], [{:+.2e}, {:+.2e}]] det={:+.2e} trace={:+.2e} -> ({},{},{})",
                    k, d11, d21, d21, d22, det, trace, p_add, n_add, z_add
                );
            }
            pos += p_add;
            neg += n_add;
            zer += z_add;
            k += 2;
        } else {
            // 1x1 block
            if d11.abs() < 1e-6 {
                println!("  k={:3} 1x1: d={:+.3e}", k, d11);
            }
            if d11.abs() <= params.zero_tol {
                zer += 1;
            } else if d11 > 0.0 {
                pos += 1;
            } else {
                neg += 1;
            }
            k += 1;
        }
    }
    println!("  manual walk: ({}, {}, {})", pos, neg, zer);
    println!("  kernel reported: {}", f2.inertia);

    // Summary table
    println!("\n=== summary ===");
    println!("  expected                     : {}", expected);
    println!("  factor_single_front          : {}", i1);
    println!("  factor_frontal (no eq, nat)  : {}", f2.inertia);
    println!("  factorize_multifrontal (AMD) : {}", i3);
    println!("  factor_frontal (no eq, zt=0) : {}", f4.inertia);
}
