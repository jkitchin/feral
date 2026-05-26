//! δ-cascade probe for issue #54.
//!
//! Loads `nuffield2_trap_iter1.mtx`, classifies each row by its
//! original diagonal sign (positive → "primal" / H block; zero →
//! "constraint" / C block), then sweeps the IPM-style perturbation
//!
//!     A_δ = A  +  δ_x · diag([row is primal])
//!              -  δ_c · diag([row is constraint])
//!
//! and reports `num_negative_eigenvalues()` at each δ. The fix in
//! issue #54 is supposed to make this trajectory monotonic (pounce
//! comment on gh#52 showed `13035 → 13042 → 12615 → 13218` — note the
//! mid-cascade backwards jump 13042→12615 that triggered the IPM
//! stall).

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::Solver;
use feral::read_mtx;
use feral::symbolic::SupernodeParams;
use feral::CscMatrix;
use std::path::PathBuf;

fn perturbed(a: &CscMatrix, primal: &[bool], delta_x: f64, delta_c: f64) -> CscMatrix {
    let n = a.n;
    let col_ptr = a.col_ptr.clone();
    let row_idx = a.row_idx.clone();
    let mut values = a.values.clone();
    // Build a diagonal-position map so we can perturb in place.
    let mut diag_added = vec![false; n];
    for col in 0..n {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        for k in start..end {
            if row_idx[k] == col {
                let add = if primal[col] { delta_x } else { -delta_c };
                values[k] += add;
                diag_added[col] = true;
                break;
            }
        }
    }
    // Insert missing diagonals in column-sorted order — required for
    // CscMatrix::from_triplets to accept.
    let mut rows = Vec::with_capacity(row_idx.len());
    let mut cols = Vec::with_capacity(row_idx.len());
    let mut vals = Vec::with_capacity(row_idx.len());
    for col in 0..n {
        let s = col_ptr[col];
        let e = col_ptr[col + 1];
        let mut inserted = diag_added[col];
        for k in s..e {
            let r = row_idx[k];
            let v = values[k];
            if !inserted && r > col {
                // Insert diagonal first (lower-triangular: row >= col).
                let add = if primal[col] { delta_x } else { -delta_c };
                if add != 0.0 {
                    rows.push(col);
                    cols.push(col);
                    vals.push(add);
                }
                inserted = true;
            }
            rows.push(r);
            cols.push(col);
            vals.push(v);
        }
        if !inserted {
            let add = if primal[col] { delta_x } else { -delta_c };
            if add != 0.0 {
                rows.push(col);
                cols.push(col);
                vals.push(add);
            }
        }
    }
    match CscMatrix::from_triplets(n, &rows, &cols, &vals) {
        Ok(m) => m,
        Err(e) => panic!("perturbed csc rebuild: {}", e),
    }
}

fn classify_primal(a: &CscMatrix) -> Vec<bool> {
    let n = a.n;
    let mut is_primal = vec![false; n];
    for (col, slot) in is_primal.iter_mut().enumerate().take(n) {
        let s = a.col_ptr[col];
        let e = a.col_ptr[col + 1];
        for k in s..e {
            if a.row_idx[k] == col {
                *slot = a.values[k] > 0.0;
                break;
            }
        }
    }
    is_primal
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("dev/repros/issue-54/nuffield2_trap_iter1.mtx"));

    let mtx = read_mtx(&path).expect("read mtx");
    let a0 = mtx.to_csc().expect("to_csc");
    let n = a0.n;
    let primal = classify_primal(&a0);
    let n_primal: usize = primal.iter().filter(|p| **p).count();
    let n_constr = n - n_primal;
    println!("matrix     = {:?}", path);
    println!("n          = {}", n);
    println!(
        "block size = {} primal (positive diag) / {} constraint (zero diag)",
        n_primal, n_constr
    );

    // Cascade follows pounce's escalation pattern:
    // δ_x ∈ {0, 1e-8, 1e-4, 2e-1, 1e2, 6.99e19}
    // δ_c = √(δ_x · μ) with μ = 0.1 (typical mid-cascade)
    let mu = 0.1f64;
    let dx_cascade = [0.0, 1e-8, 1e-6, 1e-4, 2e-1, 1.0, 1e2, 1e6, 1e12, 6.99e19];
    println!(
        "\n {:<12} {:<12} {:<8} {:<8} {:<6} {:<10} status",
        "delta_x", "delta_c", "neg", "zero", "pos", "neg+zero"
    );
    let expected_neg = 13202usize;
    let mut last_neg_zero: Option<isize> = None;
    let mut backwards_jumps = 0;
    for &dx in &dx_cascade {
        let dc = if dx > 0.0 { (dx * mu).sqrt() } else { 0.0 };
        let a_pert = perturbed(&a0, &primal, dx, dc);
        let mut solver = Solver::with_params(NumericParams::default(), SupernodeParams::default());
        let status = solver.factor(&a_pert, None);
        let inertia = solver.inertia().cloned();
        match (status, inertia) {
            (s, Some(inertia)) => {
                let nz = (inertia.negative + inertia.zero) as isize;
                let diff_neg = inertia.negative as isize - expected_neg as isize;
                let diff_nz = nz - expected_neg as isize;
                let trend = match last_neg_zero {
                    None => "",
                    Some(prev) => {
                        if nz < prev {
                            backwards_jumps += 1;
                            " (BACKWARDS)"
                        } else if nz == prev {
                            " (flat)"
                        } else {
                            ""
                        }
                    }
                };
                println!(
                    " {:<12.3e} {:<12.3e} {:<8} {:<8} {:<6} {:<10} {:?} diff_neg={:+} diff_nz={:+}{}",
                    dx,
                    dc,
                    inertia.negative,
                    inertia.zero,
                    inertia.positive,
                    nz,
                    s,
                    diff_neg,
                    diff_nz,
                    trend,
                );
                last_neg_zero = Some(nz);
            }
            (s, None) => {
                println!(" {:<12.3e} {:<12.3e} status={:?}", dx, dc, s);
            }
        }
    }
    println!(
        "\nBackwards jumps in (neg+zero) across cascade: {}",
        backwards_jumps
    );
    println!(
        "Expected: 0 if SSIDS-aligned accounting fixes the non-monotonicity \
        that caused the IPM stall (pounce gh#52)."
    );
}
