//! Triage the 6 remaining residual-margin failures from the parity panel:
//! AVION2, HAHN1, MEYER3NE, CERI651C, CERI651ELS, SSI. All pass inertia
//! against MUMPS but their residual is 1.2× to 1560× larger than MUMPS's.
//!
//! Working hypothesis: feral's solve_sparse_refined stops after 3 steps
//! with dx/x threshold = eps*sqrt(n), while MUMPS's refinement (when
//! enabled) runs up to 10 steps with a bundled component-wise stopping
//! criterion. This script instruments an extended refinement loop and
//! reports the residual trajectory per matrix so we can tell whether
//! (a) more steps would close the gap, (b) refinement stalls at a
//! feral-specific floor, or (c) the factorization is the limiter.
//!
//! Run with: cargo run --release --example triage_residual_margin

use std::path::Path;

use feral::numeric::factorize::{factorize_multifrontal, SparseFactors};
use feral::numeric::solve::{solve_sparse, solve_sparse_refined};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn sparse_params() -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    }
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        let r = ax[i] - b[i];
        rs += r * r;
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

/// Extended refinement loop: up to `max_steps`, printing residual per step.
/// Same structure as `solve_sparse_refined` but externally observable.
fn extended_refine(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    max_steps: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = matrix.n;
    let mut x = solve_sparse(factors, rhs).expect("solve_sparse");

    let mut r = vec![0.0; n];
    let mut ax = vec![0.0; n];
    matrix.symv(&x, &mut ax);
    for i in 0..n {
        r[i] = rhs[i] - ax[i];
    }

    let mut residuals = Vec::with_capacity(max_steps + 1);
    residuals.push(rel_residual(matrix, &x, rhs));

    let mut best_x = x.clone();
    let mut best_rel = residuals[0];

    for _ in 0..max_steps {
        let dx = solve_sparse(factors, &r).expect("correction solve");
        for i in 0..n {
            x[i] += dx[i];
        }
        let mut ax_new = vec![0.0; n];
        matrix.symv(&x, &mut ax_new);
        for i in 0..n {
            r[i] = rhs[i] - ax_new[i];
        }
        let rel = rel_residual(matrix, &x, rhs);
        residuals.push(rel);
        if rel < best_rel {
            best_rel = rel;
            best_x.copy_from_slice(&x);
        }
        // Divergence guard (matches solve_sparse_refined).
        let rnorm = norm2(&r);
        if rnorm > 100.0 * best_rel * norm2(rhs) {
            break;
        }
    }
    (best_x, residuals)
}

fn read_mumps_residual(path: &Path) -> Option<f64> {
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    data.get("residual_2norm_relative")?.as_f64()
}

fn triage(family: &str, stem: &str) {
    let base = format!("data/matrices/kkt/{}/{}", family, stem);
    let mtx_path = format!("{}.mtx", base);
    let sidecar_path = format!("{}.json", base);
    let mumps_path = format!("{}.mumps.json", base);

    let mtx = read_mtx(Path::new(&mtx_path)).expect("mtx");
    let csc = mtx.to_csc().expect("csc");
    let sc = read_sidecar(Path::new(&sidecar_path)).expect("sidecar");
    let rhs = sc.finite_rhs().expect("rhs");
    let mumps_res = read_mumps_residual(Path::new(&mumps_path));

    println!("\n=== {} ({} x {}) ===", stem, csc.n, csc.n);
    let b_norm = norm2(&rhs);
    println!("||b||₂ = {:.3e}", b_norm);
    if let Some(m) = mumps_res {
        println!("MUMPS residual: {:.3e}", m);
    }

    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("sym");
    let params = sparse_params();
    let (factors, inertia) =
        factorize_multifrontal(&csc, &sym, &params).expect("factorize_multifrontal");
    println!("inertia: {}", inertia);

    // Plain solve (no refinement)
    let x_plain = solve_sparse(&factors, &rhs).expect("solve_sparse");
    let rel_plain = rel_residual(&csc, &x_plain, &rhs);
    println!("plain           rel_res = {:.3e}", rel_plain);

    // Production refined (max 3 steps)
    let x_refined = solve_sparse_refined(&csc, &factors, &rhs).expect("solve_sparse_refined");
    let rel_refined = rel_residual(&csc, &x_refined, &rhs);
    println!("refined (max 3) rel_res = {:.3e}", rel_refined);

    // Extended refinement (10 steps) with per-step trajectory
    let (x_ext, trajectory) = extended_refine(&csc, &factors, &rhs, 10);
    let rel_ext = rel_residual(&csc, &x_ext, &rhs);
    println!("extended (10)   rel_res = {:.3e}", rel_ext);
    print!("trajectory:");
    for r in &trajectory {
        print!(" {:.2e}", r);
    }
    println!();

    if let Some(m) = mumps_res {
        let ratio_refined = rel_refined / m;
        let ratio_ext = rel_ext / m;
        println!(
            "vs MUMPS: refined ratio {:.1}×, extended ratio {:.1}×",
            ratio_refined, ratio_ext
        );
    }
}

fn main() {
    let cases = [
        ("avion2", "AVION2_0510"),
        ("hahn1", "HAHN1_0004"),
        ("meyer3ne", "MEYER3NE_0253"),
        ("ceri651c", "CERI651C_0746"),
        ("ceri651els", "CERI651ELS_1482"),
        ("ssi", "SSI_2597"),
    ];
    for (family, stem) in &cases {
        triage(family, stem);
    }
}
