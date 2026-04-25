//! Triage the 75 REAL_BUG sparse-only residual failures from Phase 2.2.3.
//!
//! Categorization (`scripts/categorize-sparse-only.py`) split the 89
//! sparse-only failures into:
//!   - 14 ORACLE_DISAGREE  (VESUVIO* / ACOPP14 — sidecar wrong, drop)
//!   - 75 REAL_BUG         (oracle agrees with sidecar, sparse residual_ok=false)
//!
//! All 75 REAL_BUG cases pass the inertia check; they fail because the
//! sparse residual is 2.9× to 2666× over `n * eps * 1e6`. One outlier
//! (PFIT2_0300) is at 3.55e-6 — three orders of magnitude worse than the
//! rest. The other 74 sit in 8e-10 .. 1.4e-8, consistently 3-10× over
//! tolerance on tiny matrices (n = 4..21).
//!
//! Hypothesis we want to falsify or confirm: the sparse multifrontal path
//! accumulates more rounding error on tiny matrices than the
//! single-front dense path does, even though the symbolic structure
//! collapses to one supernode at this size. If true, dense and sparse
//! should produce *different* L/D for the same matrix and dense's
//! solve_refined should reach the tolerance while sparse's does not.
//!
//! For each representative we compare:
//!   - dense factor + solve_refined          (the path that's passing)
//!   - sparse factorize_multifrontal + solve_sparse_refined  (failing)
//!   - sparse + extended refinement (10 steps) — does more IR close it?
//!
//! Run with: cargo run --release --example triage_sparse_residual_edge

use std::path::Path;

use feral::numeric::factorize::{factorize_multifrontal, SparseFactors};
use feral::numeric::solve::{solve_sparse, solve_sparse_refined};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    factor as dense_factor, read_mtx, read_sidecar, solve_refined as dense_solve_refined,
    BunchKaufmanParams, CscMatrix, SymmetricMatrix, ZeroPivotAction,
};

fn sparse_params() -> feral::numeric::factorize::NumericParams {
    feral::numeric::factorize::NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    })
}

fn dense_params() -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    }
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn rel_residual_csc(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
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

fn rel_residual_dense(a: &SymmetricMatrix, x: &[f64], b: &[f64]) -> f64 {
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

/// Mirror of `solve_sparse_refined` but exposes the per-step residual trajectory.
fn extended_sparse_refine(
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
    residuals.push(rel_residual_csc(matrix, &x, rhs));
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
        let rel = rel_residual_csc(matrix, &x, rhs);
        residuals.push(rel);
        if rel < best_rel {
            best_rel = rel;
            best_x.copy_from_slice(&x);
        }
        let rnorm = norm2(&r);
        if rnorm > 100.0 * best_rel * norm2(rhs) {
            break;
        }
    }
    (best_x, residuals)
}

fn triage(family: &str, stem: &str) {
    let base = format!("data/matrices/kkt/{}/{}", family, stem);
    let mtx_path = format!("{}.mtx", base);
    let sidecar_path = format!("{}.json", base);

    let mtx = read_mtx(Path::new(&mtx_path)).expect("mtx");
    let csc = mtx.to_csc().expect("csc");
    let dense_mat = mtx.to_dense();
    let sc = read_sidecar(Path::new(&sidecar_path)).expect("sidecar");
    let rhs = sc.finite_rhs().expect("rhs");

    let n = csc.n;
    let tol = (n as f64) * f64::EPSILON * 1e6;
    let b_norm = norm2(&rhs);

    println!(
        "\n=== {} ({}x{})  tol={:.3e}  ||b||₂={:.3e} ===",
        stem, n, n, tol, b_norm
    );

    // Dense path (this is the path that PASSES per the sparse-only category).
    let (d_factors, d_inertia) = dense_factor(&dense_mat, &dense_params()).expect("dense factor");
    let x_d = dense_solve_refined(&dense_mat, &d_factors, &rhs).expect("dense_solve_refined");
    let rel_d = rel_residual_dense(&dense_mat, &x_d, &rhs);
    println!(
        "  dense   inertia={}  rel_res={:.3e}  {}",
        d_inertia,
        rel_d,
        if rel_d <= tol { "PASS" } else { "FAIL" }
    );

    // Sparse path (the path that FAILS).
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("sym");
    let (s_factors, s_inertia) =
        factorize_multifrontal(&csc, &sym, &sparse_params()).expect("factorize_multifrontal");

    let x_s_plain = solve_sparse(&s_factors, &rhs).expect("solve_sparse");
    let rel_s_plain = rel_residual_csc(&csc, &x_s_plain, &rhs);

    let x_s_ref = solve_sparse_refined(&csc, &s_factors, &rhs).expect("solve_sparse_refined");
    let rel_s_ref = rel_residual_csc(&csc, &x_s_ref, &rhs);

    let (_x_s_ext, traj) = extended_sparse_refine(&csc, &s_factors, &rhs, 10);
    let rel_s_ext = traj.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  sparse  inertia={}", s_inertia);
    println!("    plain         rel_res={:.3e}", rel_s_plain);
    println!(
        "    refined (3)   rel_res={:.3e}  {}",
        rel_s_ref,
        if rel_s_ref <= tol { "PASS" } else { "FAIL" }
    );
    println!(
        "    extended(10)  rel_res={:.3e}  {}",
        rel_s_ext,
        if rel_s_ext <= tol { "PASS" } else { "FAIL" }
    );
    print!("    trajectory: ");
    for r in &traj {
        print!(" {:.2e}", r);
    }
    println!();

    // How far apart are the two solutions?
    let mut diff = 0.0;
    let mut xn = 0.0;
    for i in 0..n {
        let d = x_d[i] - x_s_ref[i];
        diff += d * d;
        xn += x_d[i] * x_d[i];
    }
    let rel_diff = if xn > 0.0 {
        (diff / xn).sqrt()
    } else {
        diff.sqrt()
    };
    println!("  ||x_dense - x_sparse||/||x_dense|| = {:.3e}", rel_diff);

    // Are the dense and sparse matrices the same? Compute A_dense*x_d - A_csc*x_d.
    let mut ad = vec![0.0; n];
    let mut ac = vec![0.0; n];
    dense_mat.symv(&x_d, &mut ad);
    csc.symv(&x_d, &mut ac);
    let mut adn = 0.0;
    let mut diffn = 0.0;
    for i in 0..n {
        adn += ad[i] * ad[i];
        let d = ad[i] - ac[i];
        diffn += d * d;
    }
    let rel_mat = if adn > 0.0 {
        (diffn / adn).sqrt()
    } else {
        diffn.sqrt()
    };
    println!("  ||A_dense*x - A_csc*x||/||A_dense*x|| = {:.3e}", rel_mat);

    // Compare condition signal: did supernodes collapse to one front?
    println!(
        "  symbolic: {} supernodes (n={}), est_nnz_L={}",
        sym.supernodes.len(),
        n,
        sym.factor_nnz_estimate
    );
}

fn main() {
    // PFIT2_0300 — the dramatic outlier: 3.55e-6 on n=6, 2666× over tolerance.
    // If this is a real bug, it should reproduce here.
    let outlier = [("PFIT2", "PFIT2_0300")];

    // Representatives of the bulk pattern (residual 3-10× over tolerance,
    // tiny matrices). One per dominant family.
    let bulk = [
        ("HS46", "HS46_0376"),             // dominant family (16 cases)
        ("FBRAIN3LS", "FBRAIN3LS_0736"),   // 12 cases, top of family
        ("CERI651DLS", "CERI651DLS_0643"), // 9 cases, top of family
        ("CERI651ALS", "CERI651ALS_0364"), // 7 cases
        ("HATFLDFL", "HATFLDFL_0428"),     // 7 cases, top of family
        ("CERI651CLS", "CERI651CLS_0292"), // 6 cases, top
        ("PALMER1ENE", "PALMER1ENE_0107"), // 5 cases, top
        ("ALLINITA", "ALLINITA_0750"),     // 3 cases, top
        ("HS114", "HS114_0758"),           // 1 case but n=21 (largest in REAL_BUG)
    ];

    println!("=== OUTLIER (residual 1000× the others) ===");
    for (fam, stem) in &outlier {
        triage(fam, stem);
    }

    println!("\n=== BULK PATTERN (residual just above tolerance) ===");
    for (fam, stem) in &bulk {
        triage(fam, stem);
    }
}
