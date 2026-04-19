use std::path::Path;

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    factor, read_mtx, read_sidecar, solve, solve_refined, BunchKaufmanParams, Inertia,
    ZeroPivotAction,
};

fn rel_residual(a: &feral::CscMatrix, x: &[f64], b: &[f64]) -> f64 {
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

fn main() {
    let mtx_path = Path::new("data/matrices/kkt/MGH10S/MGH10S_0000.mtx");
    let json_path = Path::new("data/matrices/kkt/MGH10S/MGH10S_0000.json");

    let mtx = read_mtx(mtx_path).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let dense_mat = mtx.to_dense();
    let sc = read_sidecar(json_path).expect("read sidecar");
    let rhs = sc.finite_rhs().expect("finite rhs");
    let n = csc.n;

    let expected_inertia = Inertia {
        positive: sc.inertia.positive,
        negative: sc.inertia.negative,
        zero: sc.inertia.zero,
    };

    println!("=== MGH10S_0000 triage ===");
    println!("n = {}, expected inertia = {}", n, expected_inertia);
    println!("rhs[0..5] = {:?}", &rhs[..5.min(n)]);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };

    // ---- Dense path ----
    println!("\n--- DENSE ---");
    let (dfac, dinertia) = factor(&dense_mat, &params).expect("dense factor");
    println!(
        "dense inertia = {} ({})",
        dinertia,
        if dinertia == expected_inertia {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    println!("dense needs_refinement = {}", dfac.needs_refinement);
    let dx = solve(&dfac, &rhs).expect("dense solve");
    println!(
        "dense solve  rel_res = {:.3e}",
        rel_residual(&csc, &dx, &rhs)
    );
    let dxr = solve_refined(&dense_mat, &dfac, &rhs).expect("dense solve_refined");
    println!(
        "dense refined rel_res = {:.3e}",
        rel_residual(&csc, &dxr, &rhs)
    );

    // ---- Sparse path ----
    println!("\n--- SPARSE ---");
    let snode = SupernodeParams::default();
    let sym = symbolic_factorize(&csc, &snode).expect("symbolic");
    println!(
        "n_supernodes = {}, perm[0..min(10,n)] = {:?}",
        sym.supernodes.len(),
        &sym.perm[..10.min(sym.perm.len())]
    );
    let np = feral::numeric::factorize::NumericParams::with_bk(params.clone());
    let (sfac, sinertia) = factorize_multifrontal(&csc, &sym, &np).expect("sparse factor");
    println!(
        "sparse inertia = {} ({})",
        sinertia,
        if sinertia == expected_inertia {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    println!("sparse needs_refinement = {}", sfac.needs_refinement);
    println!("node_factors:");
    for (i, nf) in sfac.node_factors.iter().enumerate() {
        println!(
            "  node {}: first_col={} ncol={} nrow={} inertia={}",
            i, nf.first_col, nf.ncol, nf.nrow, nf.inertia
        );
    }
    let sx = solve_sparse(&sfac, &rhs).expect("sparse solve");
    println!(
        "sparse solve rel_res = {:.3e}",
        rel_residual(&csc, &sx, &rhs)
    );

    // Component-wise comparison of dense and sparse solutions
    let mut max_diff = 0.0f64;
    let mut max_dense = 0.0f64;
    for i in 0..n {
        max_diff = max_diff.max((dxr[i] - sx[i]).abs());
        max_dense = max_dense.max(dxr[i].abs());
    }
    println!(
        "\n||dxr - sx||_inf = {:.3e}, ||dxr||_inf = {:.3e}, rel = {:.3e}",
        max_diff,
        max_dense,
        max_diff / max_dense.max(1e-300)
    );
}
