use std::path::Path;

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::{solve_sparse, solve_sparse_refined};
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
    let mtx_path = Path::new("data/matrices/kkt/ERRINBAR/ERRINBAR_0824.mtx");
    let json_path = Path::new("data/matrices/kkt/ERRINBAR/ERRINBAR_0824.json");

    let mtx = read_mtx(mtx_path).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let dense_mat = mtx.to_dense();
    let sc = read_sidecar(json_path).expect("read sidecar");
    let rhs = sc.finite_rhs().expect("finite rhs");
    let n = csc.n;

    let expected = Inertia {
        positive: sc.inertia.positive,
        negative: sc.inertia.negative,
        zero: sc.inertia.zero,
    };

    println!("=== ERRINBAR_0824 triage ===");
    println!("n = {}, expected inertia = {}", n, expected);
    println!("delta_w = {:?}, delta_c = {:?}", sc.delta_w, sc.delta_c);
    println!("iteration = {}", sc.iteration);

    // Magnitude analysis
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0f64;
    for i in 0..n {
        let v = dense_mat.get(i, i).abs();
        if v > 0.0 && v < min_diag {
            min_diag = v;
        }
        if v > max_diag {
            max_diag = v;
        }
    }
    let mut max_off = 0.0f64;
    for i in 0..n {
        for j in 0..i {
            let v = dense_mat.get(i, j).abs();
            if v > max_off {
                max_off = v;
            }
        }
    }
    println!(
        "\nDiagonal range:    [{:.2e}, {:.2e}]  ratio={:.1e}",
        min_diag,
        max_diag,
        max_diag / min_diag
    );
    println!("Max |off-diag|:    {:.2e}", max_off);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    let np = feral::numeric::factorize::NumericParams::with_bk(params.clone());

    // ---- Dense path ----
    println!("\n--- DENSE ---");
    let (dfac, dinertia) = factor(&dense_mat, &params).expect("dense factor");
    println!(
        "inertia: {} ({})",
        dinertia,
        if dinertia == expected {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    println!("needs_refinement: {}", dfac.needs_refinement);
    let dxr = solve_refined(&dense_mat, &dfac, &rhs).expect("dense solve_refined");
    let dx = solve(&dfac, &rhs).expect("dense solve");
    println!("plain   rel_res = {:.3e}", rel_residual(&csc, &dx, &rhs));
    println!("refined rel_res = {:.3e}", rel_residual(&csc, &dxr, &rhs));

    // ---- Sparse path with default (multi-supernode) ----
    println!("\n--- SPARSE (default nemin=32) ---");
    let snode_default = SupernodeParams::default();
    let sym_d = symbolic_factorize(&csc, &snode_default).expect("symbolic");
    let (sfac_d, sinertia_d) = factorize_multifrontal(&csc, &sym_d, &np).expect("sparse factor");
    println!(
        "n_supernodes = {}, inertia = {} ({})",
        sym_d.supernodes.len(),
        sinertia_d,
        if sinertia_d == expected {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    println!("supernodes:");
    for (i, snode) in sym_d.supernodes.iter().enumerate() {
        println!(
            "  snode {}: first_col={} ncol={} nrow={}",
            i, snode.first_col, snode.ncol, snode.nrow
        );
    }
    let sx_d = solve_sparse(&sfac_d, &rhs).expect("solve_sparse");
    let sxr_d = solve_sparse_refined(&csc, &sfac_d, &rhs).expect("solve_sparse_refined");
    println!("plain   rel_res = {:.3e}", rel_residual(&csc, &sx_d, &rhs));
    println!("refined rel_res = {:.3e}", rel_residual(&csc, &sxr_d, &rhs));

    // ---- Sparse path with nemin=10000 (single supernode, matches bench) ----
    println!("\n--- SPARSE (nemin=10000, single supernode) ---");
    let snode_one = SupernodeParams {
        nemin: 10000,
        ..Default::default()
    };
    let sym_o = symbolic_factorize(&csc, &snode_one).expect("symbolic");
    let (sfac_o, sinertia_o) = factorize_multifrontal(&csc, &sym_o, &np).expect("sparse factor");
    println!(
        "n_supernodes = {}, inertia = {} ({})",
        sym_o.supernodes.len(),
        sinertia_o,
        if sinertia_o == expected {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    let sx_o = solve_sparse(&sfac_o, &rhs).expect("solve_sparse");
    let sxr_o = solve_sparse_refined(&csc, &sfac_o, &rhs).expect("solve_sparse_refined");
    println!("plain   rel_res = {:.3e}", rel_residual(&csc, &sx_o, &rhs));
    println!("refined rel_res = {:.3e}", rel_residual(&csc, &sxr_o, &rhs));

    // ---- Sparse path with nemin=1 (no amalgamation) ----
    println!("\n--- SPARSE (nemin=1, no amalgamation) ---");
    let snode_none = SupernodeParams {
        nemin: 1,
        ..Default::default()
    };
    let sym_n = symbolic_factorize(&csc, &snode_none).expect("symbolic");
    let (sfac_n, sinertia_n) = factorize_multifrontal(&csc, &sym_n, &np).expect("sparse factor");
    println!(
        "n_supernodes = {}, inertia = {} ({})",
        sym_n.supernodes.len(),
        sinertia_n,
        if sinertia_n == expected {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    let sx_n = solve_sparse(&sfac_n, &rhs).expect("solve_sparse");
    let sxr_n = solve_sparse_refined(&csc, &sfac_n, &rhs).expect("solve_sparse_refined");
    println!("plain   rel_res = {:.3e}", rel_residual(&csc, &sx_n, &rhs));
    println!("refined rel_res = {:.3e}", rel_residual(&csc, &sxr_n, &rhs));

    // ---- Compare dense vs sparse solutions ----
    println!("\n--- Solution comparison (dense_refined vs sparse_one) ---");
    let mut max_diff = 0.0f64;
    let mut max_diff_idx = 0;
    let mut max_dense = 0.0f64;
    for i in 0..n {
        let d = (dxr[i] - sxr_o[i]).abs();
        if d > max_diff {
            max_diff = d;
            max_diff_idx = i;
        }
        if dxr[i].abs() > max_dense {
            max_dense = dxr[i].abs();
        }
    }
    println!(
        "max |x_dense - x_sparse| = {:.3e} at idx {}",
        max_diff, max_diff_idx
    );
    println!("  x_dense[{}] = {:.6e}", max_diff_idx, dxr[max_diff_idx]);
    println!("  x_sparse[{}] = {:.6e}", max_diff_idx, sxr_o[max_diff_idx]);
    println!("  rel diff = {:.3e}", max_diff / max_dense.max(1e-300));
}
