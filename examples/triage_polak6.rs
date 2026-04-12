use std::path::Path;

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
    let mtx_path = Path::new("data/matrices/kkt/POLAK6/POLAK6_0021.mtx");
    let json_path = Path::new("data/matrices/kkt/POLAK6/POLAK6_0021.json");

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

    println!("=== POLAK6_0021 triage ===");
    println!("n = {}, expected inertia = {}", n, expected);
    println!("delta_w = {:?}, delta_c = {:?}", sc.delta_w, sc.delta_c);

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

    let mut max_rhs = 0.0f64;
    let mut min_rhs = f64::INFINITY;
    for &r in &rhs {
        let a = r.abs();
        if a > max_rhs {
            max_rhs = a;
        }
        if a > 0.0 && a < min_rhs {
            min_rhs = a;
        }
    }
    println!(
        "RHS range:         [{:.2e}, {:.2e}]  ratio={:.1e}",
        min_rhs,
        max_rhs,
        max_rhs / min_rhs
    );

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };

    println!("\n--- Feral DENSE ---");
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

    // Print equilibration vector
    println!("\nEquilibration d_eq:");
    for i in 0..n {
        println!("  d_eq[{}] = {:.3e}", i, dfac.d_eq[i]);
    }

    // Print D-block diagonals after factorization
    println!("\nD diagonal (post-factor):");
    for i in 0..n {
        println!(
            "  D[{}] = {:.3e}  (subdiag {:.3e})",
            i, dfac.d_diag[i], dfac.d_subdiag[i]
        );
    }

    // Print pivot permutation
    println!("\nBK pivot perm: {:?}", dfac.perm);

    let dx = solve(&dfac, &rhs).expect("dense solve");
    println!(
        "\nplain solve  rel_res = {:.3e}",
        rel_residual(&csc, &dx, &rhs)
    );
    let dxr = solve_refined(&dense_mat, &dfac, &rhs).expect("dense solve_refined");
    println!(
        "refined solve rel_res = {:.3e}",
        rel_residual(&csc, &dxr, &rhs)
    );

    // Try with stricter pivot threshold
    println!("\n--- Feral DENSE with stricter pivot threshold ---");
    let strict_params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        zero_tol: 1e-30,
        ..BunchKaufmanParams::default()
    };
    let (dfac2, dinertia2) = factor(&dense_mat, &strict_params).expect("strict factor");
    println!(
        "inertia: {} ({})",
        dinertia2,
        if dinertia2 == expected {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    let dxr2 = solve_refined(&dense_mat, &dfac2, &rhs).expect("strict solve");
    println!("refined rel_res = {:.3e}", rel_residual(&csc, &dxr2, &rhs));

    // Try with looser pivot threshold
    println!("\n--- Feral DENSE with looser pivot threshold ---");
    let loose_params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        zero_tol: 1e-3,
        ..BunchKaufmanParams::default()
    };
    let (dfac3, dinertia3) = factor(&dense_mat, &loose_params).expect("loose factor");
    println!(
        "inertia: {} ({})",
        dinertia3,
        if dinertia3 == expected {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    let dxr3 = solve_refined(&dense_mat, &dfac3, &rhs).expect("loose solve");
    println!("refined rel_res = {:.3e}", rel_residual(&csc, &dxr3, &rhs));
}
