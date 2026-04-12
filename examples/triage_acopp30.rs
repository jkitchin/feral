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
    let mtx_path = Path::new("data/matrices/kkt/ACOPP30/ACOPP30_0000.mtx");
    let json_path = Path::new("data/matrices/kkt/ACOPP30/ACOPP30_0000.json");

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

    println!("=== ACOPP30_0000 triage ===");
    println!("n = {}, expected inertia = {}", n, expected);
    println!("delta_w = {:?}, delta_c = {:?}", sc.delta_w, sc.delta_c);
    println!("iteration = {}", sc.iteration);

    // Magnitude analysis
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0f64;
    let mut zero_diag = 0;
    for i in 0..n {
        let v = dense_mat.get(i, i).abs();
        if v == 0.0 {
            zero_diag += 1;
        } else {
            if v < min_diag {
                min_diag = v;
            }
            if v > max_diag {
                max_diag = v;
            }
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
        "\nDiagonal range:    [{:.2e}, {:.2e}]  ratio={:.1e}  (zero diags: {})",
        min_diag,
        max_diag,
        max_diag / min_diag,
        zero_diag
    );
    println!("Max |off-diag|:    {:.2e}", max_off);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };

    println!("\n--- Default params ---");
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
    println!("zero_tol: {:.3e}", dfac.zero_tol);

    // Print which D entries are near zero
    println!("\nNear-zero D entries (|d| < 1e-10):");
    for i in 0..n {
        if dfac.d_diag[i].abs() < 1e-10 {
            println!(
                "  D[{}] = {:.3e}  (subdiag {:.3e})  d_eq[i]={:.3e}",
                i, dfac.d_diag[i], dfac.d_subdiag[i], dfac.d_eq[i]
            );
        }
    }
    let smallest = dfac
        .d_diag
        .iter()
        .map(|d| d.abs())
        .filter(|d| *d > 0.0)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let largest = dfac
        .d_diag
        .iter()
        .map(|d| d.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    println!(
        "\nD pivot magnitude range: [{:.3e}, {:.3e}]",
        smallest, largest
    );

    let dxr = solve_refined(&dense_mat, &dfac, &rhs).expect("dense solve_refined");
    println!("refined rel_res = {:.3e}", rel_residual(&csc, &dxr, &rhs));

    let dx = solve(&dfac, &rhs).expect("dense solve");
    println!("plain   rel_res = {:.3e}", rel_residual(&csc, &dx, &rhs));

    // Try with looser pivot threshold (mirroring POLAK6 experiment)
    println!("\n--- Stricter zero_tol = 1e-30 ---");
    let strict = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        zero_tol: 1e-30,
        ..BunchKaufmanParams::default()
    };
    let (sfac, sinertia) = factor(&dense_mat, &strict).expect("strict factor");
    println!(
        "inertia: {} ({})",
        sinertia,
        if sinertia == expected {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    let sxr = solve_refined(&dense_mat, &sfac, &rhs).expect("strict solve");
    println!("refined rel_res = {:.3e}", rel_residual(&csc, &sxr, &rhs));

    println!("\n--- Looser zero_tol = 1e-8 ---");
    let loose = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        zero_tol: 1e-8,
        ..BunchKaufmanParams::default()
    };
    let (lfac, linertia) = factor(&dense_mat, &loose).expect("loose factor");
    println!(
        "inertia: {} ({})",
        linertia,
        if linertia == expected {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    let lxr = solve_refined(&dense_mat, &lfac, &rhs).expect("loose solve");
    println!("refined rel_res = {:.3e}", rel_residual(&csc, &lxr, &rhs));

    println!("\n--- Stricter alpha = 0.1 (less aggressive 2x2 selection) ---");
    let alphalow = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        alpha: 0.1,
        ..BunchKaufmanParams::default()
    };
    let (afac, ainertia) = factor(&dense_mat, &alphalow).expect("alpha factor");
    println!(
        "inertia: {} ({})",
        ainertia,
        if ainertia == expected {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    let axr = solve_refined(&dense_mat, &afac, &rhs).expect("alpha solve");
    println!("refined rel_res = {:.3e}", rel_residual(&csc, &axr, &rhs));
}
