use std::path::Path;

use feral::{
    factor, read_mtx, read_sidecar, solve_refined, BunchKaufmanParams, Inertia, ZeroPivotAction,
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
    let mtx_path = Path::new("data/matrices/kkt/CERI651DLS/CERI651DLS_0534.mtx");
    let json_path = Path::new("data/matrices/kkt/CERI651DLS/CERI651DLS_0534.json");

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

    println!("=== CERI651DLS_0534 triage ===");
    println!("n = {}, expected inertia = {}", n, expected);
    println!("Consensus from MUMPS+SSIDS+rmumps: (7, 0, 0)  — SPD");
    println!("Feral reports: (6, 0, 1) — one positive labeled zero");

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
    println!(
        "\nDiagonal range: [{:.2e}, {:.2e}]  ratio={:.1e}",
        min_diag,
        max_diag,
        max_diag / min_diag
    );

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };

    println!("\n--- Default params ---");
    let (dfac, dinertia) = factor(&dense_mat, &params).expect("dense factor");
    println!(
        "inertia: {} ({})",
        dinertia,
        if dinertia == expected { "MATCH" } else { "MISMATCH" }
    );
    println!("zero_tol = {:.3e}", dfac.zero_tol);
    println!("zero_tol_2x2 = {:.3e}", dfac.zero_tol_2x2);

    println!("\nD diagonal:");
    for i in 0..n {
        let marker = if dfac.d_subdiag[i] != 0.0 { "2x2" } else { "1x1" };
        println!(
            "  D[{}] = {:+.6e}  subdiag={:+.3e}  [{}]",
            i, dfac.d_diag[i], dfac.d_subdiag[i], marker
        );
    }
    println!("\nEquilibration d_eq:");
    for i in 0..n {
        println!("  d_eq[{}] = {:.3e}", i, dfac.d_eq[i]);
    }
    println!("\nBK perm: {:?}", dfac.perm);

    let dx = solve_refined(&dense_mat, &dfac, &rhs).expect("solve");
    println!("\nrefined rel_res = {:.3e}", rel_residual(&csc, &dx, &rhs));

    // Stricter zero_tol experiment
    println!("\n--- With zero_tol = 1e-30 ---");
    let strict = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        zero_tol: 1e-30,
        zero_tol_2x2: 1e-60,
        ..BunchKaufmanParams::default()
    };
    let (sfac, sinertia) = factor(&dense_mat, &strict).expect("strict");
    println!(
        "inertia: {} ({})",
        sinertia,
        if sinertia == expected { "MATCH" } else { "MISMATCH" }
    );
    println!("\nStrict D diagonal:");
    for i in 0..n {
        let marker = if sfac.d_subdiag[i] != 0.0 { "2x2" } else { "1x1" };
        println!(
            "  D[{}] = {:+.6e}  subdiag={:+.3e}  [{}]",
            i, sfac.d_diag[i], sfac.d_subdiag[i], marker
        );
    }
    let sx = solve_refined(&dense_mat, &sfac, &rhs).expect("solve");
    println!("\nrefined rel_res = {:.3e}", rel_residual(&csc, &sx, &rhs));
}
