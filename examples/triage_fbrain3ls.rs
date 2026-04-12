use std::path::Path;

use feral::{factor, read_mtx, read_sidecar, BunchKaufmanParams, Inertia, ZeroPivotAction};

fn main() {
    let mtx_path = Path::new("data/matrices/kkt/FBRAIN3LS/FBRAIN3LS_0788.mtx");
    let json_path = Path::new("data/matrices/kkt/FBRAIN3LS/FBRAIN3LS_0788.json");

    let mtx = read_mtx(mtx_path).expect("read mtx");
    let dense_mat = mtx.to_dense();
    let sc = read_sidecar(json_path).expect("read sidecar");
    let n = dense_mat.n;

    let expected = Inertia {
        positive: sc.inertia.positive,
        negative: sc.inertia.negative,
        zero: sc.inertia.zero,
    };

    println!("=== FBRAIN3LS_0788 triage ===");
    println!("n = {}, expected = {}", n, expected);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    let (dfac, dinertia) = factor(&dense_mat, &params).expect("factor");
    println!("\nDefault: inertia = {} zero_tol = {:.3e}", dinertia, dfac.zero_tol);
    for i in 0..n {
        let m = if dfac.d_subdiag[i] != 0.0 { "2x2" } else { "1x1" };
        println!(
            "  D[{}] = {:+.6e}  subdiag={:+.3e}  [{}]",
            i, dfac.d_diag[i], dfac.d_subdiag[i], m
        );
    }

    let strict = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        zero_tol: f64::EPSILON,
        zero_tol_2x2: f64::EPSILON * f64::EPSILON,
        ..BunchKaufmanParams::default()
    };
    let (_, sinertia) = factor(&dense_mat, &strict).expect("strict");
    println!("\nWith zero_tol = eps: {} ({})",
        sinertia, if sinertia == expected { "MATCH" } else { "MISMATCH" });
}
