//! Integration tests for real KKT matrices from collect_kkt.
//!
//! These tests are `#[ignore]`d because they require data files in
//! `data/matrices/kkt/` that are not committed to git. Run with:
//!
//!     cargo test -- --ignored

use feral::{factor, read_mtx, read_sidecar, solve, BunchKaufmanParams, Inertia, ZeroPivotAction};
use std::path::Path;

#[test]
#[ignore]
fn test_kkt_matrices_inertia_and_solve() {
    let kkt_dir = Path::new("data/matrices/kkt");
    if !kkt_dir.is_dir() {
        eprintln!(
            "SKIP: {} not found. Run collect_kkt from ripopt to generate.",
            kkt_dir.display()
        );
        return;
    }

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };

    let mut n_tested = 0usize;
    let mut n_inertia_fail = 0usize;
    let mut n_residual_fail = 0usize;
    let mut failures: Vec<String> = Vec::new();

    let mut subdirs: Vec<_> = std::fs::read_dir(kkt_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    subdirs.sort_by_key(|e| e.file_name());

    for subdir in subdirs {
        let mut mtx_files: Vec<_> = std::fs::read_dir(subdir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "mtx"))
            .collect();
        mtx_files.sort_by_key(|e| e.file_name());

        for mtx_entry in mtx_files {
            let mtx_path = mtx_entry.path();
            let stem = mtx_path.file_stem().unwrap().to_string_lossy().to_string();
            let json_path = mtx_path.with_extension("json");

            if !json_path.exists() {
                continue;
            }

            let mtx = match read_mtx(&mtx_path) {
                Ok(m) => m,
                Err(e) => {
                    failures.push(format!("{}: mtx parse error: {}", stem, e));
                    continue;
                }
            };

            if mtx.n > 1000 {
                continue;
            }

            let sidecar = match read_sidecar(&json_path) {
                Ok(s) => s,
                Err(e) => {
                    failures.push(format!("{}: json parse error: {}", stem, e));
                    continue;
                }
            };

            // Skip matrices with NaN/Inf in RHS
            let rhs = match sidecar.finite_rhs() {
                Some(r) => r,
                None => continue,
            };

            let dense = mtx.to_dense();
            n_tested += 1;

            // Factor
            let (factors, inertia) = match factor(&dense, &params) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!("{}: factor failed: {}", stem, e));
                    continue;
                }
            };

            // Inertia check
            let expected = Inertia {
                positive: sidecar.inertia.positive,
                negative: sidecar.inertia.negative,
                zero: sidecar.inertia.zero,
            };
            if inertia != expected {
                n_inertia_fail += 1;
                failures.push(format!(
                    "{}: inertia mismatch: got {} expected {}",
                    stem, inertia, expected
                ));
            }

            // Solve and residual check
            if rhs.len() != dense.n {
                failures.push(format!(
                    "{}: rhs len {} != matrix dim {}",
                    stem,
                    rhs.len(),
                    dense.n
                ));
                continue;
            }
            let x = match solve(&factors, &rhs) {
                Ok(x) => x,
                Err(e) => {
                    failures.push(format!("{}: solve failed: {}", stem, e));
                    continue;
                }
            };

            let mut ax = vec![0.0; dense.n];
            dense.symv(&x, &mut ax);
            let mut res_sq = 0.0;
            let mut b_sq = 0.0;
            for i in 0..dense.n {
                let r = ax[i] - rhs[i];
                res_sq += r * r;
                b_sq += rhs[i] * rhs[i];
            }
            let rel_res = if b_sq > 0.0 {
                (res_sq / b_sq).sqrt()
            } else {
                res_sq.sqrt()
            };

            let tol = (dense.n as f64) * f64::EPSILON * 1e6;
            if rel_res > tol {
                n_residual_fail += 1;
                failures.push(format!(
                    "{}: residual {:.2e} > tol {:.2e}",
                    stem, rel_res, tol
                ));
            }
        }
    }

    eprintln!("KKT integration test: {} matrices tested", n_tested);
    eprintln!("  Inertia failures: {}", n_inertia_fail);
    eprintln!("  Residual failures: {}", n_residual_fail);

    if !failures.is_empty() {
        eprintln!("\nFailures:");
        for f in &failures {
            eprintln!("  {}", f);
        }
        panic!(
            "{} failures out of {} matrices tested",
            failures.len(),
            n_tested
        );
    }
}
