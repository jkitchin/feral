//! Compare clnlbeam shape (diag_only ratio, col-degree distribution)
//! against the VESUVIO family to verify whether arrow-KKT-shape
//! heuristics can distinguish them.
//!
//! Background: clnlbeam routes to MC64 under the Auto scaling policy
//! (`diag_only/n >= 0.3`) and MC64 hurts the IPM trajectory. VESUVIO
//! family also has diag_only ratio ≈ 33% and benefits ~240× from MC64.
//! Need a structural feature that separates the two.
//!
//! Usage: `cargo run --release --bin probe_clnlbeam_shape`

use feral::{read_mtx, CscMatrix};
use std::path::Path;

fn shape(label: &str, csc: &CscMatrix) {
    let n = csc.n;
    let mut diag_only = 0usize;
    let mut deg_1 = 0usize;
    let mut deg_2_4 = 0usize;
    let mut deg_5_32 = 0usize;
    let mut deg_gt32 = 0usize;
    let mut max_col_nnz = 0usize;
    for j in 0..n {
        let start = csc.col_ptr[j];
        let end = csc.col_ptr[j + 1];
        let nnz = end - start;
        max_col_nnz = max_col_nnz.max(nnz);
        let mut has_diag = false;
        let mut has_offdiag = false;
        for k in start..end {
            if csc.row_idx[k] == j {
                has_diag = true;
            } else {
                has_offdiag = true;
            }
        }
        if has_diag && !has_offdiag {
            diag_only += 1;
        }
        if nnz <= 1 {
            deg_1 += 1;
        } else if nnz <= 4 {
            deg_2_4 += 1;
        } else if nnz <= 32 {
            deg_5_32 += 1;
        } else {
            deg_gt32 += 1;
        }
    }
    println!(
        "{}: n={} nnz={} diag_only={} ({:.1}%) max_col_nnz={}",
        label,
        n,
        csc.row_idx.len(),
        diag_only,
        100.0 * diag_only as f64 / n as f64,
        max_col_nnz,
    );
    println!(
        "  col_deg: deg=1: {}  deg=2-4: {}  deg=5-32: {}  deg>32: {}",
        deg_1, deg_2_4, deg_5_32, deg_gt32,
    );
}

fn main() {
    let cases: &[(&str, &str)] = &[
        (
            "clnlbeam_0000",
            "data/matrices/kkt-mittelmann/clnlbeam/clnlbeam_0000.mtx",
        ),
        (
            "clnlbeam_0001",
            "data/matrices/kkt-mittelmann/clnlbeam/clnlbeam_0001.mtx",
        ),
    ];
    for (label, p) in cases {
        let path = Path::new(p);
        if !path.exists() {
            eprintln!("SKIP {}: not present", p);
            continue;
        }
        let mtx = match read_mtx(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("SKIP {}: {:?}", p, e);
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP {}: {:?}", p, e);
                continue;
            }
        };
        shape(label, &csc);
    }
}
