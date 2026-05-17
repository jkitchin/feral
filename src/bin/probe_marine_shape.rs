use feral::{read_mtx, CscMatrix};
use std::path::Path;
fn shape(label: &str, csc: &CscMatrix) {
    let n = csc.n;
    let mut diag_only = 0usize;
    let mut deg_1 = 0usize;
    let mut deg_2_4 = 0usize;
    let mut deg_5_32 = 0usize;
    let mut deg_33_128 = 0usize;
    let mut deg_gt128 = 0usize;
    let mut max_col_nnz = 0usize;
    let mut sum_nnz = 0usize;
    for j in 0..n {
        let nnz = csc.col_ptr[j + 1] - csc.col_ptr[j];
        sum_nnz += nnz;
        max_col_nnz = max_col_nnz.max(nnz);
        let has_diag = (csc.col_ptr[j]..csc.col_ptr[j + 1]).any(|k| csc.row_idx[k] == j);
        let has_off = (csc.col_ptr[j]..csc.col_ptr[j + 1]).any(|k| csc.row_idx[k] != j);
        if has_diag && !has_off {
            diag_only += 1;
        }
        if nnz <= 1 {
            deg_1 += 1;
        } else if nnz <= 4 {
            deg_2_4 += 1;
        } else if nnz <= 32 {
            deg_5_32 += 1;
        } else if nnz <= 128 {
            deg_33_128 += 1;
        } else {
            deg_gt128 += 1;
        }
    }
    println!(
        "{}: n={} nnz={} diag_only={} ({:.1}%) max_col_nnz={} avg_col_nnz={:.1}",
        label,
        n,
        sum_nnz,
        diag_only,
        100.0 * diag_only as f64 / n as f64,
        max_col_nnz,
        sum_nnz as f64 / n as f64
    );
    println!(
        "  col_deg: 1:{} 2-4:{} 5-32:{} 33-128:{} >128:{}",
        deg_1, deg_2_4, deg_5_32, deg_33_128, deg_gt128
    );
    let pick = feral::scaling::pick_scaling_strategy(csc);
    println!("  scaling pick: {:?}", pick);
}
fn main() {
    for i in 0..5 {
        let p = format!(
            "data/matrices/kkt-mittelmann/marine_1600/marine_1600_{:04}.mtx",
            i
        );
        let path = Path::new(&p);
        if !path.exists() {
            eprintln!("SKIP {}", p);
            continue;
        }
        match read_mtx(path).and_then(|m| m.to_csc()) {
            Ok(c) => shape(&p, &c),
            Err(e) => eprintln!("ERR {}: {:?}", p, e),
        }
    }
}
