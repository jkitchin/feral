use feral::scaling::pick_scaling_strategy;
use feral::sparse::csc::CscMatrix;
use std::fs::File;
use std::io::Read;
fn ru64(f: &mut File) -> u64 {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).unwrap();
    u64::from_le_bytes(b)
}
fn ri64(f: &mut File) -> i64 {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).unwrap();
    i64::from_le_bytes(b)
}
fn rf64(f: &mut File) -> f64 {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).unwrap();
    f64::from_le_bytes(b)
}
fn main() {
    for i in 0..18 {
        let p = format!("/tmp/rkt_{:03}.bin", i);
        let mut f = match File::open(&p) {
            Ok(f) => f,
            _ => continue,
        };
        let dim = ru64(&mut f) as usize;
        let nnz = ru64(&mut f) as usize;
        let _ = ru64(&mut f);
        let ia: Vec<usize> = (0..nnz).map(|_| ri64(&mut f) as usize).collect();
        let ja: Vec<usize> = (0..nnz).map(|_| ri64(&mut f) as usize).collect();
        let vals: Vec<f64> = (0..nnz).map(|_| rf64(&mut f)).collect();
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        for k in 0..nnz {
            let (i, j) = (ia[k] - 1, ja[k] - 1);
            if i >= j {
                rows.push(i);
                cols.push(j);
            } else {
                rows.push(j);
                cols.push(i);
            }
        }
        let csc = CscMatrix::from_triplets(dim, &rows, &cols, &vals).unwrap();
        let mut diag_only = 0usize;
        let mut max_col_nnz = 0usize;
        for jj in 0..dim {
            let nc = csc.col_ptr[jj + 1] - csc.col_ptr[jj];
            if nc > max_col_nnz {
                max_col_nnz = nc;
            }
            let has_d = (csc.col_ptr[jj]..csc.col_ptr[jj + 1]).any(|k| csc.row_idx[k] == jj);
            let has_o = (csc.col_ptr[jj]..csc.col_ptr[jj + 1]).any(|k| csc.row_idx[k] != jj);
            if has_d && !has_o {
                diag_only += 1;
            }
        }
        let pick = pick_scaling_strategy(&csc);
        println!(
            "rkt_{:03}: n={} nnz={} diag={} ({:.1}%) max_col_nnz={} pick={:?}",
            i,
            dim,
            nnz,
            diag_only,
            100.0 * diag_only as f64 / dim as f64,
            max_col_nnz,
            pick
        );
    }
}
