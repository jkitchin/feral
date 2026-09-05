// Replica of feral::scaling::infnorm::compute_infnorm's loop, purely to
// count how many Knight-Ruiz sweeps each matrix needs. Same tol (1e-8),
// same cap (10), same update.
use feral::read_mtx;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for p in std::env::args().skip(1) {
        let m = read_mtx(std::path::Path::new(&p)).and_then(|m| m.to_csc())?;
        let n = m.n;
        let mut d = vec![1.0f64; n];
        let mut row_max = vec![0.0f64; n];
        let mut used = 10;
        for it in 0..10 {
            for r in row_max.iter_mut() {
                *r = 0.0;
            }
            for j in 0..n {
                let dj = d[j];
                let mut col_max = row_max[j];
                for k in m.col_ptr[j]..m.col_ptr[j + 1] {
                    let i = m.row_idx[k];
                    let v = (d[i] * m.values[k] * dj).abs();
                    if i != j && v > row_max[i] {
                        row_max[i] = v;
                    }
                    if v > col_max {
                        col_max = v;
                    }
                }
                row_max[j] = col_max;
            }
            let mut max_dev = 0.0f64;
            for i in 0..n {
                let mx = row_max[i];
                if mx > 0.0 {
                    d[i] /= mx.sqrt();
                    let dev = (mx - 1.0).abs();
                    if dev > max_dev {
                        max_dev = dev;
                    }
                }
            }
            if max_dev < 1e-8 {
                used = it + 1;
                break;
            }
        }
        println!(
            "{:<24} n={:<8} nnz={:<9} KR sweeps used = {}",
            std::path::Path::new(&p)
                .file_stem()
                .unwrap()
                .to_string_lossy(),
            n,
            m.values.len(),
            used
        );
    }
    Ok(())
}
