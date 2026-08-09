//! Measures the headroom for Knight-Ruiz warm-starting (post-0.15.0 item 1).
//!
//! KR currently starts from `d = 1` on every factorization. In an IPM the
//! KKT values drift smoothly, so the previous factorization's converged `d`
//! should be a near-fixed-point. This probe quantifies that: it runs KR
//! cold, perturbs the values the way an IPM iteration would, then runs KR
//! both cold and warm-started and reports the iteration counts and how far
//! apart the two answers land.
//!
//! Usage: cargo run --release --example probe_kr_warmstart -- <mtx> [drift]

use feral::sparse::csc::CscMatrix;

/// KR iteration, instrumented, with a caller-supplied starting `d`.
/// Mirrors `scaling::infnorm::compute_infnorm` exactly.
fn kr(matrix: &CscMatrix, d0: Option<&[f64]>) -> (Vec<f64>, usize) {
    kr_traced(matrix, d0, false)
}

fn kr_traced(matrix: &CscMatrix, d0: Option<&[f64]>, trace: bool) -> (Vec<f64>, usize) {
    let n = matrix.n;
    let mut d = match d0 {
        Some(v) => v.to_vec(),
        None => vec![1.0f64; n],
    };
    let max_iter = 10;
    let tol = 1e-8;
    let mut row_max = vec![0.0f64; n];
    let mut iters = 0;
    for _ in 0..max_iter {
        iters += 1;
        for r in row_max.iter_mut() {
            *r = 0.0;
        }
        for j in 0..n {
            let dj = d[j];
            let mut col_max = row_max[j];
            for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
                let i = matrix.row_idx[k];
                let v = (d[i] * matrix.values[k] * dj).abs();
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
            let m = row_max[i];
            if m > 0.0 {
                d[i] /= m.sqrt();
                let dev = (m - 1.0).abs();
                if dev > max_dev {
                    max_dev = dev;
                }
            }
        }
        if trace {
            eprintln!("      iter {iters:>2}: max_dev = {max_dev:.6e}");
        }
        if max_dev < tol {
            break;
        }
    }
    (d, iters)
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: <mtx> [drift]");
    let drift: f64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.05);
    let m = feral::read_mtx(std::path::Path::new(&path))
        .unwrap()
        .to_csc()
        .unwrap();

    // Factorization 1: cold KR on the original values.
    let (d1, it1) = kr(&m, None);

    // Simulate one IPM step: perturb every value multiplicatively. A
    // deterministic pseudo-random walk stands in for the barrier update.
    let mut m2 = m.clone();
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    for v in m2.values.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
        *v *= 1.0 + drift * (2.0 * u - 1.0);
    }

    // Factorization 2: cold vs warm-started from d1.
    let (d_cold, it_cold) = kr(&m2, None);
    let (d_warm, it_warm) = kr(&m2, Some(&d1));

    // How far apart are the two converged answers?
    let mut max_rel = 0.0f64;
    for i in 0..m.n {
        let a = d_cold[i];
        let b = d_warm[i];
        if a != 0.0 {
            let r = ((a - b) / a).abs();
            if r > max_rel {
                max_rel = r;
            }
        }
    }
    let identical = d_cold
        .iter()
        .zip(&d_warm)
        .all(|(a, b)| a.to_bits() == b.to_bits());

    println!(
        "{}: n={} nnz={} drift={:.3}",
        path.rsplit('/').next().unwrap(),
        m.n,
        m.values.len(),
        drift
    );
    println!("  KR iters: first-factor cold={it1}  next-factor cold={it_cold}  next-factor WARM={it_warm}");
    println!(
        "  cold-vs-warm converged d: max rel diff = {max_rel:.3e}, bit-identical = {identical}"
    );
    if std::env::var("KR_TRACE").is_ok() {
        eprintln!("    convergence trajectory (cold, perturbed matrix):");
        let _ = kr_traced(&m2, None, true);
    }
}
