//! Warm-solve scaling probe for the sparse LU under update chains (issue #81).
//!
//! The current sparse rank-1 update stores a product-form eta with a *dense*
//! `τ` vector (length n) per update, so each `ftran`/`btran` costs `O(k·n)`
//! after `k` updates and the eta file grows `O(k·n)` in memory. This probe
//! measures `ftran` time and eta storage vs the update-chain length `k` on a
//! large sparse basis, to size the case for a true Forrest–Tomlin sparse
//! row-eta.
//!
//! Run: `cargo run -p feral-diagnostics --release --bin lu_update_probe`

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{LuParams, SparseLu, SparseLuSymbolic};
use std::time::Instant;

fn tridiagonal(n: usize) -> SparseColMatrix {
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (j, col) in cols.iter_mut().enumerate() {
        if j > 0 {
            col.push((j - 1, -1.0));
        }
        col.push((j, 4.0));
        if j + 1 < n {
            col.push((j + 1, -1.0));
        }
    }
    SparseColMatrix::from_sparse_columns(n, &cols).expect("tridiagonal")
}

/// A fresh, diagonally-dominant sparse replacement column for slot `j`.
fn replacement_col(n: usize, j: usize) -> Vec<f64> {
    let mut c = vec![0.0; n];
    c[j] = 5.0;
    if j > 0 {
        c[j - 1] = -0.7;
    }
    if j + 1 < n {
        c[j + 1] = -0.6;
    }
    c
}

fn avg_ftran_us(lu: &mut SparseLu, n: usize, reps: usize) -> f64 {
    let mut total = 0.0;
    for t in 0..reps {
        let mut rhs: Vec<f64> = (0..n).map(|i| 1.0 + ((i + t) % 7) as f64).collect();
        let t0 = Instant::now();
        lu.ftran(&mut rhs).expect("ftran");
        total += t0.elapsed().as_secs_f64() * 1e6;
    }
    total / reps as f64
}

fn main() {
    let n = 5000;
    let a = tridiagonal(n);
    let symbolic = SparseLuSymbolic::analyze(&a).expect("analyze");
    let params = LuParams {
        max_updates: 100_000,
        ..LuParams::default()
    };
    let mut lu = SparseLu::factor(&a, &symbolic, params).expect("factor");

    println!("n = {n}, base factor nnz = {}", lu.factor_nnz());
    println!(
        "{:>8}  {:>14}  {:>12}",
        "updates", "ftran(µs/avg)", "eta_nnz"
    );
    let base = avg_ftran_us(&mut lu, n, 50);
    println!("{:>8}  {:>14.2}  {:>12}", 0, base, 0);

    let checkpoints = [10usize, 25, 50, 100, 200, 400];
    let mut done = 0usize;
    for &k in &checkpoints {
        while done < k {
            let slot = (done * 37 + 11) % n;
            let col = replacement_col(n, slot);
            lu.update(slot, &col).expect("update");
            done += 1;
        }
        let ft = avg_ftran_us(&mut lu, n, 50);
        // eta storage = updates × n (dense τ each).
        println!("{:>8}  {:>14.2}  {:>12}", done, ft, done * n);
    }
}
