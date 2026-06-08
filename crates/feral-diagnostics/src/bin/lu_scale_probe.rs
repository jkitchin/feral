//! Scaling probe for the unsymmetric sparse LU (issue #81).
//!
//! Factors increasingly large genuinely-sparse bases (a diagonally-dominant
//! tridiagonal matrix: O(n) nonzeros, O(n) ideal factor work) and reports
//! factor / analyze / ftran time vs n, plus the empirical log-log exponent
//! between consecutive sizes. A scalable factor should show an exponent well
//! below 2.
//!
//! Run: `cargo run -p feral-diagnostics --release --bin lu_scale_probe`

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{LuParams, SparseLu, SparseLuSymbolic};
use std::time::Instant;

/// Diagonally-dominant tridiagonal basis with `n` columns (≈3n nonzeros, no
/// fill — isolates the factor traversal cost).
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

/// 5-point 2D Laplacian on a `g×g` grid (`n = g²`, ≈5n nonzeros). Bandwidth
/// `√n` induces genuine fill — a realistic sub-quadratic sparse stress case.
fn grid2d(g: usize) -> SparseColMatrix {
    let n = g * g;
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let idx = |r: usize, c: usize| r * g + c;
    for r in 0..g {
        for c in 0..g {
            let j = idx(r, c);
            cols[j].push((j, 4.0));
            if r > 0 {
                cols[j].push((idx(r - 1, c), -1.0));
            }
            if r + 1 < g {
                cols[j].push((idx(r + 1, c), -1.0));
            }
            if c > 0 {
                cols[j].push((idx(r, c - 1), -1.0));
            }
            if c + 1 < g {
                cols[j].push((idx(r, c + 1), -1.0));
            }
        }
    }
    for col in cols.iter_mut() {
        col.sort_by_key(|&(i, _)| i);
    }
    SparseColMatrix::from_sparse_columns(n, &cols).expect("grid2d")
}

fn measure(name: &str, mats: Vec<(usize, SparseColMatrix)>) {
    println!("\n=== {name} ===");
    println!(
        "{:>8}  {:>8}  {:>12}  {:>12}  {:>12}  {:>10}  {:>10}",
        "n", "nnz", "analyze(µs)", "factor(µs)", "ftran(µs)", "factor_exp", "Lnnz"
    );
    let mut prev: Option<(f64, f64)> = None; // (n, factor_us)
    for (n, a) in mats {
        let nnz = a.nnz();
        let t0 = Instant::now();
        let symbolic = SparseLuSymbolic::analyze(&a).expect("analyze");
        let analyze_us = t0.elapsed().as_secs_f64() * 1e6;

        let t1 = Instant::now();
        let mut lu = SparseLu::factor(&a, &symbolic, LuParams::default()).expect("factor");
        let factor_us = t1.elapsed().as_secs_f64() * 1e6;

        let mut rhs: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
        let t2 = Instant::now();
        lu.ftran(&mut rhs).expect("ftran");
        let ftran_us = t2.elapsed().as_secs_f64() * 1e6;

        let exp = match prev {
            Some((pn, pf)) if pf > 0.0 => (factor_us / pf).ln() / (n as f64 / pn).ln(),
            _ => f64::NAN,
        };
        prev = Some((n as f64, factor_us));
        println!(
            "{:>8}  {:>8}  {:>12.0}  {:>12.0}  {:>12.0}  {:>10.2}  {:>10}",
            n,
            nnz,
            analyze_us,
            factor_us,
            ftran_us,
            exp,
            lu.factor_nnz()
        );
    }
}

fn main() {
    let tri: Vec<(usize, SparseColMatrix)> = [1000usize, 2000, 4000, 8000, 16000, 32000, 64000]
        .iter()
        .map(|&n| (n, tridiagonal(n)))
        .collect();
    measure("tridiagonal (no fill)", tri);

    let grid: Vec<(usize, SparseColMatrix)> = [16usize, 24, 32, 48, 64, 90, 128]
        .iter()
        .map(|&g| (g * g, grid2d(g)))
        .collect();
    measure("2D 5-point grid (fill)", grid);
}
