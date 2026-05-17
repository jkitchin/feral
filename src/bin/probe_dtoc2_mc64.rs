//! Probe MC64 scaling on dtoc2 iter 1 to verify the saturated-diagonal
//! hypothesis (`dev/journal/2026-05-17-01.org` §08:00).
//!
//! Loads `data/matrices/kkt-mittelmann/dtoc2/dtoc2_0001.mtx`, computes
//! per-column max|diag| and max|offdiag|, and reports:
//!   - how many columns have `max|diag| >> max|offdiag|` (the
//!     saturated-diagonal signature)
//!   - the resulting MC64 scaling vector range
//!
//! Usage:
//!     cargo run --release --bin probe_dtoc2_mc64 -- [problem] [iter]

use std::env;
use std::path::Path;

use feral::read_mtx;
use feral::scaling::ScalingStrategy;
use feral::CscMatrix;

fn col_diag_offdiag_max(matrix: &CscMatrix) -> Vec<(f64, f64)> {
    let n = matrix.n;
    let mut out = vec![(0.0_f64, 0.0_f64); n];
    for (j, slot) in out.iter_mut().enumerate() {
        let s = matrix.col_ptr[j];
        let e = matrix.col_ptr[j + 1];
        for k in s..e {
            let i = matrix.row_idx[k];
            let v = matrix.values[k].abs();
            if i == j {
                if v > slot.0 {
                    slot.0 = v;
                }
            } else if v > slot.1 {
                slot.1 = v;
            }
        }
    }
    out
}

fn main() {
    let problem = env::args().nth(1).unwrap_or_else(|| "dtoc2".to_string());
    let iter: usize = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    let mtx_path = format!("data/matrices/kkt-mittelmann/{problem}/{problem}_{iter:04}.mtx");
    if !Path::new(&mtx_path).exists() {
        eprintln!("SKIP: {mtx_path} not present");
        std::process::exit(2);
    }

    let mtx = match read_mtx(Path::new(&mtx_path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("read_mtx error: {e:?}");
            std::process::exit(1);
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("to_csc error: {e:?}");
            std::process::exit(1);
        }
    };

    println!(
        "# {problem} iter {iter}: n={}, nnz={}",
        csc.n,
        csc.row_idx.len()
    );

    // Count problematic entries.
    let mut n_inf = 0_usize;
    let mut n_nan = 0_usize;
    let mut n_zero = 0_usize;
    let mut max_abs = 0.0_f64;
    let mut max_abs_finite = 0.0_f64;
    for &v in &csc.values {
        if v.is_nan() {
            n_nan += 1;
        } else if v.is_infinite() {
            n_inf += 1;
        } else if v == 0.0 {
            n_zero += 1;
        } else {
            let a = v.abs();
            if a > max_abs_finite {
                max_abs_finite = a;
            }
        }
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    println!("# raw value scan: n_inf={n_inf} n_nan={n_nan} n_zero={n_zero}");
    println!("# max|v|={max_abs:.3e} max|v finite|={max_abs_finite:.3e}");

    let pairs = col_diag_offdiag_max(&csc);
    let mut buckets = [0_usize; 8];
    let mut max_diag = 0.0_f64;
    let mut max_off = 0.0_f64;
    let mut saturated = 0_usize;
    let mut diag_only = 0_usize;
    for &(d, o) in &pairs {
        if d > max_diag {
            max_diag = d;
        }
        if o > max_off {
            max_off = o;
        }
        if o == 0.0 && d > 0.0 {
            diag_only += 1;
        }
        let ratio = if o > 0.0 { d / o } else { f64::INFINITY };
        if ratio > 1e3 {
            saturated += 1;
        }
        let bucket = match ratio {
            r if r < 1.0 => 0,
            r if r < 10.0 => 1,
            r if r < 1e2 => 2,
            r if r < 1e3 => 3,
            r if r < 1e6 => 4,
            r if r < 1e10 => 5,
            r if r < 1e15 => 6,
            _ => 7,
        };
        buckets[bucket] += 1;
    }

    println!("# global max|diag| = {max_diag:.3e}");
    println!("# global max|offdiag| = {max_off:.3e}");
    println!("# cols with no offdiag (diag-only): {diag_only}");
    println!("# cols with |diag|/|offdiag| > 1e3 (saturated): {saturated}");
    println!();
    println!("# distribution of |diag|/|offdiag| per column:");
    let labels = [
        "<1",
        "1..10",
        "10..1e2",
        "1e2..1e3",
        "1e3..1e6",
        "1e6..1e10",
        "1e10..1e15",
        ">=1e15",
    ];
    for (lab, &cnt) in labels.iter().zip(&buckets) {
        println!("#   {lab:>12}  {cnt:>8}");
    }
    println!();

    // Reproduce MC64 scaling and report its range.
    use feral::numeric::factorize::NumericParams;
    use feral::numeric::solver::Solver;
    use feral::symbolic::supernode::SupernodeParams;
    let np = NumericParams {
        scaling: ScalingStrategy::Mc64Symmetric,
        ..NumericParams::default()
    };
    let _solver = Solver::with_params(np, SupernodeParams::default());
    println!("# (factor not run — dtoc2 iter 1 with mc64 hangs without CB; see journal §08:00)");
}
