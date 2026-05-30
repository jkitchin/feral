//! Multi-RHS solve benchmark for feral (issue #57).
//!
//! Run with `cargo run --bin bench_multirhs --release`.
//!
//! For each test matrix the harness factors once, then times two ways of
//! solving the same `nrhs` right-hand sides:
//!   (a) `nrhs` independent single-RHS `solve_sparse` calls (looped), and
//!   (b) one `solve_sparse_many` call.
//! It prints per-RHS microseconds for each and the ratio `many / single`.
//! Lower is better; below 1.0 means the multi-RHS path is faster per RHS.
//!
//! It also reports `max |many - single|` per case so a layout regression
//! is caught immediately. Dependency-free (no BLAS, no rng crate).

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::{solve_sparse, solve_sparse_many};
use feral::sparse::csc::CscMatrix;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use std::time::Instant;

/// Build the 2D 5-point Laplacian on a `g x g` grid (SPD, order `g*g`),
/// stored with the full symmetric pattern.
fn build_laplacian_2d(g: usize) -> CscMatrix {
    let n = g * g;
    let idx = |r: usize, c: usize| r * g + c;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    // Store only the lower triangle (row >= col), as CscMatrix requires.
    let mut push_lower = |a: usize, b: usize, v: f64| {
        let (r, c) = if a >= b { (a, b) } else { (b, a) };
        rows.push(r);
        cols.push(c);
        vals.push(v);
    };
    for r in 0..g {
        for c in 0..g {
            let i = idx(r, c);
            push_lower(i, i, 4.0);
            if c + 1 < g {
                push_lower(i, idx(r, c + 1), -1.0);
            }
            if r + 1 < g {
                push_lower(i, idx(r + 1, c), -1.0);
            }
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("valid CSC")
}

/// Deterministic pseudo-random RHS, column-major `n x nrhs`.
fn make_rhs(n: usize, nrhs: usize) -> Vec<f64> {
    (0..n * nrhs)
        .map(|k| ((k.wrapping_mul(2_654_435_761) % 2000) as f64) / 1000.0 - 1.0)
        .collect()
}

fn main() {
    // Grid sizes giving n ~ 500, 1000, 2000.
    let grids = [22usize, 32, 45]; // n = 484, 1024, 2025
    let nrhs_list = [64usize, 256];

    println!(
        "{:>6} {:>6} {:>6} {:>14} {:>14} {:>8}   max|many-single|",
        "n", "nrhs", "reps", "single us/RHS", "many us/RHS", "ratio"
    );

    for &g in &grids {
        let n = g * g;
        let m = build_laplacian_2d(g);
        let sym = symbolic_factorize(&m, &SupernodeParams::default()).expect("symbolic");
        let (factors, _) =
            factorize_multifrontal(&m, &sym, &NumericParams::default()).expect("numeric");

        for &nrhs in &nrhs_list {
            let rhs = make_rhs(n, nrhs);

            let reps = if n <= 600 {
                20
            } else if n <= 1200 {
                8
            } else {
                4
            };

            // Warm up + correctness check vs the single-RHS oracle.
            let many_x = solve_sparse_many(&factors, &rhs, nrhs).expect("many warmup");
            let mut max_abs = 0.0f64;
            for c in 0..nrhs {
                let single = solve_sparse(&factors, &rhs[c * n..(c + 1) * n]).expect("single");
                for i in 0..n {
                    let d = (single[i] - many_x[c * n + i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
            }

            // (a) Looped single-RHS solves.
            let mut single_sink = 0.0f64;
            let t0 = Instant::now();
            for _ in 0..reps {
                for c in 0..nrhs {
                    let x = solve_sparse(&factors, &rhs[c * n..(c + 1) * n]).expect("single solve");
                    single_sink += x[0];
                }
            }
            let single_total = t0.elapsed();

            // (b) One multi-RHS solve.
            let mut many_sink = 0.0f64;
            let t1 = Instant::now();
            for _ in 0..reps {
                let x = solve_sparse_many(&factors, &rhs, nrhs).expect("many solve");
                many_sink += x[0];
            }
            let many_total = t1.elapsed();

            if single_sink == f64::INFINITY || many_sink == f64::INFINITY {
                println!("(unreachable sink guard)");
            }

            let total_rhs = (reps * nrhs) as f64;
            let single_us = single_total.as_secs_f64() * 1e6 / total_rhs;
            let many_us = many_total.as_secs_f64() * 1e6 / total_rhs;
            let ratio = many_us / single_us;

            println!(
                "{:>6} {:>6} {:>6} {:>14.3} {:>14.3} {:>8.3}   {:.3e}",
                n, nrhs, reps, single_us, many_us, ratio, max_abs
            );
        }
    }
}
