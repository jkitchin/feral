//! Dense-Hessian multi-RHS probe (pounce#77 follow-up).
//!
//! Build:  cargo build --release --bin bench_dense_multirhs
//! Run:    target/release/bench_dense_multirhs
//!
//! Builds a dense SPD H = MᵀM + I, stores its full lower triangle as a
//! sparse matrix, factors it, and times looped single-RHS vs batched
//! `solve_sparse_many` over a range of nrhs. This mirrors the shape of
//! pounce's dense-Hessian KKT, where the L factor is dominated by one
//! large dense supernode.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::{solve_sparse, solve_sparse_many};
use feral::sparse::csc::CscMatrix;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use std::time::Instant;

fn build_dense_spd(n: usize) -> CscMatrix {
    // Deterministic LCG, kept inline to avoid extra deps.
    let mut s: u64 = 42;
    let mut nxt = || -> f64 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    let mut m = vec![0.0f64; n * n];
    let inv_sqrt = 1.0 / (n as f64).sqrt();
    for v in m.iter_mut() {
        *v = nxt() * inv_sqrt;
    }
    // H = MᵀM + I (full symmetric, dense).
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += m[k * n + i] * m[k * n + j];
            }
            h[i * n + j] = acc;
            h[j * n + i] = acc;
        }
        h[i * n + i] += 1.0;
    }
    let mut rows = Vec::with_capacity(n * (n + 1) / 2);
    let mut cols = Vec::with_capacity(n * (n + 1) / 2);
    let mut vals = Vec::with_capacity(n * (n + 1) / 2);
    for j in 0..n {
        for i in j..n {
            rows.push(i);
            cols.push(j);
            vals.push(h[i * n + j]);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("CSC")
}

fn make_rhs(n: usize, nrhs: usize) -> Vec<f64> {
    (0..n * nrhs)
        .map(|k| ((k.wrapping_mul(2_654_435_761) % 2000) as f64) / 1000.0 - 1.0)
        .collect()
}

fn main() {
    println!(
        "{:>6} {:>6} {:>6} {:>14} {:>14} {:>8}",
        "n", "nrhs", "reps", "single us/RHS", "many us/RHS", "ratio"
    );
    for &n in &[128usize, 256, 516, 1024] {
        let m = build_dense_spd(n);
        let sym = symbolic_factorize(&m, &SupernodeParams::default()).expect("symbolic");
        let (factors, _) =
            factorize_multifrontal(&m, &sym, &NumericParams::default()).expect("numeric");

        for &nrhs in &[1usize, 4, 16, 64, 256] {
            let rhs = make_rhs(n, nrhs);
            let reps = if n <= 200 {
                50
            } else if n <= 300 {
                20
            } else if n <= 600 {
                10
            } else {
                4
            };

            // Warm up.
            let _ = solve_sparse_many(&factors, &rhs, nrhs).expect("warm many");
            let _ = solve_sparse(&factors, &rhs[..n]).expect("warm one");

            let mut sink = 0.0;
            let t0 = Instant::now();
            for _ in 0..reps {
                for c in 0..nrhs {
                    let x = solve_sparse(&factors, &rhs[c * n..(c + 1) * n]).expect("single");
                    sink += x[0];
                }
            }
            let single_total = t0.elapsed();

            let mut many_sink = 0.0;
            let t1 = Instant::now();
            for _ in 0..reps {
                let x = solve_sparse_many(&factors, &rhs, nrhs).expect("many");
                many_sink += x[0];
            }
            let many_total = t1.elapsed();

            if sink == f64::INFINITY || many_sink == f64::INFINITY {
                unreachable!();
            }

            let total = (reps * nrhs) as f64;
            let s_us = single_total.as_secs_f64() * 1e6 / total;
            let m_us = many_total.as_secs_f64() * 1e6 / total;
            println!(
                "{:>6} {:>6} {:>6} {:>14.3} {:>14.3} {:>8.3}",
                n,
                nrhs,
                reps,
                s_us,
                m_us,
                m_us / s_us
            );
        }
    }
}
