//! Tight-loop microbench of `solve_sparse` vs `solve_sparse_refined`
//! on the smallest representative matrices, to localize per-call
//! overhead (vec allocations dominate at n<200).

use feral::numeric::factorize::factorize_multifrontal;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    read_mtx, solve_sparse, solve_sparse_refined, BunchKaufmanParams, CscMatrix, ZeroPivotAction,
};
use std::path::Path;
use std::time::Instant;

fn run(family: &str, sample: &str) {
    let p = format!("data/matrices/kkt/{}/{}{}.mtx", family, family, sample);
    let path = Path::new(&p);
    let mtx = match read_mtx(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP {}: {}", p, e);
            return;
        }
    };
    let csc = mtx.to_csc().expect("csc");
    let n = csc.n;

    let snode_params = SupernodeParams::default();
    let factor_params = feral::numeric::factorize::NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    });
    let sym = symbolic_factorize(&csc, &snode_params).expect("sym");
    let (factors, _) = factorize_multifrontal(&csc, &sym, &factor_params).expect("fac");
    let rhs = vec![1.0_f64; n];

    // Bare solve repeated 10000 times
    let iters = 10_000usize;
    let mut bare_total_ns: u128 = 0;
    let mut sink: f64 = 0.0;
    for _ in 0..iters {
        let t = Instant::now();
        let x = solve_sparse(&factors, &rhs).expect("solve");
        bare_total_ns += t.elapsed().as_nanos();
        sink += x[0];
    }

    let mut refined_total_ns: u128 = 0;
    for _ in 0..iters {
        let t = Instant::now();
        let x = solve_sparse_refined(&csc, &factors, &rhs).expect("solve");
        refined_total_ns += t.elapsed().as_nanos();
        sink += x[0];
    }

    let bare_ns_per = bare_total_ns / iters as u128;
    let refined_ns_per = refined_total_ns / iters as u128;
    let n_snodes = factors.node_factors.len();
    let max_nrow = factors
        .node_factors
        .iter()
        .map(|n| n.frontal_factors.nrow)
        .max()
        .unwrap_or(0);

    println!(
        "{:<15} n={:>4} snodes={:>4} max_nrow={:>4}   bare={:>5}ns  refined={:>6}ns  (refined/bare={:.1}x)  sink={:.3e}",
        format!("{}{}", family, sample),
        n,
        n_snodes,
        max_nrow,
        bare_ns_per,
        refined_ns_per,
        refined_ns_per as f64 / bare_ns_per as f64,
        sink,
    );

    let _ = CscMatrix::from_triplets; // keep import non-warning
}

fn main() {
    println!("Per-call overhead microbench (10000 iters each, ns/call)");
    println!("{}", "-".repeat(115));
    let cases: &[(&str, &str)] = &[
        ("ALLINITC", "_0000"),
        ("HS118", "_0000"),
        ("MCONCON", "_0000"),
        ("AVION2", "_0000"),
        ("BATCH", "_0000"),
        ("HAHN1", "_0000"),
    ];
    for (fam, samp) in cases {
        run(fam, samp);
    }
}
