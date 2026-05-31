//! Premise check for the perf-review Tier-1 #1 recommendation: do real
//! worst-case matrices have a few large fronts that dominate the numeric
//! cost (and therefore serialize the tree-parallel driver near the root)?
//!
//! Per-front dense-factor cost scales ~ ncol * nrow^2. We report the
//! supernode count, the largest front, and the share of total estimated
//! factor flops carried by the top-1/top-4/top-16 fronts. A high top-1
//! share means tree parallelism starves at the root and intra-front
//! parallelism is the lever that matters.
//!
//! Run: `cargo run --release --bin probe_front_concentration -- <a.mtx> [b.mtx ...]`

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::read_mtx;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use std::path::Path;
use std::time::Instant;

fn front_flops(ncol: usize, nrow: usize) -> f64 {
    // Dense LDL^T of an nrow x nrow front eliminating ncol pivots:
    // dominant term ~ ncol * nrow^2 (panel + trailing Schur update).
    ncol as f64 * (nrow as f64) * (nrow as f64)
}

fn analyze(path: &str) {
    let mtx = match read_mtx(Path::new(path)) {
        Ok(m) => m,
        Err(e) => {
            println!("{path}: read_mtx failed: {e:?}");
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            println!("{path}: to_csc failed: {e:?}");
            return;
        }
    };
    // Best-of-3 symbolic time.
    let mut t_sym = f64::INFINITY;
    let mut sym = match symbolic_factorize(&csc, &SupernodeParams::default()) {
        Ok(s) => s,
        Err(e) => {
            println!("{path}: symbolic failed: {e:?}");
            return;
        }
    };
    for _ in 0..3 {
        let t = Instant::now();
        if let Ok(s) = symbolic_factorize(&csc, &SupernodeParams::default()) {
            let dt = t.elapsed().as_secs_f64();
            if dt < t_sym {
                t_sym = dt;
            }
            sym = s;
        }
    }
    // Best-of-3 numeric time (sequential driver).
    let params = NumericParams::default();
    let mut t_num = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        if factorize_multifrontal(&csc, &sym, &params).is_ok() {
            let dt = t.elapsed().as_secs_f64();
            if dt < t_num {
                t_num = dt;
            }
        }
    }

    let mut flops: Vec<(f64, usize, usize)> = sym
        .supernodes
        .iter()
        .map(|s| (front_flops(s.ncol, s.nrow), s.ncol, s.nrow))
        .collect();
    let total: f64 = flops.iter().map(|x| x.0).sum();
    flops.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let share = |k: usize| -> f64 {
        if total <= 0.0 {
            return 0.0;
        }
        flops.iter().take(k).map(|x| x.0).sum::<f64>() / total
    };

    let name = Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path);
    let (top_flop, top_ncol, top_nrow) = flops.first().copied().unwrap_or((0.0, 0, 0));
    println!(
        "{name}: n={} nnz={} | {} supernodes | total est flops {:.3e}",
        csc.n,
        csc.row_idx.len(),
        sym.supernodes.len(),
        total
    );
    println!(
        "   largest front: ncol={top_ncol} nrow={top_nrow}  ({:.3e} flops, {:.1}% of total)",
        top_flop,
        100.0 * share(1)
    );
    println!(
        "   flop share: top-1 {:5.1}%   top-4 {:5.1}%   top-16 {:5.1}%",
        100.0 * share(1),
        100.0 * share(4),
        100.0 * share(16),
    );
    let frac_sym = 100.0 * t_sym / (t_sym + t_num).max(1e-12);
    println!(
        "   time: symbolic {:8.3} ms   numeric {:8.3} ms   -> symbolic = {:.0}% of (sym+num)\n",
        t_sym * 1e3,
        t_num * 1e3,
        frac_sym,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        println!("usage: probe_front_concentration <a.mtx> [b.mtx ...]");
        return;
    }
    for path in &args {
        analyze(path);
    }
}
