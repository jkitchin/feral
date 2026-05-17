//! Measure `max_nrow / n` and `max_ncol / n` ratios across the symbolic
//! factorisation for the Mittelmann KKT corpus. Used to calibrate the
//! α threshold for `Solver::with_auto_cascade_break(α)` (Step 3 of the
//! refocused cascade investigation; see
//! `dev/research/warm-state-cascade-amplification-2026-05-17.md`).
//!
//! For each `<problem>_<iter>.mtx` under the kkt-mittelmann corpus, run
//! symbolic factorization and print: n, n_supernodes, max_nrow,
//! max_ncol, max_nrow/n, max_ncol/n.
//!
//! Usage:
//!     cargo run --release --bin probe_supernode_widths -- <problem> [max_iter]

use std::env;
use std::path::Path;

use feral::read_mtx;
use feral::symbolic::supernode::SupernodeParams;
use feral::symbolic::symbolic_factorize;

fn main() {
    let problem = env::args()
        .nth(1)
        .expect("usage: probe_supernode_widths <problem> [max_iter]");
    let max_iter: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let dir = format!("data/matrices/kkt-mittelmann/{problem}");
    if !Path::new(&dir).exists() {
        eprintln!("SKIP: {dir} not present");
        std::process::exit(2);
    }

    println!(
        "{:>4}  {:>8}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}",
        "iter", "n", "snodes", "max_nrow", "max_ncol", "nr/n", "nc/n"
    );

    for i in 0..max_iter {
        let mtx_path = format!("{dir}/{problem}_{i:04}.mtx");
        if !Path::new(&mtx_path).exists() {
            break;
        }
        let Ok(mtx) = read_mtx(Path::new(&mtx_path)) else {
            continue;
        };
        let Ok(csc) = mtx.to_csc() else {
            continue;
        };
        let Ok(sym) = symbolic_factorize(&csc, &SupernodeParams::default()) else {
            continue;
        };
        let n = sym.n;
        let snodes = sym.supernodes.len();
        let max_nrow = sym.supernodes.iter().map(|s| s.nrow).max().unwrap_or(0);
        let max_ncol = sym.supernodes.iter().map(|s| s.ncol).max().unwrap_or(0);
        let nr_n = max_nrow as f64 / n as f64;
        let nc_n = max_ncol as f64 / n as f64;
        println!(
            "{i:>4}  {n:>8}  {snodes:>6}  {max_nrow:>8}  {max_ncol:>8}  {nr_n:>8.3}  {nc_n:>8.3}"
        );
    }
}
