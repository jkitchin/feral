//! Inspect why CHAINWOO_0000 produces a ~1.1 GB factor for an n=4000
//! nnz=7999 matrix where MUMPS reports nnz_L = 51,964.
//!
//! Tries every available ordering method and prints
//! (method, n_supernodes, nnz_L, factor_us, max_front).

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams};
use std::path::Path;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/matrices/kkt-expansion/CHAINWOO/CHAINWOO_0000.mtx".into());
    let mtx = read_mtx(Path::new(&path)).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    println!("matrix: {}, n={}, nnz={}", path, csc.n, csc.row_idx.len());

    let snode_params = SupernodeParams::default();
    let bk = BunchKaufmanParams::default();
    let nparams = NumericParams::with_bk(bk);

    let methods = [
        OrderingMethod::Auto,
        OrderingMethod::Amd,
        OrderingMethod::Amf,
        OrderingMethod::MetisND,
        OrderingMethod::ScotchND,
        OrderingMethod::KahipND,
    ];

    println!(
        "{:>10}  {:>10}  {:>12}  {:>12}  {:>10}  {:>10}  {:>10}",
        "method", "n_snodes", "sym_nnz_est", "num_nnz_l", "max_front", "sym_us", "num_us"
    );
    for &m in &methods {
        let t_sym = Instant::now();
        let sym = match symbolic_factorize_with_method(&csc, &snode_params, m) {
            Ok(s) => s,
            Err(e) => {
                println!("{:>10}  symbolic failed: {}", format!("{:?}", m), e);
                continue;
            }
        };
        let sym_us = t_sym.elapsed().as_micros();

        let n_snodes = sym.supernodes.len();
        let max_front = sym.supernodes.iter().map(|s| s.nrow).max().unwrap_or(0);
        let sym_nnz_est: usize = sym.supernodes.iter().map(|s| s.nrow * s.ncol).sum();

        let t_num = Instant::now();
        let result = factorize_multifrontal(&csc, &sym, &nparams);
        let num_us = t_num.elapsed().as_micros();

        let num_nnz_l = match result {
            Ok((f, _)) => f.factor_nnz(),
            Err(e) => {
                println!("{:>10}  numeric failed: {}", format!("{:?}", m), e);
                continue;
            }
        };

        println!(
            "{:>10}  {:>10}  {:>12}  {:>12}  {:>10}  {:>10}  {:>10}",
            format!("{:?}", m),
            n_snodes,
            sym_nnz_est,
            num_nnz_l,
            max_front,
            sym_us,
            num_us
        );
    }
}
