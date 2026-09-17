//! Issue #203 — fill and flop probe for chain/collocation KKT patterns.
//!
//! For one `.mtx` and an optional caller-supplied permutation file
//! (one 0-based original index per line, new-to-old, i.e. the same
//! convention as [`OrderingMethod::External`]), report for each
//! ordering:
//!
//!   * `nnz(L)` — `factor_nnz_estimate`
//!   * `flops`  — `sum_j c_j^2` over the column counts of L, where
//!     `c_j` counts the *below-diagonal* entries. This is the standard
//!     dense-column-update model for LDL^T (`sum_j c_j^2` multiply-adds)
//!     and is the quantity that tracks factorization *time*, which
//!     `nnz(L)` alone does not.
//!   * `front_max` — the largest column count, i.e. the widest front.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin issue203_fill_probe \
//!       -- MATRIX.mtx [PERM.txt ...]
//!
//! Every extra argument is another permutation file, reported as
//! `ext:<file stem>`.

use feral::read_mtx;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use std::time::Instant;

fn stats(sym: &feral::symbolic::SymbolicFactorization) -> (u64, f64, u64) {
    // `col_counts[j]` includes the diagonal, so the below-diagonal
    // count is `c - 1`.
    let mut flops = 0.0f64;
    let mut wide = 0u64;
    for &c in &sym.col_counts {
        let below = c.saturating_sub(1) as f64;
        flops += below * below;
        wide = wide.max(c as u64);
    }
    (sym.factor_nnz_estimate as u64, flops, wide)
}

fn read_perm(path: &str) -> Result<Vec<usize>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut perm = Vec::new();
    for tok in text.split_ascii_whitespace() {
        perm.push(
            tok.parse::<usize>()
                .map_err(|e| format!("{path}: bad index {tok:?}: {e}"))?,
        );
    }
    Ok(perm)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(mtx_path) = args.first() else {
        eprintln!("usage: issue203_fill_probe MATRIX.mtx [PERM.txt ...]");
        std::process::exit(2);
    };

    let mtx = match read_mtx(std::path::Path::new(mtx_path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: cannot read {mtx_path}: {e}");
            std::process::exit(1);
        }
    };
    let matrix = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: csc conversion failed: {e}");
            std::process::exit(1);
        }
    };
    let params = SupernodeParams::default();

    let mut methods: Vec<(String, OrderingMethod)> = vec![
        ("amd".to_string(), OrderingMethod::Amd),
        ("amf".to_string(), OrderingMethod::Amf),
        ("metis".to_string(), OrderingMethod::MetisND),
        ("scotch".to_string(), OrderingMethod::ScotchND),
        ("auto".to_string(), OrderingMethod::Auto),
    ];
    for path in &args[1..] {
        let perm = match read_perm(path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
        let stem = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path)
            .to_string();
        methods.push((format!("ext:{stem}"), OrderingMethod::External(perm)));
    }

    println!(
        "matrix    n={} nnz_lower={}",
        matrix.n,
        matrix.row_idx.len()
    );
    println!(
        "{:<20} {:>14} {:>16} {:>10} {:>10}",
        "ordering", "nnz_L", "flops", "front_max", "sym_ms"
    );
    for (label, method) in methods {
        let t = Instant::now();
        match symbolic_factorize_with_method(&matrix, &params, method) {
            Ok(sym) => {
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let (nnz, flops, wide) = stats(&sym);
                println!(
                    "{label:<20} {nnz:>14} {flops:>16.4e} {wide:>10} {ms:>10.1}  {:?}/{:?}",
                    sym.resolved_method, sym.resolved_preprocess
                );
            }
            Err(e) => println!("{label:<20} FAILED: {e}"),
        }
    }
}
