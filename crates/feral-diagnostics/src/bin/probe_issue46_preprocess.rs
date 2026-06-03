//! Issue #46 — ground-truth probe for the ordering-preprocess predicate.
//!
//! Verifies, on the real CHO KKT, what `pick_ordering_preprocess`
//! actually returns and the column statistics it keys on:
//!   - stored degree distribution (feral stores the LOWER triangle only),
//!   - fraction of columns with stored degree <= 2 (the existing
//!     `low_degree` predicate),
//!   - fraction of columns whose diagonal is absent or stored as 0.0
//!     (the proposed zero-(2,2)-block predicate).
//!
//! Run on both the stripped `.mtx` and the diagonal-completed form.
//! This decides whether the #46 fix is an activation-predicate change
//! (Phase 1) or lies deeper (supernode split / numeric gate, Phase 2).
//!
//! Usage: cargo run --release --bin probe_issue46_preprocess [-- <kkt.mtx>]

use std::path::Path;

use feral::scaling::mc64_matching;
use feral::symbolic::{build_supermap, pick_ordering_preprocess};
use feral::{read_mtx, CscMatrix};

const DEFAULT_MTX: &str =
    "/Users/jkitchin/projects/pounce/benchmarks/cho/feral_repro/cho_iter0_kkt.mtx";

/// Insert an explicit `0.0` diagonal for every column lacking one.
fn complete_diagonal(csc: &CscMatrix) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..csc.n {
        let mut has_diag = false;
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            rows.push(csc.row_idx[k]);
            cols.push(j);
            vals.push(csc.values[k]);
            if csc.row_idx[k] == j {
                has_diag = true;
            }
        }
        if !has_diag {
            rows.push(j);
            cols.push(j);
            vals.push(0.0);
        }
    }
    CscMatrix::from_triplets(csc.n, &rows, &cols, &vals)
        .expect("diagonal-completed triplets are valid lower-triangle")
}

fn report(label: &str, m: &CscMatrix) {
    let n = m.n;
    let mut deg_le2 = 0usize;
    let mut diag_absent = 0usize;
    let mut diag_zero = 0usize;
    let mut deg_hist = [0usize; 5]; // 0,1,2,3,>=4
    for j in 0..n {
        let start = m.col_ptr[j];
        let end = m.col_ptr[j + 1];
        let deg = end - start;
        if deg <= 2 {
            deg_le2 += 1;
        }
        deg_hist[deg.min(4)] += 1;
        if start == end || m.row_idx[start] != j {
            diag_absent += 1;
        } else if m.values[start] == 0.0 {
            diag_zero += 1;
        }
    }
    let zero_or_absent = diag_absent + diag_zero;
    println!("--- {label} ---");
    println!("  n={n}  nnz(lower)={}", m.nnz());
    println!(
        "  stored-degree hist: deg0={} deg1={} deg2={} deg3={} deg>=4={}",
        deg_hist[0], deg_hist[1], deg_hist[2], deg_hist[3], deg_hist[4]
    );
    println!(
        "  low_degree (<=2):       {deg_le2:>7}  frac={:.4}  (threshold 0.30)",
        deg_le2 as f64 / n as f64
    );
    println!("  diag absent:            {diag_absent:>7}",);
    println!("  diag stored == 0.0:     {diag_zero:>7}",);
    println!(
        "  zero-or-absent diag:    {zero_or_absent:>7}  frac={:.4}  (proposed threshold 0.10)",
        zero_or_absent as f64 / n as f64
    );
    println!(
        "  pick_ordering_preprocess => {:?}",
        pick_ordering_preprocess(m)
    );

    // The decisive measurement: does the MC64 matching actually yield
    // 2-cycle pairs for `LdltCompress` to compress? If `ncmp == n` the
    // pipeline (symbolic/mod.rs:589) falls through to the UNCOMPRESSED
    // path — no pairing, cascade unavoidable.
    match mc64_matching(m) {
        Ok((perm, n_matched)) => {
            let map = build_supermap(&perm);
            let ncmp = map.ncmp();
            println!(
                "  mc64_matching: n_matched={n_matched}  build_supermap: pairs={} singletons={} ncmp={ncmp}",
                map.pairs.len(),
                map.singletons.len(),
            );
            if ncmp == n {
                println!(
                    "  >>> ncmp == n: LdltCompress FALLS THROUGH to uncompressed — NO pairing"
                );
            } else {
                println!(
                    "  >>> compression active: {} pairs co-located ({:.1}% of n)",
                    map.pairs.len(),
                    100.0 * 2.0 * map.pairs.len() as f64 / n as f64
                );
            }
        }
        Err(e) => println!("  mc64_matching ERROR: {e:?}"),
    }
    println!();
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MTX.to_string());
    let csc = read_mtx(Path::new(&path))
        .expect("read_mtx")
        .to_csc()
        .expect("to_csc");
    report("stripped (.mtx as-is)", &csc);
    let completed = complete_diagonal(&csc);
    report("diagonal-completed (POUNCE live-KKT form)", &completed);
}
