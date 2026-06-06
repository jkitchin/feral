//! Bucket probe for the scaling-aware `LdltCompress` gate
//! (dense-column follow-up, session 2026-06-06-04).
//!
//! For each corpus family (first `.mtx` per subdir of
//! `data/matrices/kkt[/...]`), computes the two independent routing
//! decisions:
//!   - `pick_ordering_preprocess`  → None | LdltCompress
//!   - `pick_scaling_strategy`     → InfNorm | Mc64Symmetric | ...
//!
//! The symbolic `LdltCompress` branch runs MC64 (the expensive
//! Hungarian matching) once per pattern. That cache is reused for
//! numeric scaling **only** when the resolved scaling is
//! `Mc64Symmetric`. So the *target bucket* for the safe win is:
//!
//!     pick_ordering_preprocess == LdltCompress  AND
//!     pick_scaling_strategy    != Mc64Symmetric
//!
//! For matrices in that bucket the symbolic MC64 is a net-new cost
//! charged entirely to compression (numeric scaling will not reuse
//! it). This probe tallies the buckets so we know the win is
//! non-vacuous before implementing the gate. It does NOT factorize —
//! both decisions are O(nnz) structural scans.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin probe_compress_scaling_bucket
//!   FERAL_KKT_ROOTS="kkt kkt-expansion" cargo run ... (extra roots)

use std::path::{Path, PathBuf};

use feral::scaling::{pick_scaling_strategy, ScalingStrategy};
use feral::symbolic::{pick_ordering_preprocess, OrderingPreprocess};
use feral::{read_mtx, CscMatrix};

fn first_mtx_in(dir: &Path) -> Option<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "mtx").unwrap_or(false))
        .collect();
    files.sort();
    files.into_iter().next()
}

fn family_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = match std::fs::read_dir(root) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
        Err(_) => Vec::new(),
    };
    dirs.sort();
    dirs
}

fn max_col_degree(m: &CscMatrix) -> usize {
    (0..m.n)
        .map(|j| m.col_ptr[j + 1] - m.col_ptr[j])
        .max()
        .unwrap_or(0)
}

fn scaling_label(s: &ScalingStrategy) -> &'static str {
    match s {
        ScalingStrategy::Identity => "Identity",
        ScalingStrategy::External(_) => "External",
        ScalingStrategy::InfNorm => "InfNorm",
        ScalingStrategy::Mc64Symmetric => "Mc64Symmetric",
        ScalingStrategy::Auto => "Auto",
    }
}

fn main() {
    let roots_env = std::env::var("FERAL_KKT_ROOTS").unwrap_or_else(|_| "kkt".into());
    let roots: Vec<PathBuf> = roots_env
        .split_whitespace()
        .map(|r| Path::new("data/matrices").join(r))
        .collect();

    // Bucket counters. The target bucket is (LdltCompress, !Mc64).
    let mut n_total = 0usize;
    let mut compress_total = 0usize;
    let mut compress_mc64 = 0usize; // LdltCompress + Mc64 scaling (MC64 shared — keep)
    let mut compress_non_mc64 = 0usize; // TARGET bucket
    let mut target_rows: Vec<(String, usize, usize, &'static str)> = Vec::new();

    for root in &roots {
        for fam_dir in family_dirs(root) {
            let Some(mtx) = first_mtx_in(&fam_dir) else {
                continue;
            };
            let m = match read_mtx(&mtx).and_then(|raw| raw.to_csc()) {
                Ok(m) => m,
                Err(_) => continue,
            };
            // skip non-finite / empty
            if m.n == 0 {
                continue;
            }
            n_total += 1;
            let pre = pick_ordering_preprocess(&m);
            if !matches!(pre, OrderingPreprocess::LdltCompress) {
                continue;
            }
            compress_total += 1;
            let scal = pick_scaling_strategy(&m);
            let name = fam_dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            if matches!(scal, ScalingStrategy::Mc64Symmetric) {
                compress_mc64 += 1;
            } else {
                compress_non_mc64 += 1;
                target_rows.push((name, m.n, max_col_degree(&m), scaling_label(&scal)));
            }
        }
    }

    println!("== compress/scaling bucket probe ==");
    println!("roots                 : {roots_env}");
    println!("families scanned      : {n_total}");
    println!("LdltCompress chosen   : {compress_total}");
    println!("  + Mc64 scaling      : {compress_mc64}  (MC64 shared — gate keeps compression)");
    println!("  + non-Mc64 scaling  : {compress_non_mc64}  (TARGET bucket — MC64 is net-new cost)");
    println!();
    if target_rows.is_empty() {
        println!("target bucket is EMPTY — the scaling-aware gate would help no corpus matrix.");
        return;
    }
    // Sort target bucket by max_col_degree desc (worst MC64 cost first).
    target_rows.sort_by_key(|b| std::cmp::Reverse(b.2));
    println!(
        "{:<22} {:>10} {:>14} {:>14}",
        "family", "n", "max_col_deg", "scaling"
    );
    for (name, n, mcd, scal) in &target_rows {
        println!("{name:<22} {n:>10} {mcd:>14} {scal:>14}");
    }
}
