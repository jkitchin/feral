//! In-tree AMD-vs-AMF nnz_L audit on the local KKT corpus.
//!
//! Walks `data/matrices/` for `*.mtx` files, samples them with
//! `FERAL_AUDIT_PER_FAMILY` (default 1) matrices per family
//! directory and `FERAL_AUDIT_MAX_N` (default 5000) on the
//! matrix dimension, then runs both `OrderingMethod::Amd` and
//! `OrderingMethod::Amf` through the standard symbolic
//! pipeline and reports per-family geomean nnz_L ratio
//! (amf / amd) plus the top regressions and improvements.
//!
//! This is preview data for the eventual MUMPS-style
//! `pick_default_method` flip ("AMF for n ≤ 10000, MetisND
//! otherwise") in `dev/plans/amf-clean-room.md` Phase D. It
//! does *not* require the MUMPS HAMF4 sidecars on disk and
//! produces no persistent state — pure stdout.
//!
//! The MUMPS HAMF4 oracle in `tests/amf_corpus_oracle.rs`
//! remains the load-bearing correctness gate (`feral nnz_L ≤
//! 1.10 × MUMPS HAMF4 nnz_L`). This bin only answers the
//! question "in the matrices feral has on disk today, how does
//! AMF compare to AMD on fill?" — which is the cluster-level
//! data the default flip needs to be defensible.
//!
//! Run:
//!     cargo run --release --bin diag_amf_vs_amd
//!     FERAL_AUDIT_PER_FAMILY=3 FERAL_AUDIT_MAX_N=2000 \
//!         cargo run --release --bin diag_amf_vs_amd
//!     FERAL_AUDIT_ROOT=data/matrices/kkt-expansion \
//!         cargo run --release --bin diag_amf_vs_amd

use feral::read_mtx;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Default, Clone)]
struct FamilyStats {
    count: usize,
    log_ratio_sum: f64, // sum of ln(amf / amd)
    amd_total: u64,
    amf_total: u64,
    n_amf_better: usize,
    n_amd_better: usize,
    n_tied: usize,
}

#[derive(Clone)]
struct PerMatrix {
    family: String,
    name: String,
    n: usize,
    amd_nnz_l: usize,
    amf_nnz_l: usize,
    ratio: f64, // amf / amd
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn main() {
    let root = std::env::var("FERAL_AUDIT_ROOT").unwrap_or_else(|_| "data/matrices".to_string());
    let per_family = env_usize("FERAL_AUDIT_PER_FAMILY", 1);
    let max_n = env_usize("FERAL_AUDIT_MAX_N", 5000);

    let root_path = PathBuf::from(&root);
    if !root_path.is_dir() {
        eprintln!("error: {} is not a directory", root);
        std::process::exit(1);
    }
    println!(
        "AMD-vs-AMF audit  root={}  per_family={}  max_n={}",
        root, per_family, max_n
    );

    let sampled = sample_matrices(&root_path, per_family);
    println!(
        "sampled {} matrices across {} families",
        sampled.iter().map(|v| v.len()).sum::<usize>(),
        sampled.len()
    );

    let snode_params = SupernodeParams::default();
    let mut per_matrix: Vec<PerMatrix> = Vec::new();
    let mut families: BTreeMap<String, FamilyStats> = BTreeMap::new();
    let mut n_skipped_too_large = 0usize;
    let mut n_failed = 0usize;

    let t_start = Instant::now();
    for fam_matrices in &sampled {
        for path in fam_matrices {
            let family = path
                .parent()
                .and_then(|p| p.file_name())
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "?".to_string());
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            let mtx = match read_mtx(path) {
                Ok(m) => m,
                Err(_) => {
                    n_failed += 1;
                    continue;
                }
            };
            if mtx.n > max_n {
                n_skipped_too_large += 1;
                continue;
            }
            let csc = match mtx.to_csc() {
                Ok(c) => c,
                Err(_) => {
                    n_failed += 1;
                    continue;
                }
            };

            let sym_amd =
                match symbolic_factorize_with_method(&csc, &snode_params, OrderingMethod::Amd) {
                    Ok(s) => s,
                    Err(_) => {
                        n_failed += 1;
                        continue;
                    }
                };
            let sym_amf =
                match symbolic_factorize_with_method(&csc, &snode_params, OrderingMethod::Amf) {
                    Ok(s) => s,
                    Err(_) => {
                        n_failed += 1;
                        continue;
                    }
                };

            // Use sum of column counts (true nnz_L of the symbolic
            // factorization, before slack) for the ratio so the
            // 1.2 factor_slack cancels.
            let amd_nnz_l: usize = sym_amd.col_counts.iter().sum();
            let amf_nnz_l: usize = sym_amf.col_counts.iter().sum();
            if amd_nnz_l == 0 {
                continue;
            }
            let ratio = amf_nnz_l as f64 / amd_nnz_l as f64;

            let stats = families.entry(family.clone()).or_default();
            stats.count += 1;
            stats.log_ratio_sum += ratio.ln();
            stats.amd_total += amd_nnz_l as u64;
            stats.amf_total += amf_nnz_l as u64;
            if amf_nnz_l < amd_nnz_l {
                stats.n_amf_better += 1;
            } else if amf_nnz_l > amd_nnz_l {
                stats.n_amd_better += 1;
            } else {
                stats.n_tied += 1;
            }

            per_matrix.push(PerMatrix {
                family,
                name,
                n: csc.n,
                amd_nnz_l,
                amf_nnz_l,
                ratio,
            });
        }
    }
    let elapsed = t_start.elapsed();
    println!(
        "audit completed in {:.2}s   ({} matrices succeeded, {} skipped n>{}, {} failed)",
        elapsed.as_secs_f64(),
        per_matrix.len(),
        n_skipped_too_large,
        max_n,
        n_failed
    );

    if per_matrix.is_empty() {
        println!("no matrices completed; nothing to report.");
        return;
    }

    println!();
    println!("=== Per-family summary (geomean ratio amf/amd; <1.0 = AMF wins) ===");
    println!(
        "{:<24}  {:>5}  {:>10}  {:>10}  {:>10}  {:>5}  {:>5}  {:>5}",
        "family", "count", "geo_ratio", "amd_total", "amf_total", "amf<", "tied", "amd<"
    );
    for (fam, stats) in &families {
        let geo = (stats.log_ratio_sum / stats.count as f64).exp();
        println!(
            "{:<24}  {:>5}  {:>10.3}  {:>10}  {:>10}  {:>5}  {:>5}  {:>5}",
            fam,
            stats.count,
            geo,
            stats.amd_total,
            stats.amf_total,
            stats.n_amf_better,
            stats.n_tied,
            stats.n_amd_better
        );
    }

    println!();
    println!("=== Corpus rollup ===");
    let total_count: usize = families.values().map(|s| s.count).sum();
    let total_log_sum: f64 = families.values().map(|s| s.log_ratio_sum).sum();
    let total_amd: u64 = families.values().map(|s| s.amd_total).sum();
    let total_amf: u64 = families.values().map(|s| s.amf_total).sum();
    let total_amf_better: usize = families.values().map(|s| s.n_amf_better).sum();
    let total_amd_better: usize = families.values().map(|s| s.n_amd_better).sum();
    let total_tied: usize = families.values().map(|s| s.n_tied).sum();
    let geo = (total_log_sum / total_count as f64).exp();
    println!(
        "{} matrices  geomean ratio = {:.3}  amd_total nnz_L = {}  amf_total nnz_L = {}",
        total_count, geo, total_amd, total_amf
    );
    println!(
        "AMF strictly better on {} matrices, tied on {}, AMD strictly better on {}",
        total_amf_better, total_tied, total_amd_better
    );

    println!();
    println!("=== Top 15 AMF improvements (lowest ratio = AMF cuts fill most) ===");
    let mut sorted = per_matrix.clone();
    sorted.sort_by(|a, b| {
        a.ratio
            .partial_cmp(&b.ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!(
        "{:<24}  {:>6}  {:>10}  {:>10}  {:>8}",
        "matrix", "n", "amd_nnz_l", "amf_nnz_l", "ratio"
    );
    for p in sorted.iter().take(15) {
        println!(
            "{:<24}  {:>6}  {:>10}  {:>10}  {:>8.3}",
            format!("{}/{}", p.family, p.name),
            p.n,
            p.amd_nnz_l,
            p.amf_nnz_l,
            p.ratio
        );
    }

    println!();
    println!("=== Top 15 AMF regressions (highest ratio = AMF inflates fill most) ===");
    sorted.reverse();
    for p in sorted.iter().take(15) {
        println!(
            "{:<24}  {:>6}  {:>10}  {:>10}  {:>8.3}",
            format!("{}/{}", p.family, p.name),
            p.n,
            p.amd_nnz_l,
            p.amf_nnz_l,
            p.ratio
        );
    }
}

/// Walk `root` for `<root>/<cluster>/<family>/<stem>.mtx` and
/// `<root>/<family>/<stem>.mtx`. Returns one `Vec<PathBuf>` per
/// family directory, each capped at `per_family` matrices
/// (sorted by file name for determinism).
fn sample_matrices(root: &Path, per_family: usize) -> Vec<Vec<PathBuf>> {
    let mut by_family: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();
    walk(root, &mut by_family);
    let mut out: Vec<Vec<PathBuf>> = Vec::with_capacity(by_family.len());
    for (_, mut paths) in by_family {
        paths.sort();
        paths.truncate(per_family);
        if !paths.is_empty() {
            out.push(paths);
        }
    }
    out
}

fn walk(dir: &Path, by_family: &mut BTreeMap<PathBuf, Vec<PathBuf>>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    let mut local_mtx: Vec<PathBuf> = Vec::new();
    let mut subdirs: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path);
        } else if path.extension().is_some_and(|e| e == "mtx") {
            local_mtx.push(path);
        }
    }
    if !local_mtx.is_empty() {
        by_family
            .entry(dir.to_path_buf())
            .or_default()
            .extend(local_mtx);
    }
    for sub in subdirs {
        walk(&sub, by_family);
    }
}
