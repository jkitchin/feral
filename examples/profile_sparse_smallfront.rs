//! Profile sparse small-frontal overhead to locate the Phase 2.5 bottleneck.
//!
//! Phase 2.8.1 partition verdict (session 2026-04-14-02 checkpoint): sparse
//! small-frontal p90 = 2.81 vs the 2.0 target. The plan's leading hypothesis
//! is that the O(n^2) column-counts implementation dominates the fixed
//! per-matrix overhead; Phase 2.5.1 (Liu's row-subtree algorithm) is the
//! called-out fix.
//!
//! This binary replicates the `symbolic_factorize` pipeline inline with
//! per-sub-phase timing, plus the numeric factorization, for every KKT
//! matrix with n <= 500 and an existing MUMPS oracle sidecar. It ranks
//! matrices by `total_feral_us / mumps_factor_us` and prints the top 30
//! with a per-phase breakdown. The goal is to see which sub-phase carries
//! the cost on the matrices driving the p90 tail — before committing to
//! a multi-hour Phase 2.5.1 implementation.
//!
//! Usage: cargo run --release --example profile_sparse_smallfront

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::ordering::amd::{amd_order, permute_pattern};
use feral::ordering::elimination_tree::EliminationTree;
use feral::ordering::postorder::postorder;
use feral::read_mtx;
use feral::sparse::csc::CscMatrix;
use feral::symbolic::column_counts::{column_counts, total_factor_nnz};
use feral::symbolic::supernode::{find_supernodes, SupernodeParams};
use feral::symbolic::SymbolicFactorization;
use feral::{BunchKaufmanParams, ZeroPivotAction};

#[derive(Debug, Clone)]
struct PhaseTimes {
    mc64_us: u128,
    amd_us: u128,
    etree_us: u128,
    colcnt_us: u128,
    snode_us: u128,
    sym_total_us: u128,
    numeric_us: u128,
    total_us: u128,
}

#[derive(Debug, Clone)]
struct Row {
    name: String,
    n: usize,
    max_front: usize,
    times: PhaseTimes,
    mumps_us: u64,
    ratio: f64,
}

/// Inline the symbolic_factorize pipeline with per-phase timing. Returns
/// (SymbolicFactorization, PhaseTimes). This duplicates the body of
/// `feral::symbolic::symbolic_factorize` so we can insert Instant::now()
/// between phases without modifying the library API.
fn timed_symbolic(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
) -> Option<(SymbolicFactorization, PhaseTimes)> {
    let n = matrix.n;

    let t_total = Instant::now();

    // Phase 1: MC64 scaling now lives in the numeric phase (β refactor);
    // this column is preserved as zero so the report layout is stable.
    let _ = snode_params;
    let mc64_us = 0u128;

    // Phase 2: AMD ordering (symmetric_pattern + amd_order)
    let t0 = Instant::now();
    let full_pattern = matrix.symmetric_pattern();
    let amd_perm = amd_order(&full_pattern);
    let amd_us = t0.elapsed().as_micros();

    // Phase 3: etree twice + postorder + perm composition + final permute.
    // Bucketed together because they are a tight cluster of small ops on
    // the permuted pattern, each dominated by the same n-sized walks.
    let t0 = Instant::now();
    let amd_pattern = permute_pattern(&full_pattern, &amd_perm);
    let amd_etree = EliminationTree::from_pattern(&amd_pattern);
    let (post, _post_inv) = postorder(&amd_etree);
    let perm: Vec<usize> = post.iter().map(|&p| amd_perm[p]).collect();
    let mut perm_inv = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        perm_inv[old] = new;
    }
    let permuted_pattern = permute_pattern(&full_pattern, &perm);
    let etree = EliminationTree::from_pattern(&permuted_pattern);
    let etree_us = t0.elapsed().as_micros();

    // Phase 4: column counts (the O(n^2) suspect)
    let t0 = Instant::now();
    let col_counts = column_counts(&permuted_pattern, &etree);
    let factor_nnz = total_factor_nnz(&col_counts);
    let colcnt_us = t0.elapsed().as_micros();

    // Phase 5: supernode detection + peak contrib
    let t0 = Instant::now();
    let supernodes = find_supernodes(&etree, &col_counts, snode_params);
    let contrib_sizes: Vec<usize> = supernodes.iter().map(|s| s.contrib_size()).collect();
    let peak_contrib_bytes = compute_peak_contrib(&supernodes, &contrib_sizes);
    let snode_us = t0.elapsed().as_micros();

    let sym_total_us = t_total.elapsed().as_micros();

    let factor_slack = 1.2;
    let sym = SymbolicFactorization {
        n,
        perm,
        perm_inv,
        supernodes,
        factor_nnz_estimate: (factor_nnz as f64 * factor_slack) as usize,
        factor_slack,
        contrib_sizes,
        peak_contrib_bytes,
        etree,
        permuted_pattern,
        col_counts,
    };

    let times = PhaseTimes {
        mc64_us,
        amd_us,
        etree_us,
        colcnt_us,
        snode_us,
        sym_total_us,
        numeric_us: 0,
        total_us: 0,
    };
    Some((sym, times))
}

/// Mirror of feral::symbolic::compute_peak_contrib (which is private).
/// Lifted here so the inline pipeline can be self-contained.
fn compute_peak_contrib(
    supernodes: &[feral::symbolic::supernode::Supernode],
    contrib_sizes: &[usize],
) -> usize {
    let n_snodes = supernodes.len();
    if n_snodes == 0 {
        return 0;
    }
    let mut live = vec![false; n_snodes];
    let mut current_size = 0usize;
    let mut peak = 0usize;
    for k in 0..n_snodes {
        current_size += contrib_sizes[k];
        live[k] = true;
        if current_size > peak {
            peak = current_size;
        }
        for &child in &supernodes[k].children {
            if live[child] {
                current_size -= contrib_sizes[child];
                live[child] = false;
            }
        }
    }
    peak * std::mem::size_of::<f64>()
}

fn read_mumps_factor_us(path: &Path) -> Option<u64> {
    let contents = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&contents).ok()?;
    v.get("factor_us")?.as_u64()
}

fn main() {
    let kkt_dir = Path::new("data/matrices/kkt");
    if !kkt_dir.is_dir() {
        eprintln!("data/matrices/kkt not found");
        return;
    }

    let snode_params = SupernodeParams::default();
    let params = NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    });

    let mut rows: Vec<Row> = Vec::new();
    let mut n_considered = 0usize;
    let mut n_loaded = 0usize;

    let subdirs: Vec<_> = std::fs::read_dir(kkt_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    for subdir in subdirs {
        let subdir_path = subdir.path();
        if !subdir_path.is_dir() {
            continue;
        }
        let mtx_files: Vec<_> = match std::fs::read_dir(&subdir_path) {
            Ok(d) => d
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "mtx"))
                .collect(),
            Err(_) => continue,
        };
        for mtx_entry in mtx_files {
            n_considered += 1;
            let mtx_path = mtx_entry.path();
            let stem = mtx_path.file_stem().unwrap().to_string_lossy().to_string();
            let mumps_path = mtx_path.with_extension("mumps.json");
            let Some(mumps_us) = read_mumps_factor_us(&mumps_path) else {
                continue;
            };
            let Ok(mtx) = read_mtx(&mtx_path) else {
                continue;
            };
            if mtx.n > 500 {
                // Focus on the small-frontal bucket — n > 500 matrices
                // cannot have max_front < 200 anyway (though max_front
                // can be less than n, the scope here is small matrices).
                continue;
            }
            if mtx.entries.iter().any(|(_, _, v)| !v.is_finite()) {
                continue;
            }
            let Ok(csc) = mtx.to_csc() else { continue };
            n_loaded += 1;

            let Some((sym, mut times)) = timed_symbolic(&csc, &snode_params) else {
                continue;
            };
            let max_front = sym.supernodes.iter().map(|s| s.nrow).max().unwrap_or(csc.n);
            if max_front >= 200 {
                continue; // Not in the small-frontal bucket.
            }

            let t_num = Instant::now();
            if factorize_multifrontal(&csc, &sym, &params).is_err() {
                continue;
            }
            times.numeric_us = t_num.elapsed().as_micros();
            times.total_us = times.sym_total_us + times.numeric_us;

            let ratio = (times.total_us.max(1) as f64) / (mumps_us.max(1) as f64);

            rows.push(Row {
                name: stem,
                n: csc.n,
                max_front,
                times,
                mumps_us,
                ratio,
            });
        }
    }

    eprintln!(
        "Considered {} matrices, loaded {}, small-frontal bucket {}",
        n_considered,
        n_loaded,
        rows.len()
    );

    // Sort by ratio descending — worst offenders first.
    rows.sort_by(|a, b| {
        b.ratio
            .partial_cmp(&a.ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Aggregate: for each phase, what is the share of total_us across the
    // whole small-frontal set? Geometric-mean-ish summary by summing all
    // microseconds.
    let sum_us =
        |pick: fn(&PhaseTimes) -> u128| -> u128 { rows.iter().map(|r| pick(&r.times)).sum() };
    let total_all: u128 = sum_us(|t| t.total_us);
    let sym_all: u128 = sum_us(|t| t.sym_total_us);
    let num_all: u128 = sum_us(|t| t.numeric_us);
    let mc64_all: u128 = sum_us(|t| t.mc64_us);
    let amd_all: u128 = sum_us(|t| t.amd_us);
    let etree_all: u128 = sum_us(|t| t.etree_us);
    let colcnt_all: u128 = sum_us(|t| t.colcnt_us);
    let snode_all: u128 = sum_us(|t| t.snode_us);

    let pct = |part: u128| -> f64 {
        if total_all == 0 {
            0.0
        } else {
            100.0 * part as f64 / total_all as f64
        }
    };

    println!(
        "\n=== Phase share across {} small-frontal matrices ===",
        rows.len()
    );
    println!("  total:     {:>10} us ({:.1}%)", total_all, pct(total_all));
    println!("  symbolic:  {:>10} us ({:.1}%)", sym_all, pct(sym_all));
    println!("    mc64:    {:>10} us ({:.1}%)", mc64_all, pct(mc64_all));
    println!("    amd:     {:>10} us ({:.1}%)", amd_all, pct(amd_all));
    println!("    etree:   {:>10} us ({:.1}%)", etree_all, pct(etree_all));
    println!(
        "    colcnt:  {:>10} us ({:.1}%)",
        colcnt_all,
        pct(colcnt_all)
    );
    println!("    snode:   {:>10} us ({:.1}%)", snode_all, pct(snode_all));
    println!("  numeric:   {:>10} us ({:.1}%)", num_all, pct(num_all));

    // Top 30 worst offenders with per-phase breakdown.
    println!(
        "\n=== Top 30 worst factor-ratio matrices in small-frontal bucket ===\n\
         (times in us; ratio = total_feral / mumps_factor)"
    );
    println!(
        "{:<22} {:>5} {:>6} {:>6} {:>5} {:>5} {:>6} {:>5} {:>5} {:>7} {:>7} {:>6}",
        "name", "n", "mf", "mc64", "amd", "etre", "colc", "snod", "numc", "feral", "mumps", "ratio"
    );
    for r in rows.iter().take(30) {
        println!(
            "{:<22} {:>5} {:>6} {:>6} {:>5} {:>5} {:>6} {:>5} {:>5} {:>7} {:>7} {:>6.2}",
            r.name,
            r.n,
            r.max_front,
            r.times.mc64_us,
            r.times.amd_us,
            r.times.etree_us,
            r.times.colcnt_us,
            r.times.snode_us,
            r.times.numeric_us,
            r.times.total_us,
            r.mumps_us,
            r.ratio,
        );
    }

    // Per-phase histogram at several percentiles: what does the
    // distribution of each phase look like? Helpful to see if colcnt
    // has a fat tail while amd doesn't.
    println!("\n=== Per-phase percentiles (us) across small-frontal bucket ===");
    type PhasePick = fn(&PhaseTimes) -> u128;
    let phases: &[(&str, PhasePick)] = &[
        ("mc64", |t| t.mc64_us),
        ("amd", |t| t.amd_us),
        ("etree", |t| t.etree_us),
        ("colcnt", |t| t.colcnt_us),
        ("snode", |t| t.snode_us),
        ("numeric", |t| t.numeric_us),
        ("total", |t| t.total_us),
    ];
    println!(
        "{:<10} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "phase", "p50", "p90", "p99", "max", "share%"
    );
    for (name, pick) in phases {
        let mut vals: Vec<u128> = rows.iter().map(|r| pick(&r.times)).collect();
        vals.sort();
        if vals.is_empty() {
            continue;
        }
        let idx = |q: f64| -> u128 {
            let k = ((vals.len() as f64 * q).ceil() as usize)
                .saturating_sub(1)
                .min(vals.len() - 1);
            vals[k]
        };
        let p50 = idx(0.50);
        let p90 = idx(0.90);
        let p99 = idx(0.99);
        let max = *vals.last().unwrap();
        let sum: u128 = vals.iter().sum();
        let share = if total_all > 0 {
            100.0 * sum as f64 / total_all as f64
        } else {
            0.0
        };
        println!(
            "{:<10} {:>8} {:>8} {:>8} {:>8} {:>8.1}",
            name, p50, p90, p99, max, share
        );
    }
}
