//! Triage DISCS_0012 AMD slowness.
//!
//! Session 2026-04-14-03 profile showed `amd_order` taking ~9200 μs on
//! DISCS_0012 (n=234, 14790 nnz) while MUMPS's entire analyse+factor runs
//! in ~450 μs. The current `src/ordering/amd.rs` is naive exact-degree
//! minimum-degree elimination with three obvious hotspots:
//!
//!   1. Linear scan over `eliminated` to find min-degree node: O(n)/step
//!   2. `adj[a].contains(&b)` inside the fill loop: O(deg) per check,
//!      O(deg²) checks per eliminated node
//!   3. `adj[nb].iter().filter().count()` to recompute degree: O(deg)
//!      per neighbor, O(deg) neighbors per step
//!
//! This binary times AMD in isolation and instruments the hot loop with
//! per-phase counters (min-search, fill-edge insertion, degree-recompute)
//! so we can see which term actually dominates. Also compares against a
//! few other small-frontal top offenders (LAKES, DMN15103, GROUPING) to
//! confirm the pattern generalizes.
//!
//! Usage: cargo run --release --example triage_discs_amd

use std::path::Path;
use std::time::Instant;

use feral::ordering::amd::amd_order;
use feral::read_mtx;
use feral::sparse::csc::CscPattern;

/// Instrumented copy of `amd_order` that reports per-phase time plus a
/// handful of scalar counters (total elim steps, total neighbors seen,
/// total contains calls, total degree recomputes).
#[derive(Debug, Default, Clone)]
struct AmdStats {
    total_us: u128,
    min_scan_us: u128,
    fill_us: u128,
    degree_us: u128,
    n_steps: usize,
    total_neighbors: usize,
    total_contains_calls: usize,
    max_neighbors_one_step: usize,
    total_fill_inserts: usize,
}

fn amd_order_instrumented(pattern: &CscPattern) -> (Vec<usize>, AmdStats) {
    let n = pattern.n;
    let mut stats = AmdStats::default();
    let t_total = Instant::now();

    if n == 0 {
        return (Vec::new(), stats);
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (j, adj_j) in adj.iter_mut().enumerate() {
        for k in pattern.col_ptr[j]..pattern.col_ptr[j + 1] {
            let i = pattern.row_idx[k];
            if i != j {
                adj_j.push(i);
            }
        }
    }

    let mut eliminated = vec![false; n];
    let mut degree = vec![0usize; n];
    for i in 0..n {
        degree[i] = adj[i].len();
    }

    let mut perm = Vec::with_capacity(n);

    for _ in 0..n {
        // Phase A: linear scan for min degree
        let t = Instant::now();
        let mut min_deg = usize::MAX;
        let mut pivot = 0;
        for i in 0..n {
            if !eliminated[i] && degree[i] < min_deg {
                min_deg = degree[i];
                pivot = i;
            }
        }
        stats.min_scan_us += t.elapsed().as_nanos() / 1000;

        eliminated[pivot] = true;
        perm.push(pivot);

        let neighbors: Vec<usize> = adj[pivot]
            .iter()
            .copied()
            .filter(|&i| !eliminated[i])
            .collect();

        stats.n_steps += 1;
        stats.total_neighbors += neighbors.len();
        if neighbors.len() > stats.max_neighbors_one_step {
            stats.max_neighbors_one_step = neighbors.len();
        }

        // Phase B: fill edges + contains
        let t = Instant::now();
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let (a, b) = (neighbors[i], neighbors[j]);
                stats.total_contains_calls += 1;
                if !adj[a].contains(&b) {
                    adj[a].push(b);
                    adj[b].push(a);
                    stats.total_fill_inserts += 1;
                }
            }
        }
        stats.fill_us += t.elapsed().as_nanos() / 1000;

        // Phase C: degree recompute
        let t = Instant::now();
        for &nb in &neighbors {
            degree[nb] = adj[nb].iter().filter(|&&x| !eliminated[x]).count();
        }
        stats.degree_us += t.elapsed().as_nanos() / 1000;
    }

    stats.total_us = t_total.elapsed().as_micros();
    (perm, stats)
}

fn triage_matrix(path: &Path) {
    let Ok(mtx) = read_mtx(path) else {
        eprintln!("  [skip] failed to read {}", path.display());
        return;
    };
    let Ok(csc) = mtx.to_csc() else {
        eprintln!("  [skip] failed to convert {}", path.display());
        return;
    };
    let pattern = csc.symmetric_pattern();
    let n = pattern.n;
    let nnz = pattern.row_idx.len();

    // Run several times to get a stable measurement (~1 ms matrices
    // need repeated runs; 10ms matrices are fine single-shot).
    let n_trials = if nnz < 5000 { 20 } else { 3 };
    let mut best: Option<AmdStats> = None;
    let mut best_lib_us: u128 = u128::MAX;
    for _ in 0..n_trials {
        let (_, stats) = amd_order_instrumented(&pattern);
        match &best {
            None => best = Some(stats),
            Some(b) if stats.total_us < b.total_us => best = Some(stats),
            _ => {}
        }
        let t = Instant::now();
        let _ = amd_order(&pattern);
        let lib_us = t.elapsed().as_micros();
        if lib_us < best_lib_us {
            best_lib_us = lib_us;
        }
    }
    let s = best.unwrap();

    println!(
        "\n{}  n={} nnz(symmetric)={}",
        path.file_stem().unwrap().to_string_lossy(),
        n,
        nnz
    );
    println!("  total       {:>8} us", s.total_us);
    println!(
        "    min-scan  {:>8} us  ({:.1}%)",
        s.min_scan_us,
        100.0 * s.min_scan_us as f64 / s.total_us.max(1) as f64
    );
    println!(
        "    fill      {:>8} us  ({:.1}%)",
        s.fill_us,
        100.0 * s.fill_us as f64 / s.total_us.max(1) as f64
    );
    println!(
        "    degree    {:>8} us  ({:.1}%)",
        s.degree_us,
        100.0 * s.degree_us as f64 / s.total_us.max(1) as f64
    );
    println!(
        "  steps={}  avg_neigh={:.1}  max_neigh={}",
        s.n_steps,
        s.total_neighbors as f64 / s.n_steps.max(1) as f64,
        s.max_neighbors_one_step
    );
    println!(
        "  contains_calls={}  fill_inserts={}  ratio={:.2}",
        s.total_contains_calls,
        s.total_fill_inserts,
        s.total_contains_calls as f64 / s.total_fill_inserts.max(1) as f64
    );
    println!(
        "  library amd_order: {:>8} us  (speedup vs naive instrumented: {:.1}×)",
        best_lib_us,
        s.total_us as f64 / best_lib_us.max(1) as f64
    );
}

fn main() {
    let candidates = [
        "data/matrices/kkt/DISCS/DISCS_0012.mtx",
        "data/matrices/kkt/DISCS/DISCS_0000.mtx",
        "data/matrices/kkt/LAKES/LAKES_0000.mtx",
        "data/matrices/kkt/DMN15103/DMN15103_0000.mtx",
        "data/matrices/kkt/GROUPING/GROUPING_0000.mtx",
    ];

    println!("=== AMD triage (instrumented amd_order) ===");
    for rel in candidates {
        let p = Path::new(rel);
        if !p.exists() {
            eprintln!("missing: {}", rel);
            continue;
        }
        triage_matrix(p);
    }
}
