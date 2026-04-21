//! Phase 2.5.1 micro-timing: column_counts (O(n²)) vs column_counts_gnp
//! (O(nnz + n·α)) on the largest KKT matrices in the corpus.
//!
//! Usage:
//!
//!     cargo run --example bench_column_counts --release

use std::path::{Path, PathBuf};
use std::time::Instant;

use feral::ordering::elimination_tree::EliminationTree;
use feral::read_mtx;
use feral::sparse::csc::{CscMatrix, CscPattern};
use feral::symbolic::column_counts::{column_counts, column_counts_gnp};

struct Matrix {
    name: String,
    csc: CscMatrix,
    pat: CscPattern,
    etree: EliminationTree,
}

fn load() -> Vec<Matrix> {
    let dir = Path::new("data/matrices/kkt");
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut subdirs: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    subdirs.sort_by_key(|e| e.file_name());

    let mut out = Vec::new();
    for sub in subdirs {
        let sp = sub.path();
        if !sp.is_dir() {
            continue;
        }
        let Ok(es) = std::fs::read_dir(&sp) else {
            continue;
        };
        let mut files: Vec<_> = es
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "mtx"))
            .collect();
        files.sort_by_key(|e| e.file_name());
        for f in files {
            let path: PathBuf = f.path();
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            let Ok(m) = read_mtx(&path) else { continue };
            let Ok(csc) = m.to_csc() else { continue };
            let pat = csc.symmetric_pattern();
            let etree = EliminationTree::from_pattern(&pat);
            out.push(Matrix {
                name: stem,
                csc,
                pat,
                etree,
            });
        }
    }
    out
}

fn main() {
    let mut mats = load();
    eprintln!("Loaded {} matrices", mats.len());

    // Sort descending by n, take top 20.
    mats.sort_by(|a, b| b.csc.n.cmp(&a.csc.n));
    let top = mats.into_iter().take(20).collect::<Vec<_>>();

    println!(
        "{:<28} {:>6} {:>10} {:>14} {:>14} {:>10}",
        "matrix", "n", "nnz(pat)", "slow (ns)", "gnp (ns)", "speedup"
    );

    for m in &top {
        let iters = 100u32;
        let mut sink = 0u64;

        // Warm up both.
        sink ^= column_counts(&m.pat, &m.etree).iter().sum::<usize>() as u64;
        sink ^= column_counts_gnp(&m.pat, &m.etree).iter().sum::<usize>() as u64;

        let t0 = Instant::now();
        for _ in 0..iters {
            let c = column_counts(&m.pat, &m.etree);
            sink ^= c[0] as u64;
        }
        let slow_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            let c = column_counts_gnp(&m.pat, &m.etree);
            sink ^= c[0] as u64;
        }
        let fast_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        println!(
            "{:<28} {:>6} {:>10} {:>14.0} {:>14.0} {:>9.1}×",
            m.name,
            m.csc.n,
            m.pat.row_idx.len(),
            slow_ns,
            fast_ns,
            slow_ns / fast_ns
        );
        // Consume sink so the optimizer can't drop the loop.
        std::hint::black_box(sink);
    }
}
