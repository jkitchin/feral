//! Characterize the per-iteration memory profile of the multifrontal
//! sparse loop. Session 04 observed RSS climb 17 GB → 36 GB across
//! 167k iterations with `FERAL_SPARSE_MAX=20000`, then plateau.
//! This diag isolates whether the growth is a real leak (RSS grows
//! on repeated runs of the same matrix) or allocator high-water-mark
//! retention (RSS spikes on a big matrix and stays at the spike).
//!
//! Usage: `cargo run --release --bin diag_sparse_memory -- <pattern> [LIMIT]`
//! where pattern is a comma-separated list of size buckets to drive
//! the experiment, e.g.
//!     small             — repeat a small matrix N times
//!     small,big,small   — small N times, then big once, then small N times
//!     varied            — sweep through all sizes once
//!
//! Reports RSS in MB after each iteration so we can see growth pattern.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Get RSS in MB via `ps` (portable across macOS/Linux for our purposes).
fn rss_mb() -> f64 {
    let pid = std::process::id();
    let out = match Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
    {
        Ok(o) => o,
        Err(_) => return f64::NAN,
    };
    let s = String::from_utf8_lossy(&out.stdout);
    let kb: f64 = s.trim().parse().unwrap_or(f64::NAN);
    kb / 1024.0
}

fn find_matrix_dir(family: &str) -> Option<PathBuf> {
    for r in [
        "data/matrices/kkt-expansion",
        "data/matrices/kkt-mittelmann",
        "data/matrices/kkt",
    ] {
        let p = Path::new(r).join(family);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn first_mtx(dir: &Path) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    let mut paths: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("mtx"))
        .collect();
    paths.sort();
    paths.into_iter().next()
}

fn run_one(path: &Path) -> Option<usize> {
    let mtx = read_mtx(path).ok()?;
    let csc = mtx.to_csc().ok()?;
    drop(mtx);
    let n = csc.n;
    let snode_params = SupernodeParams::default();
    let bk = BunchKaufmanParams::default();
    let nparams = NumericParams::with_bk(bk);
    let sym = symbolic_factorize(&csc, &snode_params).ok()?;
    let _result = factorize_multifrontal(&csc, &sym, &nparams).ok()?;
    Some(n)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pattern = args.get(1).cloned().unwrap_or_else(|| "small".to_string());
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    // Pick representative matrices in three size buckets.
    let small_path = find_matrix_dir("HS118").and_then(|d| first_mtx(&d));
    let medium_path = find_matrix_dir("EIGENALS").and_then(|d| first_mtx(&d));
    let big_path = find_matrix_dir("CHAINWOO").and_then(|d| first_mtx(&d));

    let small = match small_path {
        Some(p) => p,
        None => {
            eprintln!("no small matrix found (HS118)");
            std::process::exit(1);
        }
    };
    let medium = medium_path;
    let big = big_path;

    let buckets: Vec<&str> = pattern.split(',').collect();
    let baseline_rss = rss_mb();
    println!(
        "# baseline RSS = {:.1} MB, pattern = {:?}, limit = {}",
        baseline_rss, buckets, limit
    );
    println!(
        "{:>5}  {:>8}  {:>10}  {:>10}  {:>10}",
        "iter", "bucket", "matrix", "n", "RSS_MB"
    );

    let mut iter = 0;
    for bucket in &buckets {
        let path: &Path = match *bucket {
            "small" => &small,
            "medium" => match &medium {
                Some(p) => p,
                None => {
                    eprintln!("# no medium matrix; skipping");
                    continue;
                }
            },
            "big" => match &big {
                Some(p) => p,
                None => {
                    eprintln!("# no big matrix; skipping");
                    continue;
                }
            },
            other => {
                eprintln!("# unknown bucket {}, skipping", other);
                continue;
            }
        };
        for _ in 0..limit {
            iter += 1;
            let n = run_one(path).unwrap_or(0);
            let rss = rss_mb();
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
            println!(
                "{:>5}  {:>8}  {:>10}  {:>10}  {:>10.1}",
                iter, bucket, stem, n, rss
            );
        }
    }

    let final_rss = rss_mb();
    println!(
        "# baseline = {:.1} MB, final = {:.1} MB, delta = {:+.1} MB",
        baseline_rss,
        final_rss,
        final_rss - baseline_rss
    );
}
