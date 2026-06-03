//! Issue #50 — enumerate corpus matrices that hit
//! `choose_adaptive`'s `n > 100_000 && full_avg_deg < 5.0` branch
//! (powerflow22-class). These are the matrices whose `Auto` routing
//! actually changed under Fix A (`ScotchND → AMD`).
//!
//! No factorization — just reads each `.mtx`, computes the full
//! symmetric pattern's average degree, and emits matches.
//!
//! Usage:
//!
//! ```ignore
//! cargo run --release --bin diag_issue50_large_sparse_scan
//! ```

use std::env;
use std::path::PathBuf;

use feral::read_mtx;

fn collect_representatives(roots: &[PathBuf]) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for root in roots {
        let Ok(entries) = std::fs::read_dir(root) else {
            eprintln!("[skip] cannot read {}", root.display());
            continue;
        };
        let mut subdirs: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        subdirs.sort();
        for sub in subdirs {
            let dir_name = sub
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string();
            let Ok(files) = std::fs::read_dir(&sub) else {
                continue;
            };
            let mut mtxs: Vec<PathBuf> = files
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mtx"))
                .collect();
            mtxs.sort();
            let pick = mtxs
                .iter()
                .find(|p| {
                    p.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.ends_with("_0000"))
                        .unwrap_or(false)
                })
                .cloned()
                .or_else(|| mtxs.first().cloned());
            if let Some(p) = pick {
                out.push((dir_name, p));
            }
        }
    }
    out
}

fn main() {
    let roots: Vec<PathBuf> = env::var("ROOTS")
        .unwrap_or_else(|_| "data/matrices/kkt,data/matrices/kkt-expansion".to_string())
        .split(',')
        .map(|s| PathBuf::from(s.trim()))
        .collect();

    let reps = collect_representatives(&roots);
    eprintln!("[large-sparse-scan] {} problem dirs", reps.len());

    println!("dir,file,n,stored_nnz,full_avg_deg");
    let mut hits = 0;
    let mut scanned = 0;
    for (dir, path) in reps {
        scanned += 1;
        let mtx = match read_mtx(&path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[skip read_mtx] {}: {e:?}", path.display());
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[skip to_csc] {}: {e:?}", path.display());
                continue;
            }
        };
        let n = csc.n;
        if n <= 100_000 {
            continue;
        }
        let stored_nnz = csc.row_idx.len();
        let full = csc.symmetric_pattern();
        let full_avg_deg = full.row_idx.len() as f64 / n.max(1) as f64;
        if full_avg_deg < 5.0 {
            let file = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string();
            println!("{},{},{},{},{:.4}", dir, file, n, stored_nnz, full_avg_deg);
            hits += 1;
        }
    }
    eprintln!("[large-sparse-scan] {hits} matches in {scanned} matrices scanned");
}
