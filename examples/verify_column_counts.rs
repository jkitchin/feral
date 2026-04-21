//! Phase 2.5.1 Step 3 — corpus cross-check for `column_counts_gnp`.
//!
//! Walks the full KKT corpus at `data/matrices/kkt/`, computes column
//! counts via both the reference `column_counts` (O(n²) elimination
//! simulation) and the new `column_counts_gnp` (O(nnz + n·α) Gilbert-
//! Ng-Peyton algorithm), and reports any per-matrix disagreement.
//!
//! The algorithm is a pure refactor: the output contract is the same
//! integer vector. Every matrix in the corpus must produce bit-exact
//! equality between the two functions. Any mismatch blocks adoption.
//!
//! Usage:
//!
//!     cargo run --example verify_column_counts --release

use std::path::Path;

use feral::ordering::elimination_tree::EliminationTree;
use feral::read_mtx;
use feral::symbolic::column_counts::{column_counts, column_counts_gnp};

fn main() {
    let dir = Path::new("data/matrices/kkt");
    let Ok(subdirs) = std::fs::read_dir(dir) else {
        eprintln!("data/matrices/kkt not found");
        return;
    };
    let mut subdirs: Vec<_> = subdirs.filter_map(|e| e.ok()).collect();
    subdirs.sort_by_key(|e| e.file_name());

    let mut total = 0usize;
    let mut matched = 0usize;
    let mut mismatched: Vec<(String, usize, usize, usize)> = Vec::new();

    for sub in subdirs {
        let sp = sub.path();
        if !sp.is_dir() {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&sp) else {
            continue;
        };
        let mut mtx_files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "mtx"))
            .collect();
        mtx_files.sort_by_key(|e| e.file_name());

        for entry in mtx_files {
            let path = entry.path();
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            let Ok(m) = read_mtx(&path) else { continue };
            let Ok(csc) = m.to_csc() else { continue };

            let pat = csc.symmetric_pattern();
            let etree = EliminationTree::from_pattern(&pat);
            let slow = column_counts(&pat, &etree);
            let fast = column_counts_gnp(&pat, &etree);

            total += 1;
            if slow == fast {
                matched += 1;
            } else {
                // Find first-differing index
                let (j, (&a, &b)) = slow
                    .iter()
                    .zip(fast.iter())
                    .enumerate()
                    .find(|(_, (a, b))| a != b)
                    .unwrap();
                mismatched.push((stem, j, a, b));
            }
        }
        if total > 0 && total.is_multiple_of(20000) {
            eprintln!("  processed {} matrices ...", total);
        }
    }

    println!("=== column_counts_gnp vs column_counts ===");
    println!("Matched   : {}/{}", matched, total);
    println!("Mismatches: {}", mismatched.len());
    if !mismatched.is_empty() {
        println!(
            "\n{:<28} {:>10} {:>10} {:>10}",
            "matrix", "col_j", "slow_c", "fast_c"
        );
        for (name, j, s, f) in mismatched.iter().take(30) {
            println!("{:<28} {:>10} {:>10} {:>10}", name, j, s, f);
        }
        if mismatched.len() > 30 {
            println!("... ({} more)", mismatched.len() - 30);
        }
    }
}
