//! Phase 2.6.5 diagnostic: for each KKT matrix, run MC64 matching
//! and classify the permutation's cycle structure into
//!   - 1-cycles (fixed points, `perm[j] == j`): on-diagonal matches
//!     — compression does nothing for these.
//!   - 2-cycles (`perm[perm[j]] == j`, `perm[j] != j`): each pair
//!     contracts to one super-variable in the ICNTL(12)=2 algorithm.
//!   - longer cycles: MUMPS's `DMUMPS_SYM_MWM` decomposes these
//!     into 2-cycles + singletons via a Duff-Pralet rule; for the
//!     survey we count length-k-cycle members that become `k/2`
//!     new 2-cycles (with the odd leftover becoming a singleton).
//!
//! Compression ratio is `n_compressed / n` = `(n1 + n2/2 + longer_to_2c) / n`.
//! A ratio close to 1.0 means "no leverage"; a ratio close to 0.5
//! is the theoretical max (every variable pairs with exactly one
//! other).
//!
//! Outputs per-matrix lines, then a summary. Only reports the
//! worst-10-ratio-vs-MUMPS entries plus an overall corpus histogram
//! so the note has concrete evidence.

use feral::scaling::mc64_matching;
use feral::{read_mtx, CscMatrix};
use std::path::{Path, PathBuf};

fn collect(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().map(|e| e == "mtx").unwrap_or(false) {
            out.push(p);
        }
    }
}

fn classify_cycles(perm: &[usize]) -> (usize, usize, usize, usize) {
    let n = perm.len();
    let mut visited = vec![false; n];
    let mut n1 = 0usize; // 1-cycles
    let mut n2 = 0usize; // elements in 2-cycles (so pairs = n2/2)
    let mut n_long_members = 0usize; // elements in 3+ cycles
    let mut n_long_cycles = 0usize; // number of cycles of length ≥ 3
    for start in 0..n {
        if visited[start] {
            continue;
        }
        if perm[start] == usize::MAX {
            visited[start] = true;
            continue;
        }
        // Walk cycle.
        let mut len = 0usize;
        let mut j = start;
        loop {
            if visited[j] {
                break;
            }
            visited[j] = true;
            len += 1;
            j = perm[j];
            if j == usize::MAX {
                break;
            }
            if j == start {
                break;
            }
        }
        match len {
            1 => n1 += 1,
            2 => n2 += 2,
            k if k >= 3 => {
                n_long_members += k;
                n_long_cycles += 1;
            }
            _ => {}
        }
    }
    (n1, n2, n_long_members, n_long_cycles)
}

fn main() {
    let worst_ten = [
        "MUONSINE/MUONSINE_0000",
        "CRESC100/CRESC100_0000",
        "KIRBY2/KIRBY2_0007",
        "HAHN1/HAHN1_0259",
        "KIRBY2/KIRBY2_0006",
        "KIRBY2/KIRBY2_0008",
        "GAUSS2/GAUSS2_0000",
        "VESUVIO/VESUVIO_0011",
        "VESUVIO/VESUVIO_0019",
        "VESUVIO/VESUVIO_0013",
    ];

    eprintln!("--- Top-10 worst-ratio matrices: MC64 cycle structure ---");
    eprintln!(
        "{:32} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8}",
        "matrix", "n", "n1", "pairs", "longN", "longK", "compRat"
    );
    for name in worst_ten.iter() {
        let path = format!("data/matrices/kkt/{}.mtx", name);
        let mtx = match read_mtx(Path::new(&path)) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{}: read failed: {}", name, e);
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{}: to_csc failed: {}", name, e);
                continue;
            }
        };
        let (perm, _n_matched) = match mc64_matching(&csc) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{}: mc64 failed: {}", name, e);
                continue;
            }
        };
        let (n1, n2, n_long_m, n_long_c) = classify_cycles(&perm);
        let pairs_from_short = n2 / 2;
        // Per MUMPS DMUMPS_SYM_MWM, a cycle of length k contributes
        // floor(k/2) tentative 2-cycles. So total compressed size =
        //   n1 + (n2/2) + sum_over_long(k/2).
        // We don't have individual long-cycle lengths here; approximate
        // by n_long_m / 2.
        let pairs_from_long = n_long_m / 2;
        let n_compressed = n1 + pairs_from_short + pairs_from_long;
        let n = csc.n;
        let comp_rat = if n > 0 {
            n_compressed as f64 / n as f64
        } else {
            1.0
        };
        eprintln!(
            "{:32} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8.3}",
            name, n, n1, pairs_from_short, n_long_m, n_long_c, comp_rat
        );
    }

    // Corpus histogram: bin matrices by compression ratio.
    eprintln!();
    eprintln!("--- Corpus histogram (compression ratio bins) ---");
    let mut matrices = Vec::new();
    collect(Path::new("data/matrices/kkt"), &mut matrices);
    matrices.sort();
    let mut bins = [0usize; 11]; // bins[i] = [i/10, (i+1)/10) except last = 1.0
    let mut total = 0usize;
    let mut failures = 0usize;
    for (k, path) in matrices.iter().enumerate() {
        if k % 2000 == 0 {
            eprintln!("  ... processed {} / {}", k, matrices.len());
        }
        let csc: CscMatrix = match read_mtx(path).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(_) => {
                failures += 1;
                continue;
            }
        };
        if csc.n == 0 {
            continue;
        }
        let (perm, _nm) = match mc64_matching(&csc) {
            Ok(r) => r,
            Err(_) => {
                failures += 1;
                continue;
            }
        };
        let (n1, n2, n_long_m, _) = classify_cycles(&perm);
        let n_compressed = n1 + n2 / 2 + n_long_m / 2;
        let ratio = n_compressed as f64 / csc.n as f64;
        let mut idx = (ratio * 10.0) as usize;
        if idx >= 11 {
            idx = 10;
        }
        bins[idx] += 1;
        total += 1;
    }
    eprintln!();
    eprintln!(
        "bin[comp_rat]  count   frac   (total={}, failures={})",
        total, failures
    );
    for (i, &c) in bins.iter().enumerate() {
        let lo = i as f64 * 0.1;
        let hi = ((i + 1) as f64 * 0.1).min(1.0);
        let frac = if total > 0 {
            c as f64 / total as f64
        } else {
            0.0
        };
        eprintln!(
            "  [{:.1}, {:.1})   {:>7}  {:>5.1}%",
            lo,
            hi,
            c,
            frac * 100.0
        );
    }
}
