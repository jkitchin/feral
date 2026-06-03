//! Pure-Rust dense Schur oracle generator.
//!
//! For every matrix that has a `<id>.mumps_schur.json` sidecar, this
//! binary computes a dense Schur block via partial-pivot Gaussian
//! elimination on `[A_FF | A_FS]` and writes
//! `<id>.dense_schur.bin` (n_schur × n_schur f64 column-major,
//! same layout as the MUMPS sidecar). The Schur indices come from
//! the MUMPS sidecar so feral / MUMPS / dense are all comparing
//! the same block.
//!
//! Used by the F3.3 Option B gate
//! (`diag_schur_parity`): a per-matrix relative bound
//! `feral-vs-oracle ≤ max(1e-10, MUMPS-vs-oracle)` that adapts to
//! conditioning. See `dev/research/schur-complement.md` for the gate
//! rationale.
//!
//! Cost: O(n_F³) partial-pivot GE per matrix. n_F up to ~1900 in the
//! corpus subset = a few seconds per matrix in pure Rust. Run once;
//! cached forever.
//!
//! Usage:
//!     cargo run --release --bin produce_dense_schur
//!     cargo run --release --bin produce_dense_schur -- data/matrices/kkt
//!     FERAL_DIAG_MAX_N=2000 cargo run --release --bin produce_dense_schur

use std::path::{Path, PathBuf};

use feral::read_mtx;

const DEFAULT_ROOTS: &[&str] = &[
    "data/matrices/kkt",
    "data/matrices/kkt-expansion",
    "data/matrices/kkt-mittelmann",
];

/// Partial-pivot Gaussian elimination on `[A_FF | A_FS]` followed by
/// `S = A_SS - A_FS^T · X` where `A_FF · X = A_FS`. Returns `None`
/// when A_FF is numerically singular (any pivot below 1e-300). The
/// algorithm is intentionally simple and the same as the dense
/// oracle in `diag_acopr14.rs`; correctness is verified there.
fn dense_schur(a_full: &[Vec<f64>], schur_indices: &[usize]) -> Option<Vec<f64>> {
    let n = a_full.len();
    let n_schur = schur_indices.len();
    let in_schur: std::collections::HashSet<usize> = schur_indices.iter().copied().collect();
    let elim: Vec<usize> = (0..n).filter(|i| !in_schur.contains(i)).collect();
    let n_elim = elim.len();
    if n_elim == 0 {
        // Pathological: all variables in Schur. Block is just A_SS.
        let mut out = vec![0.0_f64; n_schur * n_schur];
        for (j, &gj) in schur_indices.iter().enumerate() {
            for (i, &gi) in schur_indices.iter().enumerate() {
                out[j * n_schur + i] = a_full[gi][gj];
            }
        }
        return Some(out);
    }

    // Build [A_FF | A_FS] in n_elim x (n_elim + n_schur) column-major
    // augmented form (row-major Vec<Vec<f64>> for indexing simplicity).
    let mut aug = vec![vec![0.0_f64; n_elim + n_schur]; n_elim];
    for (i, &gi) in elim.iter().enumerate() {
        for (j, &gj) in elim.iter().enumerate() {
            aug[i][j] = a_full[gi][gj];
        }
        for (j, &gj) in schur_indices.iter().enumerate() {
            aug[i][n_elim + j] = a_full[gi][gj];
        }
    }
    for k in 0..n_elim {
        let mut piv = k;
        let mut piv_val = aug[k][k].abs();
        for (i, row) in aug.iter().enumerate().skip(k + 1) {
            if row[k].abs() > piv_val {
                piv = i;
                piv_val = row[k].abs();
            }
        }
        if piv != k {
            aug.swap(piv, k);
        }
        if aug[k][k].abs() < 1e-300 {
            return None;
        }
        for i in (k + 1)..n_elim {
            let f = aug[i][k] / aug[k][k];
            let pivot_row = aug[k].clone();
            for (j, slot) in aug[i].iter_mut().enumerate().skip(k) {
                *slot -= f * pivot_row[j];
            }
        }
    }
    // Back-substitute for each column of X.
    let mut x = vec![vec![0.0_f64; n_schur]; n_elim];
    for c in 0..n_schur {
        for i in (0..n_elim).rev() {
            let mut s = aug[i][n_elim + c];
            for j in (i + 1)..n_elim {
                s -= aug[i][j] * x[j][c];
            }
            x[i][c] = s / aug[i][i];
        }
    }
    // Build A_SS column-major output, then subtract A_FS^T · X.
    let mut s = vec![0.0_f64; n_schur * n_schur];
    for (j, &gj) in schur_indices.iter().enumerate() {
        for (i, &gi) in schur_indices.iter().enumerate() {
            s[j * n_schur + i] = a_full[gi][gj];
        }
    }
    for j in 0..n_schur {
        for i in 0..n_schur {
            let mut acc = 0.0_f64;
            for (k, &gk) in elim.iter().enumerate() {
                acc += a_full[gk][schur_indices[i]] * x[k][j];
            }
            s[j * n_schur + i] -= acc;
        }
    }
    Some(s)
}

fn read_mumps_schur_indices(json_path: &Path) -> Option<(usize, Vec<usize>)> {
    let text = std::fs::read_to_string(json_path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    if v.get("status").and_then(|s| s.as_str()) != Some("ok") {
        return None;
    }
    let n = v.get("n")?.as_u64()? as usize;
    let idx_arr = v.get("schur_indices_0indexed")?.as_array()?;
    let mut schur: Vec<usize> = Vec::with_capacity(idx_arr.len());
    for x in idx_arr {
        let i = x.as_u64()? as usize;
        if i >= n {
            return None;
        }
        schur.push(i);
    }
    Some((n, schur))
}

#[derive(Default)]
struct Stats {
    seen: usize,
    skipped_no_mumps: usize,
    skipped_too_large: usize,
    skipped_existing: usize,
    skipped_singular: usize,
    skipped_dim_mismatch: usize,
    written: usize,
}

fn run_one(mtx_path: &Path, max_n: usize, force: bool, stats: &mut Stats, verbose: bool) {
    stats.seen += 1;
    let name = mtx_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<?>")
        .to_string();
    let mumps_json = mtx_path.with_extension("mumps_schur.json");
    let dense_bin = mtx_path.with_extension("dense_schur.bin");
    if !force && dense_bin.exists() {
        stats.skipped_existing += 1;
        return;
    }
    let (n, schur) = match read_mumps_schur_indices(&mumps_json) {
        Some(p) => p,
        None => {
            stats.skipped_no_mumps += 1;
            return;
        }
    };
    if n > max_n {
        stats.skipped_too_large += 1;
        return;
    }
    let mtx = match read_mtx(mtx_path) {
        Ok(m) if m.n == n => m,
        _ => {
            stats.skipped_dim_mismatch += 1;
            return;
        }
    };
    if mtx.entries.iter().any(|(_, _, v)| !v.is_finite()) {
        stats.skipped_dim_mismatch += 1;
        return;
    }
    let mut a_full = vec![vec![0.0_f64; n]; n];
    for &(r, c, v) in &mtx.entries {
        a_full[r][c] = v;
        if r != c {
            a_full[c][r] = v;
        }
    }
    let s = match dense_schur(&a_full, &schur) {
        Some(s) => s,
        None => {
            stats.skipped_singular += 1;
            if verbose {
                eprintln!("SINGULAR {}", name);
            }
            return;
        }
    };
    let mut bytes = Vec::with_capacity(s.len() * 8);
    for v in &s {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Err(e) = std::fs::write(&dense_bin, &bytes) {
        eprintln!("WRITE_FAIL {}: {}", name, e);
        stats.skipped_dim_mismatch += 1;
        return;
    }
    stats.written += 1;
    if verbose {
        eprintln!("OK {} n={} n_schur={}", name, n, schur.len());
    }
}

fn walk_root(root: &Path, max_n: usize, force: bool, stats: &mut Stats, verbose: bool) {
    if !root.is_dir() {
        return;
    }
    let mut entries: Vec<_> = match std::fs::read_dir(root) {
        Ok(d) => d.filter_map(|e| e.ok()).map(|e| e.path()).collect(),
        Err(_) => return,
    };
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk_root(&p, max_n, force, stats, verbose);
        } else if p.extension().is_some_and(|ext| ext == "mtx") {
            run_one(&p, max_n, force, stats, verbose);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let force = args.iter().any(|a| a == "--force");
    let positional: Vec<String> = args.into_iter().filter(|a| a != "--force").collect();
    let roots: Vec<PathBuf> = if positional.is_empty() {
        DEFAULT_ROOTS.iter().map(PathBuf::from).collect()
    } else {
        positional.iter().map(PathBuf::from).collect()
    };
    let max_n: usize = std::env::var("FERAL_DIAG_MAX_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let verbose = std::env::var("FERAL_DIAG_VERBOSE")
        .ok()
        .map(|s| s != "0")
        .unwrap_or(false);

    let mut stats = Stats::default();
    for r in &roots {
        walk_root(r, max_n, force, &mut stats, verbose);
    }
    println!("=== produce_dense_schur ===");
    println!("seen:                 {}", stats.seen);
    println!("  skipped no MUMPS:   {}", stats.skipped_no_mumps);
    println!("  skipped n>max:      {}", stats.skipped_too_large);
    println!("  skipped existing:   {}", stats.skipped_existing);
    println!("  skipped dim/parse:  {}", stats.skipped_dim_mismatch);
    println!("  skipped singular:   {}", stats.skipped_singular);
    println!("written:              {}", stats.written);
}
