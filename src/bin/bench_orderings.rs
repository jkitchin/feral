//! Comparative fill-reducing-ordering benchmark.
//!
//! For each matrix in `tests/data/parity/<family>/*.mtx`, runs
//! `symbolic_factorize_with_method` four times (AMD / METIS / SCOTCH /
//! KaHIP), records `factor_nnz_estimate` and wall-clock symbolic
//! time, and prints a per-matrix row plus geomean-ratio summary.
//!
//! Usage:
//!   cargo run --release --bin bench_orderings
//!   cargo run --release --bin bench_orderings -- tests/data/parity
//!
//! The `factor_nnz_estimate` field is the sum of column counts of L
//! computed by the post-ordering pipeline. Different orderings produce
//! different column counts on the same pattern, so the ratio
//! `fill_other / fill_amd` is a direct symbolic-fill comparison.
//!
//! Numeric factorization time and realistic-pivot factor NNZ are out
//! of scope — see B6 in `dev/plans/ordering-scotch.md` for the reason
//! that matters eventually.

use feral::read_mtx;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Row {
    fill_amd: u64,
    fill_metis: u64,
    fill_scotch: u64,
    fill_kahip: u64,
    time_amd_us: u128,
    time_metis_us: u128,
    time_scotch_us: u128,
    time_kahip_us: u128,
}

fn measure(
    matrix: &feral::sparse::csc::CscMatrix,
    params: &SupernodeParams,
    method: OrderingMethod,
) -> Option<(u64, u128)> {
    let t = Instant::now();
    let sym = symbolic_factorize_with_method(matrix, params, method).ok()?;
    let us = t.elapsed().as_micros();
    Some((sym.factor_nnz_estimate as u64, us))
}

fn roots_from_args() -> Vec<PathBuf> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        let mut out = vec![PathBuf::from("tests/data/parity")];
        let large = PathBuf::from("tests/data/large");
        if large.is_dir() {
            out.push(large);
        }
        out
    } else {
        args.into_iter().map(PathBuf::from).collect()
    }
}

// Per-family parity layout: <root>/<family>/<*.mtx>, one representative
// matrix per family. Flat layout: <root>/<*.mtx>, each file used as-is.
// The detection is simply: does the root contain any .mtx files directly?
fn collect_mtx_files(root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let rd = match std::fs::read_dir(root) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", root.display(), e);
            return out;
        }
    };
    let entries: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    let mut flat_mtxs: Vec<PathBuf> = entries
        .iter()
        .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("mtx"))
        .cloned()
        .collect();
    if !flat_mtxs.is_empty() {
        flat_mtxs.sort();
        return flat_mtxs;
    }
    let mut subs: Vec<PathBuf> = entries.into_iter().filter(|p| p.is_dir()).collect();
    subs.sort();
    for sub in subs {
        let inner = match std::fs::read_dir(&sub) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let mut mtxs: Vec<PathBuf> = inner
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mtx"))
            .collect();
        mtxs.sort();
        // One matrix per family — the parity corpus has multiple
        // dumps per IPM iteration and we only need a representative
        // shape for the ordering comparison.
        if let Some(first) = mtxs.into_iter().next() {
            out.push(first);
        }
    }
    out
}

fn geomean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    let n = vals.len() as f64;
    let sum_ln: f64 = vals.iter().map(|v| v.ln()).sum();
    (sum_ln / n).exp()
}

fn main() {
    let roots = roots_from_args();
    let mut mtx_files: Vec<PathBuf> = Vec::new();
    for root in &roots {
        eprintln!("scanning {}", root.display());
        mtx_files.extend(collect_mtx_files(root));
    }
    if mtx_files.is_empty() {
        eprintln!("no .mtx files found");
        std::process::exit(1);
    }

    let params = SupernodeParams::default();
    let mut rows: Vec<Row> = Vec::new();
    let mut skipped = 0usize;

    println!(
        "{:<18} {:>8} {:>10} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "matrix",
        "n",
        "nnz",
        "fill_amd",
        "fill_metis",
        "fill_scotch",
        "fill_kahip",
        "t_amd",
        "t_metis",
        "t_scot",
        "t_kahip",
        "m/amd",
        "s/amd",
        "k/amd",
    );
    println!("{}", "-".repeat(160));

    for path in &mtx_files {
        // For per-family parity layout the parent dir name is the
        // family; for a flat large-matrix dir the file stem is the
        // matrix name.
        let parent_name = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("?");
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let family = if parent_name == "large" {
            stem.to_string()
        } else {
            parent_name.to_string()
        };
        let mtx = match read_mtx(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("SKIP {}: {}", family, e);
                skipped += 1;
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP {}: csc conversion failed: {}", family, e);
                skipped += 1;
                continue;
            }
        };
        let n = csc.n;
        let nnz = csc.row_idx.len();
        if n == 0 {
            skipped += 1;
            continue;
        }

        let a = measure(&csc, &params, OrderingMethod::Amd);
        let m = measure(&csc, &params, OrderingMethod::MetisND);
        let s = measure(&csc, &params, OrderingMethod::ScotchND);
        let k = measure(&csc, &params, OrderingMethod::KahipND);

        let (
            Some((fill_amd, t_amd)),
            Some((fill_metis, t_metis)),
            Some((fill_scotch, t_scot)),
            Some((fill_kahip, t_kahip)),
        ) = (a, m, s, k)
        else {
            eprintln!("SKIP {}: one or more orderings failed", family);
            skipped += 1;
            continue;
        };

        let r_m = fill_metis as f64 / fill_amd.max(1) as f64;
        let r_s = fill_scotch as f64 / fill_amd.max(1) as f64;
        let r_k = fill_kahip as f64 / fill_amd.max(1) as f64;

        println!(
            "{:<18} {:>8} {:>10} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8} {:>8} {:>8} {:>8.3} {:>8.3} {:>8.3}",
            family,
            n,
            nnz,
            fill_amd,
            fill_metis,
            fill_scotch,
            fill_kahip,
            t_amd,
            t_metis,
            t_scot,
            t_kahip,
            r_m,
            r_s,
            r_k,
        );

        rows.push(Row {
            fill_amd,
            fill_metis,
            fill_scotch,
            fill_kahip,
            time_amd_us: t_amd,
            time_metis_us: t_metis,
            time_scotch_us: t_scot,
            time_kahip_us: t_kahip,
        });
    }

    // Summary.
    let ratios_m: Vec<f64> = rows
        .iter()
        .map(|r| r.fill_metis as f64 / r.fill_amd.max(1) as f64)
        .collect();
    let ratios_s: Vec<f64> = rows
        .iter()
        .map(|r| r.fill_scotch as f64 / r.fill_amd.max(1) as f64)
        .collect();
    let ratios_k: Vec<f64> = rows
        .iter()
        .map(|r| r.fill_kahip as f64 / r.fill_amd.max(1) as f64)
        .collect();
    let (wins_amd, wins_metis, wins_scotch, wins_kahip) =
        rows.iter().fold((0, 0, 0, 0), |(a, m, s, k), r| {
            let best = r
                .fill_amd
                .min(r.fill_metis)
                .min(r.fill_scotch)
                .min(r.fill_kahip);
            let mut aa = a;
            let mut mm = m;
            let mut ss = s;
            let mut kk = k;
            if r.fill_amd == best {
                aa += 1;
            }
            if r.fill_metis == best {
                mm += 1;
            }
            if r.fill_scotch == best {
                ss += 1;
            }
            if r.fill_kahip == best {
                kk += 1;
            }
            (aa, mm, ss, kk)
        });

    println!();
    println!("summary ({} matrices, {} skipped):", rows.len(), skipped);
    println!(
        "  geomean fill_metis  / fill_amd = {:.3}",
        geomean(&ratios_m)
    );
    println!(
        "  geomean fill_scotch / fill_amd = {:.3}",
        geomean(&ratios_s)
    );
    println!(
        "  geomean fill_kahip  / fill_amd = {:.3}",
        geomean(&ratios_k)
    );
    println!(
        "  minimum-fill wins: AMD = {}, METIS = {}, SCOTCH = {}, KaHIP = {} (ties count for all at min)",
        wins_amd, wins_metis, wins_scotch, wins_kahip,
    );
    println!(
        "  total symbolic time (us): AMD = {}, METIS = {}, SCOTCH = {}, KaHIP = {}",
        rows.iter().map(|r| r.time_amd_us).sum::<u128>(),
        rows.iter().map(|r| r.time_metis_us).sum::<u128>(),
        rows.iter().map(|r| r.time_scotch_us).sum::<u128>(),
        rows.iter().map(|r| r.time_kahip_us).sum::<u128>(),
    );
}
