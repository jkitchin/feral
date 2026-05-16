//! Issue #10 / #33 follow-up: empirical test of the supernode-shape
//! thesis on the 1D-banded Mittelmann panel.
//!
//! Hypothesis from `issue-10-maxfromm-phase2-corpus.md` §"remaining
//! levers": the MAXFROMM and SLB A/Bs both failed because AMD produces
//! very narrow (ncol ≈ 1..5) supernodes on 1D-banded KKTs, so neither
//! lever can engage. If a nested-dissection ordering (Metis or Scotch)
//! widens the bottom-of-tree supernodes meaningfully, the lever
//! becomes available again. If no ordering widens shape, #10/#33 are
//! jointly blocked on supernode amalgamation (a symbolic-side
//! restructure) — that's the conclusion this binary needs to settle.
//!
//! Reports per-matrix and per-family:
//!   - supernode count
//!   - mean / p50 / p90 / max eliminated-column width (`ncol`)
//!   - mean / p90 frontal-matrix `nrow`
//!   - factor time (min-of-N) at the default `TppMethod::Plain`
//!
//! Usage: `cargo run --release --bin diag_ordering_panel`

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{
    symbolic_factorize_with_method, OrderingMethod, SupernodeParams, SymbolicFactorization,
};
use feral::{read_mtx, CscMatrix};

const N_REPEAT: usize = 5;
const CORPUS: &str = "data/matrices/kkt-mittelmann";
const FAMILIES: &[&str] = &["clnlbeam", "henon120", "lane_emden120", "dirichlet120"];
const METHODS: &[(&str, OrderingMethod)] = &[
    ("Amd", OrderingMethod::Amd),
    ("MetisND", OrderingMethod::MetisND),
    ("ScotchND", OrderingMethod::ScotchND),
];

fn load_csc(path: &Path) -> Option<CscMatrix> {
    let mtx = read_mtx(path).ok()?;
    mtx.to_csc().ok()
}

fn percentile(sorted: &[usize], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)] as f64
}

struct ShapeStats {
    n_snodes: usize,
    ncol_mean: f64,
    ncol_p50: f64,
    ncol_p90: f64,
    ncol_max: usize,
    nrow_mean: f64,
}

fn shape_stats(sym: &SymbolicFactorization) -> ShapeStats {
    let mut ncols: Vec<usize> = sym.supernodes.iter().map(|s| s.ncol).collect();
    let nrows: Vec<usize> = sym.supernodes.iter().map(|s| s.nrow).collect();
    ncols.sort_unstable();
    let n_snodes = ncols.len();
    let ncol_sum: usize = ncols.iter().sum();
    let nrow_sum: usize = nrows.iter().sum();
    ShapeStats {
        n_snodes,
        ncol_mean: ncol_sum as f64 / n_snodes.max(1) as f64,
        ncol_p50: percentile(&ncols, 0.50),
        ncol_p90: percentile(&ncols, 0.90),
        ncol_max: *ncols.last().unwrap_or(&0),
        nrow_mean: nrow_sum as f64 / n_snodes.max(1) as f64,
    }
}

fn factor_time_us(csc: &CscMatrix, sym: &SymbolicFactorization, np: &NumericParams) -> u128 {
    let _ = factorize_multifrontal(csc, sym, np);
    let mut best = u128::MAX;
    for _ in 0..N_REPEAT {
        let t = Instant::now();
        if factorize_multifrontal(csc, sym, np).is_err() {
            return 0;
        }
        let us = t.elapsed().as_micros();
        if us < best {
            best = us;
        }
    }
    best
}

fn enumerate(family: &str) -> Vec<(String, std::path::PathBuf)> {
    let dir = Path::new(CORPUS).join(family);
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<_> = entries
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.extension()?.to_str()? != "mtx" {
                return None;
            }
            let stem = p.file_stem()?.to_str()?.to_string();
            Some((stem, p))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn geomean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let s: f64 = xs.iter().map(|v| v.ln()).sum();
    (s / xs.len() as f64).exp()
}

fn process_family(family: &str) {
    let matrices = enumerate(family);
    if matrices.is_empty() {
        println!("\n[{family}] MISSING corpus");
        return;
    }
    println!("\n=== {family} ({} matrices) ===", matrices.len());
    println!(
        "{:>20}  {:>8}  {:>7}  {:>9}  {:>9}  {:>9}  {:>7}  {:>9}  {:>11}",
        "label",
        "method",
        "snodes",
        "ncol_mean",
        "ncol_p50",
        "ncol_p90",
        "ncol_max",
        "nrow_mean",
        "factor_us"
    );

    // Per-family per-method aggregates for the summary at end.
    let mut per_method_factor: Vec<(String, Vec<f64>)> = METHODS
        .iter()
        .map(|(n, _)| (n.to_string(), Vec::new()))
        .collect();
    let mut per_method_ncol_mean: Vec<(String, Vec<f64>)> = METHODS
        .iter()
        .map(|(n, _)| (n.to_string(), Vec::new()))
        .collect();
    let mut per_method_ncol_p90: Vec<(String, Vec<f64>)> = METHODS
        .iter()
        .map(|(n, _)| (n.to_string(), Vec::new()))
        .collect();

    let snode_params = SupernodeParams::default();
    let np = NumericParams::default();

    for (label, path) in &matrices {
        let Some(csc) = load_csc(path) else {
            println!("{label:>20}  load FAIL");
            continue;
        };
        for (mi, (mname, method)) in METHODS.iter().enumerate() {
            let sym = match symbolic_factorize_with_method(&csc, &snode_params, *method) {
                Ok(s) => s,
                Err(e) => {
                    println!("{label:>20}  {mname:>8}  symbolic FAIL: {e}");
                    continue;
                }
            };
            let st = shape_stats(&sym);
            let us = factor_time_us(&csc, &sym, &np);
            if us == 0 {
                println!("{label:>20}  {mname:>8}  factor FAILED");
                continue;
            }
            println!(
                "{label:>20}  {mname:>8}  {:>7}  {:>9.2}  {:>9.0}  {:>9.0}  {:>7}  {:>9.2}  {:>11}",
                st.n_snodes, st.ncol_mean, st.ncol_p50, st.ncol_p90, st.ncol_max, st.nrow_mean, us,
            );
            per_method_factor[mi].1.push(us as f64);
            per_method_ncol_mean[mi].1.push(st.ncol_mean);
            per_method_ncol_p90[mi].1.push(st.ncol_p90);
        }
    }

    // Per-family per-method summary
    println!("  --- {family} summary (per method) ---");
    println!(
        "  {:>10}  {:>14}  {:>14}  {:>14}",
        "method", "geo factor_us", "geo ncol_mean", "geo ncol_p90"
    );
    for i in 0..METHODS.len() {
        let mname = &per_method_factor[i].0;
        let g_us = geomean(&per_method_factor[i].1);
        let g_nc = geomean(&per_method_ncol_mean[i].1);
        let g_p9 = geomean(&per_method_ncol_p90[i].1);
        println!("  {mname:>10}  {g_us:>14.0}  {g_nc:>14.2}  {g_p9:>14.2}");
    }

    // Speedup of each non-AMD method vs AMD baseline, paired by matrix.
    let amd_factors = &per_method_factor[0].1;
    let amd_ncol_mean = &per_method_ncol_mean[0].1;
    if !amd_factors.is_empty() {
        println!("  --- {family} relative to Amd (paired, geomean across matrices) ---");
        println!(
            "  {:>10}  {:>14}  {:>14}",
            "method", "factor_us/Amd", "ncol_mean/Amd"
        );
        for i in 1..METHODS.len() {
            let mname = &per_method_factor[i].0;
            let f = &per_method_factor[i].1;
            let nc = &per_method_ncol_mean[i].1;
            if f.len() == amd_factors.len() && nc.len() == amd_ncol_mean.len() {
                let r_f: Vec<f64> = f.iter().zip(amd_factors).map(|(x, a)| x / a).collect();
                let r_nc: Vec<f64> = nc.iter().zip(amd_ncol_mean).map(|(x, a)| x / a).collect();
                println!(
                    "  {mname:>10}  {:>14.3}  {:>14.3}",
                    geomean(&r_f),
                    geomean(&r_nc),
                );
            }
        }
    }
}

fn main() {
    println!("=== Issue #10/#33 supernode-shape A/B (1D-banded Mittelmann) ===");
    println!(
        "methods: {:?}",
        METHODS.iter().map(|(n, _)| *n).collect::<Vec<_>>()
    );
    println!("min-of-{N_REPEAT} factor timings; warm-up uncounted");

    for fam in FAMILIES {
        process_family(fam);
    }

    println!("\nInterpretation guide:");
    println!("  - ncol_mean/Amd > ~1.5  → method meaningfully widens supernodes");
    println!("  - factor_us/Amd  < ~0.9 → method gives a direct factor win");
    println!("  - both > 1                → wider but slower (per-call ND cost dominates)");
    println!("  - if no method widens, #10 is blocked on supernode amalgamation");
}
