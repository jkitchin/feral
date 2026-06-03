//! Issue #33 A/B: SmallLeafBatch::On vs Off on the 1D-banded Mittelmann
//! KKTs that pounce-feral reports as 97% scalar-1×1-bound.
//!
//! Phase 2.11 killed the global flip when the signal was within ±5%
//! measurement noise on synthetic chains. #33 hypothesizes that real
//! 1D-banded NLPs (clnlbeam etc.) may tip the signal-vs-noise
//! because the per-front kernel cost is now dominant.
//!
//! Reports min-of-N times for both configs per matrix iteration, the
//! per-matrix speedup ratio, and a per-family geomean. Decision rule
//! (per #11): default flip needs ≥10% median speedup on the panel
//! AND no per-matrix slowdown > +5%.
//!
//! Usage: `cargo run --release --bin diag_clnlbeam_slb`

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams, SmallLeafBatch};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, CscMatrix};

const N_REPEAT: usize = 7;
const CORPUS: &str = "data/matrices/kkt-mittelmann";
const FAMILIES: &[&str] = &["clnlbeam", "henon120", "lane_emden120", "dirichlet120"];

fn load_csc(path: &Path) -> Option<CscMatrix> {
    let mtx = read_mtx(path).ok()?;
    mtx.to_csc().ok()
}

fn bench_path(
    csc: &CscMatrix,
    params: &NumericParams,
    sym: &feral::symbolic::SymbolicFactorization,
) -> u128 {
    let _ = factorize_multifrontal(csc, sym, params);
    let mut best = u128::MAX;
    for _ in 0..N_REPEAT {
        let t = Instant::now();
        if factorize_multifrontal(csc, sym, params).is_err() {
            return 0;
        }
        let us = t.elapsed().as_micros();
        if us < best {
            best = us;
        }
    }
    best
}

fn params_with(slb: SmallLeafBatch) -> NumericParams {
    NumericParams {
        small_leaf: slb,
        ..NumericParams::default()
    }
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

fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = xs.len() / 2;
    if xs.len().is_multiple_of(2) {
        0.5 * (xs[m - 1] + xs[m])
    } else {
        xs[m]
    }
}

fn process_family(family: &str) -> (Vec<f64>, usize, usize) {
    let matrices = enumerate(family);
    if matrices.is_empty() {
        println!("\n[{family}] MISSING corpus");
        return (Vec::new(), 0, 0);
    }
    println!("\n=== {family} ({} matrices) ===", matrices.len());
    println!(
        "{:>20}  {:>6}  {:>6}  {:>5}  {:>10}  {:>10}  {:>7}",
        "label", "snodes", "groups", "avg", "off_us", "on_us", "speedup"
    );
    let mut speedups = Vec::new();
    let mut wins = 0usize;
    let mut losses = 0usize;
    for (label, path) in &matrices {
        let csc = match load_csc(path) {
            Some(c) => c,
            None => {
                println!("{label:>20}  load FAIL");
                continue;
            }
        };
        let sym = match symbolic_factorize(&csc, &SupernodeParams::default()) {
            Ok(s) => s,
            Err(e) => {
                println!("{label:>20}  symbolic FAIL: {e}");
                continue;
            }
        };
        let n_snodes = sym.supernodes.len();
        let n_groups = sym.small_leaf_groups.len();
        let n_grouped: usize = sym.snode_group.iter().filter(|g| g.is_some()).count();
        let avg = if n_groups > 0 {
            n_grouped as f64 / n_groups as f64
        } else {
            0.0
        };
        let off = bench_path(&csc, &params_with(SmallLeafBatch::Off), &sym);
        let on = bench_path(&csc, &params_with(SmallLeafBatch::On), &sym);
        if off == 0 || on == 0 {
            println!("{label:>20}  factor FAILED in either config");
            continue;
        }
        let speedup = off as f64 / on as f64;
        speedups.push(speedup);
        if speedup >= 1.05 {
            wins += 1;
        } else if speedup <= 0.95 {
            losses += 1;
        }
        println!(
            "{label:>20}  {n_snodes:>6}  {n_groups:>6}  {avg:>5.1}  {off:>10}  {on:>10}  {speedup:>6.2}x"
        );
    }
    (speedups, wins, losses)
}

fn main() {
    println!("=== Issue #33 SmallLeafBatch A/B (1D-banded Mittelmann) ===");
    println!("min-of-{N_REPEAT} per config; warm-up uncounted; speedup = off/on");

    let mut all_speedups = Vec::new();
    let mut total_wins = 0usize;
    let mut total_losses = 0usize;

    for fam in FAMILIES {
        let (sp, w, l) = process_family(fam);
        if !sp.is_empty() {
            let mut sp_sorted = sp.clone();
            let med = median(&mut sp_sorted);
            let geo = geomean(&sp);
            let min = sp.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = sp.iter().cloned().fold(0.0f64, f64::max);
            println!(
                "  {fam} summary: n={} geomean={geo:.2}x median={med:.2}x min={min:.2}x max={max:.2}x  wins={w}  losses={l}",
                sp.len()
            );
        }
        all_speedups.extend(sp);
        total_wins += w;
        total_losses += l;
    }

    if !all_speedups.is_empty() {
        let mut sorted = all_speedups.clone();
        let med = median(&mut sorted);
        let geo = geomean(&all_speedups);
        let min = all_speedups.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = all_speedups.iter().cloned().fold(0.0f64, f64::max);
        println!("\n=== PANEL TOTAL (n={}) ===", all_speedups.len());
        println!("  geomean speedup: {geo:.3}x");
        println!("  median  speedup: {med:.3}x");
        println!("  min:             {min:.3}x");
        println!("  max:             {max:.3}x");
        println!("  wins >=1.05x:    {total_wins}");
        println!("  losses <=0.95x:  {total_losses}");

        println!(
            "\n#11 decision criterion (median >= 1.10x AND max regression <= +5%): {}",
            if med >= 1.10 && min >= 0.95 {
                "PASS — recommend flipping default"
            } else if med >= 1.10 {
                "MIXED — median passes but worst case regresses > 5%"
            } else {
                "FAIL — signal within noise, do not flip"
            }
        );
    }
}
