//! Quantify the fixed per-supernode cost in the numeric phase.
//!
//! Two experiments:
//!
//! (A) Per-matrix scan: for a set of long-tail and bulk matrices,
//!     report supernode count, size distribution, and numeric time.
//!     Key ratio: numeric_us / num_supernodes = average cost per front.
//!
//! (B) nemin sweep on one hot matrix (ACOPR30_0067): vary the
//!     amalgamation threshold and measure how the supernode count
//!     and numeric time change. If bigger fronts → fewer supernodes
//!     → much faster, the bottleneck is per-front overhead.

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams};

fn median_u128(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn median_usize(mut v: Vec<usize>) -> usize {
    v.sort_unstable();
    v[v.len() / 2]
}

struct MatStat {
    name: String,
    n: usize,
    factor_nnz: usize,
    nsup: usize,
    sup_med: usize,
    sup_max: usize,
    num_us: u128,
    per_sup_ns: u128,
    per_nnz_ns: u128,
}

fn measure(path: &Path, params: &SupernodeParams, reps: usize) -> Option<MatStat> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<?>")
        .to_string();
    let csc = read_mtx(path).ok()?.to_csc().ok()?;
    let n = csc.n;

    let sym = symbolic_factorize(&csc, params).ok()?;
    let factor_nnz = sym.col_counts.iter().sum::<usize>();
    let nsup = sym.supernodes.len();
    let sup_sizes: Vec<usize> = sym.supernodes.iter().map(|s| s.nrow).collect();
    let sup_med = median_usize(sup_sizes.clone());
    let sup_max = *sup_sizes.iter().max().unwrap_or(&0);

    let np = NumericParams::with_bk(BunchKaufmanParams::default());
    let mut timings = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let _ = factorize_multifrontal(&csc, &sym, &np).ok()?;
        timings.push(t.elapsed().as_nanos());
    }
    let num_ns = median_u128(timings);
    let num_us = num_ns / 1000;
    let per_sup_ns = num_ns / nsup.max(1) as u128;
    let per_nnz_ns = num_ns / factor_nnz.max(1) as u128;

    Some(MatStat {
        name,
        n,
        factor_nnz,
        nsup,
        sup_med,
        sup_max,
        num_us,
        per_sup_ns,
        per_nnz_ns,
    })
}

fn header() {
    println!(
        "{:28} {:>5} {:>8} {:>5} {:>5} {:>5} {:>7} {:>8} {:>7}",
        "matrix", "n", "fact_nnz", "nsup", "med", "max", "num_us", "ns/sup", "ns/nnz",
    );
}

fn print(s: &MatStat) {
    println!(
        "{:28} {:>5} {:>8} {:>5} {:>5} {:>5} {:>7} {:>8} {:>7}",
        s.name,
        s.n,
        s.factor_nnz,
        s.nsup,
        s.sup_med,
        s.sup_max,
        s.num_us,
        s.per_sup_ns,
        s.per_nnz_ns,
    );
}

fn main() {
    println!("=== (A) per-matrix supernode cost scan ===");
    println!("ns/sup = average nanoseconds per supernode (fixed-cost proxy)");
    println!("ns/nnz = average nanoseconds per factor nonzero (work-cost proxy)");
    println!();
    header();

    let params = SupernodeParams::default();
    let targets = [
        // long-tail
        "data/matrices/kkt/CRESC100/CRESC100_0000.mtx",
        "data/matrices/kkt/ACOPR30/ACOPR30_0185.mtx",
        "data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx",
        "data/matrices/kkt/HAIFAM/HAIFAM_0082.mtx",
        "data/matrices/kkt/HAHN1/HAHN1_0049.mtx",
        "data/matrices/kkt/GAUSS2/GAUSS2_0029.mtx",
        // bulk, where feral wins
        "data/matrices/kkt/HS118/HS118_0001.mtx",
        "data/matrices/kkt/HS92/HS92_0001.mtx",
        "data/matrices/kkt/ALLINITC/ALLINITC_0223.mtx",
        "data/matrices/kkt/HATFLDH/HATFLDH_0100.mtx",
        // medium
        "data/matrices/kkt/KIRBY2/KIRBY2_0007.mtx",
        "data/matrices/kkt/AVION2/AVION2_0000.mtx",
    ];
    for p in targets {
        if let Some(s) = measure(Path::new(p), &params, 20) {
            print(&s);
        }
    }

    println!();
    println!("=== (B) nemin sweep on ACOPR30_0067 ===");
    println!("Bigger nemin → fewer, bigger supernodes. If ns/sup holds roughly");
    println!("constant while num_us falls, the fixed per-front overhead is the");
    println!("bottleneck and aggressive amalgamation would close the gap.");
    println!();
    header();

    let p = Path::new("data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx");
    for nemin in [1usize, 4, 8, 16, 32, 64, 128, 256, 512] {
        let params = SupernodeParams {
            nemin,
            ..SupernodeParams::default()
        };
        if let Some(mut s) = measure(p, &params, 30) {
            s.name = format!("ACOPR30_0067 nemin={}", nemin);
            print(&s);
        }
    }
}
