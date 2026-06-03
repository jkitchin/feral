//! Phase 2.4.4 — profile each step of the compression pipeline to
//! find the bottleneck that makes symbolic_factorize + LdltCompress
//! cost ~2x the uncompressed path.
//!
//! The compression path (src/symbolic/mod.rs:294-310) is:
//!   (a) crate::scaling::mc64_matching(matrix) — Hungarian
//!   (b) build_supermap(&matching)
//!   (c) compress_pattern(&full_pattern, &map)
//!   (d) run_external_ordering(&cpat, method) on the compressed graph
//!   (e) expand_permutation(&super_perm, &map)
//!
//! The uncompressed path only runs (d) on the full pattern. So the
//! compression overhead is (a) + (b) + (c) + (e) + delta on (d).
//!
//! This binary reports each stage in microseconds, across a few
//! representative matrices from small (KKT easy path), medium
//! (typical corpus), and tail (HAHN1/CRESC100/GAUSS2).

use std::path::Path;
use std::time::Instant;

use feral::read_mtx;
use feral::symbolic::{build_supermap, compress_pattern, expand_permutation};

fn mean_median(v: &[u128]) -> (u128, u128) {
    if v.is_empty() {
        return (0, 0);
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    let mean: u128 = v.iter().sum::<u128>() / (v.len() as u128);
    (mean, s[s.len() / 2])
}

fn profile_matrix(path: &Path, reps: usize) {
    let Ok(mtx) = read_mtx(path) else {
        println!("  SKIP — read_mtx failed");
        return;
    };
    let Ok(csc) = mtx.to_csc() else {
        println!("  SKIP — to_csc failed");
        return;
    };
    let n = csc.n;
    if n == 0 {
        println!("  SKIP — empty matrix");
        return;
    }

    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("<?>");

    // Pre-compute the full-symmetric pattern once; every run reuses it.
    let full_pattern = csc.symmetric_pattern();

    let mut t_mc64 = Vec::with_capacity(reps);
    let mut t_supermap = Vec::with_capacity(reps);
    let mut t_compress = Vec::with_capacity(reps);
    let mut t_expand = Vec::with_capacity(reps);
    let mut ncmp = 0usize;
    let mut nfull = 0usize;

    for _ in 0..reps {
        let t = Instant::now();
        let (matching, _) = match feral::scaling::mc64_matching(&csc) {
            Ok(m) => m,
            Err(_) => {
                println!("  SKIP — mc64_matching failed");
                return;
            }
        };
        t_mc64.push(t.elapsed().as_micros());

        let t = Instant::now();
        let map = build_supermap(&matching);
        t_supermap.push(t.elapsed().as_micros());

        ncmp = map.ncmp();
        nfull = n;

        let t = Instant::now();
        let cpat = compress_pattern(&full_pattern, &map);
        t_compress.push(t.elapsed().as_micros());

        // Fake super_perm = identity on ncmp so expand runs real work.
        let super_perm: Vec<usize> = (0..cpat.n).collect();
        let t = Instant::now();
        let _ = expand_permutation(&super_perm, &map);
        t_expand.push(t.elapsed().as_micros());
    }

    let (mc64_mean, mc64_med) = mean_median(&t_mc64);
    let (sm_mean, sm_med) = mean_median(&t_supermap);
    let (cp_mean, cp_med) = mean_median(&t_compress);
    let (ex_mean, ex_med) = mean_median(&t_expand);

    let total_med = mc64_med + sm_med + cp_med + ex_med;

    println!(
        "{:26} n={:>5} ncmp={:>5} ({:4.1}%) | mc64 {:>5} | smap {:>4} | comp {:>4} | exp {:>4} | total_med {:>6} μs",
        name,
        nfull,
        ncmp,
        (ncmp as f64 / nfull.max(1) as f64) * 100.0,
        mc64_med,
        sm_med,
        cp_med,
        ex_med,
        total_med,
    );
    let _ = (mc64_mean, sm_mean, cp_mean, ex_mean);
}

fn main() {
    println!("=== Phase 2.4.4 compression-pipeline profile ===");
    println!("Times are medians over N reps. Per step: MC64 (Hungarian), supermap build,");
    println!("pattern contraction, permutation expand. Lower is better.");
    println!();

    let targets: &[(&str, &str, &str)] = &[
        // tail matrices (compression SHOULD pay off)
        ("HAHN1", "HAHN1_0153", "tail"),
        ("HAHN1", "HAHN1_0404", "tail"),
        ("GAUSS2", "GAUSS2_0029", "tail"),
        ("GAUSS2", "GAUSS2_0035", "tail"),
        ("CRESC100", "CRESC100_0000", "tail"),
        ("MUONSINE", "MUONSINE_0000", "tail"),
        ("VESUVIO", "VESUVIO_0011", "tail"),
        // medium matrices (where the geomean penalty shows up)
        ("ACOPR30", "ACOPR30_0131", "medium"),
        ("KIRBY2", "KIRBY2_0007", "medium"),
        ("HS118", "HS118_0001", "medium"),
        // small matrices (bulk of the corpus, dominant in geomean)
        ("HS92", "HS92_0001", "small"),
        ("SSI", "SSI_0106", "small"),
        ("PALMER5A", "PALMER5A_0720", "small"),
        ("ALLINITC", "ALLINITC_0223", "small"),
        ("HATFLDH", "HATFLDH_0100", "small"),
    ];

    println!("-- TAIL --");
    for (fam, stem, _class) in targets.iter().filter(|(_, _, c)| *c == "tail") {
        let path = format!("data/matrices/kkt/{}/{}.mtx", fam, stem);
        print!("");
        profile_matrix(Path::new(&path), 5);
    }
    println!();
    println!("-- MEDIUM --");
    for (fam, stem, _class) in targets.iter().filter(|(_, _, c)| *c == "medium") {
        let path = format!("data/matrices/kkt/{}/{}.mtx", fam, stem);
        profile_matrix(Path::new(&path), 10);
    }
    println!();
    println!("-- SMALL --");
    for (fam, stem, _class) in targets.iter().filter(|(_, _, c)| *c == "small") {
        let path = format!("data/matrices/kkt/{}/{}.mtx", fam, stem);
        profile_matrix(Path::new(&path), 20);
    }
}
