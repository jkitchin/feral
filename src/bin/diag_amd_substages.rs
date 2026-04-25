//! Phase 2.13b step 5 — AMD per-call sub-stage probe.
//!
//! KIRBY2_0007 spends 770µs (85.5% of symbolic) in the `ordering`
//! stage. That stage is `run_external_ordering` in
//! `src/symbolic/mod.rs`, which decomposes into:
//!
//!   1. `to_contract_pattern_bufs` — i32 conversion of CSC buffers
//!   2. `feral_ordering_core::CscPattern::new` — validation
//!   3. `feral_amd::amd_order(...)` — the AMD call itself, which
//!      itself decomposes into `AmdWorkspace::new`,
//!      `run_elimination`, and `finalize_permutation`
//!   4. usize conversion of the returned i32 perm
//!
//! This probe times all four boundaries (the AMD call broken down
//! via [`feral_amd::amd_order_substages`]) on the small-n tail
//! matrices and reports the 5-run median per sub-stage. The
//! breakdown decides Phase 2.13's open question: does symbolic
//! caching pay (allocation/setup heavy → reusing the analyze cuts
//! the dominant stage outright) or does AMD per-call shrink pay
//! (algorithmic work heavy → optimizing the inner loop helps)?
//!
//! See `dev/plans/phase-2.13-tail-diagnostic.md` step 5.

use std::path::Path;
use std::time::Instant;

use feral::scaling::mc64_matching;
use feral::symbolic::{
    build_supermap, compress_pattern, expand_permutation, pick_ordering_preprocess,
    OrderingPreprocess,
};
use feral::{read_mtx, CscMatrix};

const MATRICES: &[(&str, &str)] = &[
    ("KIRBY2_0007", "data/matrices/kkt/KIRBY2/KIRBY2_0007.mtx"),
    (
        "MUONSINE_0000",
        "data/matrices/kkt/MUONSINE/MUONSINE_0000.mtx",
    ),
    ("ACOPR30_0067", "data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx"),
    (
        "CRESC100_0000",
        "data/matrices/kkt/CRESC100/CRESC100_0000.mtx",
    ),
];

const N_RUNS: usize = 5;

#[derive(Clone, Copy, Default, Debug)]
struct Sample {
    // Common AMD sub-stages.
    prep_us: u64, // i32 conversion + CscPattern::new
    workspace_new_us: u64,
    run_elimination_us: u64,
    finalize_permutation_us: u64,
    post_us: u64, // usize conversion of returned perm
    // LdltCompress-only sub-stages. Zero on the `None` path.
    mc64_us: u64,
    supermap_us: u64,
    compress_us: u64,
    expand_us: u64,
    total_us: u64,
}

fn median_u64(xs: &mut [u64]) -> u64 {
    xs.sort_unstable();
    xs[xs.len() / 2]
}

fn load_csc(path: &str) -> Option<CscMatrix> {
    if !Path::new(path).exists() {
        eprintln!("SKIP missing: {}", path);
        return None;
    }
    let mtx = read_mtx(Path::new(path)).ok()?;
    mtx.to_csc().ok()
}

/// Time the AMD call itself (i32 prep + 3 AMD sub-stages + post).
/// `pat_to_order` is the pattern actually fed to AMD — for the
/// `None` branch it is the full pattern; for `LdltCompress` it is
/// the compressed pattern.
fn time_amd_call(pat_to_order: &feral::sparse::csc::CscPattern) -> (Sample, Vec<usize>) {
    let mut s = Sample::default();

    let t = Instant::now();
    let col_buf: Vec<i32> = pat_to_order.col_ptr.iter().map(|&x| x as i32).collect();
    let row_buf: Vec<i32> = pat_to_order.row_idx.iter().map(|&x| x as i32).collect();
    let core_pat = feral_ordering_core::CscPattern::new(pat_to_order.n, &col_buf, &row_buf)
        .expect("malformed CSC");
    s.prep_us = t.elapsed().as_micros() as u64;

    let opts = feral_amd::AmdOptions::default();
    let (perm_i32, sub) = feral_amd::amd_order_substages(&core_pat, &opts).expect("amd failed");
    s.workspace_new_us = sub.workspace_new_us;
    s.run_elimination_us = sub.run_elimination_us;
    s.finalize_permutation_us = sub.finalize_permutation_us;

    let t = Instant::now();
    let perm: Vec<usize> = perm_i32.into_iter().map(|x| x as usize).collect();
    s.post_us = t.elapsed().as_micros() as u64;

    (s, perm)
}

fn one_run(m: &CscMatrix, branch: OrderingPreprocess) -> Sample {
    // The full-symmetric pattern is what `run_external_ordering` is
    // handed in `src/symbolic/mod.rs:380`. Building it is part of
    // the *upstream* `symmetric_pattern` stage, NOT the `ordering`
    // stage, so we exclude it from the timing.
    let full_pat = m.symmetric_pattern();

    let t_total = Instant::now();
    let mut s = match branch {
        OrderingPreprocess::None | OrderingPreprocess::Auto => {
            let (sample, _) = time_amd_call(&full_pat);
            sample
        }
        OrderingPreprocess::LdltCompress => {
            // Mirror the LdltCompress branch in
            // `symbolic_factorize_with_method` lines 401-420.
            let mut s = Sample::default();

            let t = Instant::now();
            let (matching, _) = mc64_matching(m).expect("mc64 failed");
            s.mc64_us = t.elapsed().as_micros() as u64;

            let t = Instant::now();
            let map = build_supermap(&matching);
            s.supermap_us = t.elapsed().as_micros() as u64;

            if map.ncmp() == m.n {
                // Same fall-through as the production code path.
                let (amd_s, _) = time_amd_call(&full_pat);
                s.prep_us = amd_s.prep_us;
                s.workspace_new_us = amd_s.workspace_new_us;
                s.run_elimination_us = amd_s.run_elimination_us;
                s.finalize_permutation_us = amd_s.finalize_permutation_us;
                s.post_us = amd_s.post_us;
            } else {
                let t = Instant::now();
                let cpat = compress_pattern(&full_pat, &map);
                s.compress_us = t.elapsed().as_micros() as u64;

                let (amd_s, super_perm) = time_amd_call(&cpat);
                s.prep_us = amd_s.prep_us;
                s.workspace_new_us = amd_s.workspace_new_us;
                s.run_elimination_us = amd_s.run_elimination_us;
                s.finalize_permutation_us = amd_s.finalize_permutation_us;
                s.post_us = amd_s.post_us;

                let t = Instant::now();
                let _ = expand_permutation(&super_perm, &map);
                s.expand_us = t.elapsed().as_micros() as u64;
            }
            s
        }
    };
    s.total_us = t_total.elapsed().as_micros() as u64;
    s
}

fn med_field(samples: &[Sample], f: impl Fn(&Sample) -> u64) -> u64 {
    let mut v: Vec<u64> = samples.iter().map(f).collect();
    median_u64(&mut v)
}

fn main() {
    println!(
        "{:<16} {:>5} {:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "matrix",
        "n",
        "branch",
        "mc64",
        "smap",
        "compr",
        "prep",
        "ws_new",
        "run_el",
        "fin_p",
        "expand",
        "post",
        "total",
    );
    for &(label, path) in MATRICES {
        let Some(m) = load_csc(path) else { continue };
        let n = m.n;
        let branch = pick_ordering_preprocess(&m);
        let branch_str = match branch {
            OrderingPreprocess::None => "None",
            OrderingPreprocess::LdltCompress => "LdltCompress",
            OrderingPreprocess::Auto => "Auto",
        };
        let samples: Vec<Sample> = (0..N_RUNS).map(|_| one_run(&m, branch)).collect();
        println!(
            "{:<16} {:>5} {:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            label,
            n,
            branch_str,
            med_field(&samples, |s| s.mc64_us),
            med_field(&samples, |s| s.supermap_us),
            med_field(&samples, |s| s.compress_us),
            med_field(&samples, |s| s.prep_us),
            med_field(&samples, |s| s.workspace_new_us),
            med_field(&samples, |s| s.run_elimination_us),
            med_field(&samples, |s| s.finalize_permutation_us),
            med_field(&samples, |s| s.expand_us),
            med_field(&samples, |s| s.post_us),
            med_field(&samples, |s| s.total_us),
        );
    }
    println!("\nLegend (all µs, 5-run median):");
    println!("  branch  = pick_ordering_preprocess output for the matrix");
    println!("  mc64    = MC64 symmetric matching (LdltCompress only)");
    println!("  smap    = build_supermap        (LdltCompress only)");
    println!("  compr   = compress_pattern       (LdltCompress only)");
    println!("  prep    = i32 buffer conversion + CscPattern::new");
    println!("  ws_new  = AmdWorkspace::new (alloc + initial degree lists)");
    println!("  run_el  = AMD main pivot/eliminate/finalize_step loop");
    println!("  fin_p   = AMD postorder + permutation emission");
    println!("  expand  = expand_permutation     (LdltCompress only)");
    println!("  post    = i32 -> usize conversion of returned perm");
    println!("  total   = wall-clock for the whole ordering branch");
}
