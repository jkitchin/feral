//! probe_mc64_hungarian — localize where the MC64 Hungarian time goes.
//!
//! Step 3 of the scaling audit. The rocket_12800 symbolic `ldlt_compress`
//! stage is the MC64 Hungarian (38 s, n=89601, max_col_degree=38401).
//! This probe reports the algorithmic work counters
//! (`augment_searches`, `touched_total`, `heap_init_slots`,
//! `phase3_inner_iters`, `main_loop_edge_scans`) so we can tell whether
//! the cost is dense-column edge scans, heap work, or phase-3 — and a
//! synthetic dense-coupling-column ladder to fit the exponent in
//! `max_col_degree`.
//!
//! Usage:
//!   cargo run -p feral-diagnostics --bin probe_mc64_hungarian -- \
//!       --manifest list.txt          # lines: mtx_path
//!   cargo run -p feral-diagnostics --bin probe_mc64_hungarian -- \
//!       --dense-ladder 2000,4000,8000,16000,32000

use feral::scaling::diagnose_mc64_matching;
use feral::{read_mtx, CscMatrix};
use std::path::Path;
use std::time::Instant;

fn report(label: &str, csc: &CscMatrix) {
    let t = Instant::now();
    match diagnose_mc64_matching(csc) {
        Ok(s) => {
            let ms = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "{label:<22} n={:>7} cost_nnz={:>9} max_deg={:>7} | \
                 searches={:>7} touched={:>10} heap_init={:>10} phase3={:>9} \
                 edge_scans={:>12} | {:>9.1} ms",
                s.n,
                s.cost_nnz,
                s.max_col_degree,
                s.augment_searches,
                s.touched_total,
                s.heap_init_slots,
                s.phase3_inner_iters,
                s.main_loop_edge_scans,
                ms,
            );
        }
        Err(e) => println!("{label:<22} ERROR {e:?}"),
    }
}

/// Deterministic LCG (no rng dependency).
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() >> 33) as usize % bound.max(1)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Bordered random-sparse SPD: a full-range random body (degree 3 per
/// column, which defeats the greedy init so the main augmenting loop
/// runs Θ(n) searches — the regime that matters) plus ONE near-dense
/// coupling column (column 0 connects to rows `0..dense_deg`),
/// mirroring rocket_12800's structure. With `dense_deg = n/2`, the max
/// column degree scales with n; if the MC64 main-loop cost is
/// O(searches * dense_deg) = O(n^2), `main_loop_edge_scans` grows
/// quadratically across the ladder. Lower triangle only.
fn gen_bordered(n: usize, _band: usize, dense_deg: usize, seed: u64) -> Option<CscMatrix> {
    let mut rng = Lcg(seed);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut absrowsum = vec![0.0f64; n];
    // Dense coupling column 0: rows 1..=dense_deg.
    let dd = dense_deg.min(n.saturating_sub(1));
    for r in 1..=dd {
        let v = (rng.unit() - 0.5) * 2.0;
        rows.push(r);
        cols.push(0);
        vals.push(v);
        absrowsum[r] += v.abs();
        absrowsum[0] += v.abs();
    }
    // Random-sparse body for columns 1..n (full-range r > c so greedy
    // init leaves a constant fraction of columns unmatched).
    for c in 1..n {
        if c + 1 >= n {
            break;
        }
        for _ in 0..3 {
            let r = c + 1 + rng.below(n - c - 1);
            let v = (rng.unit() - 0.5) * 2.0;
            rows.push(r);
            cols.push(c);
            vals.push(v);
            absrowsum[r] += v.abs();
            absrowsum[c] += v.abs();
        }
    }
    for (i, &s) in absrowsum.iter().enumerate() {
        rows.push(i);
        cols.push(i);
        vals.push(s + 1.0);
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).ok()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let mut manifest: Option<String> = None;
    let mut ladder: Vec<usize> = Vec::new();
    while i < args.len() {
        match args[i].as_str() {
            "--manifest" => {
                i += 1;
                manifest = args.get(i).cloned();
            }
            "--dense-ladder" => {
                i += 1;
                if let Some(list) = args.get(i) {
                    ladder = list
                        .split(',')
                        .filter_map(|x| x.trim().parse().ok())
                        .collect();
                }
            }
            other => eprintln!("unknown arg '{other}', ignoring"),
        }
        i += 1;
    }

    if let Some(mf) = &manifest {
        let text = match std::fs::read_to_string(mf) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("cannot read manifest {mf}: {e}");
                return;
            }
        };
        for line in text.lines() {
            let p = line.split_whitespace().next().unwrap_or("");
            if p.is_empty() || p.starts_with('#') {
                continue;
            }
            let path = Path::new(p);
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.to_string());
            match read_mtx(path).ok().and_then(|m| m.to_csc().ok()) {
                Some(csc) => report(&name, &csc),
                None => eprintln!("skip {p}: load failed"),
            }
        }
    }

    if !ladder.is_empty() {
        println!("\n=== dense-coupling-column ladder (dense_deg = n/2, band=16) ===");
        for &n in &ladder {
            if let Some(csc) = gen_bordered(n, 16, n / 2, 0xBEEF) {
                report(&format!("bordered_{n}"), &csc);
            }
        }
    }

    if manifest.is_none() && ladder.is_empty() {
        eprintln!(
            "usage: probe_mc64_hungarian --manifest FILE | \
             --dense-ladder 2000,4000,8000,16000"
        );
    }
}
