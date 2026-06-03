//! Probe — does the MC64 / scaling cache hit across *genuinely
//! different* IPM iterates, or only when the same matrix is re-fed?
//!
//! Issues #44 / #49. A real interior-point method refactors a KKT
//! whose sparsity pattern is fixed but whose numeric *values* change
//! every iterate. The value-bounded MC64 scaling cache (B2, `c990def`)
//! reuses a cached scaling only while the new values stay within a
//! bound of the cached ones. Feeding the *same* matrix repeatedly
//! (as `probe_issue49`/`probe_explicit_zeros` do) trivially satisfies
//! that bound — it does NOT model the IPM.
//!
//! This probe factors a *sequence* of distinct consecutive iterate
//! matrices on ONE warm `Solver` and reports, per call, whether the
//! symbolic and MC64 caches engaged. A flat `mc64_cache_hits` across
//! distinct iterates means the Hungarian re-ran — the cache missed.
//!
//! Diagnostic probe; relaxed probe-bin convention (unwrap/expect).
//!
//! Usage: `cargo run --release --bin probe_cache_sequence -- A.mtx B.mtx C.mtx ...`

use std::env;
use std::path::Path;
use std::time::Instant;

use feral::scaling::pick_scaling_strategy;
use feral::{read_mtx, CscMatrix, Solver};

fn load(path: &str) -> CscMatrix {
    read_mtx(Path::new(path))
        .expect("read_mtx")
        .to_csc()
        .expect("to_csc")
}

/// Factor every matrix in `seq` in order on one warm `Solver`.
fn run_sequence(label: &str, seq: &[(String, CscMatrix)]) {
    println!("== {label}: {} iterates, one warm Solver ==", seq.len());
    let mut s = Solver::new();
    let mut prev_hits = 0usize;
    let mut prev_fallbacks = 0usize;
    for (k, (name, csc)) in seq.iter().enumerate() {
        let t = Instant::now();
        let status = s.factor(csc, None);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let hits = s.mc64_cache_hit_count();
        let fallbacks = s.mc64_fallback_count();
        let d_hits = hits - prev_hits;
        let d_fb = fallbacks - prev_fallbacks;
        prev_hits = hits;
        prev_fallbacks = fallbacks;
        let scaling = s
            .scaling_info()
            .map(|si| format!("{si:?}"))
            .unwrap_or_else(|| "-".to_string());
        let inertia = s
            .inertia()
            .map(|i| format!("({},{},{})", i.positive, i.negative, i.zero))
            .unwrap_or_else(|| "-".to_string());
        let cache = if d_hits > 0 {
            "CACHE HIT "
        } else {
            "cache MISS"
        };
        println!(
            "  call {k} [{name}] factor_ms={ms:>9.1}  {cache}  \
             symbolic_calls={}  mc64_hits={hits}(+{d_hits})  \
             mc64_fallbacks={fallbacks}(+{d_fb})  scaling={scaling}  \
             inertia={inertia}  {status:?}",
            s.symbolic_call_count(),
        );
    }
    println!();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} A.mtx B.mtx C.mtx ...", args[0]);
        std::process::exit(2);
    }
    let paths: Vec<String> = args[1..].to_vec();
    let seq: Vec<(String, CscMatrix)> = paths
        .iter()
        .map(|p| {
            let name = Path::new(p)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.clone());
            (name, load(p))
        })
        .collect();

    let (_, first) = &seq[0];
    println!(
        "n={}  route={:?}  ({} distinct iterate matrices)\n",
        first.n,
        pick_scaling_strategy(first),
        seq.len()
    );

    // Control: same matrix re-fed N times — the cache SHOULD hit.
    let same: Vec<(String, CscMatrix)> = (0..seq.len())
        .map(|_| (seq[0].0.clone(), seq[0].1.clone()))
        .collect();
    run_sequence("CONTROL — iterate 0 re-fed (identical values)", &same);

    // Real IPM pattern: distinct consecutive iterates.
    run_sequence("IPM PATTERN — distinct consecutive iterates", &seq);
}
