//! Phase 2.10 per-supernode profiler binary
//! (`dev/plans/phase-2.10-supernode-profiler.md`).
//!
//! For each curated tail-archetype matrix, runs the sequential
//! multifrontal driver `N_REPEAT` times with the profiler attached,
//! picks the run with median `total_us`, and prints a JSON report
//! containing:
//!   * front-size histogram bucketed by `nrow`
//!     (≤8, 9-16, 17-32, 33-64, 65-128, >128)
//!   * `prologue_us`, `epilogue_us`, `loop_us`, `total_us`
//!   * overall overhead percentage
//!
//! Usage: `cargo run --release --bin profile_supernode_distribution`
//!
//! Matrix selection: ACOPR30_0067 and CRESC100_0000 are the archetype
//! tiny-IPM tail matrices identified in
//! `dev/research/reference-solver-comparison.md`. LAKES_0000 +
//! NELSON_0000 + SWOPF_0000 are added to surface the *un*-characterized
//! slow families discovered in the 2026-04-25 four-agent corpus tail
//! analysis. The corpus is gitignored — missing matrices are reported
//! and skipped, so this binary runs cleanly on CI.

use std::path::Path;
use std::sync::{Arc, Mutex};

use feral::numeric::factorize::{
    factorize_multifrontal_with_workspace, FactorWorkspace, NumericParams, ProfileReport, Profiler,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, ZeroPivotAction};

const N_REPEAT: usize = 5;

const MATRICES: &[(&str, &str)] = &[
    ("ACOPR30_0067", "data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx"),
    (
        "CRESC100_0000",
        "data/matrices/kkt/CRESC100/CRESC100_0000.mtx",
    ),
    ("LAKES_0000", "data/matrices/kkt/LAKES/LAKES_0000.mtx"),
    ("NELSON_0000", "data/matrices/kkt/NELSON/NELSON_0000.mtx"),
    ("SWOPF_0000", "data/matrices/kkt/SWOPF/SWOPF_0000.mtx"),
];

fn load_csc(path: &str) -> Option<CscMatrix> {
    if !Path::new(path).exists() {
        eprintln!("SKIP: {} not present (corpus is gitignored)", path);
        return None;
    }
    let mtx = match read_mtx(Path::new(path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("read_mtx({}) failed: {}", path, e);
            return None;
        }
    };
    match mtx.to_csc() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("to_csc({}) failed: {}", path, e);
            None
        }
    }
}

fn ldlt_params(profiler: Arc<Mutex<Profiler>>) -> NumericParams {
    NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        },
        scaling: Default::default(),
        small_leaf: Default::default(),
        profiler: Some(profiler),
        parallel_telemetry: None,
    }
}

fn run_one(label: &str, csc: &CscMatrix) -> Option<ProfileReport> {
    let sym = match symbolic_factorize(csc, &SupernodeParams::default()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: symbolic_factorize failed: {}", label, e);
            return None;
        }
    };

    let mut runs: Vec<ProfileReport> = Vec::with_capacity(N_REPEAT);
    let mut ws = FactorWorkspace::default();

    // Warm-up to pre-allocate workspace and stabilize caches.
    {
        let prof = Arc::new(Mutex::new(Profiler::new()));
        let params = ldlt_params(Arc::clone(&prof));
        if let Err(e) = factorize_multifrontal_with_workspace(csc, &sym, &params, &mut ws) {
            eprintln!("{}: warm-up factor failed: {}", label, e);
            return None;
        }
    }

    for _ in 0..N_REPEAT {
        let prof = Arc::new(Mutex::new(Profiler::new()));
        let params = ldlt_params(Arc::clone(&prof));
        match factorize_multifrontal_with_workspace(csc, &sym, &params, &mut ws) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("{}: factor failed: {}", label, e);
                return None;
            }
        }
        let report = match prof.lock() {
            Ok(p) => p.report(),
            Err(_) => {
                eprintln!("{}: profiler mutex poisoned", label);
                return None;
            }
        };
        runs.push(report);
    }

    runs.sort_by_key(|r| r.total_us);
    Some(runs.into_iter().nth(N_REPEAT / 2).expect("median exists"))
}

fn main() {
    let mut any = false;
    for (label, path) in MATRICES {
        let csc = match load_csc(path) {
            Some(c) => c,
            None => continue,
        };
        let report = match run_one(label, &csc) {
            Some(r) => r,
            None => continue,
        };
        let json = match serde_json::to_string_pretty(&report) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{}: json serialize failed: {}", label, e);
                continue;
            }
        };
        println!(
            "=== {} (n={}, snodes={}) ===",
            label, csc.n, report.n_supernodes
        );
        println!("{}", json);
        any = true;
    }
    if !any {
        eprintln!("No matrices ran. Place corpus under data/matrices/kkt/.");
    }
}
