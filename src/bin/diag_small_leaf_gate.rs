//! Phase 2.11 Step A.5: measure the impact of `SmallLeafBatch::On`
//! on the tiny-IPM tail matrices. The default is `Off` (per
//! `factorize.rs:48-63`); Phase 2.10's profiler binary measured the
//! generic path. This binary measures both for direct comparison.

use std::path::Path;
use std::sync::{Arc, Mutex};

use feral::numeric::factorize::{
    factorize_multifrontal_with_workspace, FactorWorkspace, NumericParams, ProfileReport, Profiler,
    SmallLeafBatch,
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
        eprintln!("SKIP: {}", path);
        return None;
    }
    let mtx = read_mtx(Path::new(path)).ok()?;
    mtx.to_csc().ok()
}

fn ldlt_params(gate: SmallLeafBatch, profiler: Arc<Mutex<Profiler>>) -> NumericParams {
    NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        },
        scaling: Default::default(),
        small_leaf: gate,
        profiler: Some(profiler),
        parallel_telemetry: None,
    }
}

fn run(label: &str, csc: &CscMatrix, gate: SmallLeafBatch) -> Option<ProfileReport> {
    let sym = symbolic_factorize(csc, &SupernodeParams::default()).ok()?;
    let mut runs: Vec<ProfileReport> = Vec::with_capacity(N_REPEAT);
    let mut ws = FactorWorkspace::default();
    {
        let prof = Arc::new(Mutex::new(Profiler::new()));
        let p = ldlt_params(gate, Arc::clone(&prof));
        if factorize_multifrontal_with_workspace(csc, &sym, &p, &mut ws).is_err() {
            eprintln!("{}: factor failed", label);
            return None;
        }
    }
    for _ in 0..N_REPEAT {
        let prof = Arc::new(Mutex::new(Profiler::new()));
        let p = ldlt_params(gate, Arc::clone(&prof));
        if factorize_multifrontal_with_workspace(csc, &sym, &p, &mut ws).is_err() {
            return None;
        }
        let r = prof.lock().ok()?.report();
        runs.push(r);
    }
    runs.sort_by_key(|r| r.total_us);
    runs.into_iter().nth(N_REPEAT / 2)
}

fn main() {
    println!(
        "{:>16} | {:>10} {:>10} | {:>10} {:>10} | {:>8}",
        "matrix", "Off total", "Off loop", "On total", "On loop", "ratio"
    );
    println!("{:-<88}", "");
    for (label, path) in MATRICES {
        let csc = match load_csc(path) {
            Some(c) => c,
            None => continue,
        };
        let off = run(label, &csc, SmallLeafBatch::Off);
        let on = run(label, &csc, SmallLeafBatch::On);
        match (off, on) {
            (Some(o), Some(n)) => {
                let ratio = n.total_us as f64 / o.total_us as f64;
                println!(
                    "{:>16} | {:>10} {:>10} | {:>10} {:>10} | {:>8.3}",
                    label, o.total_us, o.loop_us, n.total_us, n.loop_us, ratio
                );
            }
            _ => println!("{:>16} | run failed", label),
        }
    }
}
