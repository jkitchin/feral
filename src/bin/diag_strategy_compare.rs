//! Phase 2.12 Step F: end-to-end timing comparison of
//! `AmalgamationStrategy::{Adjacency, Renumber}` on the tiny-IPM
//! tail. 5-run median per strategy per matrix using the Phase 2.10
//! profiler for the supernode-loop accounting.

use std::path::Path;
use std::sync::{Arc, Mutex};

use feral::numeric::factorize::{
    factorize_multifrontal_with_workspace, FactorWorkspace, NumericParams, ProfileReport, Profiler,
    SmallLeafBatch,
};
use feral::symbolic::{
    symbolic_factorize_with_method, AmalgamationStrategy, OrderingMethod, SupernodeParams,
};
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

fn ldlt_params(profiler: Arc<Mutex<Profiler>>) -> NumericParams {
    NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        },
        scaling: Default::default(),
        small_leaf: SmallLeafBatch::Off,
        profiler: Some(profiler),
        parallel_telemetry: None,
        fma: false,
    }
}

fn run(
    label: &str,
    csc: &CscMatrix,
    strategy: AmalgamationStrategy,
) -> Option<(usize, ProfileReport)> {
    let snode_params = SupernodeParams {
        amalgamation_strategy: strategy,
        ..Default::default()
    };
    let sym = symbolic_factorize_with_method(csc, &snode_params, OrderingMethod::Amd).ok()?;
    let n_snodes = sym.supernodes.len();

    let mut runs: Vec<ProfileReport> = Vec::with_capacity(N_REPEAT);
    let mut ws = FactorWorkspace::default();
    {
        let prof = Arc::new(Mutex::new(Profiler::new()));
        let p = ldlt_params(Arc::clone(&prof));
        if factorize_multifrontal_with_workspace(csc, &sym, &p, &mut ws).is_err() {
            eprintln!("{}: warm-up factor failed", label);
            return None;
        }
    }
    for _ in 0..N_REPEAT {
        let prof = Arc::new(Mutex::new(Profiler::new()));
        let p = ldlt_params(Arc::clone(&prof));
        if factorize_multifrontal_with_workspace(csc, &sym, &p, &mut ws).is_err() {
            return None;
        }
        let r = prof.lock().ok()?.report();
        runs.push(r);
    }
    runs.sort_by_key(|r| r.total_us);
    Some((n_snodes, runs.into_iter().nth(N_REPEAT / 2)?))
}

fn main() {
    println!(
        "{:>16} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} | {:>8}",
        "matrix",
        "Adj snodes",
        "Adj total",
        "Adj loop",
        "Ren snodes",
        "Ren total",
        "Ren loop",
        "ratio"
    );
    println!("{:-<108}", "");
    for (label, path) in MATRICES {
        let csc = match load_csc(path) {
            Some(c) => c,
            None => continue,
        };
        let adj = run(label, &csc, AmalgamationStrategy::Adjacency);
        let ren = run(label, &csc, AmalgamationStrategy::Renumber);
        match (adj, ren) {
            (Some((a_n, a)), Some((r_n, r))) => {
                let ratio = r.total_us as f64 / a.total_us as f64;
                println!(
                    "{:>16} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} | {:>8.3}",
                    label, a_n, a.total_us, a.loop_us, r_n, r.total_us, r.loop_us, ratio
                );
            }
            _ => println!("{:>16} | run failed", label),
        }
    }
}
