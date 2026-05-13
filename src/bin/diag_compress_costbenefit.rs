//! Phase 2.13c probe — cost/benefit of `LdltCompress` per matrix.
//!
//! For each matrix, runs the full symbolic + numeric factor twice:
//!   1. `OrderingPreprocess::None`        (no MC64, no compression)
//!   2. `OrderingPreprocess::LdltCompress` (current Auto choice for
//!      n >= 128 && low_degree/n >= 0.30)
//!
//! Reports the 5-run-median wall-clock of each phase for both
//! preprocesses. Negative `delta_us = LdltCompress - None` means
//! compression wins; positive means it loses (we want to gate it
//! out).
//!
//! The matrix list mixes:
//!   - Suspected losses (KIRBY2_0007, MUONSINE_0000) that motivated
//!     this phase.
//!   - Phase 2.12 known wins (LAKES, NELSON, SWOPF, ACOPR30,
//!     CRESC100) so the gate-tightening does not regress them.
//!
//! See `dev/plans/phase-2.13-tail-diagnostic.md` section 2.13c.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use feral::numeric::factorize::{
    factorize_multifrontal_with_workspace, FactorWorkspace, Profiler, SmallLeafBatch,
};
use feral::symbolic::{
    symbolic_factorize_with_method, OrderingMethod, OrderingPreprocess, SupernodeParams,
};
use feral::{read_mtx, CscMatrix};
use feral::{BunchKaufmanParams, NumericParams, ZeroPivotAction};

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
    ("LAKES_0000", "data/matrices/kkt/LAKES/LAKES_0000.mtx"),
    ("NELSON_0000", "data/matrices/kkt/NELSON/NELSON_0000.mtx"),
    ("SWOPF_0000", "data/matrices/kkt/SWOPF/SWOPF_0000.mtx"),
];

const N_RUNS: usize = 5;

#[derive(Clone, Copy, Default, Debug)]
struct Sample {
    symbolic_us: u64,
    numeric_us: u64,
    total_us: u64,
}

fn median_u64(xs: &mut [u64]) -> u64 {
    xs.sort_unstable();
    xs[xs.len() / 2]
}

fn med_field(samples: &[Sample], f: impl Fn(&Sample) -> u64) -> u64 {
    let mut v: Vec<u64> = samples.iter().map(f).collect();
    median_u64(&mut v)
}

fn ldlt_params() -> NumericParams {
    NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        },
        scaling: Default::default(),
        small_leaf: SmallLeafBatch::Off,
        profiler: Some(Arc::new(Mutex::new(Profiler::new()))),
        parallel_telemetry: None,
        fma: false,
        allow_delayed_pivots: true,
        cascade_break_ratio: None,
        cascade_break_eps: None,
    }
}

fn one_run(csc: &CscMatrix, branch: OrderingPreprocess) -> Option<Sample> {
    let snode = SupernodeParams {
        preprocess: branch,
        ..SupernodeParams::default()
    };

    let t = Instant::now();
    let sym = symbolic_factorize_with_method(csc, &snode, OrderingMethod::Amd).ok()?;
    let symbolic_us = t.elapsed().as_micros() as u64;

    let mut ws = FactorWorkspace::default();
    let p = ldlt_params();
    let t = Instant::now();
    factorize_multifrontal_with_workspace(csc, &sym, &p, &mut ws).ok()?;
    let numeric_us = t.elapsed().as_micros() as u64;

    Some(Sample {
        symbolic_us,
        numeric_us,
        total_us: symbolic_us + numeric_us,
    })
}

fn run_branch(csc: &CscMatrix, branch: OrderingPreprocess) -> Option<Sample> {
    // Warm-up factor: same pattern, primes any allocator caching.
    one_run(csc, branch)?;
    let mut samples: Vec<Sample> = Vec::with_capacity(N_RUNS);
    for _ in 0..N_RUNS {
        samples.push(one_run(csc, branch)?);
    }
    Some(Sample {
        symbolic_us: med_field(&samples, |s| s.symbolic_us),
        numeric_us: med_field(&samples, |s| s.numeric_us),
        total_us: med_field(&samples, |s| s.total_us),
    })
}

fn load_csc(path: &str) -> Option<CscMatrix> {
    if !Path::new(path).exists() {
        eprintln!("SKIP missing: {}", path);
        return None;
    }
    let mtx = read_mtx(Path::new(path)).ok()?;
    mtx.to_csc().ok()
}

fn main() {
    println!(
        "{:<16} {:>5} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>10} {:>9} {:>10}",
        "matrix",
        "n",
        "sym_n",
        "num_n",
        "tot_n",
        "sym_c",
        "num_c",
        "tot_c",
        "delta_tot",
        "delta_%",
        "verdict",
    );
    println!(
        "{:<16} {:>5} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>10} {:>9} {:>10}",
        "", "", "(med)", "(med)", "(med)", "(med)", "(med)", "(med)", "us", "of None", "",
    );
    for &(label, path) in MATRICES {
        let Some(m) = load_csc(path) else { continue };
        let n = m.n;
        let Some(none_s) = run_branch(&m, OrderingPreprocess::None) else {
            eprintln!("{}: None branch failed", label);
            continue;
        };
        let Some(comp_s) = run_branch(&m, OrderingPreprocess::LdltCompress) else {
            eprintln!("{}: LdltCompress branch failed", label);
            continue;
        };
        let delta = comp_s.total_us as i64 - none_s.total_us as i64;
        let delta_pct = 100.0 * delta as f64 / none_s.total_us.max(1) as f64;
        let verdict = if delta < 0 {
            "compress"
        } else if delta > (none_s.total_us as i64 / 20).max(2) {
            "NONE wins"
        } else {
            "neutral"
        };
        println!(
            "{:<16} {:>5} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>+10} {:>+8.1}% {:>10}",
            label,
            n,
            none_s.symbolic_us,
            none_s.numeric_us,
            none_s.total_us,
            comp_s.symbolic_us,
            comp_s.numeric_us,
            comp_s.total_us,
            delta,
            delta_pct,
            verdict,
        );
    }
    println!("\nLegend: sym = symbolic_factorize wall-time, num = numeric_factor");
    println!("        wall-time, tot = sym+num. _n = OrderingPreprocess::None,");
    println!("        _c = OrderingPreprocess::LdltCompress.");
    println!("        delta_tot < 0 -> LdltCompress wins; > 0 -> None wins.");
    println!("        verdict 'NONE wins' triggers when delta > max(5%, 2us).");
}
