//! Path-driven cost/benefit of `LdltCompress` (dense-column follow-up,
//! session 2026-06-06-04). Generalises `diag_compress_costbenefit` to
//! take matrix paths on the command line so the scaling-aware-gate
//! target bucket (large dense-column + non-Mc64 scaling matrices) can
//! be measured without editing the committed hard-coded list.
//!
//! For each matrix runs symbolic + numeric twice — `None` vs
//! `LdltCompress` — reporting the 5-run-median wall-clock of each
//! phase. Negative `delta_tot = LdltCompress - None` means compression
//! wins; large positive means the MC64 overhead dominates with no
//! numeric-fill payoff.
//!
//! Result (session 2026-06-06-04): the scaling-reuse signal does NOT
//! predict this verdict — ROSEPETAL (InfNorm, won't reuse MC64) wins
//! -75% via an 8x numeric speedup, while ORTHREGF (also InfNorm,
//! similar dense column) loses +90%. See
//! `dev/research/mc64-symbolic-skip-2026-06-06.md`.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin probe_compress_costbenefit_argv -- \
//!     data/matrices/kkt-expansion/INDEFM/INDEFM_0000.mtx ...

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
        ..NumericParams::default()
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
    one_run(csc, branch)?; // warm-up
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

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: probe_compress_costbenefit_argv <matrix.mtx> [...]");
        return;
    }
    println!(
        "{:<16} {:>7} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>10} {:>9} {:>10}",
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
    for path in &paths {
        if !Path::new(path).exists() {
            eprintln!("SKIP missing: {path}");
            continue;
        }
        let label = Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());
        let m = match read_mtx(Path::new(path)).and_then(|raw| raw.to_csc()) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("SKIP {label}: {e:?}");
                continue;
            }
        };
        let n = m.n;
        let Some(none_s) = run_branch(&m, OrderingPreprocess::None) else {
            eprintln!("{label}: None branch failed");
            continue;
        };
        let Some(comp_s) = run_branch(&m, OrderingPreprocess::LdltCompress) else {
            eprintln!("{label}: LdltCompress branch failed");
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
            "{:<16} {:>7} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>+10} {:>+8.1}% {:>10}",
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
}
