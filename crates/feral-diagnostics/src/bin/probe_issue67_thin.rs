//! Issue #67 ordering A/B: on uniformly-thin large matrices (n>10_000, no
//! arrow signature) the `n>10_000 → MetisND` default loses to AMF on fill
//! (bratu3d 1.6×, cont-201 1.3×). The issue demands a corpus-wide A/B that
//! weighs numeric **factor+solve wall-time**, not nnz_L alone, and guards the
//! large-3-D / powerflow-class that #50 protects.
//!
//! For each matrix path given on the command line this factors under three
//! orderings via the full `Solver` path:
//!   - `Auto`     — records what production actually dispatches
//!     (`resolved_method`), so MetisND-routed non-arrow matrices are the
//!     in-scope population.
//!   - `Amf`      — the candidate.
//!   - `MetisND`  — the incumbent for n>10_000.
//!
//! It reports, per matrix: n, avg/max degree (symmetric pattern), the Auto
//! dispatch, post-pivot nnz_L for AMF and MetisND, and the median factor and
//! solve wall-times (RHS = ones). The MetisND/AMF ratios on nnz_L and on
//! factor+solve time are the decision signal.
//!
//! Run:
//!   cargo run --release --bin probe_issue67_thin -- [--reps K] <matrix.mtx>...
//!     --reps K   median of K factor/solve timings (default 3)
//!
//! Throwaway diagnostic, not a test.

use feral::symbolic::OrderingMethod;
use feral::{read_mtx, FactorStatus, Solver};
use std::time::Instant;

fn median_us(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

/// Factor `m` under `method` `reps` times; return
/// (median factor µs, median solve µs, post-pivot nnz_L, inertia string).
fn measure(
    m: &feral::sparse::csc::CscMatrix,
    method: OrderingMethod,
    reps: usize,
) -> Option<(u128, u128, usize, OrderingMethod, String)> {
    let rhs = vec![1.0f64; m.n];
    let mut factor_t = Vec::with_capacity(reps);
    let mut solve_t = Vec::with_capacity(reps);
    let mut nnz_l = 0usize;
    let mut resolved = method.clone();
    let mut inertia_s = String::new();
    for _ in 0..reps.max(1) {
        let mut s = Solver::new().with_ordering(method.clone());
        let t = Instant::now();
        let st = s.factor(m, None);
        factor_t.push(t.elapsed().as_micros());
        if !matches!(
            st,
            FactorStatus::Success | FactorStatus::WrongInertia { .. }
        ) {
            return None;
        }
        let f = s.factors()?;
        nnz_l = f.factor_nnz();
        resolved = f.resolved_method.clone();
        if let Some(i) = s.inertia() {
            inertia_s = format!("({},{},{})", i.positive, i.negative, i.zero);
        }
        let t = Instant::now();
        let _ = s.solve(&rhs);
        solve_t.push(t.elapsed().as_micros());
    }
    Some((
        median_us(factor_t),
        median_us(solve_t),
        nnz_l,
        resolved,
        inertia_s,
    ))
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut reps = 3usize;
    if let Some(p) = args.iter().position(|a| a == "--reps") {
        if let Some(v) = args.get(p + 1).and_then(|s| s.parse().ok()) {
            reps = v;
        }
        args.drain(p..=p + 1);
    }
    if args.is_empty() {
        eprintln!("usage: probe_issue67_thin [--reps K] <matrix.mtx>...");
        return;
    }

    println!(
        "{:<16} {:>8} {:>6} {:>6} {:>9} {:>11} {:>11} {:>7} {:>9} {:>9} {:>9} {:>9} {:>7}",
        "matrix",
        "n",
        "avg",
        "maxd",
        "auto",
        "nnzL_amf",
        "nnzL_met",
        "fill_r",
        "fac_amf",
        "fac_met",
        "slv_amf",
        "slv_met",
        "time_r",
    );

    for path in &args {
        let p = std::path::Path::new(path);
        let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let Ok(m) = read_mtx(p).and_then(|x| x.to_csc()) else {
            eprintln!("skip {name}: read/convert failed");
            continue;
        };
        let pat = m.symmetric_pattern();
        let n = pat.n;
        if n == 0 {
            continue;
        }
        let full_nnz = pat.row_idx.len();
        let avg = full_nnz as f64 / n as f64;
        let max_deg = (0..n)
            .map(|j| pat.col_ptr[j + 1] - pat.col_ptr[j])
            .max()
            .unwrap_or(0);

        let auto = measure(&m, OrderingMethod::Auto, 1);
        let amf = measure(&m, OrderingMethod::Amf, reps);
        let met = measure(&m, OrderingMethod::MetisND, reps);

        let auto_m = auto
            .as_ref()
            .map(|a| format!("{:?}", a.3))
            .unwrap_or_else(|| "FAIL".into());
        match (amf, met) {
            (Some(a), Some(mt)) => {
                let fill_r = mt.2 as f64 / a.2.max(1) as f64;
                let total_a = (a.0 + a.1) as f64;
                let total_m = (mt.0 + mt.1) as f64;
                let time_r = total_m / total_a.max(1.0);
                let inertia_flag = if a.4 == mt.4 { "" } else { " INERTIA-DIFF" };
                println!(
                    "{:<16} {:>8} {:>6.2} {:>6} {:>9} {:>11} {:>11} {:>7.2} {:>9} {:>9} {:>9} {:>9} {:>7.2}{}",
                    name, n, avg, max_deg, auto_m, a.2, mt.2, fill_r, a.0, mt.0, a.1, mt.1, time_r,
                    inertia_flag,
                );
            }
            (a, mt) => {
                eprintln!(
                    "skip {name}: amf={} metis={}",
                    if a.is_some() { "ok" } else { "FAIL" },
                    if mt.is_some() { "ok" } else { "FAIL" }
                );
            }
        }
    }
}
