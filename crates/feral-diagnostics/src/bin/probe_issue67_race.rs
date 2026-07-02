//! Issue #67 AutoRace overhead measurement.
//!
//! An `AutoRace(Amf, MetisND)` for the band would run *both* candidates'
//! symbolic analyses, pick the smaller-fill one, then factor+solve once on
//! the winner. Its overhead over the bounded-threshold approach (run only
//! AMF's symbolic) is exactly the **extra losing candidate's symbolic time**.
//!
//! This probe times, per matrix, with `--reps` median:
//!   t_sym_amf    — symbolic analysis under AMF (the winner on the band)
//!   t_sym_met    — symbolic analysis under MetisND (the extra race cost)
//!   t_num_amf    — numeric factorization on the AMF winner
//!   t_solve_amf  — one solve (RHS = ones) on the AMF winner
//! and reports the race overhead as a fraction of the bounded-approach total
//! (t_sym_amf + t_num_amf + t_solve_amf):
//!   overhead_% = 100 * t_sym_met / (t_sym_amf + t_num_amf + t_solve_amf)
//!
//! Run: cargo run --release --bin probe_issue67_race -- [--reps K] <m.mtx>...
//! Throwaway diagnostic, not a test.

use feral::numeric::factorize::{factorize_multifrontal_parallel_with_workspace, FactorWorkspace};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, NumericParams};
use std::time::Instant;

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn sym_us(
    m: &feral::sparse::csc::CscMatrix,
    sp: &SupernodeParams,
    method: OrderingMethod,
    reps: usize,
) -> Option<u128> {
    let mut t = Vec::with_capacity(reps);
    for _ in 0..reps.max(1) {
        let t0 = Instant::now();
        symbolic_factorize_with_method(m, sp, method.clone()).ok()?;
        t.push(t0.elapsed().as_micros());
    }
    Some(median(t))
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
        eprintln!("usage: probe_issue67_race [--reps K] <matrix.mtx>...");
        return;
    }

    let sp = SupernodeParams::default();
    let params = NumericParams::default();

    println!(
        "{:<16} {:>8} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "matrix", "n", "sym_amf", "sym_met", "num_amf", "slv_amf", "race_ovh%"
    );

    for path in &args {
        let p = std::path::Path::new(path);
        let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let Ok(m) = read_mtx(p).and_then(|x| x.to_csc()) else {
            eprintln!("skip {name}: read/convert failed");
            continue;
        };
        let rhs = vec![1.0f64; m.n];

        let Some(t_sym_amf) = sym_us(&m, &sp, OrderingMethod::Amf, reps) else {
            eprintln!("skip {name}: amf symbolic failed");
            continue;
        };
        let Some(t_sym_met) = sym_us(&m, &sp, OrderingMethod::MetisND, reps) else {
            eprintln!("skip {name}: metis symbolic failed");
            continue;
        };

        // Numeric + solve on the AMF winner.
        let mut num_t = Vec::with_capacity(reps);
        let mut slv_t = Vec::with_capacity(reps);
        for _ in 0..reps.max(1) {
            let Ok(sym) = symbolic_factorize_with_method(&m, &sp, OrderingMethod::Amf) else {
                continue;
            };
            let mut ws = FactorWorkspace::new();
            let t0 = Instant::now();
            let Ok((factors, _inertia)) =
                factorize_multifrontal_parallel_with_workspace(&m, &sym, &params, &mut ws)
            else {
                continue;
            };
            num_t.push(t0.elapsed().as_micros());
            let t0 = Instant::now();
            let _ = solve_sparse_refined(&m, &factors, &rhs);
            slv_t.push(t0.elapsed().as_micros());
        }
        if num_t.is_empty() {
            eprintln!("skip {name}: numeric failed");
            continue;
        }
        let t_num = median(num_t);
        let t_slv = median(slv_t);
        let bounded_total = (t_sym_amf + t_num + t_slv).max(1);
        let ovh = 100.0 * t_sym_met as f64 / bounded_total as f64;
        println!(
            "{:<16} {:>8} {:>9} {:>9} {:>9} {:>9} {:>10.1}",
            name, m.n, t_sym_amf, t_sym_met, t_num, t_slv, ovh
        );
    }
}
