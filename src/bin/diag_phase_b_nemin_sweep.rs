//! Phase B sweep — `nemin ∈ {8, 16, 24, 32, 48}` on shape-stratified
//! fixtures to decide whether shape-dispatched `nemin` is worth
//! implementing inside `AmalgamationStrategy::Auto`. See
//! `dev/research/phase-b-shape-dispatched-nemin.md`.
//!
//! Run: `cargo run --release --bin diag_phase_b_nemin_sweep`

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, ZeroPivotAction};

const FIXTURES: &[(&str, &str, &str)] = &[
    (
        "MUONSINE",
        "data/matrices/kkt/MUONSINE/MUONSINE_0000.mtx",
        "path-like",
    ),
    (
        "KIRBY2",
        "data/matrices/kkt/KIRBY2/KIRBY2_0007.mtx",
        "bushy",
    ),
    (
        "ACOPR30",
        "data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx",
        "bushy-mid",
    ),
    (
        "SWOPF",
        "data/matrices/kkt/SWOPF/SWOPF_0000.mtx",
        "bushy-mid",
    ),
];

const NEMINS: &[usize] = &[8, 16, 24, 32, 48];
const REPS: usize = 3;

fn main() {
    println!(
        "{:<10} {:<10} {:>5} {:>10} {:>14} {:>10}",
        "fixture", "shape", "nemin", "factor_nnz", "factor_med_us", "ratio_vs_16"
    );

    for (label, path, shape) in FIXTURES {
        let mtx = match read_mtx(Path::new(path)) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("skip {} (read): {:?}", label, e);
                continue;
            }
        };
        let m = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {} (csc): {:?}", label, e);
                continue;
            }
        };

        let mut nnz_at_16: Option<usize> = None;
        let mut us_at_16: Option<u128> = None;
        let mut rows: Vec<(usize, usize, u128)> = Vec::new();

        for &nemin in NEMINS {
            let snode_params = SupernodeParams {
                nemin,
                ..Default::default()
            };
            let sym = match symbolic_factorize_with_method(&m, &snode_params, OrderingMethod::Amd) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("{} nemin={}: symbolic err {}", label, nemin, e);
                    continue;
                }
            };

            let nparams = NumericParams {
                bk: BunchKaufmanParams {
                    pivot_threshold: 1e-8,
                    on_zero_pivot: ZeroPivotAction::ForceAccept,
                    zero_tol: 1e-10,
                    ..Default::default()
                },
                scaling: ScalingStrategy::Identity,
                ..Default::default()
            };

            let mut nnz = 0;
            let mut times: Vec<u128> = Vec::new();
            for _ in 0..REPS {
                let t = Instant::now();
                let r = factorize_multifrontal(&m, &sym, &nparams);
                let us = t.elapsed().as_micros();
                match r {
                    Ok((factors, _)) => {
                        nnz = factors.factor_nnz();
                        times.push(us);
                    }
                    Err(e) => {
                        eprintln!("{} nemin={}: numeric err {}", label, nemin, e);
                        break;
                    }
                }
            }
            if times.is_empty() {
                continue;
            }
            times.sort_unstable();
            let med = times[times.len() / 2];
            rows.push((nemin, nnz, med));
            if nemin == 16 {
                nnz_at_16 = Some(nnz);
                us_at_16 = Some(med);
            }
        }

        for (nemin, nnz, med) in &rows {
            let ratio = match (nnz_at_16, us_at_16) {
                (Some(n0), Some(t0)) if n0 > 0 && t0 > 0 => format!(
                    "nnz={:.2}× t={:.2}×",
                    *nnz as f64 / n0 as f64,
                    *med as f64 / t0 as f64
                ),
                _ => "—".to_string(),
            };
            println!(
                "{:<10} {:<10} {:>5} {:>10} {:>14} {:>20}",
                label, shape, nemin, nnz, med, ratio
            );
        }
        println!();
    }
}
