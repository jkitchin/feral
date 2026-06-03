//! Hot-loop profile target for samply.
//!
//! Loops over a curated mix of matrices that span the regimes seen in
//! `profile_sparse`:
//!   - tiny (HS118, ALLINITC) where overhead dominates
//!   - mid (BATCH, AVION2, HAHN1) where feral is 5-18x slower than MUMPS
//!   - large (VESUVIO, CRESC132) where dense kernels are exercised
//!
//! Runs each through the full symbolic + numeric + refined-solve
//! pipeline `N_REPS` times so samply collects enough samples in the
//! hot paths. Designed to be invoked as:
//!   samply record target/release/profile_hot

use feral::numeric::factorize::factorize_multifrontal;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, solve_sparse_refined, BunchKaufmanParams, ZeroPivotAction};
use std::path::PathBuf;

const N_REPS: usize = 200;

const MATRICES: &[(&str, &str)] = &[
    ("HS118_0000", "data/matrices/kkt/HS118/HS118_0000.mtx"),
    ("BATCH_0000", "data/matrices/kkt/BATCH/BATCH_0000.mtx"),
    ("BATCH_0500", "data/matrices/kkt/BATCH/BATCH_0500.mtx"),
    ("AVION2_0000", "data/matrices/kkt/AVION2/AVION2_0000.mtx"),
    ("HAHN1_0000", "data/matrices/kkt/HAHN1/HAHN1_0000.mtx"),
    ("VESUVIO_0000", "data/matrices/kkt/VESUVIO/VESUVIO_0000.mtx"),
    (
        "CRESC132_0000",
        "data/matrices/kkt/CRESC132/CRESC132_0000.mtx",
    ),
];

fn main() {
    let snode_params = SupernodeParams::default();
    let bk = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    let factor_params = feral::numeric::factorize::NumericParams::with_bk(bk);

    // Pre-load matrices to keep the hot loop focused on solver code.
    let mut loaded: Vec<(String, feral::CscMatrix)> = Vec::new();
    for (name, path) in MATRICES {
        let p = PathBuf::from(path);
        match read_mtx(&p) {
            Ok(mtx) => match mtx.to_csc() {
                Ok(csc) => loaded.push((name.to_string(), csc)),
                Err(e) => eprintln!("SKIP {}: csc {}", name, e),
            },
            Err(e) => eprintln!("SKIP {}: read {}", name, e),
        }
    }
    println!(
        "loaded {} matrices, looping {} reps each",
        loaded.len(),
        N_REPS
    );

    let mut total_factor_us = 0u128;
    let mut total_sym_us = 0u128;
    let mut total_solve_us = 0u128;

    for rep in 0..N_REPS {
        for (name, csc) in &loaded {
            let n = csc.n;
            let rhs = vec![1.0_f64; n];

            let t = std::time::Instant::now();
            let sym = match symbolic_factorize(csc, &snode_params) {
                Ok(s) => s,
                Err(e) => {
                    if rep == 0 {
                        eprintln!("symbolic FAIL {}: {}", name, e);
                    }
                    continue;
                }
            };
            total_sym_us += t.elapsed().as_micros();

            let t = std::time::Instant::now();
            let factors = match factorize_multifrontal(csc, &sym, &factor_params) {
                Ok((f, _)) => f,
                Err(e) => {
                    if rep == 0 {
                        eprintln!("factor FAIL {}: {}", name, e);
                    }
                    continue;
                }
            };
            total_factor_us += t.elapsed().as_micros();

            let t = std::time::Instant::now();
            let _ = solve_sparse_refined(csc, &factors, &rhs);
            total_solve_us += t.elapsed().as_micros();
        }
    }

    println!(
        "totals over {} reps: sym={}us factor={}us solve_refined={}us",
        N_REPS, total_sym_us, total_factor_us, total_solve_us
    );
    let total = total_sym_us + total_factor_us + total_solve_us;
    if total > 0 {
        println!(
            "  fractions: sym={:.1}%  factor={:.1}%  solve={:.1}%",
            100.0 * total_sym_us as f64 / total as f64,
            100.0 * total_factor_us as f64 / total as f64,
            100.0 * total_solve_us as f64 / total as f64,
        );
    }
}
