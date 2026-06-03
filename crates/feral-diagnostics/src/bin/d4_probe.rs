//! D.4 stage-1 probe — compare pre-D.4 vs post-D.4 wall time on the
//! six tiny-n top-10 rows observed in the 2026-04-20-01 bench run.
//!
//! Pre-D.4  = force the multifrontal path via
//!            `factorize_multifrontal_supernodal` (gate bypass).
//! Post-D.4 = `factorize_multifrontal` dispatcher, which now routes
//!            tiny-n matrices to `dense_fast_factor` under the
//!            broadened gate.
//!
//! For each matrix we report the per-phase wall time using 50 cold
//! reps (min, p50) to avoid the single-shot noise that produced the
//! spurious HS85_0022 "80× regression" in the stage-3 D.3 bench.
//! Spec: `dev/plans/sparse-tail-d4.md` §Measurement plan.

use feral::numeric::factorize::{
    factorize_multifrontal, factorize_multifrontal_supernodal, should_use_dense_fast_path,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, NumericParams, ZeroPivotAction};
use std::path::PathBuf;
use std::time::Instant;

const COLD_REPS: usize = 50;

struct Target {
    family: &'static str,
    sample: &'static str,
}

const TARGETS: &[Target] = &[
    Target {
        family: "HS73",
        sample: "_0308",
    },
    Target {
        family: "PALMER1E",
        sample: "_0484",
    },
    Target {
        family: "HATFLDH",
        sample: "_0083",
    },
    Target {
        family: "PALMER1A",
        sample: "_0034",
    },
    Target {
        family: "KIRBY2LS",
        sample: "_0274",
    },
    Target {
        family: "HEART6LS",
        sample: "_0418",
    },
];

fn params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    })
}

fn load(t: &Target) -> CscMatrix {
    let path = PathBuf::from(format!(
        "data/matrices/kkt/{}/{}{}.mtx",
        t.family, t.family, t.sample
    ));
    let mtx = read_mtx(&path).expect("read_mtx");
    mtx.to_csc().expect("to_csc")
}

fn cold_summary<F: FnMut() -> u128>(reps: usize, mut f: F) -> (u128, u128) {
    let mut samples: Vec<u128> = (0..reps).map(|_| f()).collect();
    samples.sort_unstable();
    (samples[0], samples[reps / 2])
}

fn main() {
    println!(
        "D.4 stage-1 probe — pre/post wall time on tiny-n targets ({} cold reps each)",
        COLD_REPS
    );
    println!();
    println!(
        "{:<20} {:>4} {:>5} {:>6} {:>8} | {:>10} {:>10} | {:>10} {:>10} | {:>8}",
        "name", "n", "nnz", "rho", "gate", "pre_min", "pre_p50", "post_min", "post_p50", "p50_x"
    );
    println!("{}", "-".repeat(120));

    let p = params();
    let sn = SupernodeParams::default();

    for t in TARGETS {
        let csc = load(t);
        let n = csc.n;
        let nnz_lower = csc.row_idx.len();
        let rho = nnz_lower as f64 / (n * (n + 1) / 2) as f64;
        let gate = should_use_dense_fast_path(n, nnz_lower);

        // Pre-D.4: bypass entry point. Includes symbolic + numeric cost
        // because the bench harness measures both together.
        let (pre_min, pre_p50) = cold_summary(COLD_REPS, || {
            let t0 = Instant::now();
            let sym = symbolic_factorize(&csc, &sn).expect("symbolic");
            let _ = factorize_multifrontal_supernodal(&csc, &sym, &p).expect("multi");
            t0.elapsed().as_nanos()
        });

        // Post-D.4: dispatcher. On a gate hit this skips symbolic
        // entirely via dense_fast_factor. We still produce a symbolic
        // outside the timed region for API shape (the dispatcher
        // takes a &SymbolicFactorization), but on a gate hit the
        // dispatcher ignores it — so the timed region mirrors what
        // the bench harness does (sym+numeric inside the Instant
        // block). To make the comparison honest, we also include
        // symbolic in the post path even though it's discarded on
        // gate-hit routes.
        let (post_min, post_p50) = cold_summary(COLD_REPS, || {
            let t0 = Instant::now();
            let sym = symbolic_factorize(&csc, &sn).expect("symbolic");
            let _ = factorize_multifrontal(&csc, &sym, &p).expect("dispatch");
            t0.elapsed().as_nanos()
        });

        let ratio = pre_p50 as f64 / post_p50 as f64;
        let name = format!("{}{}", t.family, t.sample);
        println!(
            "{:<20} {:>4} {:>5} {:>6.3} {:>8} | {:>7.2}us {:>7.2}us | {:>7.2}us {:>7.2}us | {:>7.2}x",
            name,
            n,
            nnz_lower,
            rho,
            if gate { "DENSE" } else { "MULTI" },
            pre_min as f64 / 1000.0,
            pre_p50 as f64 / 1000.0,
            post_min as f64 / 1000.0,
            post_p50 as f64 / 1000.0,
            ratio
        );
    }
    println!();
    println!("Legend: pre_* = forced multifrontal (bypass); post_* = gated dispatcher.");
    println!("        p50_x = pre_p50 / post_p50 — speedup at the p50 of the cold distribution.");
}
