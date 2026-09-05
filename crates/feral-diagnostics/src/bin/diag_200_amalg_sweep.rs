//! Issue #200 — is the small-front regime a *tuning* choice or a
//! *failure* of amalgamation?
//!
//! `SupernodeParams::nemin` already defaults to 32 (SSIDS's value), yet
//! on IPM-KKT matrices the median frontal matrix is 3x3 and 60-86% of
//! fronts have `nrow <= 8`. Every per-front cost issue #200 is about
//! (malloc traffic, the L/D and contribution extract copies, the driver's
//! per-supernode bookkeeping) is paid once per front, so front size is
//! the lever that moves all of them at once.
//!
//! Two candidate explanations:
//!   (a) the size rule `child_ncol < nemin && parent_ncol < nemin` is
//!       binding -- then raising `nemin` grows fronts;
//!   (b) the column-adjacency precondition
//!       (`s_first + s_ncol != p_first => continue`) rejects the merges
//!       before the size rule is ever consulted -- then `nemin` does
//!       nothing and the fix is structural.
//!
//! This sweeps `nemin` under both `AmalgamationStrategy` values and
//! reports, per configuration: supernode count, median/max front `nrow`,
//! the fraction of tiny fronts, uninstrumented factor time (min of
//! `reps`, warm solver, phase timing OFF), and the scaled residual so a
//! speedup is never reported without its accuracy cost.
//!
//! Accuracy context: lowering `nemin` (4, 8) and the `merge_flop_budget`
//! guard were both rejected on 2026-08-09 for costing up to seven digits
//! of residual (HATFLDG 7.1e-15 -> 7.4e-08, VESUVIOU_0030 1.9e-06 ->
//! 5.4e-03). Both levers *reduce* merging. This sweep goes the other
//! way, so it is not a re-run of that experiment -- but the residual
//! column is mandatory regardless.
use feral::symbolic::supernode::{AmalgamationStrategy, SupernodeParams};
use feral::{read_mtx, CscMatrix, NumericParams, Solver};
use std::time::Instant;

fn residual_inf(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut r = b.to_vec();
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let v = a.values[k];
            r[i] -= v * x[j];
            if i != j {
                r[j] -= v * x[i];
            }
        }
    }
    let rn = r.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let bn = b.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
    rn / bn
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut reps = 7usize;
    let mut paths: Vec<String> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--reps" {
            if let Some(v) = it.next() {
                reps = v.parse()?;
            }
        } else {
            paths.push(a.clone());
        }
    }
    if paths.is_empty() {
        eprintln!("usage: diag_200_amalg_sweep [--reps N] <matrix.mtx>...");
        std::process::exit(2);
    }

    let configs: Vec<(&str, usize, AmalgamationStrategy)> = vec![
        (
            "renumber n=32 (DEFAULT)",
            32,
            AmalgamationStrategy::Renumber,
        ),
        ("renumber n=64", 64, AmalgamationStrategy::Renumber),
        ("renumber n=128", 128, AmalgamationStrategy::Renumber),
        ("renumber n=512", 512, AmalgamationStrategy::Renumber),
        ("adjacency n=32", 32, AmalgamationStrategy::Adjacency),
        ("adjacency n=512", 512, AmalgamationStrategy::Adjacency),
    ];

    for p in &paths {
        let csc = read_mtx(std::path::Path::new(p)).and_then(|m| m.to_csc())?;
        let n = csc.n;
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
        let name = std::path::Path::new(p)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.clone());
        println!("=== {name}  (n={n}) ===");
        println!(
            "{:<26}{:>9}{:>8}{:>8}{:>9}{:>11}{:>8}{:>12}",
            "config", "snodes", "med", "max", "%nrow<=8", "factor us", "ratio", "residual"
        );
        let mut base_us = 0.0_f64;
        for (label, nemin, strat) in &configs {
            let sp = SupernodeParams {
                nemin: *nemin,
                amalgamation_strategy: *strat,
                ..SupernodeParams::default()
            };
            let sym = match feral::symbolic::symbolic_factorize(&csc, &sp) {
                Ok(s) => s,
                Err(e) => {
                    println!("{label:<26}  symbolic failed: {e}");
                    continue;
                }
            };
            let mut rows: Vec<usize> = sym.supernodes.iter().map(|s| s.nrow).collect();
            rows.sort_unstable();
            let ns = rows.len();
            let med = if ns == 0 { 0 } else { rows[ns / 2] };
            let mx = rows.last().copied().unwrap_or(0);
            let le8 = rows.iter().filter(|&&r| r <= 8).count();

            let mut solver = Solver::with_params(NumericParams::default(), sp.clone());
            // Three warm-up calls: the permute-value-map cache is not
            // built until call 2 and not read until call 3.
            for _ in 0..3 {
                let _ = solver.factor(&csc, None);
            }
            let mut best = f64::MAX;
            for _ in 0..reps {
                let t = Instant::now();
                let _ = solver.factor(&csc, None);
                best = best.min(t.elapsed().as_secs_f64() * 1e6);
            }
            let res = match solver.solve(&b) {
                Ok(x) => residual_inf(&csc, &x, &b),
                Err(_) => f64::NAN,
            };
            if *nemin == 32 && matches!(strat, AmalgamationStrategy::Renumber) {
                base_us = best;
            }
            println!(
                "{:<26}{:>9}{:>8}{:>8}{:>8.1}%{:>11.0}{:>8.2}{:>12.2e}",
                label,
                ns,
                med,
                mx,
                100.0 * le8 as f64 / ns.max(1) as f64,
                best,
                if base_us > 0.0 { best / base_us } else { 1.0 },
                res
            );
        }
        println!();
    }
    Ok(())
}
