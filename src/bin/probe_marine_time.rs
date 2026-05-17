//! Time feral factor on marine_1600 KKT dumps under different scaling strategies.
//!
//! Marine_1600 is the largest outstanding feral/MA57 wall regression
//! (587x on the May-16 ipopt bench, 470s vs 0.8s). IPM trajectory is
//! bit-identical between the two solvers (13 iters, same objective at
//! every step), so the gap is pure per-factor cost. This probe times
//! a single fresh-Solver factor on each of the 5 dumped KKTs under
//! Auto / InfNorm / Mc64Symmetric to localize where the cost lives.

use feral::scaling::ScalingStrategy;
use feral::symbolic::supernode::SupernodeParams;
use feral::{read_mtx, NumericParams, Solver};
use std::path::Path;
use std::time::Instant;

fn run(label: &str, csc: &feral::CscMatrix, scaling: ScalingStrategy) {
    let params = NumericParams {
        scaling: scaling.clone(),
        ..NumericParams::default()
    };
    let mut solver = Solver::with_params(params, SupernodeParams::default());
    let t1 = Instant::now();
    let st1 = solver.factor(csc, None);
    let dt1 = t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    let st2 = solver.factor(csc, None);
    let dt2 = t2.elapsed().as_secs_f64();
    println!(
        "  {} scaling={:?}: factor #1={:.3}s ({:?}) #2={:.3}s ({:?})",
        label, scaling, dt1, st1, dt2, st2
    );
}

fn main() {
    let kkts: Vec<(usize, feral::CscMatrix)> = (0..5)
        .filter_map(|i| {
            let p = format!(
                "data/matrices/kkt-mittelmann/marine_1600/marine_1600_{:04}.mtx",
                i
            );
            let path = Path::new(&p);
            if !path.exists() {
                eprintln!("SKIP {}", p);
                return None;
            }
            let csc = read_mtx(path).ok()?.to_csc().ok()?;
            Some((i, csc))
        })
        .collect();

    println!("\n=== FRESH solver per iter (one Solver per factor) ===");
    for (i, csc) in &kkts {
        println!("-- iter {} (n={}, nnz={}) --", i, csc.n, csc.row_idx.len());
        run("auto    ", csc, ScalingStrategy::Auto);
        run("infnorm ", csc, ScalingStrategy::InfNorm);
        run("mc64sym ", csc, ScalingStrategy::Mc64Symmetric);
    }

    println!("\n=== WARM solver across iters (one Solver, sequential factors) ===");
    for (cb_label, cb_ratio) in [("cb=off", None), ("cb=on(0.5)", Some(0.5))] {
        for sc in [
            ScalingStrategy::Auto,
            ScalingStrategy::InfNorm,
            ScalingStrategy::Mc64Symmetric,
        ] {
            let params = NumericParams {
                scaling: sc.clone(),
                ..NumericParams::default()
            };
            let mut solver = Solver::with_params(params, SupernodeParams::default());
            if let Some(r) = cb_ratio {
                solver = solver.with_cascade_break(r).with_cascade_break_eps(1e-10);
            }
            println!("-- warm {} scaling={:?} --", cb_label, sc);
            for (i, csc) in &kkts {
                let t = Instant::now();
                let st = solver.factor(csc, None);
                let dt = t.elapsed().as_secs_f64();
                println!("  iter {}: factor={:.3}s status={:?}", i, dt, st);
            }
        }
    }
}
