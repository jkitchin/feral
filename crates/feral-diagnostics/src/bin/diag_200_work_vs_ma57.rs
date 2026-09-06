//! Issue #200 — normalise the feral-vs-MA57 wallclock gap by the work
//! each solver actually performs.
//!
//! Every prior investigation profiled feral in isolation, which can only
//! locate feral's time, never explain why there is more of it than
//! MA57's. A wallclock ratio is only an overhead statement if both
//! solvers do the same arithmetic. This reports feral's `nnz(L)` and its
//! elimination flop count so they can be put beside MA57's `INFO(14)`
//! and `RINFO(4)` from `external_benchmarks/ma57_oracle`.
//!
//! Flops are counted from the symbolic supernode shapes with the same
//! model MA57's `RINFO(4)` uses for a symmetric indefinite elimination:
//! a front of `ncol` eliminated columns and height `nrow` costs
//! `sum_{k<ncol} (nrow-k)^2` multiply-adds. Counting from the symbolic
//! structure (not instrumenting the kernels) keeps this free of the
//! probe distortion documented in `diag_200_probe_tax`.
use feral::symbolic::{supernode::SupernodeParams, symbolic_factorize};
use feral::{read_mtx, NumericParams, Solver};
use std::time::Instant;

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
    println!(
        "{:<22}{:>8}{:>12}{:>14}{:>10}{:>12}{:>10}",
        "matrix", "snodes", "nnz_L", "flops_elim", "med nrow", "factor us", "MFlop/s"
    );
    for p in &paths {
        let csc = read_mtx(std::path::Path::new(p)).and_then(|m| m.to_csc())?;
        let sp = SupernodeParams::default();
        let sym = symbolic_factorize(&csc, &sp)?;
        let mut flops: f64 = 0.0;
        let mut nnz_l_sym: u64 = 0;
        let mut rows: Vec<usize> = Vec::with_capacity(sym.supernodes.len());
        for s in &sym.supernodes {
            let (nrow, ncol) = (s.nrow, s.ncol);
            rows.push(nrow);
            for k in 0..ncol {
                let h = nrow - k;
                flops += (h * h) as f64;
            }
            // Stored L entries: the trapezoid below the diagonal block.
            nnz_l_sym += (ncol * (ncol + 1) / 2 + ncol * (nrow - ncol)) as u64;
        }
        rows.sort_unstable();
        let med = rows.get(rows.len() / 2).copied().unwrap_or(0);

        let mut solver = Solver::with_params(NumericParams::default(), sp);
        for _ in 0..3 {
            let _ = solver.factor(&csc, None);
        }
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let _ = solver.factor(&csc, None);
            best = best.min(t.elapsed().as_secs_f64() * 1e6);
        }
        let nnz_l = solver
            .last_factor_stats()
            .map(|s| s.nnz_l as u64)
            .unwrap_or(nnz_l_sym);
        let name = std::path::Path::new(p)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        println!(
            "{:<22}{:>8}{:>12}{:>14.4e}{:>10}{:>12.0}{:>10.0}",
            name,
            sym.supernodes.len(),
            nnz_l,
            flops,
            med,
            best,
            flops / best
        );
    }
    Ok(())
}
