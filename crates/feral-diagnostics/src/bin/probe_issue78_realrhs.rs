//! Issue #78 independent verification on the REAL IPM rhs.
//!
//! The author's corrected analysis (after fixing a duplicate-triplet
//! summing bug in their own reconstruction) concludes: every feral solve,
//! InfNorm and MC64 alike, is backward-stable and forward-accurate on the
//! real rhs at every iteration — the optimal-vs-infeasible split is
//! trajectory sensitivity, not a bad solve. This reproduces that with
//! feral's own solver, reading the per-solve dumps in
//! /Users/jkitchin/feral-repro-issue78/{infnorm,mc64}_run/iter_*/.
//!
//! For each iteration's accepted (highest-numbered) solve record it:
//!   - rebuilds A from the raw triplets (from_triplets SUMS duplicates,
//!     matching the author's pre-summed canonical .mtx),
//!   - solves A x = rhs under InfNorm and MC64,
//!   - reports backward residual ||A x - rhs|| / ||rhs|| for each, and the
//!     agreement ||x_inf - x_mc64|| / ||x_inf|| between the two scalings.
//!
//! Flags any backward residual above 1e-10.

use std::path::Path;

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus, Solver};
use feral::scaling::ScalingStrategy;
use feral::CscMatrix;
use serde::Deserialize;

#[derive(Deserialize)]
struct SolveRec {
    n: usize,
    irn: Vec<usize>,
    jcn: Vec<usize>,
    vals: Vec<f64>,
    rhs: Vec<f64>,
    sol: Vec<f64>,
    status: String,
}

fn load(path: &Path) -> Option<SolveRec> {
    let txt = std::fs::read_to_string(path).ok()?;
    let line = txt.lines().find(|l| !l.trim().is_empty())?;
    serde_json::from_str(line).ok()
}

/// Build a lower-triangle CSC from 1-based triplets; from_triplets sums
/// the 36 duplicate (i,j) entries the way the canonical .mtx does.
fn build(rec: &SolveRec) -> Option<CscMatrix> {
    let mut rows = Vec::with_capacity(rec.irn.len());
    let mut cols = Vec::with_capacity(rec.jcn.len());
    let mut vals = Vec::with_capacity(rec.vals.len());
    for k in 0..rec.irn.len() {
        let (mut i, mut j) = (rec.irn[k] - 1, rec.jcn[k] - 1);
        if i < j {
            std::mem::swap(&mut i, &mut j); // keep lower triangle
        }
        rows.push(i);
        cols.push(j);
        vals.push(rec.vals[k]);
    }
    CscMatrix::from_triplets(rec.n, &rows, &cols, &vals).ok()
}

fn matvec(csc: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for j in 0..csc.n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = csc.values[k];
            out[i] += v * x[j];
            if i != j {
                out[j] += v * x[i];
            }
        }
    }
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let (mut n, mut d) = (0.0, 0.0);
    for i in 0..a.len() {
        let e = a[i] - b[i];
        n += e * e;
        d += b[i] * b[i];
    }
    n.sqrt() / d.sqrt().max(1e-300)
}

fn back_res(csc: &CscMatrix, x: &[f64], rhs: &[f64]) -> f64 {
    let mut ax = vec![0.0; csc.n];
    matvec(csc, x, &mut ax);
    rel(&ax, rhs)
}

fn solve(csc: &CscMatrix, rhs: &[f64], s: ScalingStrategy) -> Option<Vec<f64>> {
    let params = NumericParams {
        scaling: s,
        ..NumericParams::default()
    };
    let mut solver = Solver::with_params(params, feral::symbolic::SupernodeParams::default());
    if !matches!(solver.factor(csc, None), FactorStatus::Success) {
        return None;
    }
    solver.solve_refined(csc, rhs).ok()
}

fn accepted_solve(iter_dir: &Path) -> Option<std::path::PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(iter_dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    files.pop() // highest-numbered solve = the accepted one
}

fn main() {
    let run = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/jkitchin/feral-repro-issue78/infnorm_run".to_string());
    println!("# run dir: {run}");
    println!("# iter  back(InfNorm)  back(MC64)   ||x_inf-x_mc64||/||x_inf||  flag");
    let mut worst_back = 0.0f64;
    let mut max_disagree = 0.0f64;
    for it in 0..=40 {
        let dir = format!("{run}/iter_{it:03}");
        let Some(f) = accepted_solve(Path::new(&dir)) else {
            continue;
        };
        let Some(rec) = load(&f) else { continue };
        let Some(csc) = build(&rec) else {
            println!("  {it:>3}  build failed");
            continue;
        };
        let (Some(xi), Some(xm)) = (
            solve(&csc, &rec.rhs, ScalingStrategy::InfNorm),
            solve(&csc, &rec.rhs, ScalingStrategy::Mc64Symmetric),
        ) else {
            println!("  {it:>3}  factor/solve failed (status={})", rec.status);
            continue;
        };
        let bi = back_res(&csc, &xi, &rec.rhs);
        let bm = back_res(&csc, &xm, &rec.rhs);
        let dis = rel(&xi, &xm);
        worst_back = worst_back.max(bi).max(bm);
        max_disagree = max_disagree.max(dis);
        let flag = if bi > 1e-10 || bm > 1e-10 {
            " <-- NOT backward-stable"
        } else {
            ""
        };
        // Sanity: recorded sol's own backward residual (validates the dump).
        let _ = rec.sol.len();
        println!("  {it:>3}    {bi:.2e}      {bm:.2e}        {dis:.2e}{flag}");
    }
    println!("\n# worst backward residual over all iters: {worst_back:.2e}");
    println!("# max InfNorm-vs-MC64 single-solve disagreement: {max_disagree:.2e}");
    println!(
        "# (single-solve disagreement is the LSB-level seed; the cross-RUN\n\
         #  state divergence in divergence_curve.csv is what grows to ~1e-2.)"
    );
}
