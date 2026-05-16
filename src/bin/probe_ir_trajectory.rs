//! Per-matrix IR trajectory probe for issue #30.
//!
//! Reads the same manifest format as `bench_one_matrix`
//! (`mtx_path rhs_path out_path` per line). For each entry, performs
//! a symbolic + numeric factor and then calls
//! `solve_sparse_refined_with_diagnostics` to record the full
//! refinement trajectory.
//!
//! Output sidecar fields (key value, one per line):
//!   solver       feral-<ver>-ir-probe
//!   n            matrix dimension
//!   nnz          stored entries (lower-triangular convention)
//!   anorm_1      exact ||A||_1
//!   kappa_1_est  Hager-Higham 1-norm condition estimate
//!   ir_steps     <k>   number of trajectory points (1 = unrefined only)
//!   ir_returned  <s>   index of the iterate returned (best ||r||_2)
//!   ir_step_<i>  res2=<...> rel_res=<...> fwd_bound=<...> improved=<0|1>
//!   status       ok / fail
//!
//! Usage: `cargo run --release --bin probe_ir_trajectory -- manifest.txt`

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use feral::numeric::factorize::factorize_multifrontal_parallel_with_workspace;
use feral::numeric::factorize::FactorWorkspace;
use feral::numeric::solve::solve_sparse_refined_with_diagnostics;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, NumericParams};

fn read_rhs(path: &str, n: usize) -> std::io::Result<Vec<f64>> {
    let f = File::open(path)?;
    let mut b = Vec::with_capacity(n);
    for line in BufReader::new(f).lines() {
        let line = line?;
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        match s.parse::<f64>() {
            Ok(v) => b.push(v),
            Err(e) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("rhs parse: {}", e),
                ))
            }
        }
    }
    if b.len() != n {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("rhs length {} != n {}", b.len(), n),
        ));
    }
    Ok(b)
}

fn probe_one(mtx_path: &str, rhs_path: &str, out_path: &str) -> std::io::Result<()> {
    let mut out = File::create(out_path)?;
    let ver = env!("CARGO_PKG_VERSION");
    writeln!(out, "solver feral-{}-ir-probe", ver)?;

    let mtx = match read_mtx(Path::new(mtx_path)) {
        Ok(m) => m,
        Err(e) => {
            writeln!(out, "status fail")?;
            writeln!(out, "fail_reason read_mtx_{:?}", e)?;
            return Ok(());
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            writeln!(out, "status fail")?;
            writeln!(out, "fail_reason to_csc_{:?}", e)?;
            return Ok(());
        }
    };

    writeln!(out, "n {}", csc.n)?;
    writeln!(out, "nnz {}", csc.row_idx.len())?;

    let b = match read_rhs(rhs_path, csc.n) {
        Ok(b) => b,
        Err(e) => {
            writeln!(out, "status fail")?;
            writeln!(out, "fail_reason rhs_{}", e)?;
            return Ok(());
        }
    };

    let snode = SupernodeParams::default();
    let sym = match symbolic_factorize(&csc, &snode) {
        Ok(s) => s,
        Err(e) => {
            writeln!(out, "status fail")?;
            writeln!(out, "fail_reason symbolic_{:?}", e)?;
            return Ok(());
        }
    };

    let params = NumericParams::default();
    let mut ws = FactorWorkspace::new();
    let (factors, inertia) =
        match factorize_multifrontal_parallel_with_workspace(&csc, &sym, &params, &mut ws) {
            Ok(p) => p,
            Err(e) => {
                writeln!(out, "status fail")?;
                writeln!(out, "fail_reason factor_{:?}", e)?;
                return Ok(());
            }
        };

    let (_, diag) = match solve_sparse_refined_with_diagnostics(&csc, &factors, &b) {
        Ok(p) => p,
        Err(e) => {
            writeln!(out, "status fail")?;
            writeln!(out, "fail_reason solve_{:?}", e)?;
            return Ok(());
        }
    };

    writeln!(out, "inertia_pos {}", inertia.positive)?;
    writeln!(out, "inertia_neg {}", inertia.negative)?;
    writeln!(out, "inertia_zero {}", inertia.zero)?;
    writeln!(out, "anorm_1 {:.17e}", diag.anorm_1)?;
    writeln!(out, "kappa_1_est {:.17e}", diag.kappa_1_est)?;
    writeln!(out, "ir_steps {}", diag.steps.len())?;
    writeln!(out, "ir_returned {}", diag.returned_step)?;
    for s in &diag.steps {
        writeln!(
            out,
            "ir_step_{} res2={:.17e} rel_res={:.17e} fwd_bound={:.17e} improved={}",
            s.step,
            s.residual_2norm,
            s.relative_residual,
            s.forward_error_bound,
            if s.improved { 1 } else { 0 }
        )?;
    }
    writeln!(out, "status ok")?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} manifest.txt", args[0]);
        std::process::exit(2);
    }
    let f = File::open(&args[1])?;
    let mut n_done = 0usize;
    let mut n_fail = 0usize;
    for line in BufReader::new(f).lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }
        let mtx = parts[0];
        let rhs = parts[1];
        let out = parts[2];
        if probe_one(mtx, rhs, out).is_err() {
            n_fail += 1;
        }
        n_done += 1;
        eprintln!("[{}] {} -> {}", n_done, mtx, out);
    }
    eprintln!("done {} (failures {})", n_done, n_fail);
    Ok(())
}
