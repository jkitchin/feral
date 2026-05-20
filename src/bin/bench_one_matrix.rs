//! Per-matrix bench driver for the cross-solver comparison harness.
//!
//! Reads a manifest of `(mtx_path, rhs_path, out_path)` triples
//! (whitespace-separated). For each entry:
//!   1. Reads the .mtx file and converts to CSC.
//!   2. Reads RHS from rhs_path (one f64 per line).
//!   3. Runs symbolic + numeric factor + solve.
//!   4. Computes rel_res = ||A x - b||_2 / ||b||_2.
//!   5. Writes per-key text sidecar mirroring the MA97 / MUMPS schema:
//!      `solver feral-<ver>`, `n`, `nnz`, `inertia_*`, `analyse_us`,
//!      `factor_us`, `solve_us`, `rel_res`, `status`.
//!
//! Usage: `cargo run --release --bin bench_one_matrix -- manifest.txt`

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::factorize_multifrontal_parallel_with_workspace;
use feral::numeric::factorize::FactorWorkspace;
use feral::numeric::solve::solve_sparse_refined;
use feral::scaling::{Mc64FallbackReason, ScalingInfo};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, CscMatrix, NumericParams};

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

fn rel_res_2norm(csc: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = csc.n;
    let mut r = b.iter().map(|v| -v).collect::<Vec<f64>>();
    for j in 0..n {
        for p in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[p];
            let a = csc.values[p];
            r[i] += a * x[j];
            if i != j {
                r[j] += a * x[i];
            }
        }
    }
    let rn: f64 = r.iter().map(|v| v * v).sum();
    let bn: f64 = b.iter().map(|v| v * v).sum();
    if bn == 0.0 {
        0.0
    } else {
        (rn / bn).sqrt()
    }
}

fn solve_one(mtx_path: &str, rhs_path: &str, out_path: &str) -> std::io::Result<()> {
    let mut out = File::create(out_path)?;
    let ver = env!("CARGO_PKG_VERSION");
    writeln!(out, "solver feral-{}", ver)?;

    let mtx = match read_mtx(Path::new(mtx_path)) {
        Ok(m) => m,
        Err(e) => {
            writeln!(out, "status fail")?;
            writeln!(
                out,
                "fail_reason read_mtx_{}",
                format!("{:?}", e).replace(' ', "_")
            )?;
            return Ok(());
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            writeln!(out, "status fail")?;
            writeln!(
                out,
                "fail_reason to_csc_{}",
                format!("{:?}", e).replace(' ', "_")
            )?;
            return Ok(());
        }
    };

    writeln!(out, "n {}", csc.n)?;
    writeln!(out, "nnz {}", csc.row_idx.len())?;

    let b = match read_rhs(rhs_path, csc.n) {
        Ok(b) => b,
        Err(e) => {
            writeln!(out, "status fail")?;
            writeln!(
                out,
                "fail_reason rhs_{}",
                format!("{}", e).replace(' ', "_")
            )?;
            return Ok(());
        }
    };

    let snode = SupernodeParams::default();
    let t0 = Instant::now();
    let sym = match symbolic_factorize(&csc, &snode) {
        Ok(s) => s,
        Err(e) => {
            writeln!(out, "status fail")?;
            writeln!(
                out,
                "fail_reason symbolic_{}",
                format!("{:?}", e).replace(' ', "_")
            )?;
            return Ok(());
        }
    };
    let analyse_us = t0.elapsed().as_micros() as u64;

    // NumericParams::default() ⇒ `ScalingStrategy::Auto` (MC64 or
    // InfNorm picked by matrix shape) and BK pivot threshold = 1e-8.
    // This matches the high-level `Solver::new()` defaults; bench
    // measures what library users get out of the box.
    let params = NumericParams::default();
    let mut ws = FactorWorkspace::new();
    let t0 = Instant::now();
    let factor_res = factorize_multifrontal_parallel_with_workspace(&csc, &sym, &params, &mut ws);
    let factor_us = t0.elapsed().as_micros() as u64;
    let (factors, inertia) = match factor_res {
        Ok(pair) => pair,
        Err(e) => {
            writeln!(out, "status fail")?;
            writeln!(
                out,
                "fail_reason factor_{}",
                format!("{:?}", e).replace(' ', "_")
            )?;
            writeln!(out, "analyse_us {}", analyse_us)?;
            return Ok(());
        }
    };

    // Iterative refinement against the original matrix. MUMPS callers
    // get this via ICNTL(10)>0; MA97 callers via solve_fredholm /
    // explicit refinement loop. Without this feral was timing solve
    // against the un-refined back-substitution and losing 4-5 orders
    // of magnitude in residual quality on ill-conditioned matrices.
    let t0 = Instant::now();
    let x = match solve_sparse_refined(&csc, &factors, &b) {
        Ok(x) => x,
        Err(e) => {
            writeln!(out, "status fail")?;
            writeln!(
                out,
                "fail_reason solve_{}",
                format!("{:?}", e).replace(' ', "_")
            )?;
            writeln!(out, "analyse_us {}", analyse_us)?;
            writeln!(out, "factor_us {}", factor_us)?;
            return Ok(());
        }
    };
    let solve_us = t0.elapsed().as_micros() as u64;

    let rel = rel_res_2norm(&csc, &x, &b);

    writeln!(out, "inertia_pos {}", inertia.positive)?;
    writeln!(out, "inertia_neg {}", inertia.negative)?;
    writeln!(out, "inertia_zero {}", inertia.zero)?;
    writeln!(out, "analyse_us {}", analyse_us)?;
    writeln!(out, "factor_us {}", factor_us)?;
    writeln!(out, "solve_us {}", solve_us)?;
    writeln!(out, "rel_res {:.17e}", rel)?;
    writeln!(out, "refined yes")?;
    // Issue #24: surface the silent MC64 → InfNorm fallback so a
    // bench-row reader can distinguish "Auto resolved to MC64"
    // from "Auto promised MC64 but fell back to InfNorm". The
    // field is always present (yes/no) like `refined`.
    let (fallback_flag, fallback_reason) = match &factors.scaling_info {
        ScalingInfo::Mc64FallbackToInfnorm { reason } => {
            let r = match reason {
                Mc64FallbackReason::InfNormSpreadAcceptable => "infnorm_spread_acceptable",
                Mc64FallbackReason::Mc64WorseThanInfnorm => "mc64_worse_than_infnorm",
                Mc64FallbackReason::Mc64ScalingDegenerate => "mc64_scaling_degenerate",
            };
            ("yes", Some(r))
        }
        _ => ("no", None),
    };
    writeln!(out, "mc64_fallback {}", fallback_flag)?;
    if let Some(r) = fallback_reason {
        writeln!(out, "mc64_fallback_reason {}", r)?;
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
        let rc = solve_one(mtx, rhs, out);
        if rc.is_err() {
            n_fail += 1;
        }
        n_done += 1;
        eprintln!("[{}] {} -> {}", n_done, mtx, out);
    }
    eprintln!("done {} (failures {})", n_done, n_fail);
    Ok(())
}
