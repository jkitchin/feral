//! Issue #50 — per-stage symbolic profiler probe for the powerflow22
//! KKT (`gams/nlpbench/feral_repro/powerflow22/kkt_solve_*.bin`).
//!
//! Diagnostic question: of the 113 s `Solver::factor` time the
//! reporter measured on the 2.8M-dim power-grid KKT, how much lives
//! in the ordering call (`MetisND` / `ScotchND` itself) vs the
//! symbolic post-pass (etree, column counts, supernode detect)?
//!
//! Runs `symbolic_factorize_with_method` once per ordering on the
//! supplied `.bin` matrix and prints the per-stage breakdown plus
//! `resolved_method` (settles whether `Auto` lands on `MetisND` or
//! `ScotchND` on this matrix — see Q4 in the research note).
//!
//! Matrix file format (little-endian, no padding):
//!
//! ```text
//! u64  dim
//! u64  nnz                # triplet count
//! u64  nrhs
//! i64  irn[nnz]           # 1-based row indices
//! i64  jcn[nnz]           # 1-based col indices
//! f64  vals[nnz]
//! f64  rhs[dim * nrhs]    # ignored here
//! ```
//!
//! See `dev/research/issue-50-metisnd-symbolic-cost.md`.

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use feral::symbolic::{
    symbolic_factorize_with_method, OrderingMethod, SupernodeParams, SymbolicProfileReport,
    SymbolicProfiler,
};
use feral::{CscMatrix, FeralError};

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64_vec(r: &mut impl Read, n: usize) -> std::io::Result<Vec<i64>> {
    let mut out = vec![0i64; n];
    let mut buf = [0u8; 8];
    for v in out.iter_mut() {
        r.read_exact(&mut buf)?;
        *v = i64::from_le_bytes(buf);
    }
    Ok(out)
}

fn read_f64_vec(r: &mut impl Read, n: usize) -> std::io::Result<Vec<f64>> {
    let mut out = vec![0.0f64; n];
    let mut buf = [0u8; 8];
    for v in out.iter_mut() {
        r.read_exact(&mut buf)?;
        *v = f64::from_le_bytes(buf);
    }
    Ok(out)
}

fn load_bin(path: &Path) -> Result<CscMatrix, String> {
    let f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut r = BufReader::new(f);
    let dim = read_u64(&mut r).map_err(|e| format!("read dim: {e}"))? as usize;
    let nnz = read_u64(&mut r).map_err(|e| format!("read nnz: {e}"))? as usize;
    let _nrhs = read_u64(&mut r).map_err(|e| format!("read nrhs: {e}"))? as usize;
    let irn = read_i64_vec(&mut r, nnz).map_err(|e| format!("read irn: {e}"))?;
    let jcn = read_i64_vec(&mut r, nnz).map_err(|e| format!("read jcn: {e}"))?;
    let vals = read_f64_vec(&mut r, nnz).map_err(|e| format!("read vals: {e}"))?;

    // 1-based -> 0-based; canonicalize to lower triangle.
    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    for k in 0..nnz {
        let i = (irn[k] - 1) as usize;
        let j = (jcn[k] - 1) as usize;
        if i >= j {
            rows.push(i);
            cols.push(j);
        } else {
            rows.push(j);
            cols.push(i);
        }
    }
    CscMatrix::from_triplets(dim, &rows, &cols, &vals)
        .map_err(|e: FeralError| format!("from_triplets: {e:?}"))
}

fn run(
    matrix: &CscMatrix,
    method: OrderingMethod,
) -> Result<(SymbolicProfileReport, OrderingMethod, usize), String> {
    let prof = Arc::new(Mutex::new(SymbolicProfiler::new()));
    let params = SupernodeParams {
        symbolic_profiler: Some(Arc::clone(&prof)),
        ..SupernodeParams::default()
    };
    let sym = symbolic_factorize_with_method(matrix, &params, method.clone())
        .map_err(|e| format!("symbolic_factorize_with_method({method:?}): {e:?}"))?;
    let report = prof
        .lock()
        .map_err(|e| format!("profiler lock poisoned: {e}"))?
        .report();
    Ok((report, sym.resolved_method, sym.factor_nnz_estimate))
}

fn print_report(
    label: &str,
    method: OrderingMethod,
    resolved: OrderingMethod,
    factor_nnz: usize,
    r: &SymbolicProfileReport,
) {
    println!("=== ordering = {label} (requested {method:?}; resolved {resolved:?}) ===");
    println!(
        "total = {:.3} s ({} us), accounted = {:.3} s, overhead = {:.1}%",
        r.total_us as f64 / 1.0e6,
        r.total_us,
        r.accounted_us as f64 / 1.0e6,
        r.overhead_pct
    );
    println!("factor_nnz_estimate = {factor_nnz}");
    println!("{:<22}  {:>12}  {:>6}", "stage", "us", "%");
    let mut sorted: Vec<_> = r.stages.iter().collect();
    sorted.sort_by_key(|s| std::cmp::Reverse(s.us));
    for s in sorted {
        println!("{:<22}  {:>12}  {:>5.1}%", s.name, s.us, s.pct_of_total);
    }
    if !r.validation_warnings.is_empty() {
        println!("WARNINGS: {:?}", r.validation_warnings);
    }
    println!();
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: diag_issue50_symbolic <path/to/kkt.bin>");
        eprintln!();
        eprintln!("Runs symbolic_factorize_with_method on the supplied .bin");
        eprintln!("KKT (issue #50 powerflow22 dump) under OrderingMethod::Auto,");
        eprintln!("MetisND, and Amd, and prints the per-stage timing breakdown.");
        return ExitCode::from(2);
    };

    let t = Instant::now();
    let matrix = match load_bin(Path::new(path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "loaded {} in {:.2} s: n = {}, stored nnz = {} (avg_deg = {:.2})",
        path,
        t.elapsed().as_secs_f64(),
        matrix.n,
        matrix.row_idx.len(),
        matrix.row_idx.len() as f64 / matrix.n.max(1) as f64,
    );
    println!();

    // Order matters: Auto first so we see what the default path picks
    // on this matrix without prior interference. Each call builds a
    // fresh symbolic factorization — no caching between runs.
    let cases: &[(&str, OrderingMethod)] = &[
        ("Auto", OrderingMethod::Auto),
        ("MetisND", OrderingMethod::MetisND),
        ("Amd", OrderingMethod::Amd),
    ];

    let mut any_failure = false;
    for (label, method) in cases {
        let t = Instant::now();
        match run(&matrix, method.clone()) {
            Ok((report, resolved, factor_nnz)) => {
                eprintln!(
                    "[{label}] symbolic done in {:.2} s wall",
                    t.elapsed().as_secs_f64()
                );
                print_report(label, method.clone(), resolved, factor_nnz, &report);
            }
            Err(e) => {
                eprintln!("[{label}] FAILED: {e}");
                any_failure = true;
            }
        }
    }

    if any_failure {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
