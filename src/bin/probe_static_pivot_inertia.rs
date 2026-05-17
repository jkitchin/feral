//! Probe: per-iter inertia on rocket_12800 KKTs under different
//! `FERAL_STATIC_PIVOT` thresholds. Goal: prove the knob shifts
//! inertia from feral's "true" 38402-negative to MA57's 38400-negative
//! by perturbing tiny negative pivots up across zero.
//!
//! Reads `/tmp/rkt_NNN.bin` (same format as probe_rocket_residuals).
//!
//! Usage:
//!   cargo run --release --bin probe_static_pivot_inertia -- /tmp/rkt [start] [stop]
//!
//! Env:
//!   STATIC_PIVOTS="0,1e-12,1e-10,1e-8,1e-6"   thresholds to sweep
//!   ONLY=N    only matrix index N

use feral::numeric::factorize::NumericParams;
use feral::sparse::csc::CscMatrix;
use feral::symbolic::supernode::SupernodeParams;
use feral::Solver;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

fn read_u64(f: &mut File) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_i64(f: &mut File) -> std::io::Result<i64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}
fn read_f64(f: &mut File) -> std::io::Result<f64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

#[allow(clippy::type_complexity)]
fn load(path: &str) -> Option<(usize, Vec<usize>, Vec<usize>, Vec<f64>)> {
    let mut f = File::open(path).ok()?;
    let dim = read_u64(&mut f).ok()? as usize;
    let nnz = read_u64(&mut f).ok()? as usize;
    let _nrhs = read_u64(&mut f).ok()? as usize;
    let mut ia = Vec::with_capacity(nnz);
    let mut ja = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);
    for _ in 0..nnz {
        ia.push(read_i64(&mut f).ok()? as usize);
    }
    for _ in 0..nnz {
        ja.push(read_i64(&mut f).ok()? as usize);
    }
    for _ in 0..nnz {
        vals.push(read_f64(&mut f).ok()?);
    }
    let mut rows_0 = Vec::with_capacity(nnz);
    let mut cols_0 = Vec::with_capacity(nnz);
    for k in 0..nnz {
        let i = ia[k] - 1;
        let j = ja[k] - 1;
        if i >= j {
            rows_0.push(i);
            cols_0.push(j);
        } else {
            rows_0.push(j);
            cols_0.push(i);
        }
    }
    Some((dim, rows_0, cols_0, vals))
}

fn run_one(matrix: &CscMatrix, threshold: Option<f64>) -> (Option<usize>, f64, bool) {
    let np = NumericParams {
        static_pivot_threshold: threshold,
        ..NumericParams::default()
    };
    let mut solver = Solver::with_params(np, SupernodeParams::default());
    let t0 = Instant::now();
    let _st = solver.factor(matrix, None);
    let dt = t0.elapsed().as_secs_f64();
    let neg = solver.inertia().map(|i| i.negative);
    let needs_ref = solver
        .factors()
        .map(|f| f.needs_refinement)
        .unwrap_or(false);
    (neg, dt, needs_ref)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prefix = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/rkt".to_string());
    let start: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let stop_arg: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(18);
    let only: Option<usize> = std::env::var("ONLY").ok().and_then(|s| s.parse().ok());

    let pivots_env =
        std::env::var("STATIC_PIVOTS").unwrap_or_else(|_| "0,1e-12,1e-10,1e-8,1e-6".to_string());
    let thresholds: Vec<Option<f64>> = pivots_env
        .split(',')
        .map(|s| {
            let v: f64 = s.trim().parse().unwrap_or(0.0);
            if v <= 0.0 {
                None
            } else {
                Some(v)
            }
        })
        .collect();

    let iter_range: Vec<usize> = if let Some(o) = only {
        vec![o]
    } else {
        (start..stop_arg).collect()
    };

    // Header
    print!("{:>5}", "iter");
    for t in &thresholds {
        match t {
            Some(v) => print!("  t={:<10.0e}", v),
            None => print!("  {:<12}", "t=0(off)"),
        }
    }
    println!();

    for k in iter_range {
        let path = format!("{}_{:03}.bin", prefix, k);
        let Some((dim, rows, cols, vals)) = load(&path) else {
            println!("no file: {path}");
            break;
        };
        let matrix = CscMatrix::from_triplets(dim, &rows, &cols, &vals).unwrap();
        print!("{:>5}", k);
        for t in &thresholds {
            let (neg, dt, needs_ref) = run_one(&matrix, *t);
            let nstr = neg
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".to_string());
            let flag = if needs_ref { "*" } else { " " };
            print!("  neg={:>5}{}({:.2}s)", nstr, flag, dt);
        }
        println!();
        let _ = dim;
    }
}
