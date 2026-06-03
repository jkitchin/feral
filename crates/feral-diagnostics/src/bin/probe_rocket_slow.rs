//! Profile the slow late-iter rocket_12800 factor on the LIVE
//! pounce-dumped KKT corpus (/tmp/rkt_NNN.bin).
//!
//! Replay observation: calls #013-017 take 0.75-1.69 s each (warm).
//! This probe loads the worst one, runs it fresh sequential under
//! the per-supernode Profiler, and reports the bucket distribution
//! plus top-N hottest supernodes to localize the cost.

use feral::numeric::factorize::Profiler;
use feral::scaling::ScalingStrategy;
use feral::sparse::csc::CscMatrix;
use feral::symbolic::supernode::SupernodeParams;
use feral::{NumericParams, Solver};
use std::fs::File;
use std::io::Read;
use std::sync::{Arc, Mutex};
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

fn load_kkt(path: &str) -> std::io::Result<CscMatrix> {
    let mut f = File::open(path)?;
    let dim = read_u64(&mut f)? as usize;
    let nnz = read_u64(&mut f)? as usize;
    let _nrhs = read_u64(&mut f)? as usize;
    let mut ia = Vec::with_capacity(nnz);
    let mut ja = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);
    for _ in 0..nnz {
        ia.push(read_i64(&mut f)? as usize);
    }
    for _ in 0..nnz {
        ja.push(read_i64(&mut f)? as usize);
    }
    for _ in 0..nnz {
        vals.push(read_f64(&mut f)?);
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
    CscMatrix::from_triplets(dim, &rows_0, &cols_0, &vals)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{:?}", e)))
}

fn profile_one(label: &str, csc: &CscMatrix, scaling: ScalingStrategy) {
    let prof = Arc::new(Mutex::new(Profiler::new()));
    let np = NumericParams {
        profiler: Some(prof.clone()),
        scaling: scaling.clone(),
        ..NumericParams::default()
    };
    let mut solver = Solver::with_params(np, SupernodeParams::default()).with_parallel(false);

    // First call (cold; will include symbolic).
    let t0 = Instant::now();
    let _ = solver.factor(csc, None);
    let cold_ms = t0.elapsed().as_secs_f64() * 1e3;

    if let Ok(mut p) = prof.lock() {
        *p = Profiler::new();
    }
    let t0 = Instant::now();
    let st = solver.factor(csc, None);
    let warm_ms = t0.elapsed().as_secs_f64() * 1e3;

    let prof = match prof.lock() {
        Ok(p) => p.clone(),
        Err(_) => return,
    };
    let report = prof.report();
    println!(
        "{} scaling={:?}  cold={:.1} ms  warm={:.1} ms  status={:?}",
        label, scaling, cold_ms, warm_ms, st,
    );
    println!(
        "  prologue={:.1} ms  loop={:.1} ms  epilogue={:.1} ms  total={:.1} ms  overhead={:.1}%  n_snodes={}",
        report.prologue_us as f64 / 1e3,
        report.loop_us as f64 / 1e3,
        report.epilogue_us as f64 / 1e3,
        report.total_us as f64 / 1e3,
        report.overhead_pct,
        report.n_supernodes,
    );
    println!("  bucket            count    sum_ms     pct      avg_us");
    for b in &report.buckets {
        if b.count == 0 {
            continue;
        }
        println!(
            "    nrow {:>7}  {:>6}  {:>8.1}  {:>5.1}%  {:>8.1}",
            b.range,
            b.count,
            b.sum_us as f64 / 1e3,
            b.pct_of_total,
            b.avg_us,
        );
    }
    let top: Vec<_> = {
        let mut v: Vec<_> = prof.timings().iter().collect();
        v.sort_by_key(|t| std::cmp::Reverse(t.us));
        v.into_iter().take(10).collect()
    };
    println!("  top 10 supernodes:");
    for t in top {
        println!(
            "    snode {:>5}  nrow={:>5} ncol={:>5}  {:>8.2} ms",
            t.snode_idx,
            t.nrow,
            t.ncol,
            t.us as f64 / 1e3,
        );
    }
}

fn main() {
    let target = std::env::args().nth(1).unwrap_or_else(|| "017".to_string());
    let path = format!("/tmp/rkt_{}.bin", target);
    let csc = match load_kkt(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("load failed: {:?}", e);
            return;
        }
    };
    println!("Loaded {} : n={} nnz={}", path, csc.n, csc.row_idx.len());

    profile_one("auto      ", &csc, ScalingStrategy::Auto);
    println!();
    profile_one("infnorm   ", &csc, ScalingStrategy::InfNorm);
    println!();
    profile_one("mc64sym   ", &csc, ScalingStrategy::Mc64Symmetric);
    println!();
    profile_one("identity  ", &csc, ScalingStrategy::Identity);
}
