//! Probe: per-iter residuals on rocket_12800 KKTs under CB on/off,
//! with and without iterative refinement, against the actual IPM RHS
//! dumped alongside the matrix.
//!
//! Issue #38 Failure B investigation. For each of the 18 dumped IPM
//! iter KKTs we report:
//!   - inertia (negative, positive, zero) under CB=off vs CB=on
//!   - unrefined `||Ax - b||_inf / ||b||_inf` (raw solve quality)
//!   - refined residual via Solver::solve_refined (matches the default
//!     ipopt-feral path through `feral_solve` since 597a90a)
//!
//! Reads `/tmp/rkt_NNN.bin` (pounce-linsol dump format):
//!   header: u64 dim, u64 nnz, u64 nrhs
//!   triplets: nnz × i64 ia (1-based), nnz × i64 ja (1-based), nnz × f64 vals
//!   rhs: dim × nrhs × f64 (column-major)
//!
//! Usage:
//!   cargo run --release --bin probe_rocket_residuals -- /tmp/rkt [start] [stop]
//!
//! Env:
//!   ONLY=N    — run only matrix index N (overrides start/stop)
//!   SYNTH=1   — use a deterministic synthetic RHS instead of the dumped IPM RHS

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
fn load(path: &str) -> Option<(usize, Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>)> {
    let mut f = File::open(path).ok()?;
    let dim = read_u64(&mut f).ok()? as usize;
    let nnz = read_u64(&mut f).ok()? as usize;
    let nrhs = read_u64(&mut f).ok()? as usize;
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
    let mut rhs = Vec::with_capacity(dim * nrhs);
    for _ in 0..(dim * nrhs) {
        rhs.push(read_f64(&mut f).ok()?);
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
    Some((dim, rows_0, cols_0, vals, rhs))
}

fn build_solver(cb: bool) -> Solver {
    let mut np = NumericParams::default();
    if let Ok(s) = std::env::var("PIVTOL") {
        if let Ok(v) = s.parse::<f64>() {
            np.bk.pivot_threshold = v;
        }
    }
    let mut s = Solver::with_params(np, SupernodeParams::default());
    if cb {
        s = s.with_cascade_break(0.5).with_cascade_break_eps(1e-10);
    }
    s
}

fn norm_inf(v: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for &x in v {
        let a = x.abs();
        if a > m {
            m = a;
        }
    }
    m
}

fn norm2(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

fn run_one(label: &str, cb: bool, matrix: &CscMatrix, b: &[f64], k: usize) {
    let mut solver = build_solver(cb);
    let t0 = Instant::now();
    let st = solver.factor(matrix, None);
    let dt = t0.elapsed().as_secs_f64();
    let inert = solver.inertia().cloned();
    let neg = solver.num_negative_eigenvalues();
    let pos = inert.as_ref().map(|i| i.positive).unwrap_or(0);
    let zer = inert.as_ref().map(|i| i.zero).unwrap_or(0);
    println!(
        "call #{k:03} [{label}]: factor={dt:.3}s status={st:?} inertia=(neg={neg}, pos={pos}, zero={zer})"
    );

    let n = matrix.n;
    let b_inf = norm_inf(b);
    let b_2 = norm2(b);

    // Unrefined
    let t1 = Instant::now();
    let x = match solver.solve(b) {
        Ok(v) => v,
        Err(e) => {
            println!("  unrefined solve FAIL: {:?}", e);
            return;
        }
    };
    let dts = t1.elapsed().as_secs_f64();
    let mut ax = vec![0.0f64; n];
    matrix.symv(&x, &mut ax);
    let r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
    let r_inf = norm_inf(&r);
    let r_2 = norm2(&r);
    let rel_inf = if b_inf > 0.0 { r_inf / b_inf } else { r_inf };
    let rel_2 = if b_2 > 0.0 { r_2 / b_2 } else { r_2 };
    let x_inf = norm_inf(&x);
    println!(
        "  unrefined: solve={dts:.3}s ||r||_inf={r_inf:.3e} ||b||_inf={b_inf:.3e} rel_inf={rel_inf:.3e} rel_2={rel_2:.3e} ||x||_inf={x_inf:.3e}"
    );

    // Refined (the actual production path; default since 597a90a)
    let t2 = Instant::now();
    let xr = match solver.solve_refined(matrix, b) {
        Ok(v) => v,
        Err(e) => {
            println!("  refined solve FAIL: {:?}", e);
            return;
        }
    };
    let dtr = t2.elapsed().as_secs_f64();
    let mut ax2 = vec![0.0f64; n];
    matrix.symv(&xr, &mut ax2);
    let r2: Vec<f64> = (0..n).map(|i| b[i] - ax2[i]).collect();
    let r2_inf = norm_inf(&r2);
    let r2_2 = norm2(&r2);
    let rel2_inf = if b_inf > 0.0 { r2_inf / b_inf } else { r2_inf };
    let rel2_2 = if b_2 > 0.0 { r2_2 / b_2 } else { r2_2 };
    let xr_inf = norm_inf(&xr);
    println!(
        "  refined:   solve={dtr:.3}s ||r||_inf={r2_inf:.3e} rel_inf={rel2_inf:.3e} rel_2={rel2_2:.3e} ||x||_inf={xr_inf:.3e}"
    );

    // Diff between unrefined and refined x (how much did refinement move it?)
    let mut diff = 0.0f64;
    let mut xnorm = 0.0f64;
    for i in 0..n {
        let d = (x[i] - xr[i]).abs();
        if d > diff {
            diff = d;
        }
        let xa = xr[i].abs();
        if xa > xnorm {
            xnorm = xa;
        }
    }
    let rel_change = if xnorm > 0.0 { diff / xnorm } else { diff };
    println!("  refinement moved x by ||dx||_inf={diff:.3e} (rel to ||xr||_inf: {rel_change:.3e})");
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
    let synth = std::env::var("SYNTH").is_ok();

    let iter_range: Vec<usize> = if let Some(o) = only {
        vec![o]
    } else {
        (start..stop_arg).collect()
    };

    for k in iter_range {
        let path = format!("{}_{:03}.bin", prefix, k);
        let Some((dim, rows, cols, vals, rhs)) = load(&path) else {
            println!("no file: {path}");
            break;
        };
        let matrix = CscMatrix::from_triplets(dim, &rows, &cols, &vals).unwrap();
        let b: Vec<f64> = if synth {
            (0..dim).map(|i| ((i % 17) as f64 - 8.0) * 1.0e-3).collect()
        } else {
            rhs[..dim].to_vec()
        };
        println!(
            "=== {path} dim={dim} nnz_triplets={} ||rhs||_inf={:.3e} ===",
            vals.len(),
            norm_inf(&b),
        );
        run_one("CBoff", false, &matrix, &b, k);
        run_one("CBon ", true, &matrix, &b, k);
    }
}
