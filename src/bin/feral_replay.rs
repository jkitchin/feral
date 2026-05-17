#![allow(clippy::type_complexity)]
// Replay pounce-dumped KKT call sequence through ONE feral::Solver.
//
// Run with:
//   cd ~/projects/feral && cargo build --release --bin feral_replay
//   ./target/release/feral_replay /tmp/rkt
//
// Loads /tmp/rkt_000.bin .. /tmp/rkt_NNN.bin and feeds them sequentially.

use feral::numeric::factorize::NumericParams;
use feral::scaling::ScalingStrategy;
use feral::sparse::csc::CscMatrix;
use feral::symbolic::supernode::SupernodeParams;
use feral::Solver;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

fn read_u64(f: &mut File) -> u64 {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).unwrap();
    u64::from_le_bytes(b)
}
fn read_i64(f: &mut File) -> i64 {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).unwrap();
    i64::from_le_bytes(b)
}
fn read_f64(f: &mut File) -> f64 {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).unwrap();
    f64::from_le_bytes(b)
}

fn load(path: &str) -> Option<(usize, Vec<usize>, Vec<usize>, Vec<f64>)> {
    let mut f = File::open(path).ok()?;
    let dim = read_u64(&mut f) as usize;
    let nnz = read_u64(&mut f) as usize;
    let _nrhs = read_u64(&mut f) as usize;
    let mut ia = Vec::with_capacity(nnz);
    let mut ja = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);
    for _ in 0..nnz {
        ia.push(read_i64(&mut f) as usize);
    }
    for _ in 0..nnz {
        ja.push(read_i64(&mut f) as usize);
    }
    for _ in 0..nnz {
        vals.push(read_f64(&mut f));
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

fn build_solver() -> Solver {
    let mut np = NumericParams::default();
    match std::env::var("SCALING").as_deref() {
        Ok("identity") => np.scaling = ScalingStrategy::Identity,
        Ok("infnorm") => np.scaling = ScalingStrategy::InfNorm,
        Ok("mc64") => np.scaling = ScalingStrategy::Mc64Symmetric,
        _ => {}
    }
    let mut solver = Solver::with_params(np, SupernodeParams::default());
    if std::env::var("CB").is_ok() {
        solver = solver.with_cascade_break(0.5).with_cascade_break_eps(1e-10);
    }
    if matches!(std::env::var("PAR").as_deref(), Ok("0") | Ok("off")) {
        solver = solver.with_parallel(false);
    }
    solver
}

fn main() {
    let prefix = std::env::args()
        .nth(1)
        .expect("usage: feral_replay <prefix>");
    let fresh_each = std::env::var("FRESH").is_ok();
    let start: usize = std::env::var("START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let stop: usize = std::env::var("STOP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    if std::env::var("CB").is_ok() {
        eprintln!("[replay] cascade_break ON");
    }
    if matches!(std::env::var("PAR").as_deref(), Ok("0") | Ok("off")) {
        eprintln!("[replay] parallel OFF");
    }
    if fresh_each {
        eprintln!("[replay] FRESH solver each call");
    }
    let mut solver = build_solver();
    for k in start..stop {
        let path = format!("{}_{:03}.bin", prefix, k);
        let Some((dim, rows, cols, vals)) = load(&path) else {
            break;
        };
        let t0 = Instant::now();
        let matrix = CscMatrix::from_triplets(dim, &rows, &cols, &vals).unwrap();
        let t_build = t0.elapsed().as_secs_f64();
        if fresh_each {
            solver = build_solver();
        }
        let t1 = Instant::now();
        let st = solver.factor(&matrix, None);
        let dt = t1.elapsed().as_secs_f64();
        println!(
            "call #{k:03}: build={t_build:.3}s factor={dt:.3}s status={st:?} neg={} dim={dim} nnz={}",
            solver.num_negative_eigenvalues(),
            vals.len(),
        );
    }
}
