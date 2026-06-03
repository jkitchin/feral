// Standalone: load pounce-dumped KKT (dim, nnz, nrhs, ia[], ja[], vals[], rhs[]
// — see pounce_linsol/t_sym_solver.rs:188) and time a single feral factor.
//
// Run with:
//   cd ~/projects/feral && cargo run --release --example feral_time /tmp/pounce_kkt.bin

use feral::sparse::csc::CscMatrix;
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

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: feral_time <kkt.bin>");
    let mut f = File::open(&path).unwrap();
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
    // pounce dumps 1-based MA57-style triplets; convert to 0-based and
    // canonicalize to lower triangle, matching what pounce-feral does in
    // initialize_structure (lib.rs:241).
    let mut rows_0: Vec<usize> = Vec::with_capacity(nnz);
    let mut cols_0: Vec<usize> = Vec::with_capacity(nnz);
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
    println!("dim={dim}, nnz={nnz}");
    let t_build = Instant::now();
    let matrix = CscMatrix::from_triplets(dim, &rows_0, &cols_0, &vals).unwrap();
    println!("from_triplets: {:.3}s", t_build.elapsed().as_secs_f64());

    let mut solver = Solver::new(); // default: cb=off, parallel=on
    let t_fac1 = Instant::now();
    let st1 = solver.factor(&matrix, None);
    let dt1 = t_fac1.elapsed().as_secs_f64();
    println!("factor #1: {dt1:.3}s status={st1:?}");

    let t_fac2 = Instant::now();
    let st2 = solver.factor(&matrix, None);
    let dt2 = t_fac2.elapsed().as_secs_f64();
    println!("factor #2: {dt2:.3}s status={st2:?}");
}
