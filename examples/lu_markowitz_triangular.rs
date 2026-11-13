//! Issue #232: time `SparseLu::factor_markowitz` on a near-triangular LP-basis
//! shape, and print a bitwise digest of the factorization so a refactor of the
//! elimination's bookkeeping can be checked to leave every pivot and every
//! value unchanged.
//!
//! The issue's basis (MIPLIB piperout-d27, m = 17 470, 50 762 nnz, 13 217
//! one-entry columns, 91.6 % peeled by singletons, 1 459 x 1 459 kernel with
//! 5 201 nnz) lives in discopt. This generator reproduces that shape:
//! `[[K, B], [0, T]]` with `T` upper triangular (peeled bottom-up as row
//! singletons, plus the pure-diagonal columns as column singletons) and `K` a
//! random sparse kernel.
//!
//! Also reads Matrix Market bases given on the command line.
//!
//! Usage: `cargo run --release --example lu_markowitz_triangular [reps] [file.mtx ...]`
//! (`reps = 0`: digest and fill only; `FERAL_SEEDS=n`: n synthetic bases).

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{LuParams, SparseLu};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn piperout_shape(seed: u64) -> SparseColMatrix {
    let m = 17_470;
    let k = 1_459;
    let mut rng = Rng(seed);
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    // Kernel: diagonal plus ~2.6 entries per column (≈ 5 200 nnz).
    for (j, col) in cols.iter_mut().enumerate().take(k) {
        col.push((j, 1.0 + rng.f64()));
        for _ in 0..3 {
            if rng.f64() < 0.85 {
                // Mostly below the diagonal (no fill), a few anywhere, so the
                // kernel is a genuine bump with little fill like the real one.
                let i = if j + 1 < k && rng.f64() < 0.97 {
                    j + 1 + rng.below(k - j - 1)
                } else {
                    rng.below(k)
                };
                if !col.iter().any(|&(r, _)| r == i) {
                    col.push((i, rng.f64() - 0.5));
                }
            }
        }
    }
    // Triangular part: ~80 % one-entry columns, the rest carry a few entries
    // in rows above the diagonal (kernel rows included).
    for (j, col) in cols.iter_mut().enumerate().skip(k) {
        col.push((j, 2.0 + rng.f64()));
        if rng.f64() < 0.21 {
            let extra = 2 + rng.below(9);
            for _ in 0..extra {
                let i = rng.below(j);
                if !col.iter().any(|&(r, _)| r == i) {
                    col.push((i, rng.f64() - 0.5));
                }
            }
        }
    }
    let mut a = SparseColMatrix::from_sparse_columns(m, &cols).expect("basis");
    // A permutation so the triangular structure is not in index order.
    let mut p: Vec<usize> = (0..m).collect();
    for i in (1..m).rev() {
        p.swap(i, rng.below(i + 1));
    }
    let mut pc: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for j in 0..m {
        for idx in a.col_ptr[j]..a.col_ptr[j + 1] {
            pc[p[j]].push((p[a.row_idx[idx]], a.values[idx]));
        }
    }
    a = SparseColMatrix::from_sparse_columns(m, &pc).expect("basis");
    a
}

fn read_mtx(path: &str) -> SparseColMatrix {
    let text = std::fs::read_to_string(path).expect("read");
    let mut lines = text.lines().filter(|l| !l.starts_with('%'));
    let hdr: Vec<usize> = lines
        .next()
        .expect("header")
        .split_whitespace()
        .map(|t| t.parse().expect("int"))
        .collect();
    let m = hdr[0];
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for l in lines {
        let t: Vec<&str> = l.split_whitespace().collect();
        if t.len() < 3 {
            continue;
        }
        let i: usize = t[0].parse().expect("i");
        let j: usize = t[1].parse().expect("j");
        let v: f64 = t[2].parse().expect("v");
        cols[j - 1].push((i - 1, v));
    }
    SparseColMatrix::from_sparse_columns(m, &cols).expect("basis")
}

fn digest(lu: &mut SparseLu) -> u64 {
    let m = lu.dim();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    for &p in lu.perm() {
        mix(p as u64);
    }
    for &q in lu.qcol() {
        mix(q as u64);
    }
    mix(lu.factor_nnz() as u64);
    let mut x: Vec<f64> = (0..m).map(|i| 1.0 + (i % 7) as f64).collect();
    lu.ftran(&mut x).expect("ftran");
    for v in &x {
        mix(v.to_bits());
    }
    let mut y: Vec<f64> = (0..m).map(|i| 1.0 + (i % 5) as f64).collect();
    lu.btran(&mut y).expect("btran");
    for v in &y {
        mix(v.to_bits());
    }
    h
}

fn run(name: &str, a: &SparseColMatrix, reps: usize) {
    let params = LuParams::default();
    let mut lu = SparseLu::factor_markowitz(a, params.clone()).expect("factor");
    let d = digest(&mut lu);
    if reps == 0 {
        println!("{name}: factor_nnz={} digest={d:016x}", lu.factor_nnz());
        return;
    }
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let lu = SparseLu::factor_markowitz(a, params.clone()).expect("factor");
        times.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(lu);
    }
    times.sort_by(|x, y| x.total_cmp(y));
    println!(
        "{name}: m={} nnz={} factor_nnz={} median={:.3} ms min={:.3} ms digest={d:016x}",
        a.m,
        a.values.len(),
        lu.factor_nnz(),
        times[times.len() / 2],
        times[0],
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(100);
    // `FERAL_SEEDS=n` widens the synthetic set (e.g. a fill sweep with reps 0).
    let nseeds: u64 = std::env::var("FERAL_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    for seed in 1..=nseeds {
        run(
            &format!("piperout-shape seed {seed}"),
            &piperout_shape(seed),
            reps,
        );
    }
    for path in args.iter().skip(1) {
        run(path, &read_mtx(path), reps);
    }
}
