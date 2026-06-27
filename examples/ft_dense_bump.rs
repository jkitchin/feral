//! Issue #89 reproducer B (standalone, feral-only): Forrest–Tomlin `update()`
//! cost on dense-inverse (set-covering) bases.
//!
//! The entering column always has a fixed, small nnz, yet `update()` cost grows
//! with the factor fill. This also prints `eta_ops()` / `last_eta_ops()` so the
//! mismatch between the recorded eta op-count and the true per-update build cost
//! is visible: an O(bump/nnz) update would be ~flat in time but the time tracks
//! `factor_nnz`, and the recorded eta op-count badly undercounts the real work.

use feral::{LuParams, SparseColMatrix, SparseLu, SparseLuSymbolic};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
    fn unit(&mut self) -> f64 {
        (self.next() as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

// per_col entries at random rows with random values (=> dense inverse, the
// covering regime) plus a unit diagonal so the basis is nonsingular.
fn covering_col(diag: usize, m: usize, per_col: usize, rng: &mut Lcg) -> Vec<(usize, f64)> {
    let mut rows = vec![diag];
    while rows.len() < per_col {
        let r = rng.below(m);
        if !rows.contains(&r) {
            rows.push(r);
        }
    }
    rows.into_iter()
        .map(|r| (r, if r == diag { 1.0 } else { 0.5 * rng.unit() }))
        .collect()
}

fn factor(m: usize, cols: &[Vec<(usize, f64)>], p: LuParams) -> SparseLu {
    let a = SparseColMatrix::from_sparse_columns(m, cols).unwrap();
    let sym = SparseLuSymbolic::analyze(&a).unwrap();
    SparseLu::factor(&a, &sym, p).unwrap()
}

fn run(m: usize, per_col: usize) {
    let p = LuParams::default(); // max_updates = 64
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15 ^ (m as u64).wrapping_mul(2654435761));
    let basis: Vec<_> = (0..m)
        .map(|t| covering_col(t, m, per_col, &mut rng))
        .collect();
    let mut lu = factor(m, &basis, p.clone());
    let nnz0 = lu.factor_nnz();
    let mut dense = vec![0.0f64; m];
    let mut times = Vec::new();
    let mut last_etas = Vec::new();
    let mut last_work = Vec::new();
    for t in 0..50usize.min(m - 1) {
        let s = covering_col(t, m, per_col, &mut rng);
        for v in dense.iter_mut() {
            *v = 0.0;
        }
        for &(r, v) in &s {
            dense[r] = v;
        }
        let t0 = Instant::now();
        if lu.update(t, &dense).is_ok() {
            times.push(t0.elapsed().as_secs_f64() * 1e6);
            last_etas.push(lu.last_eta_ops());
            last_work.push(lu.last_update_work());
        }
    }
    let n = times.len().max(1);
    let avg = times.iter().sum::<f64>() / n as f64;
    let avg_eta = last_etas.iter().sum::<usize>() as f64 / n as f64;
    let avg_work = last_work.iter().sum::<usize>() as f64 / n as f64;
    println!(
        "m={m:>5}  factor_nnz/m={:>7.1}  update_avg={:>12.1} us/upd  \
         avg_last_eta_ops={:>7.0}  avg_last_update_work={:>9.0}  (entering nnz = {per_col})",
        nnz0 as f64 / m as f64,
        avg,
        avg_eta,
        avg_work,
    );
}

fn main() {
    println!("# entering column always has 8 nonzeros; an O(bump/nnz) update should be ~flat");
    for &m in &[256usize, 512, 1024, 2048] {
        run(m, 8);
    }
}
