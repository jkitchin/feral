//! Phase-timing microbench for feral's unsymmetric `SparseLu`, sized like
//! discopt's B&B node bases (m ≈ 130–256, a few % dense). It answers the
//! question behind issue #130 for the discopt-simplex use case: on these
//! bases, is the per-refactorization cost in the *symbolic analyze*, the
//! *numeric factor*, or the *ftran/btran solves*? #130 only speeds up the
//! solves — if analyze/factor dominate, it is the wrong lever for discopt.
//!
//! Usage: cargo run --release --bin probe_lu_phases -- [m] [density_pct]

use std::time::Instant;

use feral::{LuParams, SparseColMatrix, SparseLu, SparseLuSymbolic};

// Deterministic LCG (no Instant/rand needed for reproducibility).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 32) as u32 as f64 / u32::MAX as f64
    }
}

/// Build a random nonsingular m×m sparse matrix with ~`density` off-diagonal
/// fill, strong diagonal (so it factors without trouble). Columns are
/// `Vec<(row, val)>` as feral's `from_sparse_columns` wants.
fn random_basis(m: usize, density: f64, seed: u64) -> Vec<Vec<(usize, f64)>> {
    let mut rng = Rng(seed);
    let mut cols = vec![Vec::new(); m];
    for (j, col) in cols.iter_mut().enumerate() {
        for i in 0..m {
            if i == j {
                col.push((i, m as f64)); // dominant diagonal
            } else if rng.unit() < density {
                col.push((i, rng.unit() * 2.0 - 1.0));
            }
        }
    }
    cols
}

fn main() {
    let mut args = std::env::args().skip(1);
    let m: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    let density_pct: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let density = density_pct / 100.0;

    let cols = random_basis(m, density, 0xC0FFEE);
    let a = SparseColMatrix::from_sparse_columns(m, &cols).expect("valid basis");
    let nnz: usize = cols.iter().map(|c| c.len()).sum();
    let params = LuParams::default();

    // Warm + time each phase separately. Analyze and factor are the
    // per-refactorization costs; ftran/btran are the per-pivot solve costs.
    let reps_af = 200usize;
    let reps_solve = 2000usize;

    // analyze
    let mut best_analyze = u128::MAX;
    for _ in 0..reps_af {
        let t = Instant::now();
        let _sym = SparseLuSymbolic::analyze(&a).expect("analyze");
        best_analyze = best_analyze.min(t.elapsed().as_nanos());
    }
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");

    // factor
    let mut best_factor = u128::MAX;
    for _ in 0..reps_af {
        let t = Instant::now();
        let _lu = SparseLu::factor(&a, &sym, params.clone()).expect("factor");
        best_factor = best_factor.min(t.elapsed().as_nanos());
    }
    let mut lu = SparseLu::factor(&a, &sym, params.clone()).expect("factor");
    let fnnz = lu.factor_nnz();

    // ftran / btran (dense unit RHS — the pessimistic, non-hypersparse case;
    // a sparse-e_i RHS would be the hyper-sparse case #130 targets).
    let mut best_ftran = u128::MAX;
    for k in 0..reps_solve {
        let mut rhs = vec![0.0f64; m];
        rhs[k % m] = 1.0;
        let t = Instant::now();
        lu.ftran(&mut rhs).expect("ftran");
        best_ftran = best_ftran.min(t.elapsed().as_nanos());
    }
    let mut best_btran = u128::MAX;
    for k in 0..reps_solve {
        let mut rhs = vec![0.0f64; m];
        rhs[k % m] = 1.0;
        let t = Instant::now();
        lu.btran(&mut rhs).expect("btran");
        best_btran = best_btran.min(t.elapsed().as_nanos());
    }

    let us = |ns: u128| ns as f64 / 1000.0;
    println!("m={m} density={density_pct:.1}% nnz(A)={nnz} nnz(factor)={fnnz}",);
    println!(
        "  analyze={:.2}us  factor={:.2}us  ftran={:.3}us  btran={:.3}us",
        us(best_analyze),
        us(best_factor),
        us(best_ftran),
        us(best_btran),
    );
    let refactor = best_analyze + best_factor;
    let solve = best_ftran + best_btran;
    println!(
        "  refactor(analyze+factor)={:.2}us  one ftran+btran={:.3}us  \
         ratio refactor/solve={:.0}x",
        us(refactor),
        us(solve),
        refactor as f64 / solve.max(1) as f64,
    );
}
