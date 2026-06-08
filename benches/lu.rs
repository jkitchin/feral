//! Criterion benchmarks for the unsymmetric LU basis engine (issue #81).
//!
//! Three stories:
//!   * `dense_*` / `sparse_*` — factor + warm-solve (`ftran`) baselines.
//!   * `update` — the cost of one rank-1 column-replacement (dense
//!     Bartels–Golub, sparse Forrest–Tomlin).
//!   * `update_vs_refactor` — the crossover the whole feature exists for:
//!     a single FT update vs a full refactor of the same basis. The ratio is
//!     how much the in-place update saves over re-factoring.
#![allow(clippy::needless_range_loop)]

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{DenseLu, LuParams, SparseLu, SparseLuSymbolic};

/// Reproducible xorshift64 for bench inputs.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }
    fn f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// Dense diagonally-dominant `m`×`m` basis as column vectors.
fn dense_basis(m: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let mut cols = vec![vec![0.0; m]; m];
    for j in 0..m {
        for i in 0..m {
            cols[j][i] = rng.f64();
        }
        cols[j][j] += m as f64;
    }
    cols
}

/// Block-diagonal sparse basis: `nblocks` dense diagonally-dominant blocks of
/// size `bs`. Within-block column replacements have localized spikes.
fn block_diag(nblocks: usize, bs: usize, seed: u64) -> SparseColMatrix {
    let n = nblocks * bs;
    let mut rng = Rng::new(seed);
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for b in 0..nblocks {
        let base = b * bs;
        for cc in 0..bs {
            for rr in 0..bs {
                let v = if rr == cc {
                    bs as f64 + rng.f64().abs()
                } else {
                    rng.f64()
                };
                cols[base + cc].push((base + rr, v));
            }
        }
    }
    SparseColMatrix::from_sparse_columns(n, &cols).expect("block_diag")
}

/// A within-block diagonally-dominant replacement column for `slot`.
fn within_block_col(n: usize, bs: usize, slot: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let base = (slot / bs) * bs;
    let mut c = vec![0.0; n];
    for rr in 0..bs {
        let i = base + rr;
        c[i] = if i == slot { bs as f64 } else { rng.f64() };
    }
    c
}

fn bench_dense(c: &mut Criterion) {
    let mut g = c.benchmark_group("dense");
    for &m in &[16usize, 64, 256] {
        let cols = dense_basis(m, 1);
        g.bench_with_input(BenchmarkId::new("factor", m), &m, |b, _| {
            b.iter(|| {
                let lu = DenseLu::factor(black_box(&cols), m, LuParams::default());
                black_box(lu.is_ok())
            })
        });
        let lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
        let rhs: Vec<f64> = (0..m).map(|i| 1.0 + (i % 7) as f64).collect();
        g.bench_with_input(BenchmarkId::new("ftran", m), &m, |b, _| {
            b.iter_batched(
                || (lu.clone(), rhs.clone()),
                |(mut lu, mut r)| {
                    lu.ftran(&mut r).expect("ftran");
                    black_box(r[0])
                },
                BatchSize::SmallInput,
            )
        });
        let new_col: Vec<f64> = (0..m).map(|i| 1.0 + (i % 3) as f64).collect();
        g.bench_with_input(BenchmarkId::new("update", m), &m, |b, _| {
            b.iter_batched(
                || lu.clone(),
                |mut lu| {
                    let _ = lu.update(black_box(m / 2), black_box(&new_col));
                },
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

fn bench_sparse(c: &mut Criterion) {
    let bs = 25;
    let mut g = c.benchmark_group("sparse");
    for &nblocks in &[8usize, 40, 200] {
        let n = nblocks * bs;
        let a = block_diag(nblocks, bs, 7);
        let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
        g.bench_with_input(BenchmarkId::new("factor", n), &n, |b, _| {
            b.iter(|| {
                let lu = SparseLu::factor(black_box(&a), &sym, LuParams::default());
                black_box(lu.is_ok())
            })
        });
        let lu = SparseLu::factor(&a, &sym, LuParams::default()).expect("factor");
        let rhs: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
        g.bench_with_input(BenchmarkId::new("ftran", n), &n, |b, _| {
            b.iter_batched(
                || (lu.clone(), rhs.clone()),
                |(mut lu, mut r)| {
                    lu.ftran(&mut r).expect("ftran");
                    black_box(r[0])
                },
                BatchSize::SmallInput,
            )
        });
        let slot = n / 2;
        let col = within_block_col(n, bs, slot, 99);
        g.bench_with_input(BenchmarkId::new("ft_update", n), &n, |b, _| {
            b.iter_batched(
                || lu.clone(),
                |mut lu| {
                    let _ = lu.update(black_box(slot), black_box(&col));
                },
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

fn bench_update_vs_refactor(c: &mut Criterion) {
    let bs = 25;
    let mut g = c.benchmark_group("update_vs_refactor");
    for &nblocks in &[40usize, 200] {
        let n = nblocks * bs;
        let a = block_diag(nblocks, bs, 13);
        let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
        let lu = SparseLu::factor(&a, &sym, LuParams::default()).expect("factor");
        let slot = n / 2;
        let col = within_block_col(n, bs, slot, 31);

        g.bench_with_input(BenchmarkId::new("ft_update", n), &n, |b, _| {
            b.iter_batched(
                || lu.clone(),
                |mut lu| {
                    let _ = lu.update(black_box(slot), black_box(&col));
                },
                BatchSize::SmallInput,
            )
        });
        g.bench_with_input(BenchmarkId::new("refactor", n), &n, |b, _| {
            b.iter_batched(
                || lu.clone(),
                |mut lu| {
                    lu.refactor(black_box(&a), &sym).expect("refactor");
                },
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

criterion_group!(benches, bench_dense, bench_sparse, bench_update_vs_refactor);
criterion_main!(benches);
