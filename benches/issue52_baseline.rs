//! Pre-issue-52 baseline bench. Uses only the `Solver` API that
//! exists on `main` (no `with_profiling` call), so this same file
//! can be checked out on a `main` worktree and run unchanged to
//! produce a fair before/after comparison for issue #52's hard
//! constraint: the default-off path must be within bench noise of
//! the pre-issue-52 baseline.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use feral::scaling::ScalingStrategy;
use feral::{CscMatrix, FactorStatus, Solver};

fn tridiagonal_spd(n: usize) -> CscMatrix {
    let mut rows = Vec::with_capacity(2 * n);
    let mut cols = Vec::with_capacity(2 * n);
    let mut vals = Vec::with_capacity(2 * n);
    for j in 0..n {
        rows.push(j);
        cols.push(j);
        vals.push(4.0);
        if j + 1 < n {
            rows.push(j + 1);
            cols.push(j);
            vals.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("tridiagonal_spd")
}

fn run_factor(matrix: &CscMatrix) {
    let mut solver = Solver::new().with_scaling(ScalingStrategy::Identity);
    let s1 = solver.factor(matrix, None);
    assert!(matches!(s1, FactorStatus::Success));
    let s2 = solver.factor(matrix, None);
    assert!(matches!(s2, FactorStatus::Success));
    black_box(solver.factors());
}

fn bench_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("issue52_baseline");
    for &n in &[64usize, 256, 1024] {
        let m = tridiagonal_spd(n);
        group.bench_with_input(BenchmarkId::new("factor", n), &m, |b, m| {
            b.iter(|| run_factor(black_box(m)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_baseline);
criterion_main!(benches);
