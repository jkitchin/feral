//! Issue #52 — overhead bench for the opt-in `with_profiling` knob.
//!
//! The plan (`dev/plans/issue-52-opt-in-stats.md` §B2) requires the
//! default-off path to stay essentially free: a debugging knob that
//! is not on must not cost real users measurable time. This bench
//! empirically validates that constraint by measuring `Solver::factor`
//! with profiling disabled vs enabled on three representative
//! tridiagonal SPD shapes (n = 64, 256, 1024). The n = 64 shape
//! routes through the dense fast path; n = 256 and n = 1024 route
//! through the multifrontal sparse driver and exercise the
//! per-supernode profiler timing loop.
//!
//! Acceptance: default-off `factor_default/*` must be within criterion
//! noise of a pre-issue-52 baseline. The `factor_with_profiling/*`
//! numbers document the cost users pay when they opt in.
//!
//! Each iteration constructs a fresh `Solver`, so the first factor
//! is always a cache miss — that is the worst case for symbolic
//! profiling because the symbolic profiler is wired in only on the
//! miss branch. A second factor on the same matrix is included to
//! exercise the cache-hit branch (where profiling overhead is
//! limited to the numeric per-supernode timer).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use feral::scaling::ScalingStrategy;
use feral::{CscMatrix, FactorStatus, Solver};

/// SPD tridiagonal with `4` on the diagonal and `-1` on the first
/// sub-diagonal. Lower-triangle CSC.
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

fn run_factor(matrix: &CscMatrix, profiling: bool) {
    let mut solver = Solver::new()
        .with_scaling(ScalingStrategy::Identity)
        .with_profiling(profiling);
    let s1 = solver.factor(matrix, None);
    assert!(matches!(s1, FactorStatus::Success));
    // Second factor exercises the symbolic-cache-hit branch so the
    // bench reflects both halves of a realistic re-solve workflow.
    let s2 = solver.factor(matrix, None);
    assert!(matches!(s2, FactorStatus::Success));
    black_box(solver.factors());
}

fn bench_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("issue52_overhead");
    for &n in &[64usize, 256, 1024] {
        let m = tridiagonal_spd(n);
        group.bench_with_input(BenchmarkId::new("factor_default", n), &m, |b, m| {
            b.iter(|| run_factor(black_box(m), false));
        });
        group.bench_with_input(BenchmarkId::new("factor_with_profiling", n), &m, |b, m| {
            b.iter(|| run_factor(black_box(m), true));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_overhead);
criterion_main!(benches);
