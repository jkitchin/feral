//! Criterion benchmark for the Forrest–Tomlin update chain on a real
//! wide-bump basis: the MINLPLib `casctanks` McCormick LP relaxation extracted
//! from discopt's revised simplex (discopt#229). This is the workload that
//! drives `SparseLu::update` / `eliminate_bump` into its O(bump²) worst case
//! (avg bump width ≈ 574 on the full trace).
//!
//! It times only the **update chain**: each segment's basis is factored fresh
//! in untimed `iter_batched` setup, and the timed routine applies that
//! segment's column-replacement updates. This isolates the bump-elimination
//! cost that the sub-diagonal-index and dense-workspace optimizations target
//! (see `dev/research/bump-elimination-speedup-2026-06-18.md`).
//!
//! Default fixture: `tests/data/lu_trace/casctanks.txt` (3 widest-bump
//! segments). Point `FERAL_LU_TRACE` at the full extracted trace to benchmark
//! all 36 segments / 1702 updates.

use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{LuParams, SparseLu, SparseLuSymbolic};

type SparseCol = Vec<(usize, f64)>;

struct Segment {
    basis: Vec<SparseCol>,
    updates: Vec<(usize, SparseCol)>,
}

fn parse_sparse_col(fields: &[&str]) -> SparseCol {
    let mut col: SparseCol = fields
        .iter()
        .filter_map(|tok| {
            let (r, v) = tok.split_once(':')?;
            Some((r.parse::<usize>().ok()?, v.parse::<f64>().ok()?))
        })
        .collect();
    col.sort_by_key(|&(r, _)| r);
    col
}

fn load_trace() -> (usize, Vec<Segment>) {
    let path = std::env::var("FERAL_LU_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/lu_trace/casctanks.txt")
        });
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read trace {}: {e}", path.display()));
    let mut m = 0usize;
    let mut segments: Vec<Segment> = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("M") => m = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            Some("REFACTOR") => segments.push(Segment {
                basis: Vec::new(),
                updates: Vec::new(),
            }),
            Some("BCOL") => {
                let rest: Vec<&str> = it.collect();
                let col = parse_sparse_col(&rest[2.min(rest.len())..]);
                segments.last_mut().unwrap().basis.push(col);
            }
            Some("UPDATE") => {
                let rest: Vec<&str> = it.collect();
                let slot: usize = rest[0].parse().unwrap();
                let col = parse_sparse_col(&rest[2.min(rest.len())..]);
                segments.last_mut().unwrap().updates.push((slot, col));
            }
            _ => {}
        }
    }
    (m, segments)
}

fn to_dense(col: &SparseCol, m: usize) -> Vec<f64> {
    let mut d = vec![0.0; m];
    for &(r, v) in col {
        d[r] = v;
    }
    d
}

fn params() -> LuParams {
    LuParams {
        max_updates: 1_000_000,
        max_growth: 1e30,
        ..LuParams::default()
    }
}

fn bench_update_chain(c: &mut Criterion) {
    let (m, segments) = load_trace();
    let total_updates: usize = segments.iter().map(|s| s.updates.len()).sum();

    // Pre-build the dense entering columns once (not part of the update cost).
    struct Prepared<'a> {
        basis: &'a [SparseCol],
        updates: Vec<(usize, Vec<f64>)>,
    }
    let prepared: Vec<Prepared<'_>> = segments
        .iter()
        .map(|seg| Prepared {
            basis: &seg.basis,
            updates: seg
                .updates
                .iter()
                .map(|(slot, col)| (*slot, to_dense(col, m)))
                .collect(),
        })
        .collect();

    let mut group = c.benchmark_group("casctanks_ft_update");
    group.sample_size(10);
    group.bench_function(format!("chain_{}_updates_m{}", total_updates, m), |b| {
        b.iter_batched(
            // Untimed setup: factor every segment fresh (updates mutate the factor).
            || {
                prepared
                    .iter()
                    .map(|p| {
                        let a = SparseColMatrix::from_sparse_columns(m, p.basis).expect("matrix");
                        let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
                        let lu = SparseLu::factor(&a, &sym, params()).expect("factor");
                        (lu, p.updates.clone())
                    })
                    .collect::<Vec<_>>()
            },
            // Timed: the update chain only.
            |segs| {
                for (mut lu, ups) in segs {
                    for (slot, dcol) in &ups {
                        let _ = black_box(lu.update(*slot, dcol));
                    }
                    black_box(&lu);
                }
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_update_chain);
criterion_main!(benches);
