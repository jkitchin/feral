//! Replay of a real Forrest–Tomlin LU update chain extracted from discopt's
//! revised-simplex solve of the MINLPLib `casctanks` McCormick LP relaxation
//! (discopt#229). This is the *wide non-localized-spike* regime that drives
//! `SparseLu::update` (`src/lu/sparse_update.rs:eliminate_bump`) into its
//! O(bump²) worst case (avg bump width ≈ 574, max ≈ m = 2169 on the full trace).
//!
//! The fixture is a segmented trace: at each refactorization the full basis is
//! dumped (`BCOL` lines), followed by the column-replacement updates applied
//! before the next refactor (`UPDATE` lines). Replay = {factor each segment's
//! basis, apply its updates in order}.
//!
//! This test is a **correctness regression guard**: it pins that the sparse FT
//! update produces correct solves on this real chain, so the bump-elimination
//! speedups (sub-diagonal index, dense workspace — see
//! `dev/research/bump-elimination-speedup-2026-06-18.md`) can be proven
//! numerics-preserving. It checks two independent oracles per update:
//!   1. oracle-free true residual  ‖B·ftran(e) − e‖∞ / ‖e‖∞  on the tracked
//!      current basis B (drives the *actual* residual down, no self-truth), and
//!   2. cross-agreement of `SparseLu::ftran` against the `DenseLu` factor.
//!
//! Default fixture: `tests/data/lu_trace/casctanks.txt` (a reduced, widest-bump
//! subset). Point `FERAL_LU_TRACE` at the full extracted trace to replay all
//! segments. Regenerate the full trace per the commands in
//! `dev/journal/2026-06-18-01.org`.

use std::path::PathBuf;

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{DenseLu, LuParams, SparseLu, SparseLuSymbolic};

/// One sparse column: (row, value) pairs, row-sorted.
type SparseCol = Vec<(usize, f64)>;

struct Segment {
    /// `m` basis columns at the refactor that opened this segment.
    basis: Vec<SparseCol>,
    /// Column replacements applied in order: (leaving_slot, entering_col).
    updates: Vec<(usize, SparseCol)>,
}

struct Trace {
    m: usize,
    segments: Vec<Segment>,
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

fn parse_trace(text: &str) -> Trace {
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
                // BCOL <j> <nnz> <r:v>...  — j is emitted in order 0..m-1.
                let rest: Vec<&str> = it.collect();
                let col = parse_sparse_col(&rest[2.min(rest.len())..]);
                segments
                    .last_mut()
                    .expect("BCOL before REFACTOR")
                    .basis
                    .push(col);
            }
            Some("UPDATE") => {
                // UPDATE <slot> <nnz> <r:v>...
                let rest: Vec<&str> = it.collect();
                let slot: usize = rest[0].parse().expect("update slot");
                let col = parse_sparse_col(&rest[2.min(rest.len())..]);
                segments
                    .last_mut()
                    .expect("UPDATE before REFACTOR")
                    .updates
                    .push((slot, col));
            }
            _ => {}
        }
    }
    Trace { m, segments }
}

fn to_dense(col: &SparseCol, m: usize) -> Vec<f64> {
    let mut d = vec![0.0; m];
    for &(r, v) in col {
        d[r] = v;
    }
    d
}

/// Generous params so feral does not self-trigger NeedsRefactor mid-segment;
/// the trace's REFACTOR markers drive segmentation, not feral's internal budget.
fn replay_params() -> LuParams {
    LuParams {
        max_updates: 1_000_000,
        max_growth: 1e30,
        ..LuParams::default()
    }
}

/// ‖B·x − rhs‖∞ / ‖rhs‖∞ for the current basis `b` (sparse columns), x solved.
fn true_residual(b: &[SparseCol], x: &[f64], rhs: &[f64]) -> f64 {
    let mut r = vec![0.0; rhs.len()];
    for (j, col) in b.iter().enumerate() {
        let xj = x[j];
        for &(i, v) in col {
            r[i] += v * xj;
        }
    }
    let num = r
        .iter()
        .zip(rhs)
        .map(|(&ri, &bi)| (ri - bi).abs())
        .fold(0.0, f64::max);
    let den = rhs.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300);
    num / den
}

fn rel_inf_diff(a: &[f64], b: &[f64]) -> f64 {
    let num = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max);
    let den = a.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300);
    num / den
}

fn fixture_text() -> Option<String> {
    let path = std::env::var("FERAL_LU_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/lu_trace/casctanks.txt")
        });
    match std::fs::read_to_string(&path) {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!(
                "skipping casctanks replay: cannot read {}: {e}",
                path.display()
            );
            None
        }
    }
}

#[test]
fn casctanks_update_chain_stays_correct() {
    let Some(text) = fixture_text() else { return };
    let trace = parse_trace(&text);
    let m = trace.m;
    assert!(m > 0 && !trace.segments.is_empty(), "empty/invalid trace");

    // A fixed, well-spread RHS for the solve checks.
    let rhs: Vec<f64> = (0..m).map(|i| 1.0 + ((i % 7) as f64) * 0.5).collect();

    // The DenseLu cross-oracle needs an O(m³) dense factor per segment — too
    // expensive for a debug-build CI run on m≈2169. The oracle-free true
    // residual is the always-on correctness check; opt into the second oracle
    // (DenseLu) with FERAL_LU_TRACE_DENSE=1 for release verification.
    let use_dense_oracle = std::env::var("FERAL_LU_TRACE_DENSE").is_ok();

    let mut worst_resid = 0.0_f64;
    let mut worst_oracle = 0.0_f64;
    let mut applied = 0usize;
    let mut refactor_signals = 0usize;

    for (s, seg) in trace.segments.iter().enumerate() {
        assert_eq!(
            seg.basis.len(),
            m,
            "segment {s} basis has {} cols, expected {m}",
            seg.basis.len()
        );

        let factor_sparse = |basis: &[SparseCol]| -> SparseLu {
            let a = SparseColMatrix::from_sparse_columns(m, basis).expect("basis matrix");
            let symbolic = SparseLuSymbolic::analyze(&a).expect("analyze");
            SparseLu::factor(&a, &symbolic, replay_params()).expect("sparse factor")
        };
        let factor_dense = |basis: &[SparseCol]| -> DenseLu {
            let cols: Vec<Vec<f64>> = basis.iter().map(|c| to_dense(c, m)).collect();
            DenseLu::factor(&cols, m, replay_params()).expect("dense factor")
        };

        // Track the current basis so we can form the true residual and re-seed
        // either factor when it declines an incremental update.
        let mut basis = seg.basis.clone();
        let mut sparse = factor_sparse(&basis);
        let mut dense = use_dense_oracle.then(|| factor_dense(&basis));

        for (slot, col) in &seg.updates {
            let dcol = to_dense(col, m);
            // Apply to the tracked basis first; the factor must end up
            // representing this post-update basis, whether incrementally or by
            // a fresh refactor (the system-under-test picks its own refactor
            // cadence — a NeedsRefactor is not a failure).
            basis[*slot] = col.clone();
            applied += 1;

            if sparse.update(*slot, &dcol).is_err() {
                refactor_signals += 1;
                sparse = factor_sparse(&basis);
            }

            // Oracle-free: the solve must drive the true residual to zero.
            let mut xs = rhs.clone();
            sparse.ftran(&mut xs).expect("sparse ftran");
            worst_resid = worst_resid.max(true_residual(&basis, &xs, &rhs));

            // Second oracle (opt-in): an independent DenseLu factor of the same
            // basis must produce the same solve.
            if let Some(dense) = dense.as_mut() {
                if dense.update(*slot, &dcol).is_err() {
                    *dense = factor_dense(&basis);
                }
                let mut xd = rhs.clone();
                dense.ftran(&mut xd).expect("dense ftran");
                worst_oracle = worst_oracle.max(rel_inf_diff(&xs, &xd));
            }
        }
    }

    eprintln!(
        "casctanks replay: m={m} segments={} applied_updates={applied} refactor_signals={refactor_signals} \
         worst_true_residual={worst_resid:.3e} worst_sparse_vs_dense={worst_oracle:.3e}",
        trace.segments.len()
    );

    // The chain is well-conditioned (coeff range ~4 orders); FT solves should be
    // tight. These bounds match the existing lu_sparse.rs residual expectations.
    assert!(
        worst_resid < 1e-7,
        "true residual too large: {worst_resid:.3e}"
    );
    assert!(
        worst_oracle < 1e-7,
        "sparse vs dense disagreement: {worst_oracle:.3e}"
    );
    assert!(
        applied > 0,
        "no updates applied — trace did not exercise the update path"
    );
}
