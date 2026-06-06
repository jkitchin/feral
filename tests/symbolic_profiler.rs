//! Phase 2.13b symbolic profiler smoke tests
//! (`dev/research/phase-2.13b-symbolic-profiler.md`).
//!
//! Verifies the per-stage symbolic profiler records the expected
//! stages on a tiny block-diagonal SPD matrix that does not depend
//! on the gitignored corpus.
//!
//! Invariants:
//!   * Profiler-None path produces an identical SymbolicFactorization
//!     to the profiler-Some path (no observable behavior change).
//!   * Every expected stage name appears at most once per call.
//!   * `accounted_us` is non-negative; `total_us >= accounted_us`
//!     (no validation warnings emitted).

use std::sync::{Arc, Mutex};

use feral::symbolic::supernode::OrderingPreprocess;
use feral::symbolic::{
    pick_ordering_preprocess, symbolic_factorize, SupernodeParams, SymbolicProfiler,
};
use feral::CscMatrix;

/// Arrow matrix: a dense hub column 0 plus `n-1` degree-1 leaf columns.
/// Stored as the lower triangle, columns `1..n` each hold only their
/// diagonal, so the `nnz_col <= 2` share is `(n-1)/n` — well above the
/// 0.30 `LOW_DEGREE_THRESHOLD`. This is the arrow/slack-KKT signature
/// that `pick_ordering_preprocess` routes to `LdltCompress` (issue #80),
/// so it is the minimal pattern that exercises the MC64 preprocessor.
fn arrow_kkt(n: usize) -> CscMatrix {
    let mut rows = vec![0usize];
    let mut cols = vec![0usize];
    let mut vals = vec![2.0f64];
    for j in 1..n {
        // off-diagonal (j, 0) — stored in column 0 (the hub)
        rows.push(j);
        cols.push(0);
        vals.push(1.0);
        // diagonal (j, j)
        rows.push(j);
        cols.push(j);
        vals.push(2.0 + j as f64);
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("build arrow matrix")
}

fn block_diag_spd(k: usize) -> CscMatrix {
    let n = 2 * k;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for b in 0..k {
        let base = 2 * b;
        rows.push(base);
        cols.push(base);
        vals.push(4.0);
        rows.push(base + 1);
        cols.push(base);
        vals.push(1.0);
        rows.push(base + 1);
        cols.push(base + 1);
        vals.push(3.0);
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("build block-diag matrix")
}

fn expected_stages_minimum() -> &'static [&'static str] {
    // The Renumber path is conditional on `bias.iter().any(...)`;
    // on a block-diagonal SPD with no opportunity for amalgamation
    // the bias may be empty, so the renumber stage records 0 µs but
    // is still recorded. Listing only stages that always run.
    &[
        "symmetric_pattern",
        "pick_preprocess",
        "ordering",
        "permute1",
        "etree_initial",
        "postorder",
        "perm_compose",
        "permute2",
        "etree_relabel",
        "col_counts",
        "renumber",
        "find_supernodes",
        "small_leaf_groups",
        "peak_contrib",
    ]
}

#[test]
fn profiler_none_succeeds_and_matches() {
    let csc = block_diag_spd(20);

    let sym_a = symbolic_factorize(&csc, &SupernodeParams::default()).expect("none");

    let prof = Arc::new(Mutex::new(SymbolicProfiler::new()));
    let params = SupernodeParams {
        symbolic_profiler: Some(Arc::clone(&prof)),
        ..SupernodeParams::default()
    };
    let sym_b = symbolic_factorize(&csc, &params).expect("some");

    assert_eq!(sym_a.perm, sym_b.perm);
    assert_eq!(sym_a.supernodes.len(), sym_b.supernodes.len());
}

#[test]
fn profiler_records_all_expected_stages() {
    let csc = block_diag_spd(20);
    let prof = Arc::new(Mutex::new(SymbolicProfiler::new()));
    let params = SupernodeParams {
        symbolic_profiler: Some(Arc::clone(&prof)),
        ..SupernodeParams::default()
    };
    let _ = symbolic_factorize(&csc, &params).expect("symbolic");

    let p = prof.lock().expect("lock");
    let names: Vec<&'static str> = p.stages().iter().map(|s| s.name).collect();
    for &expected in expected_stages_minimum() {
        assert!(
            names.contains(&expected),
            "stage `{}` not recorded; got {:?}",
            expected,
            names
        );
    }
}

#[test]
fn profiler_total_bounds_accounted() {
    let csc = block_diag_spd(20);
    let prof = Arc::new(Mutex::new(SymbolicProfiler::new()));
    let params = SupernodeParams {
        symbolic_profiler: Some(Arc::clone(&prof)),
        ..SupernodeParams::default()
    };
    let _ = symbolic_factorize(&csc, &params).expect("symbolic");
    let report = prof.lock().expect("lock").report();

    assert!(
        report.accounted_us <= report.total_us,
        "accounted {} > total {}",
        report.accounted_us,
        report.total_us
    );
    assert!(
        report.validation_warnings.is_empty(),
        "validation warnings: {:?}",
        report.validation_warnings
    );
}

#[test]
fn ldlt_compress_recorded_separately_from_ordering() {
    // Issue #80: on the arrow/slack-KKT signature the MC64 matching in
    // the LdltCompress preprocessor used to be folded into the "ordering"
    // stage, mis-attributing ~53s of MC64 on pf22 to the fill-reducing
    // ordering (which is ~0.3s). The preprocessor must get its own stage.
    let csc = arrow_kkt(256);
    assert_eq!(
        pick_ordering_preprocess(&csc),
        OrderingPreprocess::LdltCompress,
        "test matrix must route to LdltCompress"
    );

    let prof = Arc::new(Mutex::new(SymbolicProfiler::new()));
    let params = SupernodeParams {
        symbolic_profiler: Some(Arc::clone(&prof)),
        ..SupernodeParams::default()
    };
    let _ = symbolic_factorize(&csc, &params).expect("symbolic");

    let p = prof.lock().expect("lock");
    let names: Vec<&'static str> = p.stages().iter().map(|s| s.name).collect();
    assert!(
        names.contains(&"ldlt_compress"),
        "MC64/LdltCompress preprocessor must have its own stage; got {names:?}"
    );
    assert!(
        names.contains(&"ordering"),
        "the fill-reducing ordering must still record an `ordering` stage; got {names:?}"
    );
    // Each must appear at most once per call.
    assert_eq!(
        names.iter().filter(|&&s| s == "ldlt_compress").count(),
        1,
        "ldlt_compress recorded more than once"
    );
    assert_eq!(
        names.iter().filter(|&&s| s == "ordering").count(),
        1,
        "ordering recorded more than once"
    );
}

#[test]
fn profiler_pct_of_total_consistent() {
    let csc = block_diag_spd(20);
    let prof = Arc::new(Mutex::new(SymbolicProfiler::new()));
    let params = SupernodeParams {
        symbolic_profiler: Some(Arc::clone(&prof)),
        ..SupernodeParams::default()
    };
    let _ = symbolic_factorize(&csc, &params).expect("symbolic");
    let report = prof.lock().expect("lock").report();

    if report.total_us == 0 {
        // Sub-microsecond on tiny n; nothing to check.
        return;
    }
    let stage_pct_sum: f64 = report.stages.iter().map(|s| s.pct_of_total).sum();
    // overhead_pct + sum(stage_pct) ≈ 100 (within rounding from u64 us).
    assert!(
        (stage_pct_sum + report.overhead_pct - 100.0).abs() < 1e-6,
        "stage_pct_sum={} overhead_pct={} sum should be 100",
        stage_pct_sum,
        report.overhead_pct
    );
}
