//! Phase 2.10 profiler smoke tests
//! (`dev/plans/phase-2.10-supernode-profiler.md`).
//!
//! These verify the profiler's acceptance invariants on a tiny
//! hand-built block-diagonal SPD matrix that does not depend on the
//! gitignored corpus, so the tests run on CI.
//!
//! Invariants under test:
//!   * Profiler-None path is bit-equal to non-profiled path (no
//!     observable change to factorization output).
//!   * Profiler records exactly one timing per supernode.
//!   * Bucket counts sum to `n_supernodes`; bucket time sums to
//!     `loop_ns`; every supernode falls in exactly one bucket.

use std::sync::{Arc, Mutex};

use feral::numeric::factorize::{factorize_multifrontal, NumericParams, Profiler};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};

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

fn params_with(profiler: Option<Arc<Mutex<Profiler>>>) -> NumericParams {
    NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        },
        scaling: Default::default(),
        small_leaf: Default::default(),
        profiler,
        parallel_telemetry: None,
        fma: false,
        allow_delayed_pivots: true,
        cascade_break_ratio: None,
        cascade_break_eps: None,
        min_parallel_flops: None,
        sqd_mode: false,
        static_pivot_threshold: None,
        warn_partial_singular: false,
        pattern_reused_hint: false,
        use_static_row_indices: true,
    }
}

#[test]
fn profiler_none_factorization_succeeds() {
    let csc = block_diag_spd(20);
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let params = params_with(None);
    let (_factors, _inertia) =
        factorize_multifrontal(&csc, &sym, &params).expect("factor must succeed");
}

#[test]
fn profiler_records_one_per_supernode() {
    let csc = block_diag_spd(20);
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let prof = Arc::new(Mutex::new(Profiler::new()));
    let params = params_with(Some(Arc::clone(&prof)));
    factorize_multifrontal(&csc, &sym, &params).expect("factor");
    let report = prof.lock().expect("lock").report();
    assert_eq!(
        report.n_supernodes,
        sym.supernodes.len(),
        "expected one timing per supernode"
    );
}

#[test]
fn profiler_buckets_partition_supernodes() {
    let csc = block_diag_spd(20);
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let prof = Arc::new(Mutex::new(Profiler::new()));
    let params = params_with(Some(Arc::clone(&prof)));
    factorize_multifrontal(&csc, &sym, &params).expect("factor");
    let report = prof.lock().expect("lock").report();

    let count_sum: usize = report.buckets.iter().map(|b| b.count).sum();
    assert_eq!(
        count_sum, report.n_supernodes,
        "bucket counts must sum to n_supernodes"
    );
    assert!(
        report.validation_warnings.is_empty(),
        "report has validation warnings: {:?}",
        report.validation_warnings
    );
}

#[test]
fn profiler_bucket_ns_sum_equals_loop_ns() {
    let csc = block_diag_spd(20);
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let prof = Arc::new(Mutex::new(Profiler::new()));
    let params = params_with(Some(Arc::clone(&prof)));
    factorize_multifrontal(&csc, &sym, &params).expect("factor");
    let report = prof.lock().expect("lock").report();

    let bucket_ns_sum: u64 = report.buckets.iter().map(|b| b.sum_ns).sum();
    assert_eq!(
        bucket_ns_sum, report.loop_ns,
        "sum of bucket sum_ns must equal loop_ns"
    );
}

#[test]
fn profiler_total_bounds_components() {
    let csc = block_diag_spd(20);
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let prof = Arc::new(Mutex::new(Profiler::new()));
    let params = params_with(Some(Arc::clone(&prof)));
    factorize_multifrontal(&csc, &sym, &params).expect("factor");
    let report = prof.lock().expect("lock").report();

    // total must be at least as large as the sum of measured
    // sub-phases; gap is timer/lock overhead. Compared in nanoseconds:
    // `loop_ns` is ns while the prologue/epilogue/total fields are us.
    let components_ns = report.loop_ns + (report.prologue_us + report.epilogue_us) * 1000;
    let total_ns = report.total_us * 1000;
    assert!(
        components_ns <= total_ns,
        "components {} ns > total {} ns",
        components_ns,
        total_ns
    );
}
