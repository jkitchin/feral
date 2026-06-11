//! N3 (dev/research/repo-review-2026-06-09.md): the parallel multifrontal
//! driver — the `Solver` default whenever `should_parallelize_assembly`
//! fires — ignores `NumericParams::profiler`. The sequential driver
//! records one `SupernodeTiming` per supernode (profiler_smoke.rs guards
//! that), but `factorize_multifrontal_supernodal_parallel` never touches
//! `params.profiler`, so `Solver::with_profiling(true)` returns an empty
//! report on the default dispatch — directly contradicting the
//! `with_profiling` / `profile_report` documentation in solver.rs.
//!
//! This is the cleanly-reproducible facet of N3. (N3 also notes the
//! parallel driver ignores `pattern_reused_hint`'s permute cache and
//! `params.small_leaf`; those are not addressed here.)
//!
//! Reproduction: attach a fresh `Profiler` and run the parallel driver
//! directly (the same gate-bypass `parallel_parity::deep_chain_tree_*`
//! uses, so the test does not need a corpus matrix large enough to clear
//! the flop gate). After the factor, the profiler must hold exactly one
//! timing per supernode.
//!
//!   * Pre-fix: the parallel driver drops the profiler → `n_supernodes`
//!     is 0 (RED).
//!   * Post-fix: the parallel driver records one timing per supernode →
//!     `n_supernodes == sym.supernodes.len()` (GREEN).

use std::sync::{Arc, Mutex};

use feral::numeric::factorize::{
    factorize_multifrontal_supernodal_parallel, NumericParams, Profiler,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::CscMatrix;

/// Tridiagonal SPD (diagonal 4, subdiagonal -1): strictly diagonally
/// dominant ⇒ SPD ⇒ LDLᵀ needs no pivoting. Amalgamates into a deep
/// supernode chain, giving many supernodes for the profiler to record.
fn tridiagonal_spd(n: usize) -> CscMatrix {
    let mut rows = Vec::with_capacity(2 * n);
    let mut cols = Vec::with_capacity(2 * n);
    let mut vals = Vec::with_capacity(2 * n);
    for c in 0..n {
        rows.push(c);
        cols.push(c);
        vals.push(4.0);
        if c + 1 < n {
            rows.push(c + 1);
            cols.push(c);
            vals.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("tridiagonal triplets")
}

#[test]
fn n3_parallel_driver_records_profiler_timings() {
    let a = tridiagonal_spd(2000);
    let sym = symbolic_factorize(&a, &SupernodeParams::default()).expect("symbolic");
    assert!(
        sym.supernodes.len() > 1,
        "test needs a multi-supernode tree (got {})",
        sym.supernodes.len()
    );

    let prof = Arc::new(Mutex::new(Profiler::new()));
    let params = NumericParams {
        profiler: Some(Arc::clone(&prof)),
        ..NumericParams::default()
    };

    // Drive the task-graph parallel driver directly so the parallel path
    // runs regardless of the flop/size gate in the public wrapper.
    let (_factors, _inertia) = factorize_multifrontal_supernodal_parallel(&a, &sym, &params)
        .expect("parallel factor must succeed");

    let report = prof.lock().expect("profiler lock").report();
    assert_eq!(
        report.n_supernodes,
        sym.supernodes.len(),
        "the parallel driver must record one profiler timing per supernode \
         (n_supernodes == 0 means it dropped params.profiler, the N3 bug — \
         Solver::with_profiling(true) would return an empty report on the \
         default parallel dispatch)"
    );
}
