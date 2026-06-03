//! Issue #67 regression: on uniformly-thin large matrices in the
//! `(10_000, 100_000]` band (no arrow/border signature) the default
//! ordering must prefer AMF over MetisND. A full corpus A/B on factor+solve
//! wall-time found AMF wins or ties MetisND across the whole band; MetisND's
//! nested-dissection separators do not pay off on these discretization
//! patterns at this scale, and its symbolic ordering is 2–5× more expensive.
//!
//! The fixtures `tests/data/large/{bratu3d,cont-201}.mtx` are SuiteSparse
//! downloads (gitignored, fetched on demand via
//! `dev/scripts/fetch_large_matrices.sh`). When absent (e.g. CI) each check
//! prints a SKIP line and passes — a local/opt-in regression guard, not a
//! CI gate.
//!
//! Oracle (external — measured symbolic fill_auto vs fill_metis from
//! `bench_orderings`, session 2026-06-03-04):
//!   bratu3d (n=27792):  AMF ≈ 6.22M nnz_L, MetisND ≈ 9.86M.
//!   cont-201 (n=80595): AMF ≈ 4.50M nnz_L, MetisND ≈ 5.95M.
//! The `resolved_method == Amf` assertion is the routing regression; the
//! nnz_L ceiling separates AMF from the MetisND fill with margin.

use feral::read_mtx;
use feral::symbolic::{symbolic_factorize, OrderingMethod, SupernodeParams};
use std::path::Path;

fn check(file: &str, n_expect: usize, nnz_l_ceiling: usize, metis_nnz_l: usize) {
    let path = Path::new(file);
    if !path.is_file() {
        eprintln!(
            "SKIP: {} not present. Fetch with dev/scripts/fetch_large_matrices.sh.",
            path.display()
        );
        return;
    }

    let m = read_mtx(path)
        .and_then(|mtx| mtx.to_csc())
        .unwrap_or_else(|e| panic!("read {file}: {e}"));
    assert_eq!(m.n, n_expect, "unexpected dimension for {file}");

    let sym =
        symbolic_factorize(&m, &SupernodeParams::default()).expect("symbolic factorize fixture");
    let nnz_l: usize = sym.col_counts.iter().sum();

    assert_eq!(
        sym.resolved_method,
        OrderingMethod::Amf,
        "issue #67: thin-large band ({file}) must route to AMF, not {:?}",
        sym.resolved_method
    );
    assert!(
        nnz_l < nnz_l_ceiling,
        "issue #67: {file} nnz_L = {nnz_l} should be < {nnz_l_ceiling} \
         (AMF wins; the MetisND fill is ≈ {metis_nnz_l})",
    );
}

#[test]
fn bratu3d_thin_band_routes_to_amf() {
    // n=27792, avg_deg 6.25, max_deg 7 — uniformly thin 3-D PDE, no arrow.
    check("tests/data/large/bratu3d.mtx", 27792, 8_000_000, 9_860_296);
}

#[test]
fn cont_201_thin_band_routes_to_amf() {
    // n=80595, avg_deg 5.44, max_deg 6 — uniformly thin, no arrow.
    check("tests/data/large/cont-201.mtx", 80595, 5_200_000, 5_948_611);
}
