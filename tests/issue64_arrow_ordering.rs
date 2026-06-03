//! Issue #64 regression: the default ordering must not pick MetisND on
//! r05's arrow/bordered IPM KKT, where nested dissection blows the LDLᵀ
//! factor up ~7–9× vs AMF/AMD.
//!
//! The fixture `tests/data/large/r05_kkt.mtx` is a generated IPM KKT
//! (n=14842, nnz=118968), not a SuiteSparse download, so it is
//! gitignored and regenerated on demand via
//! `dev/scripts/regen_r05_kkt.sh`. When the fixture is absent (e.g. CI),
//! this test prints a SKIP line and passes — it is a local/opt-in
//! regression guard, not a CI gate.
//!
//! Oracle (external — the issue's measured fill): AMF/AMD ≈ 0.53–0.61M,
//! MetisND ≈ 3.6–4.4M. The `< 1.0e6` threshold separates them with wide
//! margin and is robust to ordering-impl / METIS-seed drift.

use feral::read_mtx;
use feral::symbolic::{symbolic_factorize, OrderingMethod, SupernodeParams};
use std::path::Path;

#[test]
fn r05_kkt_default_ordering_avoids_metis_fill_blowup() {
    let path = Path::new("tests/data/large/r05_kkt.mtx");
    if !path.is_file() {
        eprintln!(
            "SKIP: {} not present. Regenerate with dev/scripts/regen_r05_kkt.sh.",
            path.display()
        );
        return;
    }

    let m = read_mtx(path)
        .and_then(|mtx| mtx.to_csc())
        .expect("read r05_kkt.mtx");
    assert_eq!(m.n, 14842, "unexpected r05 KKT dimension");

    let sym = symbolic_factorize(&m, &SupernodeParams::default()).expect("symbolic factorize r05");
    let nnz_l: usize = sym.col_counts.iter().sum();

    assert_ne!(
        sym.resolved_method,
        OrderingMethod::MetisND,
        "issue #64: default ordering must not route r05's arrow KKT to MetisND"
    );
    assert!(
        nnz_l < 1_000_000,
        "issue #64: nnz_L = {} should be < 1.0e6 (AMF/AMD ≈ 0.5-0.6M; the \
         MetisND regression is ≈ 3.6-4.4M)",
        nnz_l
    );
}
