//! Issue #91 regression: `OrderingPreprocess::Auto` must not inflate fill
//! on a quasi-definite conic KKT.
//!
//! The qap15 conic KKT (n=50880, nnz=168105) is saturated with degree-≤2
//! columns (the diagonal regularization rows), so the structural predicate
//! `pick_ordering_preprocess` recommends `LdltCompress` — exactly where
//! MC64 compression *inflates* fill: simplicial nnz_L jumps from ≈7.16M
//! (preprocess=None) to ≈45.4M, and the numeric factor from ≈0.8s to ≈15s.
//! The fix makes `Auto` verify fill (race None vs LdltCompress) instead of
//! trusting the predicate, so `Auto` fill ≤ `None` fill always.
//!
//! The fixture `tests/data/large/qap15_kkt.mtx` is a generated conic-IPM
//! KKT (pounce-convex on qap15.mps), not a SuiteSparse download, so it is
//! gitignored and regenerated on demand via
//! `dev/scripts/regen_qap15_kkt.sh`. When absent (e.g. CI), this test
//! prints a SKIP line and passes — a local/opt-in regression guard, not a
//! CI gate.
//!
//! Oracle (external — measured on the real matrix, cf.
//! `dev/research/issue-91-preprocess-misfire.md`): preprocess=None ≈7.16M,
//! the buggy Auto-via-predicate ≈45.4M. The `< 20M` threshold separates
//! them with wide margin and is robust to ordering-impl drift.

use feral::read_mtx;
use feral::symbolic::{
    pick_ordering_preprocess, symbolic_factorize_with_method, total_factor_nnz, OrderingMethod,
    OrderingPreprocess, SupernodeParams,
};
use std::path::Path;

#[test]
fn qap15_auto_preprocess_does_not_inflate_fill() {
    let path = Path::new("tests/data/large/qap15_kkt.mtx");
    if !path.is_file() {
        eprintln!(
            "SKIP: {} not present. Regenerate with dev/scripts/regen_qap15_kkt.sh.",
            path.display()
        );
        return;
    }

    let m = read_mtx(path)
        .and_then(|mtx| mtx.to_csc())
        .expect("read qap15_kkt.mtx");
    assert_eq!(m.n, 50880, "unexpected qap15 KKT dimension");

    // The predicate *does* recommend compression on this pattern — that is
    // the trap the race must defuse. (Documents the regression trigger; if
    // the predicate ever stops firing here, this matrix no longer guards
    // the bug and the fixture should be revisited.)
    assert_eq!(
        pick_ordering_preprocess(&m),
        OrderingPreprocess::LdltCompress,
        "issue #91: predicate is expected to (mistakenly) recommend LdltCompress here"
    );

    // Compare Auto vs None under the *same* ordering method to isolate the
    // preprocessing effect.
    let auto_params = SupernodeParams {
        preprocess: OrderingPreprocess::Auto,
        ..SupernodeParams::default()
    };
    let none_params = SupernodeParams {
        preprocess: OrderingPreprocess::None,
        ..SupernodeParams::default()
    };
    let auto = symbolic_factorize_with_method(&m, &auto_params, OrderingMethod::Amd)
        .expect("symbolic factorize qap15 (Auto)");
    let none = symbolic_factorize_with_method(&m, &none_params, OrderingMethod::Amd)
        .expect("symbolic factorize qap15 (None)");

    let auto_nnz = total_factor_nnz(&auto.col_counts);
    let none_nnz = total_factor_nnz(&none.col_counts);

    // Core invariant: the verified race can never do worse than None.
    assert!(
        auto_nnz <= none_nnz,
        "issue #91: Auto fill {auto_nnz} must not exceed None fill {none_nnz}"
    );

    // Regression guard: the buggy predicate path produced ≈45.4M; the fixed
    // race lands at ≈7.16M. 20M separates them with wide margin.
    assert!(
        auto_nnz < 20_000_000,
        "issue #91: Auto nnz_L = {auto_nnz} should be < 20M \
         (None ≈ 7.16M; the predicate-trusting regression was ≈ 45.4M)",
        auto_nnz = auto_nnz
    );

    // And the verified Auto must have rejected LdltCompress here.
    assert_eq!(
        auto.resolved_preprocess,
        OrderingPreprocess::None,
        "issue #91: verified Auto must resolve to None on qap15 (LdltCompress inflates fill)"
    );
}
