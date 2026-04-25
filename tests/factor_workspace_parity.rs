//! Parity tests for `factorize_multifrontal_with_workspace` vs
//! `factorize_multifrontal`.
//!
//! Contract: for any input `(matrix, symbolic, params)` the workspace-
//! reusing path must produce a `SparseFactors` whose retained data is
//! bit-equal to the no-workspace path. The workspace pools scratch
//! memory ONLY; it must not influence the factors, inertia, or any
//! solver-visible state.
//!
//! These tests are the guardrail for the rollout plan in
//! `dev/plans/factor-workspace.md`. They are green on the trivial-
//! delegator implementation that ships in the skeleton commit and
//! must remain green as each scratch site is pooled in subsequent
//! commits.
//!
//! Coverage chosen from the 10-matrix panel in
//! `src/bin/alloc_probe.rs` plus cross-matrix reuse (AVION2 ->
//! BATCH -> VESUVIO -> AVION2) to detect un-cleared scratch state
//! between calls. A rank-deficient matrix with delayed pivots is
//! also included.

use feral::numeric::factorize::{
    factorize_multifrontal, factorize_multifrontal_with_workspace, FactorWorkspace, NodeFactors,
    NumericParams, SparseFactors,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};
use std::path::Path;

fn load_csc(path: &str) -> CscMatrix {
    let mtx = match read_mtx(Path::new(path)) {
        Ok(m) => m,
        Err(e) => panic!("read_mtx({}) failed: {}", path, e),
    };
    match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => panic!("to_csc({}) failed: {}", path, e),
    }
}

/// `data/matrices/` is gitignored; corpus is regenerated from ripopt
/// CUTEst runs and not present on CI. Returns `true` when missing so
/// the caller can early-return with a SKIP message instead of
/// panicking through `read_mtx`.
fn skip_if_missing(path: &str) -> bool {
    if !Path::new(path).exists() {
        eprintln!("SKIP: {} not present (corpus is gitignored)", path);
        true
    } else {
        false
    }
}

fn default_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    })
}

/// Assert that two `SparseFactors` are bit-equal across every field
/// consumed by the solve path. Panics with a specific message on the
/// first mismatch.
fn assert_factors_equal(a: &SparseFactors, b: &SparseFactors, ctx: &str) {
    assert_eq!(a.n, b.n, "{}: n differs", ctx);
    assert_eq!(a.perm, b.perm, "{}: perm differs", ctx);
    assert_eq!(a.perm_inv, b.perm_inv, "{}: perm_inv differs", ctx);
    assert_eq!(
        a.needs_refinement, b.needs_refinement,
        "{}: needs_refinement differs",
        ctx
    );
    assert_eq!(a.scaling.len(), b.scaling.len(), "{}: scaling len", ctx);
    for (i, (x, y)) in a.scaling.iter().zip(b.scaling.iter()).enumerate() {
        assert!(
            x.to_bits() == y.to_bits(),
            "{}: scaling[{}] bits differ ({} vs {})",
            ctx,
            i,
            x,
            y
        );
    }
    assert_eq!(
        a.node_factors.len(),
        b.node_factors.len(),
        "{}: node count",
        ctx
    );
    for (k, (na, nb)) in a.node_factors.iter().zip(b.node_factors.iter()).enumerate() {
        assert_node_eq(na, nb, &format!("{}/node[{}]", ctx, k));
    }
}

fn assert_node_eq(a: &NodeFactors, b: &NodeFactors, ctx: &str) {
    assert_eq!(a.first_col, b.first_col, "{}: first_col", ctx);
    assert_eq!(a.ncol, b.ncol, "{}: ncol", ctx);
    assert_eq!(a.nelim, b.nelim, "{}: nelim", ctx);
    assert_eq!(a.n_delayed_in, b.n_delayed_in, "{}: n_delayed_in", ctx);
    assert_eq!(a.nrow, b.nrow, "{}: nrow", ctx);
    assert_eq!(a.row_indices, b.row_indices, "{}: row_indices", ctx);
    assert_inertia_eq(&a.inertia, &b.inertia, &format!("{}/inertia", ctx));
    let fa = &a.frontal_factors;
    let fb = &b.frontal_factors;
    assert_eq!(fa.nrow, fb.nrow, "{}: ff.nrow", ctx);
    assert_eq!(fa.ncol, fb.ncol, "{}: ff.ncol", ctx);
    assert_eq!(fa.nelim, fb.nelim, "{}: ff.nelim", ctx);
    assert_eq!(fa.contrib_dim, fb.contrib_dim, "{}: ff.contrib_dim", ctx);
    assert_eq!(fa.n_delayed, fb.n_delayed, "{}: ff.n_delayed", ctx);
    assert_eq!(fa.perm, fb.perm, "{}: ff.perm", ctx);
    assert_eq!(fa.perm_inv, fb.perm_inv, "{}: ff.perm_inv", ctx);
    assert_bits_eq(&fa.l, &fb.l, &format!("{}/ff.l", ctx));
    assert_bits_eq(&fa.d_diag, &fb.d_diag, &format!("{}/ff.d_diag", ctx));
    assert_bits_eq(
        &fa.d_subdiag,
        &fb.d_subdiag,
        &format!("{}/ff.d_subdiag", ctx),
    );
    assert_bits_eq(&fa.contrib, &fb.contrib, &format!("{}/ff.contrib", ctx));
    assert_eq!(
        fa.needs_refinement, fb.needs_refinement,
        "{}: ff.needs_refinement",
        ctx
    );
    assert!(
        fa.zero_tol.to_bits() == fb.zero_tol.to_bits(),
        "{}: ff.zero_tol bits",
        ctx
    );
    assert!(
        fa.zero_tol_2x2.to_bits() == fb.zero_tol_2x2.to_bits(),
        "{}: ff.zero_tol_2x2 bits",
        ctx
    );
    assert_inertia_eq(&fa.inertia, &fb.inertia, &format!("{}/ff.inertia", ctx));
}

fn assert_inertia_eq(a: &Inertia, b: &Inertia, ctx: &str) {
    assert_eq!(a.positive, b.positive, "{}: positive", ctx);
    assert_eq!(a.negative, b.negative, "{}: negative", ctx);
    assert_eq!(a.zero, b.zero, "{}: zero", ctx);
}

fn assert_bits_eq(a: &[f64], b: &[f64], ctx: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{}: length {} vs {}",
        ctx,
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            x.to_bits() == y.to_bits(),
            "{}[{}]: bits differ ({} vs {})",
            ctx,
            i,
            x,
            y
        );
    }
}

/// Run both paths and return both results + inertia for a given matrix.
fn factor_both(
    csc: &CscMatrix,
    ws: &mut FactorWorkspace,
) -> ((SparseFactors, Inertia), (SparseFactors, Inertia)) {
    let snode_params = SupernodeParams::default();
    let sym = match symbolic_factorize(csc, &snode_params) {
        Ok(s) => s,
        Err(e) => panic!("symbolic_factorize failed: {}", e),
    };
    let params = default_params();
    let baseline = match factorize_multifrontal(csc, &sym, &params) {
        Ok(r) => r,
        Err(e) => panic!("factorize_multifrontal (baseline) failed: {}", e),
    };
    let with_ws = match factorize_multifrontal_with_workspace(csc, &sym, &params, ws) {
        Ok(r) => r,
        Err(e) => panic!("factorize_multifrontal_with_workspace failed: {}", e),
    };
    (baseline, with_ws)
}

fn assert_parity(path: &str, ws: &mut FactorWorkspace) {
    if skip_if_missing(path) {
        return;
    }
    let csc = load_csc(path);
    let ((f0, i0), (f1, i1)) = factor_both(&csc, ws);
    assert_inertia_eq(&i0, &i1, &format!("{}/total_inertia", path));
    assert_factors_equal(&f0, &f1, path);
}

#[test]
fn parity_avion2_0000() {
    let mut ws = FactorWorkspace::new();
    assert_parity("data/matrices/kkt/AVION2/AVION2_0000.mtx", &mut ws);
}

#[test]
fn parity_batch_0000() {
    let mut ws = FactorWorkspace::new();
    assert_parity("data/matrices/kkt/BATCH/BATCH_0000.mtx", &mut ws);
}

#[test]
fn parity_lakes_1199() {
    let mut ws = FactorWorkspace::new();
    assert_parity("data/matrices/kkt/LAKES/LAKES_1199.mtx", &mut ws);
}

#[test]
fn parity_vesuvio_0000() {
    let mut ws = FactorWorkspace::new();
    assert_parity("data/matrices/kkt/VESUVIO/VESUVIO_0000.mtx", &mut ws);
}

#[test]
fn parity_hahn1_0000() {
    let mut ws = FactorWorkspace::new();
    assert_parity("data/matrices/kkt/HAHN1/HAHN1_0000.mtx", &mut ws);
}

/// Delayed-pivot exercise: the AVION2 / BATCH samples above cover the
/// typical postorder traversal. This test adds a matrix with known
/// delayed pivots (MSS1_0009 has sign indefiniteness that forces BK
/// delays in the middle of the supernode tree) to ensure the
/// contribution-block pooling (when it lands) doesn't mishandle the
/// delayed-column layout.
#[test]
fn parity_mss1_0009_delayed_pivots() {
    let mut ws = FactorWorkspace::new();
    assert_parity("data/matrices/kkt/MSS1/MSS1_0009.mtx", &mut ws);
}

/// Cross-matrix reuse: feed the same workspace through a sequence of
/// different matrices, including a return to a previously-seen one.
/// This catches stale scratch (e.g., a `row_map` not cleared on exit)
/// that would produce wrong results only on the second call.
#[test]
fn parity_cross_matrix_reuse() {
    let mut ws = FactorWorkspace::new();
    let sequence = [
        "data/matrices/kkt/AVION2/AVION2_0000.mtx",
        "data/matrices/kkt/BATCH/BATCH_0000.mtx",
        "data/matrices/kkt/VESUVIO/VESUVIO_0000.mtx",
        "data/matrices/kkt/AVION2/AVION2_0000.mtx",
        "data/matrices/kkt/HAHN1/HAHN1_0000.mtx",
        "data/matrices/kkt/BATCH/BATCH_0000.mtx",
    ];
    for (step, path) in sequence.iter().enumerate() {
        if skip_if_missing(path) {
            return;
        }
        let ctx = format!("step {} ({})", step, path);
        let csc = load_csc(path);
        let ((f0, i0), (f1, i1)) = factor_both(&csc, &mut ws);
        assert_inertia_eq(&i0, &i1, &format!("{}/total_inertia", ctx));
        assert_factors_equal(&f0, &f1, &ctx);
    }
}

/// Repeat-same-matrix reuse: the lifetime of a typical IPM iteration
/// is "same sparsity, new values" but here we exercise the cheaper
/// invariant of identical inputs. Any drift between the first and
/// second call implicates un-cleared scratch.
#[test]
fn parity_repeat_same_matrix() {
    let mut ws = FactorWorkspace::new();
    let path = "data/matrices/kkt/BATCH/BATCH_0500.mtx";
    if skip_if_missing(path) {
        return;
    }
    let csc = load_csc(path);
    let ((f_baseline, _), _) = factor_both(&csc, &mut ws);
    for step in 0..5 {
        let ctx = format!("{} repeat iter {}", path, step);
        let snode_params = SupernodeParams::default();
        let sym = match symbolic_factorize(&csc, &snode_params) {
            Ok(s) => s,
            Err(e) => panic!("symbolic: {}", e),
        };
        let params = default_params();
        let (f_ws, _) = match factorize_multifrontal_with_workspace(&csc, &sym, &params, &mut ws) {
            Ok(r) => r,
            Err(e) => panic!("with_workspace: {}", e),
        };
        assert_factors_equal(&f_baseline, &f_ws, &ctx);
    }
}
