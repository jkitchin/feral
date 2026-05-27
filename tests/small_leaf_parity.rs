//! Parity tests for Phase 2.9 SmallLeafSubtree batching
//! (`dev/plans/phase-2.9-small-leaf-subtree.md`).
//!
//! Contract: for every supported input the batched-leaf path
//! (`NumericParams::small_leaf == On`) must produce bit-equal
//! `SparseFactors` to the scalar path (`small_leaf == Off`). The
//! batched path is numeric-only — scaling and the symbolic factorization
//! are shared — so the only observable effect on factors should be None.
//!
//! Corpus rationale:
//! * A tiny hand-built block-diagonal matrix — exercises the
//!   multi-group/multi-member code path with a known-answer case.
//! * ACOPR30_0067 — the archetype long-tail IPM matrix that motivated
//!   the phase.
//! * CRESC100_0000 — long-tail bulk.
//! * HAIFAM_0082 — different tree shape.
//! * VESUVIO_0000 — drawn from the `factor_workspace_parity.rs` corpus
//!   as a cross-phase regression canary.

use feral::numeric::factorize::{
    factorize_multifrontal, NodeFactors, NumericParams, SmallLeafBatch, SparseFactors,
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

fn params_off() -> NumericParams {
    NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        },
        scaling: Default::default(),
        small_leaf: SmallLeafBatch::Off,
        profiler: None,
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
    }
}

fn params_on() -> NumericParams {
    NumericParams {
        small_leaf: SmallLeafBatch::On,
        ..params_off()
    }
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
    assert_inertia_eq(&fa.inertia, &fb.inertia, &format!("{}/ff.inertia", ctx));
}

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

fn assert_parity(csc: &CscMatrix, ctx: &str) {
    let snode_params = SupernodeParams::default();
    let sym = match symbolic_factorize(csc, &snode_params) {
        Ok(s) => s,
        Err(e) => panic!("{}: symbolic_factorize failed: {}", ctx, e),
    };

    let p_off = params_off();
    let p_on = params_on();

    let (f_off, i_off) = match factorize_multifrontal(csc, &sym, &p_off) {
        Ok(r) => r,
        Err(e) => panic!("{}: Off path failed: {}", ctx, e),
    };
    let (f_on, i_on) = match factorize_multifrontal(csc, &sym, &p_on) {
        Ok(r) => r,
        Err(e) => panic!("{}: On path failed: {}", ctx, e),
    };

    assert_inertia_eq(&i_off, &i_on, &format!("{}/total_inertia", ctx));
    assert_factors_equal(&f_off, &f_on, ctx);
}

fn assert_parity_path(path: &str) {
    if !Path::new(path).exists() {
        eprintln!("SKIP: {} not present (corpus is gitignored)", path);
        return;
    }
    let csc = load_csc(path);
    assert_parity(&csc, path);
}

/// Tiny hand-built matrix with several true-leaf supernodes. A
/// block-diagonal SPD matrix: each 2×2 block is a standalone supernode
/// with no children and full native pattern — exactly the shape that
/// the batched-leaf path is designed to accelerate.
fn block_diag_spd(k: usize) -> CscMatrix {
    let n = 2 * k;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for b in 0..k {
        let base = 2 * b;
        // Column base (lower triangle including diagonal)
        rows.push(base);
        cols.push(base);
        vals.push(4.0);
        rows.push(base + 1);
        cols.push(base);
        vals.push(1.0);
        // Column base+1
        rows.push(base + 1);
        cols.push(base + 1);
        vals.push(3.0);
    }
    match CscMatrix::from_triplets(n, &rows, &cols, &vals) {
        Ok(c) => c,
        Err(e) => panic!("block_diag_spd: {}", e),
    }
}

#[test]
fn small_leaf_parity_block_diag_spd() {
    let csc = block_diag_spd(20);
    assert_parity(&csc, "block_diag_spd(20)");
}

#[test]
fn small_leaf_parity_block_diag_spd_small() {
    let csc = block_diag_spd(3);
    assert_parity(&csc, "block_diag_spd(3)");
}

#[test]
fn small_leaf_parity_vesuvio_0000() {
    assert_parity_path("data/matrices/kkt/VESUVIO/VESUVIO_0000.mtx");
}

#[test]
fn small_leaf_parity_acopr30_0067() {
    assert_parity_path("data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx");
}

#[test]
fn small_leaf_parity_cresc100_0000() {
    assert_parity_path("data/matrices/kkt/CRESC100/CRESC100_0000.mtx");
}

#[test]
fn small_leaf_parity_haifam_0082() {
    assert_parity_path("data/matrices/kkt/HAIFAM/HAIFAM_0082.mtx");
}

/// Sanity check: the long-tail IPM archetypes actually produce
/// non-empty small-leaf groups (otherwise the parity tests above
/// would silently degenerate to the Off path being compared with
/// itself). If this test fails the parity tests become meaningless —
/// investigate the `SmallLeafParams` defaults or the grouping logic
/// in `src/symbolic/small_leaf.rs` before trusting parity.
#[test]
fn small_leaf_groups_nonempty_on_archetypes() {
    for path in &[
        "data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx",
        "data/matrices/kkt/CRESC100/CRESC100_0000.mtx",
        "data/matrices/kkt/HAIFAM/HAIFAM_0082.mtx",
    ] {
        if !Path::new(path).exists() {
            eprintln!("SKIP: {} not present (corpus is gitignored)", path);
            continue;
        }
        let csc = load_csc(path);
        let snode_params = SupernodeParams::default();
        let sym = match symbolic_factorize(&csc, &snode_params) {
            Ok(s) => s,
            Err(e) => panic!("{}: symbolic_factorize failed: {}", path, e),
        };
        let n_grouped: usize = sym.snode_group.iter().filter(|g| g.is_some()).count();
        assert!(
            !sym.small_leaf_groups.is_empty(),
            "{}: expected non-empty small_leaf_groups (got {} snodes, {} grouped)",
            path,
            sym.supernodes.len(),
            n_grouped
        );
        assert!(
            n_grouped > 0,
            "{}: expected at least one grouped snode",
            path
        );
    }
}
