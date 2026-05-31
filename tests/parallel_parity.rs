//! Parity tests for the Phase 2.5.2 parallel multifrontal driver.
//!
//! Contract: `factorize_multifrontal_supernodal_parallel` must produce
//! a `SparseFactors` that is bit-equal to the sequential
//! `factorize_multifrontal` on the same input. The parallel driver
//! uses one task per supernode with mutex-protected contribution-block
//! exchange and per-thread workspaces; FP-order determinism rests on
//! each supernode's extend-add loop running atomically (in
//! `snode.children` order) just like in the sequential path.
//!
//! These tests are the guardrail for the Step C exit criterion in
//! `dev/plans/phase-2.5.2-rayon-assembly-tree.md`.

use feral::numeric::factorize::{
    factorize_multifrontal, factorize_multifrontal_supernodal_parallel, NodeFactors, NumericParams,
    SparseFactors,
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

fn default_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    })
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

fn assert_inertia_eq(a: &Inertia, b: &Inertia, ctx: &str) {
    assert_eq!(a.positive, b.positive, "{}: positive", ctx);
    assert_eq!(a.negative, b.negative, "{}: negative", ctx);
    assert_eq!(a.zero, b.zero, "{}: zero", ctx);
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
}

fn assert_factors_equal(a: &SparseFactors, b: &SparseFactors, ctx: &str) {
    assert_eq!(a.n, b.n, "{}: n", ctx);
    assert_eq!(a.perm, b.perm, "{}: perm", ctx);
    assert_eq!(a.perm_inv, b.perm_inv, "{}: perm_inv", ctx);
    assert_eq!(
        a.needs_refinement, b.needs_refinement,
        "{}: needs_refinement",
        ctx
    );
    assert_bits_eq(&a.scaling, &b.scaling, &format!("{}/scaling", ctx));
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

fn assert_parity(path: &str) {
    if !Path::new(path).exists() {
        eprintln!("SKIP: {} not present (corpus is gitignored)", path);
        return;
    }
    let csc = load_csc(path);
    let snode_params = SupernodeParams::default();
    let sym = match symbolic_factorize(&csc, &snode_params) {
        Ok(s) => s,
        Err(e) => panic!("symbolic_factorize({}) failed: {}", path, e),
    };
    let params = default_params();
    let (seq_factors, seq_inertia) = match factorize_multifrontal(&csc, &sym, &params) {
        Ok(r) => r,
        Err(e) => panic!("sequential factorize({}) failed: {}", path, e),
    };
    let (par_factors, par_inertia) =
        match factorize_multifrontal_supernodal_parallel(&csc, &sym, &params) {
            Ok(r) => r,
            Err(e) => panic!("parallel factorize({}) failed: {}", path, e),
        };
    assert_inertia_eq(&seq_inertia, &par_inertia, &format!("{}/total", path));
    assert_factors_equal(&seq_factors, &par_factors, path);
}

#[test]
fn parallel_parity_avion2_0000() {
    assert_parity("data/matrices/kkt/AVION2/AVION2_0000.mtx");
}

#[test]
fn parallel_parity_batch_0000() {
    assert_parity("data/matrices/kkt/BATCH/BATCH_0000.mtx");
}

#[test]
fn parallel_parity_vesuvio_0000() {
    assert_parity("data/matrices/kkt/VESUVIO/VESUVIO_0000.mtx");
}

#[test]
fn parallel_parity_hahn1_0000() {
    assert_parity("data/matrices/kkt/HAHN1/HAHN1_0000.mtx");
}

#[test]
fn parallel_parity_lakes_1199() {
    assert_parity("data/matrices/kkt/LAKES/LAKES_1199.mtx");
}

#[test]
fn parallel_parity_mss1_0009_delayed_pivots() {
    assert_parity("data/matrices/kkt/MSS1/MSS1_0009.mtx");
}

/// Regression guard (pounce#79): the parallel task-graph driver must
/// keep worker-stack usage O(1) in elimination-tree height.
///
/// A tridiagonal SPD system has a chain-structured elimination tree.
/// Under the default ordering it amalgamates into a deep supernode chain
/// (measured: n = 8000 ⇒ ~500 supernodes, supernode-tree height ~500),
/// which exercises `run_parallel_task`'s leaf→root climb over a long
/// path. Because that climb is trampolined through rayon's task queue
/// (each parent is *spawned*, not called on the child's stack frame),
/// this factorizes on default rayon worker stacks without overflow. If a
/// future refactor reintroduced native leaf→root recursion, a deep chain
/// would blow the worker stack and crash this test (SIGSEGV/SIGABRT)
/// rather than fail an assertion.
///
/// Measured during the pounce#79 investigation: the deepest corpus
/// matrix `c-big` (n = 345 241, supernode-tree height 1521) factors on a
/// 32 KiB worker stack — far below the rayon ~2 MiB default — confirming
/// stack need is independent of tree height. See the doc comment on
/// `run_parallel_task` and `dev/research/parallel-stack-depth-pounce79.md`.
#[test]
fn deep_chain_tree_no_stack_overflow() {
    let n = 8000usize;
    // Tridiagonal SPD, lower triangle only (CscMatrix stores row >=
    // col): diagonal 4, subdiagonal -1. Strictly diagonally dominant ⇒
    // SPD ⇒ LDLᵀ needs no pivoting, inertia (n, 0, 0).
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
    let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("tridiagonal triplets");
    let sym = symbolic_factorize(&a, &SupernodeParams::default()).expect("symbolic");

    // Call the task-graph driver directly so the parallel path runs
    // regardless of the size/flops gate in the public
    // `factorize_multifrontal_parallel` wrapper (a thin tridiagonal
    // would otherwise fall through to the sequential driver).
    let (par_factors, par_inertia) =
        factorize_multifrontal_supernodal_parallel(&a, &sym, &NumericParams::default())
            .expect("parallel factor on deep chain");

    assert_eq!(par_inertia.positive, n, "SPD: all positive");
    assert_eq!(par_inertia.negative, 0, "SPD: none negative");
    assert_eq!(par_inertia.zero, 0, "SPD: no zero pivots");

    // Bit-exact parity with the sequential driver on the same deep tree
    // — extends the parallel/sequential contract to this chain shape.
    let (seq_factors, seq_inertia) =
        factorize_multifrontal(&a, &sym, &NumericParams::default()).expect("sequential factor");
    assert_inertia_eq(&seq_inertia, &par_inertia, "deep-chain/total");
    assert_factors_equal(&seq_factors, &par_factors, "deep-chain");
}

/// Lever 1.1 gate: the intra-front parallel Schur update must be
/// bit-exact with the serial path, and the parallel path must actually
/// fire (a front wide enough to clear `INTRAFRONT_MIN_AREA = 256*256`).
///
/// A dense, diagonally dominant SPD matrix factorizes into a single wide
/// root supernode of all-1×1 pivots — exactly the
/// `apply_blocked_schur_panel` fast path that Lever 1.1 parallelizes.
/// With `n = 1200`, the first 64-wide panel has trailing area
/// `(1200 - 64) * 64 = 72_704 >= 65_536`, so the `par_chunks_mut` branch
/// runs. We force a 4-thread pool so the split executes even on a
/// single-core CI, then assert the parallel driver (intra-front ON, set
/// internally) is byte-identical to the sequential driver (intra-front
/// OFF) on L, D, and inertia. Bit-exactness is structural — each
/// trailing column is reduced on one thread — so any thread count or
/// chunk partition must match. See
/// `dev/research/lever-1.1-intrafront-parallel-schur.md`.
#[test]
fn intrafront_parallel_schur_matches_serial() {
    let n = 1200usize;
    // Dense lower triangle (row >= col). Diagonally dominant ⇒ SPD ⇒
    // all 1×1 pivots, no 2×2, no zero pivots — routes through the
    // rank-`n_elim` panel fast path that Lever 1.1 parallelizes.
    let mut rows = Vec::with_capacity(n * (n + 1) / 2);
    let mut cols = Vec::with_capacity(n * (n + 1) / 2);
    let mut vals = Vec::with_capacity(n * (n + 1) / 2);
    for c in 0..n {
        rows.push(c);
        cols.push(c);
        vals.push(n as f64 + 1.0); // diagonal dominates the n-1 unit off-diagonals
        for r in (c + 1)..n {
            rows.push(r);
            cols.push(c);
            vals.push(1.0);
        }
    }
    let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("dense SPD triplets");
    let sym = symbolic_factorize(&a, &SupernodeParams::default()).expect("symbolic");

    // Sequential driver: intra-front stays false (serial Schur).
    let (seq_factors, seq_inertia) =
        factorize_multifrontal(&a, &sym, &NumericParams::default()).expect("sequential factor");

    // Parallel driver (sets intra-front true internally), forced onto a
    // 4-thread pool so the `par_chunks_mut` split is actually exercised.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("4-thread pool");
    let (par_factors, par_inertia) = pool.install(|| {
        factorize_multifrontal_supernodal_parallel(&a, &sym, &NumericParams::default())
            .expect("parallel factor")
    });

    assert_eq!(par_inertia.positive, n, "SPD: all positive");
    assert_eq!(par_inertia.negative, 0, "SPD: none negative");
    assert_eq!(par_inertia.zero, 0, "SPD: no zero pivots");
    assert_inertia_eq(&seq_inertia, &par_inertia, "intrafront/total");
    // `assert_factors_equal` -> `assert_node_eq` already checks L, d_diag,
    // d_subdiag, and contrib bit-exact per supernode, so this single call
    // covers the full factor (not just L).
    assert_factors_equal(&seq_factors, &par_factors, "intrafront");
}
