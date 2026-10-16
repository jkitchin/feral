//! Issue #153: per-front factor outputs (`l`, `d_diag`, `d_subdiag`,
//! `perm`, `perm_inv`, `row_indices`) are recycled from the previous
//! factor on a warm, pattern-reused `Solver::factor`.
//!
//! Contract: recycling pools memory only. A warm `Solver` that factors
//! iterate k+1 after iterate k must store a factor that is bit-identical
//! to the pre-recycling behaviour, and must solve to bit-identical
//! results.
//!
//! Oracle: a twin warm `Solver` fed the same iterate sequence but with
//! `invalidate_factors()` called before every `factor()`. That drops only
//! `last_factors`/`last_inertia`, which `factor_impl` never reads, so the
//! twin keeps every other piece of warm state (cached symbolic, scaling
//! caches, sticky routing) but has nothing to recycle: it is the
//! pre-change allocation path. A *fresh* `Solver` is not a valid oracle,
//! because warm state such as the cached value-dependent analysis
//! legitimately changes the factor.
//!
//! Plan: `dev/plans/factor-output-recycling.md`.

use feral::numeric::factorize::{NodeFactors, SparseFactors};
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::Path;

fn load(path: &str) -> CscMatrix {
    let m = match read_mtx(Path::new(path)) {
        Ok(m) => m,
        Err(e) => panic!("read_mtx({path}): {e}"),
    };
    match m.to_csc() {
        Ok(c) => c,
        Err(e) => panic!("to_csc({path}): {e}"),
    }
}

/// Same pattern, multiplicatively perturbed values (a stand-in for one
/// IPM step). Deterministic xorshift.
fn perturb(m: &CscMatrix, seed: u64, drift: f64) -> CscMatrix {
    let mut out = m.clone();
    let mut s = seed | 1;
    for v in out.values.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s >> 11) as f64 / (1u64 << 53) as f64;
        *v *= 1.0 + drift * (2.0 * u - 1.0);
    }
    out
}

fn bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn assert_node_eq(a: &NodeFactors, b: &NodeFactors, ctx: &str, i: usize) {
    assert_eq!(a.first_col, b.first_col, "{ctx}: node {i} first_col");
    assert_eq!(a.ncol, b.ncol, "{ctx}: node {i} ncol");
    assert_eq!(a.nelim, b.nelim, "{ctx}: node {i} nelim");
    assert_eq!(
        a.n_delayed_in, b.n_delayed_in,
        "{ctx}: node {i} n_delayed_in"
    );
    assert_eq!(a.nrow, b.nrow, "{ctx}: node {i} nrow");
    assert_eq!(a.row_indices, b.row_indices, "{ctx}: node {i} row_indices");
    assert_eq!(a.inertia, b.inertia, "{ctx}: node {i} inertia");
    let (fa, fb) = (&a.frontal_factors, &b.frontal_factors);
    assert_eq!(fa.nrow, fb.nrow, "{ctx}: node {i} ff.nrow");
    assert_eq!(fa.nelim, fb.nelim, "{ctx}: node {i} ff.nelim");
    assert!(bits_eq(&fa.l, &fb.l), "{ctx}: node {i} l bits differ");
    assert!(
        bits_eq(&fa.d_diag, &fb.d_diag),
        "{ctx}: node {i} d_diag bits differ"
    );
    assert!(
        bits_eq(&fa.d_subdiag, &fb.d_subdiag),
        "{ctx}: node {i} d_subdiag bits differ"
    );
    assert_eq!(fa.perm, fb.perm, "{ctx}: node {i} perm");
    assert_eq!(fa.perm_inv, fb.perm_inv, "{ctx}: node {i} perm_inv");
}

fn assert_factors_eq(a: &SparseFactors, b: &SparseFactors, ctx: &str) {
    assert_eq!(a.n, b.n, "{ctx}: n");
    assert_eq!(a.perm, b.perm, "{ctx}: perm");
    assert!(bits_eq(&a.scaling, &b.scaling), "{ctx}: scaling bits");
    assert_eq!(
        a.node_factors.len(),
        b.node_factors.len(),
        "{ctx}: node count"
    );
    for (i, (x, y)) in a.node_factors.iter().zip(&b.node_factors).enumerate() {
        assert_node_eq(x, y, ctx, i);
    }
}

fn factor_ok(s: &mut Solver, m: &CscMatrix, ctx: &str) {
    match s.factor(m, None) {
        FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
        other => panic!("{ctx}: factor failed: {other:?}"),
    }
}

fn total_delays(f: &SparseFactors) -> usize {
    f.node_factors.iter().map(|n| n.n_delayed_in).sum()
}

/// Factor `iterates` in sequence on a recycling warm `Solver` and on its
/// non-recycling twin; after each, compare the two. Returns the per-iterate
/// total delayed-pivot counts (for coverage assertions).
fn check_sequence(iterates: &[CscMatrix], parallel: bool, ctx: &str) -> Vec<usize> {
    let mut warm = Solver::new().with_parallel(parallel);
    let mut twin = Solver::new().with_parallel(parallel);
    let mut delays = Vec::new();
    for (k, m) in iterates.iter().enumerate() {
        let ctx_k = format!("{ctx} iterate {k} parallel={parallel}");
        factor_ok(&mut warm, m, &ctx_k);
        twin.invalidate_factors();
        factor_ok(&mut twin, m, &ctx_k);
        let (wf, ff) = match (warm.factors(), twin.factors()) {
            (Some(a), Some(b)) => (a, b),
            _ => panic!("{ctx_k}: missing stored factor"),
        };
        assert_factors_eq(wf, ff, &ctx_k);
        assert_eq!(warm.inertia(), twin.inertia(), "{ctx_k}: inertia");
        delays.push(total_delays(wf));

        let rhs: Vec<f64> = (0..m.n).map(|i| 1.0 + (i % 7) as f64).collect();
        let (xw, xf) = match (warm.solve(&rhs), twin.solve(&rhs)) {
            (Ok(a), Ok(b)) => (a, b),
            (a, b) => panic!("{ctx_k}: solve failed: {:?} / {:?}", a.err(), b.err()),
        };
        assert!(bits_eq(&xw, &xf), "{ctx_k}: solve bits differ");
    }
    delays
}

const FIXTURES: &[&str] = &[
    "tests/data/parity/vesuviou/VESUVIOU_0030.mtx",
    "tests/data/parity/cresc132/CRESC132_0000.mtx",
    "tests/data/parity/hahn1/HAHN1_0004.mtx",
    "tests/data/parity/muonsine/MUONSINE_0027.mtx",
    "tests/data/parity/hydcar20/HYDCAR20_0000.mtx",
    "tests/data/large/sawpath_kkt.mtx",
    "tests/data/large/twirism1_kkt.mtx",
];

#[test]
fn warm_refactor_bit_identical_to_no_recycle_twin_sequential() {
    for path in FIXTURES {
        let a = load(path);
        let seq = [
            a.clone(),
            perturb(&a, 0x9E37, 0.05),
            perturb(&a, 0x7F4A, 0.20),
        ];
        check_sequence(&seq, false, path);
    }
}

#[test]
fn warm_refactor_bit_identical_to_no_recycle_twin_parallel() {
    for path in FIXTURES {
        let a = load(path);
        let seq = [
            a.clone(),
            perturb(&a, 0x51ED, 0.05),
            perturb(&a, 0x2545, 0.20),
        ];
        check_sequence(&seq, true, path);
    }
}

/// Delay-count change between iterates: the recycled buffers for a node
/// were sized by the previous iterate's `nelim`/`nrow`, so the reuse path
/// must grow or shrink them correctly. Iterates alternate between a
/// strongly diagonal matrix (no delays) and the same pattern with the
/// diagonal of the leading columns driven to ~0 (BK must delay).
#[test]
fn warm_refactor_bit_identical_when_delay_count_changes() {
    let a = load("tests/data/parity/hahn1/HAHN1_0004.mtx");
    let mut weak = a.clone();
    for j in 0..weak.n {
        for k in weak.col_ptr[j]..weak.col_ptr[j + 1] {
            if weak.row_idx[k] == j && j % 3 == 0 {
                weak.values[k] *= 1e-14;
            }
        }
    }
    let seq = [a.clone(), weak.clone(), a.clone(), weak];
    for parallel in [false, true] {
        let delays = check_sequence(&seq, parallel, "hahn1 delay toggle");
        assert!(
            delays.iter().any(|&d| d != delays[0]),
            "fixture does not change the delay count across iterates ({delays:?}); \
             the test no longer covers buffer resizing"
        );
    }
}
