//! Issue #227 Step 0: the predicted / actual dense-front work counters.
//!
//! `F(c, r) = Σ_{k<c} (r − k)²` per front. Predicted sums it over the
//! analysis's supernodes `(ncol, nrow)`; actual over the factor's fronts
//! `(nelim, nrow)` (nrow including delayed-in columns). The oracles below
//! are hand sums.

use feral::numeric::factorize::{factorize_multifrontal_supernodal, NumericParams};
use feral::scaling::ScalingStrategy;
use feral::symbolic::{
    predicted_front_flops, symbolic_factorize, symbolic_factorize_with_method, OrderingMethod,
    SupernodeParams,
};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, Solver, ZeroPivotAction};
use std::path::{Path, PathBuf};

fn tridiag(n: usize) -> CscMatrix {
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        r.push(j);
        c.push(j);
        v.push(4.0);
        if j + 1 < n {
            r.push(j + 1);
            c.push(j);
            v.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &r, &c, &v).expect("tridiag")
}

/// `tests/delayed_pivoting.rs`'s arrow-apex fixture: column 0's pivot
/// 1e-3 fails the 0.01 threshold against its off-diagonal 1 and is
/// delayed from its leaf into the root.
fn arrow_apex() -> CscMatrix {
    CscMatrix::from_triplets(
        4,
        &[0, 3, 1, 2, 3, 2, 3, 3],
        &[0, 0, 1, 1, 1, 2, 2, 3],
        &[1e-3, 1.0, 10.0, 0.5, 1.0, 10.0, 1.0, 2.0],
    )
    .expect("arrow")
}

fn nemin1() -> SupernodeParams {
    SupernodeParams {
        nemin: 1,
        ..Default::default()
    }
}

#[test]
fn predicted_on_unamalgamated_tridiagonal_is_the_hand_sum() {
    // nemin = 1 leaves only the fundamental supernodes of a path: columns
    // 0..3 each eliminate against one row below (F(1, 2) = 2² = 4); the
    // last two columns form one fundamental supernode, a dense 2×2 front
    // (F(2, 2) = 2² + 1² = 5). Total 3·4 + 5 = 17 — the same as one
    // supernode per column (4·4 + 1), as it must be: no fill either way.
    let a = tridiag(5);
    let sym = symbolic_factorize_with_method(&a, &nemin1(), OrderingMethod::Amd).expect("sym");
    let shapes: Vec<(usize, usize)> = sym.supernodes.iter().map(|s| (s.ncol, s.nrow)).collect();
    assert_eq!(shapes, vec![(1, 2), (1, 2), (1, 2), (2, 2)]);
    assert_eq!(predicted_front_flops(&sym), 17);
}

#[test]
fn predicted_and_actual_on_a_delayed_pivot_are_the_hand_sums() {
    let a = arrow_apex();
    let sym = symbolic_factorize(&a, &nemin1()).expect("sym");
    // Two supernodes: the leaf {0} with rows {0, 3}, and the root
    // {1, 2, 3}: F(1, 2) + F(3, 3) = 4 + (9 + 4 + 1) = 18.
    let shapes: Vec<(usize, usize)> = sym.supernodes.iter().map(|s| (s.ncol, s.nrow)).collect();
    assert_eq!(shapes, vec![(1, 2), (3, 3)]);
    assert_eq!(predicted_front_flops(&sym), 18);

    let params = NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        },
        scaling: ScalingStrategy::Identity,
        ..NumericParams::default()
    };
    let (f, _) = factorize_multifrontal_supernodal(&a, &sym, &params).expect("factor");
    // The leaf eliminates nothing (nelim 0, F(0, 2) = 0); the root
    // receives the delayed column (nrow 4) and eliminates all four:
    // F(4, 4) = 16 + 9 + 4 + 1 = 30.
    let fronts: Vec<(usize, usize)> = f.node_factors.iter().map(|n| (n.nelim, n.nrow)).collect();
    assert_eq!(fronts, vec![(0, 2), (4, 4)]);
    assert_eq!(f.actual_front_flops(), 30);
}

#[test]
fn spd_tridiagonal_has_actual_equal_to_predicted() {
    for n in [5usize, 64, 500] {
        let a = tridiag(n);
        let mut s = Solver::new();
        let _ = s.factor(&a, None);
        let st = s.last_factor_stats().expect("stats");
        assert!(st.predicted_flops > 0);
        assert_eq!(st.actual_flops, st.predicted_flops, "n = {n}");
    }
}

fn parity_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir("tests/data/parity")
        .expect("parity dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    dirs.sort();
    for d in dirs {
        if let Ok(rd) = std::fs::read_dir(&d) {
            let mut fs: Vec<PathBuf> = rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
                .collect();
            fs.sort();
            out.extend(fs);
        }
    }
    out
}

fn flops(path: &Path, parallel: bool) -> (u128, u128) {
    let a = read_mtx(path).and_then(|m| m.to_csc()).expect("mtx");
    let mut s = Solver::new().with_parallel(parallel);
    let _ = s.factor(&a, None);
    match s.last_factor_stats() {
        Some(st) => (st.predicted_flops, st.actual_flops),
        None => (0, 0),
    }
}

#[test]
fn counters_are_deterministic_across_runs_and_thread_counts() {
    let files = parity_files();
    assert!(!files.is_empty());
    let mut delayed = 0;
    for p in &files {
        let par = flops(p, true);
        let seq = flops(p, false);
        let again = flops(p, true);
        assert_eq!(par, seq, "{p:?}: parallel vs sequential");
        assert_eq!(par, again, "{p:?}: run to run");
        if par.1 != par.0 {
            delayed += 1;
        }
    }
    // The corpus exercises delays, so the two counters are not trivially
    // the same number.
    assert!(delayed > 0, "no parity matrix moved work by delaying");
}
