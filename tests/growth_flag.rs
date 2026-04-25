//! End-to-end regression for the pivot-growth refinement flag.
//!
//! The helper that scans `L` for `|L_ij| > 1e6` and sets
//! `needs_refinement` is unit-tested directly inside
//! `src/dense/factor.rs::growth_flag_tests`. This integration test
//! suite covers the factor-level wiring:
//!
//!   1. A well-conditioned SPD matrix factors cleanly and must NOT
//!      flag refinement (negative-case sanity check).
//!   2. Bratu3d under default `Solver` parameters reaches max|L| ≈ 8e16
//!      and must flag `needs_refinement = true` so callers using plain
//!      `Solver::solve` get a programmatic signal that the factor is
//!      too unstable for plain forward/back substitution. Gated
//!      `#[ignore]` because the matrix lives outside the repo and the
//!      factor takes ~120 s on M-series. See
//!      dev/journal/2026-04-25-02.org for the motivating triage.
//!
//! Constructing a small synthetic matrix that triggers max|L| > 1e6
//! through the production paths (dense `factor()`, `factor_frontal`,
//! and the multifrontal sparse path simultaneously) is finicky:
//! `factor()` equilibrates by default, and the BK alpha test interacts
//! with `pivot_threshold = 0.0` and the 2×2 pivot rule in ways that
//! make a "minimal" pathological example brittle. The unit tests cover
//! the helper; the bratu3d gated test covers the wiring.

use feral::{factor as dense_factor, BunchKaufmanParams, SymmetricMatrix};

#[test]
fn well_conditioned_spd_does_not_flag_growth() {
    // 4×4 SPD diagonally-dominant: a clean factorization should leave
    // max|L| well below 1e6 and not flag refinement.
    let mut m = SymmetricMatrix::zeros(4);
    for i in 0..4 {
        m.set(i, i, 4.0);
    }
    for i in 0..3 {
        m.set(i + 1, i, 1.0);
    }

    let (f, _inertia) = dense_factor(&m, &BunchKaufmanParams::default()).expect("factor SPD");
    let max_l = f.l.iter().map(|x| x.abs()).fold(0f64, f64::max);
    assert!(max_l < 10.0, "well-conditioned SPD: max|L| = {max_l:.3e}");
    assert!(!f.needs_refinement);
}

/// End-to-end regression for the bratu3d motivating case. Gated
/// `#[ignore]` because the matrix lives outside the repo and the
/// factor is slow (~120 s on M-series). Run alongside the smoke test:
///
///     cargo test --release --test growth_flag -- --ignored --nocapture
#[test]
#[ignore]
fn bratu3d_default_factor_flags_refinement() {
    use feral::numeric::solver::{FactorStatus, Solver};
    use feral::read_mtx;
    use std::path::Path;

    let path = Path::new("tests/data/large/bratu3d.mtx");
    if !path.exists() {
        eprintln!(
            "SKIP: {} not found; run dev/scripts/fetch_large_matrices.sh.",
            path.display()
        );
        return;
    }
    let mtx = read_mtx(path).expect("mtx");
    let csc = mtx.to_csc().expect("csc");
    let mut solver = Solver::new();
    match solver.factor(&csc, None) {
        FactorStatus::Success => {}
        other => panic!("expected Success, got {other:?}"),
    }
    let factors = solver.factors().expect("factors");
    let max_l = factors
        .node_factors
        .iter()
        .flat_map(|nf| nf.frontal_factors.l.iter())
        .map(|x| x.abs())
        .fold(0f64, f64::max);
    eprintln!("bratu3d: max|L| = {max_l:.3e}");
    assert!(
        max_l > 1e6,
        "bratu3d under default params should still produce growth (max|L|={max_l:.3e}); \
         if this fires, the underlying defaults changed and the test premise is stale"
    );
    assert!(
        factors.needs_refinement,
        "needs_refinement must flag the catastrophic factor"
    );
}
