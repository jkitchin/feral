//! Issue #163: `SparseLuSymbolic::analyze` must be whole-basis AMD, not the
//! Suhl–Suhl peel.
//!
//! # Why this is a contract test and not a numerical one
//!
//! PR #160 made `analyze` peel. The peel is opt-in again because it broke a
//! downstream regression — an ill-conditioned LP that had certified `Optimal`
//! started returning `Numerical`, losing its dual bound, with the dense-bump
//! route the peel exists to enable still switched off.
//!
//! It is tempting to pin that with a residual assertion here. **The evidence
//! says there is nothing to assert.** Every basis that LP's simplex handed feral
//! was dumped and re-factored under both orderings: backward error is ~1e-16
//! under both on all of them, and forward error against a known solution — which
//! reaches 2.6e-11, the basis genuinely being ill-conditioned — is *never worse*
//! under the peel, ratios 0.0x–1.0x across all 30 bases of the failing run. The
//! peel does not produce a worse factorization. It produces a different rounding
//! trajectory, which that LP was sensitive enough to diverge on.
//!
//! So the defect a feral-side test can actually catch is the ordering change
//! itself, and that is what these assert. The end-to-end behavior is pinned
//! downstream, by the test that reported the issue.
//!
//! See `dev/research/lu-ordering-and-kernel-2026-08-13.md`.

use feral::{LuOrderingParams, LuParams, SparseColMatrix, SparseLu, SparseLuSymbolic};

/// An LP-shaped basis: a triangular border the peel can strip, wrapped around a
/// bump it cannot. `nfront` column singletons, then a `bump`x`bump` dense block,
/// then `nback` row singletons.
fn lp_like_basis(nfront: usize, bump: usize, nback: usize) -> SparseColMatrix {
    let m = nfront + bump + nback;
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    // Front: strict column singletons — one entry each, on their own row.
    for (j, col) in cols.iter_mut().enumerate().take(nfront) {
        col.push((j, 2.0 + (j % 5) as f64));
    }
    // Bump: dense block on rows/cols [nfront, nfront+bump).
    for jj in 0..bump {
        for ii in 0..bump {
            let v = if ii == jj {
                4.0
            } else {
                0.25 + 0.03 * (ii + 2 * jj) as f64
            };
            cols[nfront + jj].push((nfront + ii, v));
        }
    }
    // Back: rows below the bump are touched by exactly one column each, so the
    // back peel strips them; each such column also reaches into the bump, which
    // is what makes the peel a *fixpoint* rather than a single pass.
    for k in 0..nback {
        let j = nfront + bump + k;
        cols[j].push((nfront + bump + k, 3.0 + (k % 3) as f64));
        if bump > 0 {
            cols[j].push((nfront + (k % bump), 0.5));
        }
    }
    SparseColMatrix::from_sparse_columns(m, &cols).expect("basis")
}

#[test]
fn analyze_is_whole_basis_amd_and_does_not_peel() {
    let (nfront, bump, nback) = (40usize, 12usize, 25usize);
    let a = lp_like_basis(nfront, bump, nback);
    let m = a.m;

    let peeled = SparseLuSymbolic::analyze_triangularized(&a).expect("triangularized");
    // Non-vacuity: the fixture must actually be peelable, or "analyze did not
    // peel" says nothing. `analyze_triangularized` is the witness.
    assert!(peeled.triangularized, "the peeling constructor must peel");
    assert!(
        peeled.bump_hi - peeled.bump_lo <= bump,
        "fixture is not peelable: peel left a bump of {} on a basis whose \
         irreducible core is {bump}",
        peeled.bump_hi - peeled.bump_lo
    );

    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    assert!(
        !sym.triangularized,
        "issue #163: `analyze` must not triangularize — the peel is opt-in via \
         `analyze_triangularized`"
    );
    assert_eq!(
        (sym.bump_lo, sym.bump_hi),
        (0, m),
        "a non-peeling ordering must claim the whole basis as bump"
    );

    // And it must be *the same* ordering as `analyze_amd_only`, not merely some
    // other non-peeling one: `analyze_amd_only` is the name the benchmark arms
    // and `examples/basis_refactor.rs` use for the pre-#160 behavior.
    let amd = SparseLuSymbolic::analyze_amd_only(&a).expect("amd_only");
    assert_eq!(
        sym.qcol, amd.qcol,
        "`analyze` must be exactly `analyze_amd_only`"
    );
}

#[test]
fn dense_bump_route_needs_the_peel_and_the_cap_together() {
    // The two are opted into as a pair (issue #163). Setting the cap without
    // the peeling constructor must be an inert no-op, not a partial opt-in.
    let a = lp_like_basis(40, 12, 25);
    let params = LuParams {
        dense_bump_max_dim: 4096,
        ..LuParams::default()
    };

    let with_peel = SparseLuSymbolic::analyze_triangularized(&a).expect("triangularized");
    let lu = SparseLu::factor(&a, &with_peel, params.clone()).expect("factor");
    assert!(
        lu.used_dense_bump(),
        "cap + peel must take the dense route, or the no-op check below is \
         vacuous"
    );

    let default_sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    let lu = SparseLu::factor(&a, &default_sym, params).expect("factor");
    assert!(
        !lu.used_dense_bump(),
        "`dense_bump_max_dim` must be inert under the default ordering"
    );
}

/// **Issue #165 — the ordering is reachable as a parameter.**
///
/// `analyze_with` is the form a caller can carry in its own config and A/B
/// without a code change, which is the whole point: the right ordering varies by
/// instance (1.306x geomean across 14 QPLIB relaxations, 0.389x on one of them),
/// and the maintainer had to patch a downstream solver to find that out.
///
/// So the contract is that it is not a *third* ordering. Both settings must
/// reproduce the corresponding constructor exactly — same permutation, same
/// bump, same `triangularized` provenance flag, which is what
/// `LuParams::dense_bump_max_dim` gates on — and the default must be the
/// non-peeling one, matching `analyze`.
#[test]
fn analyze_with_reproduces_both_constructors_exactly() {
    let a = lp_like_basis(40, 12, 25);

    let same = |x: &SparseLuSymbolic, y: &SparseLuSymbolic| {
        x.qcol == y.qcol
            && (x.bump_lo, x.bump_hi) == (y.bump_lo, y.bump_hi)
            && x.triangularized == y.triangularized
    };

    let plain = SparseLuSymbolic::analyze(&a).expect("analyze");
    let peeled = SparseLuSymbolic::analyze_triangularized(&a).expect("triangularized");
    // Non-vacuity: the two constructors must actually differ on this fixture,
    // or "reproduces both" is one claim, not two.
    assert!(
        !same(&plain, &peeled),
        "fixture is useless: the two orderings agree on it"
    );

    let default = LuOrderingParams::default();
    assert!(
        !default.triangularize,
        "the default must be whole-basis AMD (issue #163)"
    );
    let via_default = SparseLuSymbolic::analyze_with(&a, default).expect("analyze_with default");
    assert!(
        same(&via_default, &plain),
        "analyze_with(default) must be exactly `analyze`"
    );

    let via_peel = SparseLuSymbolic::analyze_with(
        &a,
        LuOrderingParams {
            triangularize: true,
        },
    )
    .expect("analyze_with peel");
    assert!(
        same(&via_peel, &peeled),
        "analyze_with(triangularize: true) must be exactly `analyze_triangularized`"
    );
}
