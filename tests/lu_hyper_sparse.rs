//! Differential tests for the reach-limited ("hyper-sparse") triangular solves
//! (issue #161B).
//!
//! The route is a *silent* optimization: a solve whose reach overruns
//! `hyper_sparse_max_density · m` falls back to the dense sweep and produces the
//! same answer. That makes vacuous passes the main hazard here — a test that
//! only checks "the answer is right" would still pass if the reach-limited code
//! never ran at all. Every test below that means to exercise the route asserts
//! on `hyper_sparse_sweeps()`, which counts the sweeps that actually took it.
//!
//! The oracle throughout is the **dense route on the same factor**: the two
//! routes are two ways of evaluating the same substitution, so agreement to
//! round-off is the contract. Where an independent check is cheap, the residual
//! `‖Bx − a‖∞` against the original basis is asserted too, so a bug that
//! corrupts both routes identically cannot hide.

use feral::{FeralError, LuParams, SparseColMatrix, SparseLu, SparseLuSymbolic};

/// Deterministic LCG — same generator as `src/bin/probe_lu_phases.rs`, so a
/// failing seed can be replayed there.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 32) as u32 as f64 / u32::MAX as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n.max(1)
    }
}

/// An LP-simplex-shaped basis: a sparse near-triangular skeleton with a
/// non-triangular "bump" block, then row- and column-permuted so nothing about
/// the ordering is handed to the solver. This is the structural signature issue
/// #161 reports for real QPLIB bases — overwhelmingly triangular, with a
/// residual bump that cannot be peeled — and it is what makes `B⁻¹eᵢ` sparse.
fn lp_basis(m: usize, bump: usize, band: usize, seed: u64) -> Vec<Vec<(usize, f64)>> {
    let mut rng = Rng(seed);
    let bump_lo = m / 3;
    let bump_hi = (bump_lo + bump).min(m);
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];

    for (j, col) in cols.iter_mut().enumerate() {
        col.push((j, 4.0 + rng.unit())); // strong diagonal
                                         // Sub-diagonal band: keeps the basis triangular and the inverse sparse.
                                         // At most one off-diagonal per column, so the reach out of a unit vector
                                         // is a *path* rather than a branching tree — which is what puts the
                                         // fixture at the ~2.3 nonzeros/column real LP bases have (QPLIB_3852 in
                                         // issue #161 measures 2.27/col) and what makes `B⁻¹eᵢ` genuinely sparse.
                                         // Two per column instead of one branches the reach and drives the median
                                         // solution past 25% dense, at which point the route stops firing at all.
        if rng.unit() < 0.7 {
            let i = j + 1 + rng.below(band);
            if i < m {
                col.push((i, rng.unit() * 2.0 - 1.0));
            }
        }
        // Bump: entries *above* the diagonal, so this block is not triangular
        // in any order and must be genuinely eliminated.
        if j >= bump_lo && j < bump_hi {
            for _ in 0..2 {
                let i = bump_lo + rng.below(bump_hi - bump_lo);
                if i != j {
                    col.push((i, rng.unit() * 2.0 - 1.0));
                }
            }
        }
    }

    // Permute rows and columns.
    let mut rperm: Vec<usize> = (0..m).collect();
    let mut cperm: Vec<usize> = (0..m).collect();
    for k in (1..m).rev() {
        rperm.swap(k, rng.below(k + 1));
        cperm.swap(k, rng.below(k + 1));
    }
    let mut out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for (j, col) in cols.into_iter().enumerate() {
        let dst = &mut out[cperm[j]];
        for (i, v) in col {
            dst.push((rperm[i], v));
        }
        dst.sort_by_key(|&(i, _)| i);
        dst.dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 += a.1;
                true
            } else {
                false
            }
        });
    }
    out
}

fn params(density: f64) -> LuParams {
    LuParams {
        hyper_sparse_max_density: density,
        ..LuParams::default()
    }
}

/// Factor the same basis twice: once with the route off, once on.
fn factor_pair(cols: &[Vec<(usize, f64)>], m: usize, density: f64) -> (SparseLu, SparseLu) {
    let a = SparseColMatrix::from_sparse_columns(m, cols).expect("basis");
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    let off = SparseLu::factor(&a, &sym, params(0.0)).expect("factor off");
    let on = SparseLu::factor(&a, &sym, params(density)).expect("factor on");
    (off, on)
}

fn max_abs_diff(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()))
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

/// The core contract: on unit-vector right-hand sides — what a simplex
/// ftran/btran actually receives — the reach-limited route must agree with the
/// dense route to round-off, and must actually have fired.
#[test]
fn reach_route_agrees_with_dense_route_on_unit_rhs() {
    let m = 400;
    let cols = lp_basis(m, 40, 6, 0xF00D);
    let a = SparseColMatrix::from_sparse_columns(m, &cols).expect("basis");
    let (mut off, mut on) = factor_pair(&cols, m, 0.25);
    assert_eq!(
        off.factor_nnz(),
        on.factor_nnz(),
        "the route must not change the factor, only how it is traversed"
    );

    let mut worst_f = 0.0_f64;
    let mut worst_b = 0.0_f64;
    let mut worst_resid = 0.0_f64;
    for k in 0..m {
        let mut e = vec![0.0; m];
        e[k] = 1.0;

        let (mut xf_off, mut xf_on) = (e.clone(), e.clone());
        off.ftran(&mut xf_off).expect("ftran off");
        on.ftran(&mut xf_on).expect("ftran on");
        worst_f = worst_f.max(max_abs_diff(&xf_off, &xf_on));

        let (mut xb_off, mut xb_on) = (e.clone(), e.clone());
        off.btran(&mut xb_off).expect("btran off");
        on.btran(&mut xb_on).expect("btran on");
        worst_b = worst_b.max(max_abs_diff(&xb_off, &xb_on));

        // Independent check: B x = e, so a bug shared by both routes still shows.
        let mut bx = vec![0.0; m];
        a.matvec(&xf_on, &mut bx);
        bx[k] -= 1.0;
        worst_resid = worst_resid.max(inf_norm(&bx) / inf_norm(&xf_on).max(1.0));
    }

    assert_eq!(
        off.hyper_sparse_sweeps(),
        0,
        "density 0.0 must never take the reach route"
    );
    assert!(
        on.hyper_sparse_sweeps() > m,
        "the reach route must have fired on most of the {m} ftran + {m} btran \
         sweeps, got {} — otherwise this test passes vacuously",
        on.hyper_sparse_sweeps()
    );
    assert!(
        worst_f < 1e-12 && worst_b < 1e-12,
        "reach vs dense route disagree: ftran {worst_f:e}, btran {worst_b:e}"
    );
    assert!(worst_resid < 1e-10, "residual {worst_resid:e}");
}

/// A dense right-hand side must produce a reach that overruns the cap, fall back
/// to the dense sweep, and still be correct. This is the guard on the fallback
/// itself: if `abandon()` failed to restore the all-false `mark` invariant, the
/// *next* solve's reach would be silently truncated and wrong — so the dense-rhs
/// solves are interleaved with sparse ones rather than run in a block.
#[test]
fn dense_rhs_falls_back_and_leaves_the_workspace_clean() {
    let m = 300;
    let cols = lp_basis(m, 30, 5, 0xBEEF);
    let (mut off, mut on) = factor_pair(&cols, m, 0.25);

    let mut rng = Rng(7);
    for k in 0..m {
        // Alternate: a dense rhs (must abandon a partial reach) then a unit rhs
        // (must still get a correct, untruncated reach right afterwards).
        let dense: Vec<f64> = (0..m).map(|_| rng.unit() + 0.5).collect();
        let (mut d_off, mut d_on) = (dense.clone(), dense.clone());
        off.ftran(&mut d_off).expect("ftran off");
        on.ftran(&mut d_on).expect("ftran on");
        assert!(
            max_abs_diff(&d_off, &d_on) < 1e-12,
            "dense-rhs ftran diverged at k={k}"
        );
        let (mut d_off, mut d_on) = (dense.clone(), dense.clone());
        off.btran(&mut d_off).expect("btran off");
        on.btran(&mut d_on).expect("btran on");
        assert!(
            max_abs_diff(&d_off, &d_on) < 1e-12,
            "dense-rhs btran diverged at k={k}"
        );

        let mut e = vec![0.0; m];
        e[k] = 1.0;
        let (mut e_off, mut e_on) = (e.clone(), e.clone());
        off.ftran(&mut e_off).expect("ftran off");
        on.ftran(&mut e_on).expect("ftran on");
        assert!(
            max_abs_diff(&e_off, &e_on) < 1e-12,
            "unit-rhs ftran after a dense-rhs fallback diverged at k={k} — \
             the abandoned reach did not restore its marker"
        );
    }
    assert!(
        on.hyper_sparse_sweeps() > 0,
        "the interleaved unit-rhs solves must still have taken the route"
    );
}

/// The route must survive a chain of Forrest–Tomlin updates. This is the test
/// that would catch the two ordering traps: `U` is upper triangular in `uperm`
/// order (not position order) once an update has permuted it, so a reach sorted
/// by position index would sweep in a wrong order; and `u_above` widens to hold
/// off-diagonal entries at positions *below* their column, so a reach that
/// assumed strict-upper adjacency would miss edges.
#[test]
fn reach_route_survives_forrest_tomlin_updates() {
    let m = 250;
    let cols = lp_basis(m, 25, 5, 0x5EED);
    let a = SparseColMatrix::from_sparse_columns(m, &cols).expect("basis");
    let (mut off, mut on) = factor_pair(&cols, m, 0.25);

    let mut rng = Rng(11);
    let mut applied = 0usize;
    for step in 0..24 {
        let slot = rng.below(m);
        // Entering column `B[:,slot] + 0.3·B[:,other]`. Replacing column `slot`
        // by this multiplies `det(B)` by exactly 1 (the transform is
        // `I + 0.3·e_other e_slotᵀ`, unit triangular for `other != slot`), so the
        // updated basis is guaranteed nonsingular and the FT update has a pivot
        // to find. A column of random nonzeros instead is almost always
        // rejected with `NeedsRefactor`, which would silently reduce this test
        // to the no-update case — hence the `applied` floor asserted below.
        let other = (slot + 1 + rng.below(m - 1)) % m;
        let mut entering = vec![0.0; m];
        for &(i, v) in cols[slot].iter() {
            entering[i] += v;
        }
        for &(i, v) in cols[other].iter() {
            entering[i] += 0.3 * v;
        }

        let r_off = off.update(slot, &entering);
        let r_on = on.update(slot, &entering);
        assert_eq!(
            r_off.is_ok(),
            r_on.is_ok(),
            "the routes must agree on whether update {step} is possible"
        );
        if r_off.is_err() {
            continue;
        }
        applied += 1;

        for k in [step % m, (step * 7 + 3) % m] {
            let mut e = vec![0.0; m];
            e[k] = 1.0;
            let (mut f_off, mut f_on) = (e.clone(), e.clone());
            let (o, n) = (off.ftran(&mut f_off), on.ftran(&mut f_on));
            assert_eq!(o.is_ok(), n.is_ok());
            if o.is_ok() {
                assert!(
                    max_abs_diff(&f_off, &f_on) < 1e-9,
                    "ftran diverged after {applied} updates (rhs e{k}): {:e}",
                    max_abs_diff(&f_off, &f_on)
                );
            }
            let (mut b_off, mut b_on) = (e.clone(), e.clone());
            let (o, n) = (off.btran(&mut b_off), on.btran(&mut b_on));
            assert_eq!(o.is_ok(), n.is_ok());
            if o.is_ok() {
                assert!(
                    max_abs_diff(&b_off, &b_on) < 1e-9,
                    "btran diverged after {applied} updates (rhs e{k}): {:e}",
                    max_abs_diff(&b_off, &b_on)
                );
            }
        }
    }
    assert!(
        applied >= 8,
        "only {applied} updates applied — test too weak"
    );
    assert!(
        on.hyper_sparse_sweeps() > 0,
        "the reach route never fired across the update chain"
    );
    let _ = a;
}

/// `hyper_sparse_max_density = 0.0` is the documented off switch: no reach
/// sweeps, and none of the route's state is allocated. The observable proxy for
/// "not allocated" is that a clone of the factor is not larger, which we cannot
/// see from here — so this asserts the behavioral half (no sweeps, identical
/// answers) and leaves the allocation claim to the code.
#[test]
fn density_zero_is_off() {
    let m = 200;
    let cols = lp_basis(m, 20, 4, 0xABCD);
    let (mut off, mut on) = factor_pair(&cols, m, 0.5);
    for k in 0..m {
        let mut e = vec![0.0; m];
        e[k] = 1.0;
        let (mut x_off, mut x_on) = (e.clone(), e.clone());
        off.ftran(&mut x_off).expect("ftran");
        on.ftran(&mut x_on).expect("ftran");
        assert!(max_abs_diff(&x_off, &x_on) < 1e-12);
    }
    assert_eq!(off.hyper_sparse_sweeps(), 0);
    assert!(on.hyper_sparse_sweeps() > 0);
}

/// A cap so small that no reach can fit under it must route everything to the
/// dense sweep — the degenerate end of the fallback, where the `cap == 0`
/// arithmetic (`m as f64 * d` truncating to zero) has to behave.
#[test]
fn tiny_cap_routes_everything_dense() {
    let m = 64;
    let cols = lp_basis(m, 8, 3, 0x1234);
    // 64 * 0.001 == 0.064, truncating to a cap of 0 nodes.
    let (mut off, mut on) = factor_pair(&cols, m, 0.001);
    for k in 0..m {
        let mut e = vec![0.0; m];
        e[k] = 1.0;
        let (mut x_off, mut x_on) = (e.clone(), e.clone());
        off.ftran(&mut x_off).expect("ftran");
        on.ftran(&mut x_on).expect("ftran");
        assert!(max_abs_diff(&x_off, &x_on) < 1e-12);
        let (mut y_off, mut y_on) = (e.clone(), e.clone());
        off.btran(&mut y_off).expect("btran");
        on.btran(&mut y_on).expect("btran");
        assert!(max_abs_diff(&y_off, &y_on) < 1e-12);
    }
    assert_eq!(
        on.hyper_sparse_sweeps(),
        0,
        "a zero cap must abort every reach"
    );
}

/// An all-zero right-hand side reaches nothing, which is the one input where the
/// route's narrowed diagnostic is observable: the dense sweep divides `0` by
/// every `U` diagonal on the way past, the reach route touches no row at all.
/// Both must return the zero solution.
#[test]
fn zero_rhs_solves_to_zero_on_both_routes() {
    let m = 120;
    let cols = lp_basis(m, 12, 4, 0x0F0F);
    let (mut off, mut on) = factor_pair(&cols, m, 0.25);
    for lu in [&mut off, &mut on] {
        let mut z = vec![0.0; m];
        lu.ftran(&mut z).expect("ftran zero");
        assert!(z.iter().all(|&v| v == 0.0), "ftran(0) must be 0");
        let mut z = vec![0.0; m];
        lu.btran(&mut z).expect("btran zero");
        assert!(z.iter().all(|&v| v == 0.0), "btran(0) must be 0");
    }
}

/// `hyper_sparse_max_density` outside `[0, 1]` is rejected before any
/// factorization consumes it — including `NaN`, which would otherwise cast to a
/// cap of `0` and silently disable the route rather than report the mistake.
#[test]
fn out_of_range_density_is_rejected() {
    let m = 32;
    let cols = lp_basis(m, 4, 3, 0x9);
    let a = SparseColMatrix::from_sparse_columns(m, &cols).expect("basis");
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    for bad in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                SparseLu::factor(&a, &sym, params(bad)),
                Err(FeralError::InvalidInput(_))
            ),
            "hyper_sparse_max_density = {bad} must be rejected"
        );
    }
    for ok in [0.0, 0.25, 1.0] {
        assert!(SparseLu::factor(&a, &sym, params(ok)).is_ok());
    }
}

/// Two-sided scaling wraps the core solve in permutations and diagonal scalings.
/// The reach runs *inside* that wrapper, on the factored `Ã`, so a route that
/// leaked scaled coordinates into unscaled ones would show up here and nowhere
/// else.
#[test]
fn reach_route_is_correct_under_scaling() {
    use feral::LuScaling;
    let m = 220;
    let cols = lp_basis(m, 22, 5, 0xDEAD);
    let a = SparseColMatrix::from_sparse_columns(m, &cols).expect("basis");
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    let with = |d: f64| LuParams {
        hyper_sparse_max_density: d,
        scaling: LuScaling::InfNorm,
        ..LuParams::default()
    };
    let mut off = SparseLu::factor(&a, &sym, with(0.0)).expect("factor off");
    let mut on = SparseLu::factor(&a, &sym, with(0.25)).expect("factor on");

    for k in 0..m {
        let mut e = vec![0.0; m];
        e[k] = 1.0;
        let (mut x_off, mut x_on) = (e.clone(), e.clone());
        off.ftran(&mut x_off).expect("ftran off");
        on.ftran(&mut x_on).expect("ftran on");
        assert!(
            max_abs_diff(&x_off, &x_on) < 1e-11,
            "scaled ftran diverged at k={k}"
        );
        let (mut y_off, mut y_on) = (e.clone(), e.clone());
        off.btran(&mut y_off).expect("btran off");
        on.btran(&mut y_on).expect("btran on");
        assert!(
            max_abs_diff(&y_off, &y_on) < 1e-11,
            "scaled btran diverged at k={k}"
        );
    }
    assert!(on.hyper_sparse_sweeps() > 0);
}

/// The reach must be work-proportional, not just correct. A unit-vector rhs on
/// a near-triangular basis has to touch a small fraction of the factor; if a
/// future change made the reach traverse the whole DAG the answers would still
/// be right and every other test here would still pass.
///
/// The witness is the sweep count against `factor_nnz`: the route only fires
/// when the reach stayed under `0.25·m`, so a high fire rate over unit-vector
/// solves *is* the statement that the solutions are sparse.
#[test]
fn unit_rhs_solutions_are_sparse_enough_to_route() {
    let m = 500;
    let cols = lp_basis(m, 50, 6, 0xC0FFEE);
    let (_, mut on) = factor_pair(&cols, m, 0.25);
    let mut sparse_solutions = 0usize;
    for k in 0..m {
        let mut e = vec![0.0; m];
        e[k] = 1.0;
        on.ftran(&mut e).expect("ftran");
        let nnz = e.iter().filter(|&&v| v != 0.0).count();
        if nnz * 4 <= m {
            sparse_solutions += 1;
        }
    }
    assert!(
        sparse_solutions * 2 > m,
        "only {sparse_solutions} of {m} unit-rhs solutions were under 25% dense \
         — the fixture is not LP-shaped enough to exercise the route"
    );
    assert!(
        on.hyper_sparse_sweeps() >= sparse_solutions,
        "sweeps {} < sparse solutions {sparse_solutions}",
        on.hyper_sparse_sweeps()
    );
}

/// **Composition with the dense-bump route (issue #161 part A, PR #160).**
///
/// The reach-limited sweeps walk `u_above` and the row-wise index of `L`. The
/// dense-bump route rebuilds part of both by splicing a dense factorization
/// into them. This checks the reach-limited route against the dense sweep on
/// such a factor, with both `used_dense_bump()` and `hyper_sparse_sweeps()`
/// asserted so neither half can be silently absent.
#[test]
fn reach_route_composes_with_the_dense_bump_route() {
    let m = 400;
    let cols = lp_basis(m, 40, 6, 0xB0BA);
    let a = SparseColMatrix::from_sparse_columns(m, &cols).expect("basis");
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    let mk = |d: f64| LuParams {
        dense_bump_max_dim: 512,
        hyper_sparse_max_density: d,
        ..LuParams::default()
    };
    let mut off = SparseLu::factor(&a, &sym, mk(0.0)).expect("factor off");
    let mut on = SparseLu::factor(&a, &sym, mk(0.25)).expect("factor on");
    assert!(
        off.used_dense_bump() && on.used_dense_bump(),
        "the dense-bump route did not fire on both arms"
    );
    assert_eq!(
        off.factor_nnz(),
        on.factor_nnz(),
        "the reach route must not change the factor the bump route produced"
    );

    let mut worst = 0.0_f64;
    for k in 0..m {
        let mut e = vec![0.0; m];
        e[k] = 1.0;
        let (mut f_off, mut f_on) = (e.clone(), e.clone());
        off.ftran(&mut f_off).expect("ftran off");
        on.ftran(&mut f_on).expect("ftran on");
        worst = worst.max(max_abs_diff(&f_off, &f_on));
        let (mut b_off, mut b_on) = (e.clone(), e.clone());
        off.btran(&mut b_off).expect("btran off");
        on.btran(&mut b_on).expect("btran on");
        worst = worst.max(max_abs_diff(&b_off, &b_on));
    }
    assert_eq!(off.hyper_sparse_sweeps(), 0);
    assert!(
        on.hyper_sparse_sweeps() > m,
        "the reach route fired only {} times on a dense-bump factor",
        on.hyper_sparse_sweeps()
    );
    assert!(
        worst < 1e-11,
        "routes diverged on a dense-bump factor: {worst:e}"
    );
}

/// The shipped default cap is pinned, because getting it wrong is silent.
///
/// This PR originally shipped `0.25`, chosen from a sweep on the in-tree
/// synthetic fixture where "the win is flat from 0.05 to 1.00". That was true
/// for `ftran` and false for `btran`: on the real QPLIB_1157 basis, `0.25`
/// admits `btran` reaches sitting between 10% and 25% of `m`, which the sorted
/// sweep loses on — a **1.45x regression** for no `ftran` gain.
///
/// The fixture could not show it. The effect only appears when a basis's
/// solution-density profile straddles the cap, and `lp_basis()` produces
/// solutions that are either tiny or nearly `m`, never in that band. So this
/// test cannot re-derive the right value either — it just pins it, so that
/// raising it again is a deliberate act that has to come with evidence from a
/// basis whose solutions live in the admitted band and *win* there.
#[test]
fn default_density_cap_is_pinned_to_the_measured_value() {
    let d = LuParams::default().hyper_sparse_max_density;
    assert!(
        (d - 0.10).abs() < 1e-12,
        "default hyper_sparse_max_density is {d}, expected 0.10. Raising this \
         needs a real basis measurement, not a synthetic sweep — see the \
         parameter's doc comment for the QPLIB_1157 table that set it."
    );
}
