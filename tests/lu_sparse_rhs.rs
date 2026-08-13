//! Tests for the sparse-in / sparse-out `ftran_sparse` / `btran_sparse`
//! (issue #161B, second half).
//!
//! The oracle is the **dense entry point on the same factor**. `ftran_sparse`
//! is not a different algorithm — it is the same substitution reached through a
//! signature that does not force an `O(m)` pass — so agreement to round-off is
//! the contract, and disagreement is a bug in the new path rather than a
//! tolerance question.
//!
//! Two hazards get their own tests beyond plain agreement:
//!
//! * **The zeroed-accumulator invariant.** The sparse path keeps a dense
//!   length-`m` accumulator that must be all-zero between calls, restored in
//!   `O(touched)`. A leak does not corrupt the solve that caused it — it
//!   corrupts the *next* one. So the tests here interleave rather than batch,
//!   and deliberately run failing solves in between succeeding ones.
//! * **Work-proportionality.** The whole point is an asymptotic claim, and a
//!   wall-clock benchmark cannot pin an asymptote. `last_sparse_solve_work()`
//!   is asserted to stay flat while `m` grows 8x.

use feral::{FeralError, LuParams, LuScaling, SparseColMatrix, SparseLu, SparseLuSymbolic};

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

/// An LP-simplex-shaped basis; see `tests/lu_hyper_sparse.rs` for the rationale
/// (near-triangular skeleton at ~2 nonzeros/column plus a non-triangular bump,
/// row- and column-permuted). `bump = 0` gives a purely triangular basis whose
/// `B⁻¹eᵢ` support does not grow with `m` — which is what the
/// work-proportionality test needs.
fn lp_basis(m: usize, bump: usize, band: usize, seed: u64) -> Vec<Vec<(usize, f64)>> {
    let mut rng = Rng(seed);
    let bump_lo = m / 3;
    let bump_hi = (bump_lo + bump).min(m);
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for (j, col) in cols.iter_mut().enumerate() {
        col.push((j, 4.0 + rng.unit()));
        if rng.unit() < 0.7 {
            let i = j + 1 + rng.below(band);
            if i < m {
                col.push((i, rng.unit() * 2.0 - 1.0));
            }
        }
        if j >= bump_lo && j < bump_hi {
            for _ in 0..2 {
                let i = bump_lo + rng.below(bump_hi - bump_lo);
                if i != j {
                    col.push((i, rng.unit() * 2.0 - 1.0));
                }
            }
        }
    }
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

fn factor(cols: &[Vec<(usize, f64)>], m: usize, scaling: LuScaling) -> SparseLu {
    let a = SparseColMatrix::from_sparse_columns(m, cols).expect("basis");
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    SparseLu::factor(
        &a,
        &sym,
        LuParams {
            scaling,
            ..LuParams::default()
        },
    )
    .expect("factor")
}

/// Expand a sparse solution to dense for comparison against the dense API.
fn densify(sparse: &[(usize, f64)], m: usize) -> Vec<f64> {
    let mut v = vec![0.0; m];
    for &(i, x) in sparse.iter() {
        v[i] = x;
    }
    v
}

fn max_abs_diff(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()))
}

/// The core contract, on the right-hand sides a simplex actually sends: unit
/// vectors. The sparse and dense entry points must produce the same solution.
#[test]
fn sparse_and_dense_ftran_btran_agree_on_unit_rhs() {
    let m = 400;
    let cols = lp_basis(m, 40, 6, 0xF00D);
    let mut lu = factor(&cols, m, LuScaling::None);
    let mut out = Vec::new();
    let mut worst_f = 0.0_f64;
    let mut worst_b = 0.0_f64;

    for k in 0..m {
        let mut dense = vec![0.0; m];
        dense[k] = 1.0;
        let mut expect = dense.clone();
        lu.ftran(&mut expect).expect("ftran");
        lu.ftran_sparse(&[(k, 1.0)], &mut out)
            .expect("ftran_sparse");
        worst_f = worst_f.max(max_abs_diff(&expect, &densify(&out, m)));

        let mut expect = dense.clone();
        lu.btran(&mut expect).expect("btran");
        lu.btran_sparse(&[(k, 1.0)], &mut out)
            .expect("btran_sparse");
        worst_b = worst_b.max(max_abs_diff(&expect, &densify(&out, m)));
    }
    assert!(
        worst_f < 1e-12 && worst_b < 1e-12,
        "sparse vs dense entry point: ftran {worst_f:e}, btran {worst_b:e}"
    );
}

/// Multi-nonzero right-hand sides, including ones dense enough that the reach
/// covers most of the factor. The sparse path has no dense fallback, so this is
/// where it has to be right rather than merely fast.
#[test]
fn sparse_and_dense_agree_on_multi_nonzero_rhs() {
    let m = 300;
    let cols = lp_basis(m, 30, 5, 0xBEEF);
    let mut lu = factor(&cols, m, LuScaling::None);
    let mut rng = Rng(3);
    let mut out = Vec::new();
    let mut worst = 0.0_f64;

    for nnz in [1usize, 2, 5, 20, 100, m] {
        for _ in 0..8 {
            let mut dense = vec![0.0; m];
            let mut sparse: Vec<(usize, f64)> = Vec::new();
            for _ in 0..nnz {
                let i = rng.below(m);
                let v = rng.unit() * 2.0 - 1.0;
                if dense[i] == 0.0 && v != 0.0 {
                    dense[i] = v;
                    sparse.push((i, v));
                }
            }
            sparse.sort_by_key(|&(i, _)| i);

            let mut expect = dense.clone();
            lu.ftran(&mut expect).expect("ftran");
            lu.ftran_sparse(&sparse, &mut out).expect("ftran_sparse");
            worst = worst.max(max_abs_diff(&expect, &densify(&out, m)));

            let mut expect = dense.clone();
            lu.btran(&mut expect).expect("btran");
            lu.btran_sparse(&sparse, &mut out).expect("btran_sparse");
            worst = worst.max(max_abs_diff(&expect, &densify(&out, m)));
        }
    }
    assert!(worst < 1e-11, "sparse vs dense on multi-nnz rhs: {worst:e}");
}

/// **The point of the exercise.** With the local structure held fixed and `m`
/// grown 8x, the scalar work of a sparse solve must not grow. The dense entry
/// point's cost necessarily does — it reads `m` values and writes `m` values —
/// which is exactly the floor this API exists to get under.
///
/// A wall-clock benchmark cannot make this assertion: an `O(m)` term
/// reintroduced by a future change would look like a constant-factor slowdown,
/// which is indistinguishable from machine noise at these sizes. The work
/// counter is deterministic, so the asymptote can be pinned directly.
#[test]
fn sparse_solve_work_does_not_grow_with_m() {
    // Purely triangular (bump = 0): the reach out of a unit vector is a path
    // whose expected length is set by the band and the off-diagonal density,
    // both of which are held fixed as `m` varies.
    let median_work = |m: usize| -> usize {
        let cols = lp_basis(m, 0, 3, 0xC0FFEE);
        let mut lu = factor(&cols, m, LuScaling::None);
        let mut out = Vec::new();
        let mut work: Vec<usize> = Vec::new();
        for t in 0..256 {
            let k = (t * 97 + 11) % m;
            lu.ftran_sparse(&[(k, 1.0)], &mut out)
                .expect("ftran_sparse");
            work.push(lu.last_sparse_solve_work());
            lu.btran_sparse(&[(k, 1.0)], &mut out)
                .expect("btran_sparse");
            work.push(lu.last_sparse_solve_work());
        }
        work.sort_unstable();
        work[work.len() / 2]
    };

    let small = median_work(1000);
    let large = median_work(8000);
    assert!(small > 0, "the work counter is not being recorded");
    assert!(
        large <= 3 * small,
        "sparse solve work grew with m: {small} at m=1000 vs {large} at m=8000 \
         (m grew 8x; an O(m) term would show ~8x here)"
    );
    // And in absolute terms it is a tiny fraction of the basis, which is what
    // "hyper-sparse" means. A solve that quietly touched the whole factor would
    // satisfy the ratio above if it did so at both sizes.
    assert!(
        large * 50 < 8000,
        "median work {large} is not small relative to m=8000"
    );
}

/// The dense accumulator must come back all-zero from **every** exit path,
/// including the error path out of the `U` solve. A leak corrupts the next
/// solve, not the one that leaked, so the failing solves are interleaved with
/// good ones and the good ones are checked against the dense oracle.
#[test]
fn failed_solves_leave_the_accumulator_clean() {
    let m = 200;
    let cols = lp_basis(m, 20, 4, 0x5EED);
    let mut lu = factor(&cols, m, LuScaling::None);
    let mut out = Vec::new();

    // Establish the oracle before corrupting anything.
    let mut expect = vec![0.0; m];
    expect[7] = 1.0;
    lu.ftran(&mut expect).expect("ftran");

    for round in 0..16 {
        // A solve that must fail: an out-of-range index, caught during the
        // scatter, after some positions may already have been touched.
        let bad = vec![(3usize, 1.0), (m + round, 2.0)];
        assert!(
            matches!(
                lu.ftran_sparse(&bad, &mut out),
                Err(FeralError::DimensionMismatch { .. })
            ),
            "an out-of-range index must be rejected"
        );
        assert!(
            out.is_empty(),
            "a failed solve must not leave output behind"
        );

        // ... and the very next solve must be unaffected.
        lu.ftran_sparse(&[(7, 1.0)], &mut out)
            .expect("ftran_sparse");
        let got = densify(&out, m);
        assert!(
            max_abs_diff(&expect, &got) < 1e-12,
            "round {round}: a solve after a failed one diverged — the \
             accumulator was not restored"
        );
    }
}

/// Repeated indices in the right-hand side accumulate, matching what scattering
/// into a dense vector would do. Explicit zeros contribute nothing and are not
/// emitted.
#[test]
fn duplicate_indices_accumulate_and_zeros_are_dropped() {
    let m = 120;
    let cols = lp_basis(m, 12, 4, 0xABCD);
    let mut lu = factor(&cols, m, LuScaling::None);
    let mut out = Vec::new();

    let mut dense = vec![0.0; m];
    dense[5] = 1.5 + 2.5; // the sum of the duplicates below
    dense[9] = -1.0;
    let mut expect = dense.clone();
    lu.ftran(&mut expect).expect("ftran");

    let rhs = vec![(5usize, 1.5), (9, -1.0), (5, 2.5), (11, 0.0)];
    lu.ftran_sparse(&rhs, &mut out).expect("ftran_sparse");
    assert!(max_abs_diff(&expect, &densify(&out, m)) < 1e-12);
    assert!(
        out.iter().all(|&(_, v)| v != 0.0),
        "explicit zeros must not be emitted"
    );
    assert!(
        out.windows(2).all(|w| w[0].0 < w[1].0),
        "output must be sorted by index and duplicate-free"
    );
}

/// Two-sided scaling wraps the solve in permutations and diagonal scalings that
/// the sparse path applies per-nonzero rather than per-element. A mapping error
/// there would be invisible without scaling turned on.
#[test]
fn sparse_solves_are_correct_under_scaling() {
    let m = 220;
    let cols = lp_basis(m, 22, 5, 0xDEAD);
    for scaling in [LuScaling::InfNorm, LuScaling::Mc64] {
        let mut lu = factor(&cols, m, scaling);
        let mut out = Vec::new();
        let mut worst = 0.0_f64;
        for k in 0..m {
            let mut dense = vec![0.0; m];
            dense[k] = 1.0;

            let mut expect = dense.clone();
            lu.ftran(&mut expect).expect("ftran");
            lu.ftran_sparse(&[(k, 1.0)], &mut out)
                .expect("ftran_sparse");
            worst = worst.max(max_abs_diff(&expect, &densify(&out, m)));

            let mut expect = dense.clone();
            lu.btran(&mut expect).expect("btran");
            lu.btran_sparse(&[(k, 1.0)], &mut out)
                .expect("btran_sparse");
            worst = worst.max(max_abs_diff(&expect, &densify(&out, m)));
        }
        assert!(worst < 1e-10, "scaling {scaling:?}: diverged by {worst:e}");
    }
}

/// Across a Forrest–Tomlin update chain. This is where the eta replay matters:
/// the sparse path walks the same ops as `FtEta::apply_forward` /
/// `apply_transpose` but must additionally extend its pattern with whatever
/// they make nonzero. Missing that would drop entries from the solution — and
/// the missing entries would be exactly the ones the update introduced, which
/// no test on a fresh factor can see.
#[test]
fn sparse_solves_track_forrest_tomlin_updates() {
    let m = 250;
    let cols = lp_basis(m, 25, 5, 0x1234);
    let mut lu = factor(&cols, m, LuScaling::None);
    let mut out = Vec::new();
    let mut rng = Rng(11);
    let mut applied = 0usize;

    for step in 0..24 {
        let slot = rng.below(m);
        // `B[:,slot] + 0.3·B[:,other]` keeps the basis nonsingular by
        // construction (see tests/lu_hyper_sparse.rs for why), so the update
        // has a pivot to find instead of being rejected.
        let other = (slot + 1 + rng.below(m - 1)) % m;
        let mut entering = vec![0.0; m];
        for &(i, v) in cols[slot].iter() {
            entering[i] += v;
        }
        for &(i, v) in cols[other].iter() {
            entering[i] += 0.3 * v;
        }
        if lu.update(slot, &entering).is_err() {
            continue;
        }
        applied += 1;

        for k in [step % m, (step * 7 + 3) % m] {
            let mut dense = vec![0.0; m];
            dense[k] = 1.0;

            let mut expect = dense.clone();
            let dense_ok = lu.ftran(&mut expect).is_ok();
            let sparse_res = lu.ftran_sparse(&[(k, 1.0)], &mut out);
            assert_eq!(dense_ok, sparse_res.is_ok(), "routes disagree on failure");
            if dense_ok {
                assert!(
                    max_abs_diff(&expect, &densify(&out, m)) < 1e-9,
                    "ftran diverged after {applied} updates: {:e}",
                    max_abs_diff(&expect, &densify(&out, m))
                );
            }

            let mut expect = dense.clone();
            let dense_ok = lu.btran(&mut expect).is_ok();
            let sparse_res = lu.btran_sparse(&[(k, 1.0)], &mut out);
            assert_eq!(dense_ok, sparse_res.is_ok(), "routes disagree on failure");
            if dense_ok {
                assert!(
                    max_abs_diff(&expect, &densify(&out, m)) < 1e-9,
                    "btran diverged after {applied} updates: {:e}",
                    max_abs_diff(&expect, &densify(&out, m))
                );
            }
        }
    }
    assert!(
        applied >= 8,
        "only {applied} updates applied — test too weak"
    );
}

/// An empty right-hand side solves to an empty solution, and a right-hand side
/// of nothing but explicit zeros does too — without touching the factor.
#[test]
fn empty_rhs_gives_empty_solution() {
    let m = 64;
    let cols = lp_basis(m, 8, 3, 0x9);
    let mut lu = factor(&cols, m, LuScaling::None);
    let mut out = vec![(0usize, 1.0)]; // must be cleared
    lu.ftran_sparse(&[], &mut out).expect("ftran_sparse");
    assert!(out.is_empty(), "empty rhs must give empty solution");
    lu.btran_sparse(&[(3, 0.0), (5, 0.0)], &mut out)
        .expect("btran_sparse");
    assert!(out.is_empty(), "all-zero rhs must give empty solution");
    assert_eq!(
        lu.last_sparse_solve_work(),
        0,
        "an empty solve must do no work"
    );
}

/// The sparse entry points do not depend on `hyper_sparse_max_density`: that
/// parameter governs whether the *dense* entry points take their reach-limited
/// route. A sparse-in/sparse-out solve is reach-based by construction, and must
/// work — and be work-proportional — even with the dense route switched off.
#[test]
fn sparse_entry_points_work_with_the_dense_route_disabled() {
    let m = 200;
    let cols = lp_basis(m, 20, 4, 0x77);
    let a = SparseColMatrix::from_sparse_columns(m, &cols).expect("basis");
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    let mut lu = SparseLu::factor(
        &a,
        &sym,
        LuParams {
            hyper_sparse_max_density: 0.0,
            ..LuParams::default()
        },
    )
    .expect("factor");

    let mut out = Vec::new();
    let mut worst = 0.0_f64;
    let mut worst_work = 0usize;
    for k in 0..m {
        let mut expect = vec![0.0; m];
        expect[k] = 1.0;
        lu.ftran(&mut expect).expect("ftran");
        lu.ftran_sparse(&[(k, 1.0)], &mut out)
            .expect("ftran_sparse");
        worst = worst.max(max_abs_diff(&expect, &densify(&out, m)));
        worst_work = worst_work.max(lu.last_sparse_solve_work());

        let mut expect = vec![0.0; m];
        expect[k] = 1.0;
        lu.btran(&mut expect).expect("btran");
        lu.btran_sparse(&[(k, 1.0)], &mut out)
            .expect("btran_sparse");
        worst = worst.max(max_abs_diff(&expect, &densify(&out, m)));
    }
    assert_eq!(
        lu.hyper_sparse_sweeps(),
        0,
        "the dense route must still be off"
    );
    assert!(worst < 1e-12, "diverged by {worst:e}");
    assert!(
        worst_work < m,
        "worst-case work {worst_work} should stay under m={m} — the lazily \
         built L row index is missing and the Lᵀ reach silently degenerated"
    );
}

/// **Composition with the dense-bump route (issue #161 part A, PR #160).**
///
/// That route factors the post-triangularization bump with the *dense* kernel
/// and splices the result back into the sparse `L`/`U`. The sparse solves here
/// navigate those same structures by reach — `u_above` for the `U` sweep, the
/// row-wise index of `L` for the `Lᵀ` sweep — so a splice that produced entries
/// those indices did not describe would give a silently truncated reach and a
/// wrong answer.
///
/// Neither branch could test this: part A predates the reach work and part B
/// was written against a tree without the splice. It is the merge's risk
/// surface, so it is the merge's test. `used_dense_bump()` is asserted so the
/// test cannot pass vacuously against the sparse fallback the route takes when
/// the peel does not hold.
#[test]
fn sparse_solves_compose_with_the_dense_bump_route() {
    let m = 400;
    let cols = lp_basis(m, 40, 6, 0xB0BA);
    let a = SparseColMatrix::from_sparse_columns(m, &cols).expect("basis");
    // `analyze_triangularized`, not `analyze`: since issue #163 the peel is
    // opt-in, and `dense_bump_max_dim` only applies to a symbolic that peeled.
    let sym = SparseLuSymbolic::analyze_triangularized(&a).expect("analyze");
    let mut lu = SparseLu::factor(
        &a,
        &sym,
        LuParams {
            dense_bump_max_dim: 512,
            ..LuParams::default()
        },
    )
    .expect("factor");
    assert!(
        lu.used_dense_bump(),
        "the dense-bump route did not fire — this test would pass vacuously \
         against the plain sparse factorization"
    );

    let mut out = Vec::new();
    let mut worst = 0.0_f64;
    for k in 0..m {
        let mut expect = vec![0.0; m];
        expect[k] = 1.0;
        lu.ftran(&mut expect).expect("ftran");
        lu.ftran_sparse(&[(k, 1.0)], &mut out)
            .expect("ftran_sparse");
        worst = worst.max(max_abs_diff(&expect, &densify(&out, m)));

        let mut expect = vec![0.0; m];
        expect[k] = 1.0;
        lu.btran(&mut expect).expect("btran");
        lu.btran_sparse(&[(k, 1.0)], &mut out)
            .expect("btran_sparse");
        worst = worst.max(max_abs_diff(&expect, &densify(&out, m)));
    }
    assert!(
        worst < 1e-10,
        "sparse solves diverged from the dense entry point on a dense-bump \
         factor: {worst:e}"
    );
}
