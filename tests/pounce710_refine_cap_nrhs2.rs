//! pounce#710 acceptance item 3 — "verify the cap actually takes effect
//! on the `nrhs = 2` path".
//!
//! `Solver::solve_many_refined_into` dispatches on
//! `BLAS3_REFINE_THRESHOLD = 16`: wide solves refine through the batched
//! panel kernel (issue #58), narrow ones run a per-column loop. `nrhs =
//! 2` is the predictor-corrector shape an interior-point host actually
//! calls, so it is the *only* arm pounce would exercise — and it is the
//! arm that duplicated the dispatch, so a cap that reached the batched
//! path and missed this one would look fixed from every test issue #178
//! shipped.
//!
//! Issue #178's own tests cover the free functions
//! (`solve_sparse_many_refined_opts`) and cover `Solver` only at
//! `RefineOptions::default()`. This file covers the combination pounce
//! consumes: the stateful `Solver`, a non-default cap, and `nrhs` on
//! both sides of the threshold.
//!
//! The factor here is a genuinely perturbed approximate inverse — the
//! solver factors `A + ΔA` and refines against `A`, the same
//! construction as `tests/issue178_refine_cap.rs` — because that is the
//! only regime where the default budget runs long enough for a cap to be
//! observable. With an exact factor, refinement bottoms out on roundoff
//! in two or three steps and `k = 1` would be indistinguishable from
//! `k = 10` for reasons that have nothing to do with the cap.

use feral::numeric::factorize::NumericParams;
use feral::numeric::solve::RefineOptions;
use feral::symbolic::SupernodeParams;
use feral::{BunchKaufmanParams, CscMatrix, Solver, ZeroPivotAction};

const N: usize = 20;

/// Both sides of the `BLAS3_REFINE_THRESHOLD = 16` dispatch. `2` is the
/// shape pounce calls; `20` is here so a fix that moved the cap into the
/// batched arm alone cannot pass by covering the other one.
const NRHS_CASES: [usize; 2] = [2, 20];

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    })
}

/// Symmetric tridiagonal, `-1` off-diagonals, caller-chosen diagonal.
fn tridiag(n: usize, diag: impl Fn(usize) -> f64) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        rows.push(j);
        cols.push(j);
        vals.push(diag(j));
        if j + 1 < n {
            rows.push(j + 1);
            cols.push(j);
            vals.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("tridiag")
}

/// `(A, solver holding a factor of A + ΔA)`. The 1 % diagonal
/// perturbation makes refinement converge linearly rather than bottom
/// out, so the default run uses its whole budget.
fn perturbed_case() -> (CscMatrix, Solver) {
    let a = tridiag(N, |_| 2.0);
    let perturbed = tridiag(N, |j| 2.0 + 1e-2 * (1.0 + (j as f64) * 0.01));
    let mut s = Solver::with_params(ldlt_params(), SupernodeParams::default());
    let status = s.factor(&perturbed, None);
    assert!(
        s.factors().is_some(),
        "factor failed with status {status:?}"
    );
    (a, s)
}

/// Column-major right-hand sides, each column scaled differently so a
/// per-column cap cannot be confused with a whole-batch one.
fn rhs_for(nrhs: usize) -> Vec<f64> {
    (0..N * nrhs)
        .map(|k| (((k % N) % 7) as f64 - 3.0) * (1.0 + (k / N) as f64 * 0.25))
        .collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; a.n];
    a.symv(x, &mut ax);
    let (mut rs, mut bs) = (0.0, 0.0);
    for i in 0..a.n {
        let r = ax[i] - b[i];
        rs += r * r;
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

/// Non-vacuity guard for everything below. If the default run on this
/// fixture ever stopped refining past one step, every "the cap changed
/// the answer" assertion here would pass for the wrong reason.
#[test]
fn the_default_run_refines_past_one_step_on_both_arms() {
    let (a, s) = perturbed_case();
    for nrhs in NRHS_CASES {
        let rhs = rhs_for(nrhs);
        let default = s
            .solve_many_refined(&a, &rhs, nrhs)
            .expect("solve_many_refined");
        let one = s
            .solve_many_refined_opts(&a, &rhs, nrhs, RefineOptions::with_max_steps(1))
            .expect("solve_many_refined_opts");
        assert_ne!(
            bits(&default),
            bits(&one),
            "nrhs = {nrhs}: the default run stops at one correction, so a \
             missed cap would be undetectable on this fixture"
        );
    }
}

/// The strongest form of "the cap took effect": `k = 0` must be the
/// unrefined batched solve, bit for bit. If the narrow arm ignored
/// `opts`, this would come back refined and differ.
#[test]
fn cap_zero_on_the_solver_equals_solve_many_bit_for_bit() {
    let (a, s) = perturbed_case();
    for nrhs in NRHS_CASES {
        let rhs = rhs_for(nrhs);
        let plain = s.solve_many(&rhs, nrhs).expect("solve_many");
        let capped = s
            .solve_many_refined_opts(&a, &rhs, nrhs, RefineOptions::with_max_steps(0))
            .expect("solve_many_refined_opts");
        assert_eq!(bits(&capped), bits(&plain), "nrhs = {nrhs}");
    }
}

/// The cap is per column, and the per-column result is exactly what the
/// single-RHS refiner produces at the same cap. This is the property
/// pounce depends on: it calls with `nrhs = 2` and reasons about each
/// column's residual separately.
#[test]
fn capped_columns_match_the_single_rhs_refiner_at_the_same_cap() {
    let (a, s) = perturbed_case();
    for k in [1usize, 2, 3] {
        let opts = RefineOptions::with_max_steps(k);
        for nrhs in NRHS_CASES {
            let rhs = rhs_for(nrhs);
            let many = s
                .solve_many_refined_opts(&a, &rhs, nrhs, opts)
                .expect("solve_many_refined_opts");
            for c in 0..nrhs {
                let col = &rhs[c * N..(c + 1) * N];
                let one = s
                    .solve_refined_opts(&a, col, opts)
                    .expect("solve_refined_opts");
                assert_eq!(
                    bits(&many[c * N..(c + 1) * N]),
                    bits(&one),
                    "k = {k}, nrhs = {nrhs}, column {c}"
                );
            }
        }
    }
}

/// The cap selects *how many* corrections run, not merely whether any
/// do: on a linearly-converging factor each extra step must improve the
/// residual and produce a different iterate. A cap wired as a boolean
/// would give identical answers for k = 1, 2, 3.
#[test]
fn each_extra_allowed_step_is_actually_taken() {
    let (a, s) = perturbed_case();
    for nrhs in NRHS_CASES {
        let rhs = rhs_for(nrhs);
        let mut prev: Option<(Vec<f64>, f64)> = None;
        for k in 1usize..=4 {
            let x = s
                .solve_many_refined_opts(&a, &rhs, nrhs, RefineOptions::with_max_steps(k))
                .expect("solve_many_refined_opts");
            // Worst column residual: a batch-wide cap that stopped early
            // on one column would show up here.
            let worst = (0..nrhs)
                .map(|c| rel_residual(&a, &x[c * N..(c + 1) * N], &rhs[c * N..(c + 1) * N]))
                .fold(0.0f64, f64::max);
            if let Some((px, pr)) = prev {
                assert_ne!(
                    bits(&x),
                    bits(&px),
                    "nrhs = {nrhs}: k = {k} == k = {}",
                    k - 1
                );
                assert!(
                    worst < pr,
                    "nrhs = {nrhs}: k = {k} residual {worst:e} did not improve on \
                     k = {}'s {pr:e}",
                    k - 1
                );
            }
            prev = Some((x, worst));
        }
    }
}

/// The in-place entry point pounce would use to avoid the per-solve
/// allocation honours the same cap as its allocating twin.
#[test]
fn the_in_place_entry_point_honours_the_cap_too() {
    let (a, s) = perturbed_case();
    let opts = RefineOptions::with_max_steps(1);
    for nrhs in NRHS_CASES {
        let rhs = rhs_for(nrhs);
        let want = s
            .solve_many_refined_opts(&a, &rhs, nrhs, opts)
            .expect("solve_many_refined_opts");
        let mut got = vec![f64::NAN; N * nrhs];
        s.solve_many_refined_into(&a, &rhs, nrhs, &mut got, opts)
            .expect("solve_many_refined_into");
        assert_eq!(bits(&got), bits(&want), "nrhs = {nrhs}");
    }
}

/// `RefineOptions::default()` must remain the pre-#178 behaviour on this
/// path, so adopting the option in pounce is opt-in and nothing else
/// moves.
#[test]
fn default_options_change_nothing_on_this_path() {
    let (a, s) = perturbed_case();
    for nrhs in NRHS_CASES {
        let rhs = rhs_for(nrhs);
        let old = s
            .solve_many_refined(&a, &rhs, nrhs)
            .expect("solve_many_refined");
        let new = s
            .solve_many_refined_opts(&a, &rhs, nrhs, RefineOptions::default())
            .expect("solve_many_refined_opts");
        assert_eq!(bits(&new), bits(&old), "nrhs = {nrhs}");
    }
}
