//! Issue #178 item 1 — a caller-supplied cap on iterative-refinement
//! correction steps.
//!
//! An interior-point host runs its own refinement loop over the same
//! augmented system, so FERAL's inner 10-step budget is work whose
//! residual nobody consults. `RefineOptions::max_steps` lets that caller
//! say "at most k corrections" per call. The default is unchanged
//! (`DEFAULT_REFINE_MAX_STEPS == 10`).
//!
//! Test numbering follows the verification list in issue #178:
//!   1. cap truncates the trajectory,
//!   2. cap is a cap, not a target (existing exits keep priority),
//!   3. `k = 0` equals `solve_sparse` bit-for-bit,
//!   4. any `k` returns the best-residual iterate,
//!   5. the same four on the multi-RHS path.
//!
//! Design note: `dev/research/refinement-cap-2026-08-19.md`.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams, SparseFactors};
use feral::numeric::solve::{
    solve_sparse, solve_sparse_many, solve_sparse_many_refined, solve_sparse_many_refined_opts,
    solve_sparse_refined, solve_sparse_refined_opts, solve_sparse_refined_with_diagnostics,
    solve_sparse_refined_with_diagnostics_opts, RefineOptions, DEFAULT_REFINE_MAX_STEPS,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    })
}

/// Symmetric tridiagonal with `-1` off-diagonals and a caller-chosen
/// diagonal, lower triangle in CSC.
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

fn factorize(m: &CscMatrix) -> SparseFactors {
    let sym = symbolic_factorize(m, &SupernodeParams::default()).expect("symbolic");
    let (f, _) = factorize_multifrontal(m, &sym, &ldlt_params()).expect("numeric");
    f
}

const N: usize = 20;

/// The oracle for "a solve that runs the full budget".
///
/// FERAL's 2-strike plateau exit means a matrix factored *exactly* never
/// reaches step 10 — refinement bottoms out on roundoff in two or three
/// steps. The situation the cap exists for is the one where the factor is
/// a genuinely perturbed approximate inverse (cascade-break's L-factor
/// perturbation, static pivot perturbation), so refinement converges
/// *linearly* and keeps improving for as long as it is allowed to run.
///
/// The public API takes the matrix and the factor separately, so that
/// situation is directly constructible: factor `A + ΔA` and refine
/// against `A`. With a 1 % diagonal perturbation the residual falls by
/// ~3.3× per step and never reaches `ε·√n`, so the run uses all 10
/// corrections with no stagnant step.
///
/// Returns `(A, factor-of-perturbed-A, rhs)`.
fn full_budget_case() -> (CscMatrix, SparseFactors, Vec<f64>) {
    let a = tridiag(N, |_| 2.0);
    let perturbed = tridiag(N, |j| 2.0 + 1e-2 * (1.0 + (j as f64) * 0.01));
    let factors = factorize(&perturbed);
    let rhs: Vec<f64> = (0..N).map(|i| ((i % 7) as f64) - 3.0).collect();
    (a, factors, rhs)
}

/// Same construction with a 100× smaller perturbation: refinement reaches
/// the `ε·√n` target and exits on its own, well inside the budget. This
/// is the "cap is not a target" case.
fn early_exit_case() -> (CscMatrix, SparseFactors, Vec<f64>) {
    let a = tridiag(N, |_| 2.0);
    let perturbed = tridiag(N, |j| 2.0 + 1e-4 * (1.0 + (j as f64) * 0.01));
    let factors = factorize(&perturbed);
    let rhs: Vec<f64> = (0..N).map(|i| ((i % 7) as f64) - 3.0).collect();
    (a, factors, rhs)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
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

/// Guard for the tests below: the default run really does use the whole
/// budget on this case. If this ever stops holding, the cap tests that
/// depend on it become vacuous, so it is asserted separately and loudly.
#[test]
fn default_run_uses_the_full_budget_on_a_perturbed_factor() {
    let (a, f, rhs) = full_budget_case();
    let (_, diag) = solve_sparse_refined_with_diagnostics(&a, &f, &rhs).expect("refine");
    assert_eq!(
        diag.steps.len(),
        DEFAULT_REFINE_MAX_STEPS + 1,
        "expected 1 initial + {} corrections, got {} steps: {:?}",
        DEFAULT_REFINE_MAX_STEPS,
        diag.steps.len(),
        diag.steps
            .iter()
            .map(|s| s.relative_residual)
            .collect::<Vec<_>>()
    );
    // Every step improved — no plateau exit was in play, so the run
    // stopped on the cap and nothing else.
    assert!(
        diag.steps.iter().all(|s| s.improved),
        "a step failed to improve; the run did not stop on the cap alone"
    );
}

/// #178 verification 1 — `k` correction steps means `k + 1` entries.
#[test]
fn cap_truncates_the_trajectory_to_exactly_k_corrections() {
    let (a, f, rhs) = full_budget_case();
    for k in 0..=DEFAULT_REFINE_MAX_STEPS {
        let (_, diag) = solve_sparse_refined_with_diagnostics_opts(
            &a,
            &f,
            &rhs,
            RefineOptions::with_max_steps(k),
        )
        .expect("refine");
        assert_eq!(
            diag.steps.len(),
            k + 1,
            "k = {k} should give 1 initial + {k} corrections, got {}",
            diag.steps.len()
        );
    }
}

/// #178 verification 1, the specific case in the issue text: a solve that
/// runs 11 steps by default returns exactly 2 under `k = 1` — one
/// correction, not zero and not two.
#[test]
fn cap_of_one_gives_exactly_one_correction() {
    let (a, f, rhs) = full_budget_case();
    let (_, diag) =
        solve_sparse_refined_with_diagnostics_opts(&a, &f, &rhs, RefineOptions::with_max_steps(1))
            .expect("refine");
    assert_eq!(diag.steps.len(), 2);
    assert_eq!(diag.steps[1].step, 1);
    assert!(
        diag.steps[1].improved,
        "the one correction should have improved the residual"
    );
    // And it is genuinely a correction: the answer differs from unrefined.
    let unrefined = solve_sparse(&f, &rhs).expect("solve_sparse");
    let capped =
        solve_sparse_refined_opts(&a, &f, &rhs, RefineOptions::with_max_steps(1)).expect("refine");
    assert_ne!(bits(&capped), bits(&unrefined));
}

/// #178 verification 2 — the cap is an upper bound, not a target. A
/// system that converges early must return the same trajectory and the
/// same iterate no matter how large the cap is.
#[test]
fn cap_is_a_cap_not_a_target() {
    let (a, f, rhs) = early_exit_case();
    let (x_default, d_default) =
        solve_sparse_refined_with_diagnostics(&a, &f, &rhs).expect("refine");
    assert!(
        d_default.steps.len() < DEFAULT_REFINE_MAX_STEPS + 1,
        "early-exit case unexpectedly used the whole budget ({} steps)",
        d_default.steps.len()
    );

    for k in [DEFAULT_REFINE_MAX_STEPS, 25, 100] {
        let (x, d) = solve_sparse_refined_with_diagnostics_opts(
            &a,
            &f,
            &rhs,
            RefineOptions::with_max_steps(k),
        )
        .expect("refine");
        assert_eq!(
            d.steps.len(),
            d_default.steps.len(),
            "raising the cap to {k} changed the step count"
        );
        assert_eq!(
            bits(&x),
            bits(&x_default),
            "raising the cap to {k} changed x"
        );
    }
}

/// #178 verification 3 — `k = 0` is the unrefined solve, bit-for-bit.
#[test]
fn cap_zero_equals_solve_sparse_bit_for_bit() {
    for (a, f, rhs) in [full_budget_case(), early_exit_case()] {
        let plain = solve_sparse(&f, &rhs).expect("solve_sparse");
        let capped = solve_sparse_refined_opts(&a, &f, &rhs, RefineOptions::with_max_steps(0))
            .expect("refine");
        assert_eq!(bits(&capped), bits(&plain));
    }
}

/// #178 verification 4 — under any cap the returned iterate is the
/// best-residual one, so no cap can return an answer worse than the
/// unrefined solve.
#[test]
fn capped_result_is_never_worse_than_unrefined() {
    let (a, f, rhs) = full_budget_case();
    let unrefined_res = rel_residual(&a, &solve_sparse(&f, &rhs).expect("solve_sparse"), &rhs);

    for k in 0..=DEFAULT_REFINE_MAX_STEPS {
        let opts = RefineOptions::with_max_steps(k);
        let (x, diag) =
            solve_sparse_refined_with_diagnostics_opts(&a, &f, &rhs, opts).expect("refine");
        let res = rel_residual(&a, &x, &rhs);
        assert!(
            res <= unrefined_res,
            "k = {k} returned residual {res:.3e}, worse than unrefined {unrefined_res:.3e}"
        );
        // The returned iterate is the one `returned_step` names, and that
        // step holds the smallest residual in the trajectory.
        let best = diag
            .steps
            .iter()
            .map(|s| s.residual_2norm)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(diag.steps[diag.returned_step].residual_2norm, best);
        // Non-allocating and allocating entry points agree.
        let x2 = solve_sparse_refined_opts(&a, &f, &rhs, opts).expect("refine");
        assert_eq!(bits(&x), bits(&x2));
    }
}

/// The default-constructed options reproduce the existing entry points
/// bit-for-bit — the guarantee that makes this a non-breaking change.
#[test]
fn default_options_are_bit_identical_to_the_uncapped_entry_points() {
    for (a, f, rhs) in [full_budget_case(), early_exit_case()] {
        let old = solve_sparse_refined(&a, &f, &rhs).expect("refine");
        let new =
            solve_sparse_refined_opts(&a, &f, &rhs, RefineOptions::default()).expect("refine");
        assert_eq!(bits(&old), bits(&new));

        let nrhs = 3;
        let mut wide = Vec::with_capacity(N * nrhs);
        for c in 0..nrhs {
            for b in &rhs {
                wide.push(b * ((c + 1) as f64));
            }
        }
        let old_m = solve_sparse_many_refined(&a, &f, &wide, nrhs).expect("many refine");
        let new_m = solve_sparse_many_refined_opts(&a, &f, &wide, nrhs, RefineOptions::default())
            .expect("many refine");
        assert_eq!(bits(&old_m), bits(&new_m));
    }
}

// ---------------------------------------------------------------------
// #178 verification 5 — the same four properties on the multi-RHS path,
// where each column's step count is independent.
// ---------------------------------------------------------------------

/// Two columns whose refinement trajectories differ: column 0 is the
/// full-budget case, column 1 (scaled by 1e6) converges no faster but
/// carries a different `||b||`, and the third column is the zero RHS
/// (exact in one solve). Built against the perturbed factor.
fn many_case(nrhs: usize) -> (CscMatrix, SparseFactors, Vec<f64>) {
    let (a, f, rhs) = full_budget_case();
    let mut wide = vec![0.0; N * nrhs];
    for c in 0..nrhs {
        if c == nrhs - 1 {
            continue; // last column stays the zero RHS
        }
        let scale = 10f64.powi(c as i32 * 3);
        for i in 0..N {
            wide[c * N + i] = rhs[i] * scale;
        }
    }
    (a, f, wide)
}

/// `k = 0` on the multi-RHS refiner equals `solve_sparse_many`
/// bit-for-bit, for both the per-column and the batched panel widths.
#[test]
fn many_cap_zero_equals_solve_sparse_many_bit_for_bit() {
    for nrhs in [2usize, 20] {
        let (a, f, wide) = many_case(nrhs);
        let plain = solve_sparse_many(&f, &wide, nrhs).expect("solve_sparse_many");
        let capped =
            solve_sparse_many_refined_opts(&a, &f, &wide, nrhs, RefineOptions::with_max_steps(0))
                .expect("many refine");
        assert_eq!(bits(&capped), bits(&plain), "nrhs = {nrhs}");
    }
}

/// Each column of the capped multi-RHS refiner equals the capped
/// single-RHS refiner on that column, bit-for-bit — i.e. the cap applies
/// per column, exactly as the uncapped budget does.
#[test]
fn many_cap_matches_the_single_rhs_refiner_per_column() {
    let nrhs = 3;
    let (a, f, wide) = many_case(nrhs);
    for k in [0usize, 1, 2, 5, DEFAULT_REFINE_MAX_STEPS] {
        let opts = RefineOptions::with_max_steps(k);
        let many = solve_sparse_many_refined_opts(&a, &f, &wide, nrhs, opts).expect("many refine");
        for c in 0..nrhs {
            let col = &wide[c * N..(c + 1) * N];
            let single = solve_sparse_refined_opts(&a, &f, col, opts).expect("refine");
            assert_eq!(
                bits(&many[c * N..(c + 1) * N]),
                bits(&single),
                "k = {k}, column {c}"
            );
        }
    }
}

/// The multi-RHS cap is a cap, not a target: on the early-exit case,
/// raising it past the point of convergence changes nothing.
#[test]
fn many_cap_is_a_cap_not_a_target() {
    let nrhs = 4;
    let (a, f, rhs) = early_exit_case();
    let mut wide = vec![0.0; N * nrhs];
    for c in 0..nrhs - 1 {
        for i in 0..N {
            wide[c * N + i] = rhs[i] * ((c + 1) as f64);
        }
    }
    let base = solve_sparse_many_refined(&a, &f, &wide, nrhs).expect("many refine");
    for k in [DEFAULT_REFINE_MAX_STEPS, 25, 100] {
        let x =
            solve_sparse_many_refined_opts(&a, &f, &wide, nrhs, RefineOptions::with_max_steps(k))
                .expect("many refine");
        assert_eq!(
            bits(&x),
            bits(&base),
            "raising the multi-RHS cap to {k} changed X"
        );
    }
}

/// No cap can make a column worse than its unrefined solve.
#[test]
fn many_capped_result_is_never_worse_than_unrefined() {
    let nrhs = 3;
    let (a, f, wide) = many_case(nrhs);
    let plain = solve_sparse_many(&f, &wide, nrhs).expect("solve_sparse_many");
    for k in 0..=DEFAULT_REFINE_MAX_STEPS {
        let x =
            solve_sparse_many_refined_opts(&a, &f, &wide, nrhs, RefineOptions::with_max_steps(k))
                .expect("many refine");
        for c in 0..nrhs {
            let col = &wide[c * N..(c + 1) * N];
            let r_plain = rel_residual(&a, &plain[c * N..(c + 1) * N], col);
            let r_capped = rel_residual(&a, &x[c * N..(c + 1) * N], col);
            assert!(
                r_capped <= r_plain,
                "k = {k}, column {c}: capped {r_capped:.3e} worse than unrefined {r_plain:.3e}"
            );
        }
    }
}

/// `RefineOptions::default()` is today's budget. Guards against a silent
/// default change — issue #178 explicitly does not ask for one.
#[test]
fn the_default_budget_is_unchanged() {
    assert_eq!(DEFAULT_REFINE_MAX_STEPS, 10);
    assert_eq!(RefineOptions::default().max_steps, DEFAULT_REFINE_MAX_STEPS);
}
