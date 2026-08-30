//! Issue #190 — a caller-settable *target* for iterative refinement, and
//! a report of what the refinement actually did.
//!
//! #178 made the step budget per-call. #190's finding is that the budget
//! is the wrong knob: on a large near-singular KKT the hardwired `ε·√n`
//! target is unreachable, so the loop runs to the cap on every call and
//! the cap becomes the de-facto stopping rule — one that was not monotone
//! in outcome across a 118-problem corpus.
//!
//! Two asks, tested here:
//!   1. `RefineOptions::stop` — `EpsSqrtN` (unchanged default),
//!      `RelativeResidual(t)`, `BackwardError(t)`.
//!   2. The `*_into` entry points return a `RefineOutcome` saying how
//!      many corrections ran, how good the returned iterate is, and
//!      **which exit fired** — the last being what back-solve wall time
//!      cannot tell a host.
//!
//! Design note: `dev/research/issue-190-refine-target.md`.
//!
//! **Where the target constants come from.** They are not round numbers
//! picked to look tidy — a target below what the fixture can reach tests
//! the cap, not the criterion. Measured ladder for `full_budget_case()`,
//! single RHS, one row per correction step:
//!
//! ```text
//!   k:      1        2        3        4        5
//!   rel: 2.37e-3  7.74e-4  2.55e-4  8.38e-5  2.76e-5
//!   ome: 1.21e-3  4.85e-4  1.66e-4  5.53e-5  1.82e-5
//!   k:      6        7        8        9       10
//!   rel: 9.07e-6  2.99e-6  9.82e-7  3.23e-7  1.06e-7
//!   ome: 6.01e-6  1.98e-6  6.51e-7  2.14e-7  7.05e-8
//! ```
//!
//! So `1e-4` costs 4 steps, `1e-6` costs 8, and anything below ~`1e-7`
//! is unreachable and runs to the cap. Tests that want a *reachable*
//! target use the first two; `the_cap_still_wins_over_an_unreachable_target`
//! deliberately uses the third case.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams, SparseFactors};
use feral::numeric::solve::{
    solve_sparse_many_refined_into, solve_sparse_refined, solve_sparse_refined_into, RefineOptions,
    RefineStop, StopCriterion, DEFAULT_REFINE_MAX_STEPS,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, FactorStatus, Solver, ZeroPivotAction};

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    })
}

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

/// Same construction as `tests/issue178_refine_cap.rs`: factor `A + ΔA`
/// and refine against `A`, so the correction is a genuinely approximate
/// inverse and refinement converges *linearly* instead of bottoming out
/// on roundoff. With a 1 % diagonal perturbation the residual falls ~3.3×
/// per step and never reaches `ε·√n`, so the default run uses all ten
/// corrections. That is exactly the situation #190 describes.
fn full_budget_case() -> (CscMatrix, SparseFactors, Vec<f64>) {
    let a = tridiag(N, |_| 2.0);
    let perturbed = tridiag(N, |j| 2.0 + 1e-2 * (1.0 + (j as f64) * 0.01));
    let factors = factorize(&perturbed);
    let rhs: Vec<f64> = (0..N).map(|i| ((i % 7) as f64) - 3.0).collect();
    (a, factors, rhs)
}

/// 100× smaller perturbation: refinement reaches `ε·√n` on its own, well
/// inside the budget.
fn early_exit_case() -> (CscMatrix, SparseFactors, Vec<f64>) {
    let a = tridiag(N, |_| 2.0);
    let perturbed = tridiag(N, |j| 2.0 + 1e-4 * (1.0 + (j as f64) * 0.01));
    let factors = factorize(&perturbed);
    let rhs: Vec<f64> = (0..N).map(|i| ((i % 7) as f64) - 3.0).collect();
    (a, factors, rhs)
}

fn refine(
    a: &CscMatrix,
    f: &SparseFactors,
    rhs: &[f64],
    opts: RefineOptions,
) -> (Vec<f64>, feral::RefineOutcome) {
    let mut x = vec![0.0; a.n];
    let o = solve_sparse_refined_into(a, f, rhs, &mut x, opts).expect("refine");
    (x, o)
}

// --- guard: the fixtures still do what the tests assume ---------------

#[test]
fn the_full_budget_case_really_does_run_to_the_cap() {
    // If this stops holding, every "stops earlier than the default" test
    // below becomes vacuous, so it is asserted separately and loudly.
    let (a, f, rhs) = full_budget_case();
    let (_, o) = refine(&a, &f, &rhs, RefineOptions::default());
    assert_eq!(o.steps, DEFAULT_REFINE_MAX_STEPS, "outcome: {o:?}");
    assert_eq!(o.stop, RefineStop::MaxSteps, "outcome: {o:?}");
}

#[test]
fn the_early_exit_case_really_does_converge_on_its_own() {
    let (a, f, rhs) = early_exit_case();
    let (_, o) = refine(&a, &f, &rhs, RefineOptions::default());
    assert_eq!(o.stop, RefineStop::Converged, "outcome: {o:?}");
    assert!(o.steps < DEFAULT_REFINE_MAX_STEPS, "outcome: {o:?}");
}

// --- ask 1: the target is settable -----------------------------------

#[test]
fn the_default_criterion_is_bit_for_bit_the_historical_behavior() {
    // The no-regression gate. `EpsSqrtN` keeps the strict `<` the old
    // code had, so a caller that does not opt in runs the same
    // arithmetic — not merely a close answer.
    for (a, f, rhs) in [full_budget_case(), early_exit_case()] {
        let before = solve_sparse_refined(&a, &f, &rhs).expect("refine");
        let (after, _) = refine(&a, &f, &rhs, RefineOptions::default());
        let lhs: Vec<u64> = before.iter().map(|v| v.to_bits()).collect();
        let rhs_bits: Vec<u64> = after.iter().map(|v| v.to_bits()).collect();
        assert_eq!(lhs, rhs_bits, "default options must not change any bit");
    }
}

#[test]
fn a_loose_relative_target_stops_where_the_default_runs_to_the_cap() {
    let (a, f, rhs) = full_budget_case();
    let (_, o) = refine(&a, &f, &rhs, RefineOptions::with_target(1e-6));
    assert_eq!(o.stop, RefineStop::Converged, "outcome: {o:?}");
    assert!(
        o.steps < DEFAULT_REFINE_MAX_STEPS,
        "a reachable target must cost fewer steps than the cap: {o:?}"
    );
    assert!(
        o.relative_residual <= 1e-6,
        "reported Converged, so the target must actually be met: {o:?}"
    );
}

#[test]
fn a_tighter_relative_target_costs_more_steps_than_a_looser_one() {
    // Monotonicity in the *target* is the property the step cap did not
    // have (#190's corpus table). Assert it directly.
    //
    // Targets come from the fixture's measured ladder (see the module
    // comment): it contracts ~3.0x per step from 2.4e-3, so 1e-4 lands at
    // 4 corrections and 1e-6 at 8. Both are above the 10-step floor of
    // 1.06e-7, so both are genuinely reachable and this test is about
    // monotonicity, not about the cap.
    let (a, f, rhs) = full_budget_case();
    let (_, loose) = refine(&a, &f, &rhs, RefineOptions::with_target(1e-4));
    let (_, tight) = refine(&a, &f, &rhs, RefineOptions::with_target(1e-6));
    assert_eq!(loose.stop, RefineStop::Converged, "loose: {loose:?}");
    assert_eq!(tight.stop, RefineStop::Converged, "tight: {tight:?}");
    assert!(
        loose.steps < tight.steps,
        "loose {loose:?} should be cheaper than tight {tight:?}"
    );
    assert!(loose.relative_residual <= 1e-4);
    assert!(tight.relative_residual <= 1e-6);
}

#[test]
fn a_backward_error_target_stops_early_too() {
    let (a, f, rhs) = full_budget_case();
    // 1e-6 is reachable here; the fixture's omega floor at the cap is
    // 7.05e-8, so anything tighter than ~1e-7 would test the cap instead.
    let (_, o) = refine(&a, &f, &rhs, RefineOptions::with_backward_error(1e-6));
    assert_eq!(o.stop, RefineStop::Converged, "outcome: {o:?}");
    assert!(o.steps < DEFAULT_REFINE_MAX_STEPS, "outcome: {o:?}");
}

#[test]
fn a_tighter_backward_error_target_costs_more_steps() {
    let (a, f, rhs) = full_budget_case();
    let (_, loose) = refine(&a, &f, &rhs, RefineOptions::with_backward_error(1e-4));
    let (_, tight) = refine(&a, &f, &rhs, RefineOptions::with_backward_error(1e-6));
    assert_eq!(loose.stop, RefineStop::Converged, "loose: {loose:?}");
    assert_eq!(tight.stop, RefineStop::Converged, "tight: {tight:?}");
    assert!(
        loose.steps < tight.steps,
        "loose {loose:?} should be cheaper than tight {tight:?}"
    );
}

#[test]
fn the_cap_still_wins_over_an_unreachable_target() {
    // #178's invariant, re-asserted under the new criteria: a target
    // never *extends* the run past the budget.
    let (a, f, rhs) = full_budget_case();
    let opts = RefineOptions::with_target(0.0).and_max_steps(3);
    let (_, o) = refine(&a, &f, &rhs, opts);
    assert_eq!(o.steps, 3, "outcome: {o:?}");
    assert_eq!(o.stop, RefineStop::MaxSteps, "outcome: {o:?}");
}

#[test]
fn a_met_target_never_adds_work_when_the_cap_is_raised() {
    // The other half of #178's invariant: raising `max_steps` cannot add
    // work to a run that has already reached its target. Checked under
    // all three criteria, since each has its own exit test.
    let (a, f, rhs) = early_exit_case();
    for stop in [
        StopCriterion::EpsSqrtN,
        StopCriterion::RelativeResidual(1e-6),
        StopCriterion::BackwardError(1e-6),
    ] {
        let ten = refine(&a, &f, &rhs, RefineOptions::default().and_stop(stop)).1;
        let hundred = refine(
            &a,
            &f,
            &rhs,
            RefineOptions::default().and_stop(stop).and_max_steps(100),
        )
        .1;
        assert_eq!(ten.steps, hundred.steps, "{stop:?}: {ten:?} vs {hundred:?}");
        assert_eq!(ten.stop, RefineStop::Converged, "{stop:?}: {ten:?}");
    }
}

#[test]
fn builders_compose_in_either_order() {
    let a = RefineOptions::with_target(1e-7).and_max_steps(4);
    let b = RefineOptions::with_max_steps(4).and_stop(StopCriterion::RelativeResidual(1e-7));
    assert_eq!(a, b);
    assert_eq!(a.max_steps, 4);
    assert_eq!(a.stop, StopCriterion::RelativeResidual(1e-7));
}

// --- ask 2: the outcome is surfaced ----------------------------------

#[test]
fn max_steps_zero_reports_max_steps_and_a_nan_residual() {
    // #178 requires `k = 0` to cost exactly what an unrefined solve
    // costs, so no residual is formed and there is nothing honest to
    // report. NaN says "not computed" rather than inventing a 0.0 that
    // would read as a perfect solve.
    let (a, f, rhs) = full_budget_case();
    let (_, o) = refine(&a, &f, &rhs, RefineOptions::with_max_steps(0));
    assert_eq!(o.steps, 0, "outcome: {o:?}");
    assert_eq!(o.stop, RefineStop::MaxSteps, "outcome: {o:?}");
    assert!(o.relative_residual.is_nan(), "outcome: {o:?}");
}

#[test]
fn a_stagnating_run_is_reported_as_stagnated_not_as_the_cap() {
    // Refining an *exact* factor bottoms out on roundoff in two or three
    // steps. Distinguishing that from "the budget bound" is the whole
    // point of `RefineStop` — back-solve wall time cannot.
    let a = tridiag(N, |_| 2.0);
    let f = factorize(&a);
    let rhs: Vec<f64> = (0..N).map(|i| ((i % 7) as f64) - 3.0).collect();
    let (_, o) = refine(&a, &f, &rhs, RefineOptions::with_target(0.0));
    assert_eq!(
        o.stop,
        RefineStop::Stagnated,
        "an exactly-factored system under an unreachable target must \
         bottom out, not exhaust the budget: {o:?}"
    );
    assert!(o.steps < DEFAULT_REFINE_MAX_STEPS, "outcome: {o:?}");
}

#[test]
fn the_solver_entry_point_surfaces_the_same_outcome() {
    // #190's actual ask: a host calls `Solver`, not the free functions.
    let a = tridiag(N, |_| 2.0);
    let perturbed = tridiag(N, |j| 2.0 + 1e-2 * (1.0 + (j as f64) * 0.01));
    let mut s = Solver::new();
    let st = s.factor(&perturbed, None);
    assert!(
        matches!(st, FactorStatus::Success),
        "factor must succeed for this test to mean anything, got {st:?}"
    );
    let rhs: Vec<f64> = (0..N).map(|i| ((i % 7) as f64) - 3.0).collect();
    let mut x = vec![0.0; N];
    let o = s
        .solve_refined_into(&a, &rhs, &mut x, RefineOptions::default())
        .expect("solve_refined_into");
    assert_eq!(o.steps, DEFAULT_REFINE_MAX_STEPS, "outcome: {o:?}");
    assert_eq!(o.stop, RefineStop::MaxSteps, "outcome: {o:?}");
}

// --- multi-RHS -------------------------------------------------------

/// Three columns whose per-column difficulty differs, so the aggregation
/// has something to aggregate.
fn many_rhs(n: usize, nrhs: usize) -> Vec<f64> {
    let mut b = vec![0.0; n * nrhs];
    for c in 0..nrhs {
        for i in 0..n {
            b[c * n + i] = ((i % 7) as f64) - 3.0 + (c as f64) * 0.5;
        }
    }
    b
}

#[test]
fn the_multi_rhs_outcome_is_the_max_over_columns() {
    for &nrhs in &[3usize, 20] {
        let (a, f, _) = full_budget_case();
        let b = many_rhs(N, nrhs);
        let mut x = vec![0.0; N * nrhs];
        let opts = RefineOptions::with_target(1e-6);
        let agg =
            solve_sparse_many_refined_into(&a, &f, &b, nrhs, &mut x, opts).expect("many refined");

        let mut want_steps = 0usize;
        let mut want_rel = 0.0f64;
        for c in 0..nrhs {
            let (_, o) = refine(&a, &f, &b[c * N..(c + 1) * N], opts);
            want_steps = want_steps.max(o.steps);
            want_rel = want_rel.max(o.relative_residual);
        }
        assert_eq!(agg.steps, want_steps, "nrhs = {nrhs}: {agg:?}");
        assert!(
            (agg.relative_residual - want_rel).abs() <= 1e-15 * want_rel.max(1e-300),
            "nrhs = {nrhs}: aggregated {:e} vs per-column max {want_rel:e}",
            agg.relative_residual
        );
    }
}

#[test]
fn the_multi_rhs_path_honors_each_criterion_per_column() {
    let (a, f, _) = full_budget_case();
    let nrhs = 20;
    let b = many_rhs(N, nrhs);
    // The relative target has to be looser than the single-RHS one: the
    // worst column's 10-step floor is 4.46e-6, 42x the best column's
    // 1.06e-7, because ||b|| varies across columns. omega does not spread
    // that way (2.86e-8 to 7.05e-8 across the same 20 columns), which is
    // the scale-awareness the criterion exists for.
    for (opts, name) in [
        (RefineOptions::with_target(1e-4), "relative"),
        (RefineOptions::with_backward_error(1e-6), "backward"),
    ] {
        let mut x = vec![0.0; N * nrhs];
        let o =
            solve_sparse_many_refined_into(&a, &f, &b, nrhs, &mut x, opts).expect("many refined");
        assert_eq!(o.stop, RefineStop::Converged, "{name}: {o:?}");
        assert!(o.steps < DEFAULT_REFINE_MAX_STEPS, "{name}: {o:?}");

        // Every column must independently satisfy the criterion, not
        // merely the aggregate.
        for c in 0..nrhs {
            let (single, so) = refine(&a, &f, &b[c * N..(c + 1) * N], opts);
            let batched = &x[c * N..(c + 1) * N];
            assert_eq!(so.stop, RefineStop::Converged, "{name} col {c}: {so:?}");
            let bits_a: Vec<u64> = single.iter().map(|v| v.to_bits()).collect();
            let bits_b: Vec<u64> = batched.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                bits_a, bits_b,
                "{name} col {c}: batched and per-column refiners must agree bit-for-bit"
            );
        }
    }
}

// --- ask 2b: the outcome reports ω, not just the normwise residual ----
//
// The downstream question on PR #191 was whether `RefineOutcome` carries
// the achieved backward error as well as the step count. It does, and
// these tests pin what the number means. The point of the field is that
// 0.18.0 makes ω *the thing being certified* by the default criterion —
// a host that gets `Converged` back and wants to log or threshold on the
// certificate should not have to recompute it with a second `abs_symv`.

/// ω computed straight from the Arioli-Demmel-Duff definition
/// `ω = maxᵢ |rᵢ| / (|A|·|x| + |b|)ᵢ`, with its own residual and its own
/// traversal of the stored lower triangle. Deliberately does **not** call
/// `CscMatrix::abs_symv` or anything in `numeric::solve`, so it is an
/// independent oracle for what the refiner reports.
///
/// The production version carries LAPACK `dgerfs`'s `safe1`/`safe2`
/// guard for denominators near underflow. These fixtures are the
/// `tridiag(20, 2.0)` family with an O(1) right-hand side, so every
/// denominator is O(1) — some twenty orders above the guard's cutoff —
/// and the guarded and unguarded forms agree to the last bit.
fn omega_by_definition(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0f64; n];
    let mut den = vec![0.0f64; n];
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let v = a.values[k];
            ax[i] += v * x[j];
            den[i] += v.abs() * x[j].abs();
            if i != j {
                ax[j] += v * x[i];
                den[j] += v.abs() * x[i].abs();
            }
        }
    }
    let mut om = 0.0f64;
    for i in 0..n {
        let d = den[i] + b[i].abs();
        assert!(d > 1e-8, "fixture denominator {d:e} is near the guard");
        let t = (b[i] - ax[i]).abs() / d;
        if t > om {
            om = t;
        }
    }
    om
}

#[test]
fn the_reported_omega_is_the_omega_of_the_returned_iterate() {
    let (a, f, rhs) = full_budget_case();
    // Every criterion that actually forms ω. `full_budget_case` cannot
    // reach `√ε`, so these run to the cap and the returned iterate is
    // the best one seen -- exactly the case where a host wants to know
    // how close refinement got.
    let criteria = [
        StopCriterion::BackwardError(1e-4),
        StopCriterion::BackwardError(f64::EPSILON.sqrt()),
        StopCriterion::EpsSqrtNAndBackwardError(f64::EPSILON.sqrt()),
    ];
    for stop in criteria {
        let (x, o) = refine(
            &a,
            &f,
            &rhs,
            RefineOptions {
                stop,
                ..RefineOptions::default()
            },
        );
        let expect = omega_by_definition(&a, &x, &rhs);
        assert!(
            o.backward_error.is_finite(),
            "{stop:?}: reported ω must be a measurement, got {:e}",
            o.backward_error
        );
        let rel = (o.backward_error - expect).abs() / expect;
        assert!(
            rel < 1e-12,
            "{stop:?}: reported ω {:e} != recomputed {expect:e} (rel {rel:e})",
            o.backward_error
        );
    }
}

#[test]
fn omega_is_reported_as_not_measured_under_the_normwise_criteria() {
    // `NaN` means "not formed", and is deliberately distinct from
    // `INFINITY`, which the refiner reports for a non-finite iterate.
    // Leaking the internal `INFINITY` sentinel here would read as a
    // catastrophically bad solve on a run that converged fine.
    let (a, f, rhs) = early_exit_case();
    for stop in [
        StopCriterion::EpsSqrtN,
        StopCriterion::RelativeResidual(1e-4),
    ] {
        let (_, o) = refine(
            &a,
            &f,
            &rhs,
            RefineOptions {
                stop,
                ..RefineOptions::default()
            },
        );
        assert_eq!(o.stop, RefineStop::Converged, "{stop:?}: {o:?}");
        assert!(
            o.backward_error.is_nan(),
            "{stop:?}: ω was never formed, must report NaN, got {:e}",
            o.backward_error
        );
    }
}

#[test]
fn omega_is_reported_as_not_measured_when_the_budget_is_zero() {
    // Issue #178 requires `max_steps = 0` to cost exactly an unrefined
    // solve, so no residual and no ω are formed -- same reasoning as
    // `relative_residual` being NaN on this path.
    let (a, f, rhs) = early_exit_case();
    let (_, o) = refine(
        &a,
        &f,
        &rhs,
        RefineOptions {
            max_steps: 0,
            ..RefineOptions::default()
        },
    );
    assert!(o.relative_residual.is_nan(), "{o:?}");
    assert!(o.backward_error.is_nan(), "{o:?}");
}

#[test]
fn the_multi_rhs_outcome_reports_the_worst_column_omega() {
    // Two aggregations, written independently and both checked here:
    // the batched refiner's own fold over `best_om`, and the fold in
    // `Solver::solve_many_refined_into`, which for `nrhs` below
    // BLAS3_REFINE_THRESHOLD loops per column and combines the
    // single-RHS outcomes itself. #190 requires the two dispatch paths
    // report identically, so `nrhs = 2` (the IPM width, narrow path) and
    // `nrhs = 40` (batched path) are both run through the `Solver` as
    // well as through the free function.
    let (a, f, _rhs1) = full_budget_case();
    let perturbed = tridiag(N, |j| 2.0 + 1e-2 * (1.0 + (j as f64) * 0.01));
    let mut solver = Solver::new();
    let st = solver.factor(&perturbed, None);
    assert!(matches!(st, FactorStatus::Success), "factor: {st:?}");
    let opts = RefineOptions {
        stop: StopCriterion::BackwardError(1e-4),
        ..RefineOptions::default()
    };
    for nrhs in [2usize, 40] {
        // Each column rotates the base RHS pattern and rescales by an
        // O(1) factor, so the columns have genuinely different ω and the
        // max fold is exercised. Deliberately *not* a 10^-c ladder: that
        // pushes the small columns' denominators toward the `dgerfs`
        // underflow guard, where the oracle's unguarded formula stops
        // agreeing with the production one for reasons that have nothing
        // to do with what this test is asserting.
        let mut rhs = vec![0.0; N * nrhs];
        for c in 0..nrhs {
            let s = 1.0 + 0.25 * (c as f64);
            for i in 0..N {
                rhs[c * N + i] = (((i + 2 * c) % 7) as f64 - 3.0) * s;
            }
        }
        let mut x = vec![0.0; N * nrhs];
        let o = solve_sparse_many_refined_into(&a, &f, &rhs, nrhs, &mut x, opts).expect("many");
        let worst = (0..nrhs)
            .map(|c| omega_by_definition(&a, &x[c * N..(c + 1) * N], &rhs[c * N..(c + 1) * N]))
            .fold(0.0f64, f64::max);
        assert!(
            o.backward_error.is_finite(),
            "nrhs={nrhs}: {:e}",
            o.backward_error
        );
        let rel = (o.backward_error - worst).abs() / worst;
        assert!(
            rel < 1e-12,
            "nrhs={nrhs}: reported ω {:e} != worst column {worst:e} (rel {rel:e})",
            o.backward_error
        );

        // Same problem through the `Solver`, which at nrhs = 2 takes the
        // per-column loop and its separately written fold.
        let mut xs = vec![0.0; N * nrhs];
        let os = solver
            .solve_many_refined_into(&a, &rhs, nrhs, &mut xs, opts)
            .expect("solver many");
        let rel = (os.backward_error - worst).abs() / worst;
        assert!(
            rel < 1e-12,
            "nrhs={nrhs}: Solver reported ω {:e} != worst column {worst:e} (rel {rel:e})",
            os.backward_error
        );
    }
}

#[test]
fn the_solver_multi_rhs_path_also_reports_omega_as_not_measured() {
    // The narrow path's fold is written separately from the batched
    // one, so its "not measured" case needs its own check: a max fold
    // over NaN silently yields 0.0, which would read as a *perfect*
    // backward error on a run that never formed ω at all.
    let perturbed = tridiag(N, |j| 2.0 + 1e-4 * (1.0 + (j as f64) * 0.01));
    let a = tridiag(N, |_| 2.0);
    let mut solver = Solver::new();
    let st = solver.factor(&perturbed, None);
    assert!(matches!(st, FactorStatus::Success), "factor: {st:?}");
    for nrhs in [2usize, 40] {
        let rhs = many_rhs(N, nrhs);
        let mut x = vec![0.0; N * nrhs];
        let o = solver
            .solve_many_refined_into(
                &a,
                &rhs,
                nrhs,
                &mut x,
                RefineOptions {
                    stop: StopCriterion::EpsSqrtN,
                    ..RefineOptions::default()
                },
            )
            .expect("solver many");
        assert!(
            o.backward_error.is_nan(),
            "nrhs={nrhs}: ω never formed, must be NaN, got {:e}",
            o.backward_error
        );
    }
}
