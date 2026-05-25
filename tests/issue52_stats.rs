//! Tests for issue #52: opt-in instrumentation accessors.
//!
//! Phase A targets `Solver::last_factor_stats()` and the
//! `FactorStats` snapshot type.
//!
//! Phase B targets the opt-in profiler accessors
//! `Solver::with_profiling`, `Solver::profile_report`, and
//! `Solver::symbolic_profile_report`. The plan
//! (`dev/plans/issue-52-opt-in-stats.md`) calls out that default-off
//! must stay byte-identical to current behavior; B1 below pins that
//! contract on the `Option<...>` return surface.
//!
//! Test catalogue:
//! - A1: `last_factor_stats_returns_none_before_factor`
//! - A2: `last_factor_stats_after_success_populates_all_fields`
//! - A3: `pattern_reused_false_first_factor_true_second`
//! - A4: `pattern_reused_false_after_pattern_change`
//! - B1: `profile_report_none_when_profiling_disabled`
//! - B2: `profile_report_some_when_profiling_enabled`
//! - B3: `symbolic_profile_report_only_on_cache_miss_factor`

use feral::scaling::ScalingStrategy;
use feral::{CscMatrix, FactorStats, FactorStatus, ProfileReport, Solver, SymbolicProfileReport};

/// Build a 2×2 SPD diagonal matrix `diag(2, 2)` (lower-triangle CSC).
fn diag2_spd() -> CscMatrix {
    CscMatrix::from_triplets(2, &[0, 1], &[0, 1], &[2.0, 2.0]).expect("from_triplets")
}

/// Build a different-pattern 3×3 SPD matrix to force a fingerprint
/// mismatch in A4. Lower triangle: diag(3, 3, 3) plus a (2,1) entry
/// so `nnz != n` and the pattern is unmistakably distinct from A2's
/// 2×2 pure-diagonal pattern.
fn tri3_spd_with_off_diag() -> CscMatrix {
    // a_00 = 3, a_11 = 3, a_22 = 3, a_21 = 1.
    CscMatrix::from_triplets(3, &[0, 1, 2, 2], &[0, 1, 2, 1], &[3.0, 3.0, 3.0, 1.0])
        .expect("from_triplets")
}

/// Build an n×n SPD tridiagonal matrix: `4` on the diagonal, `-1` on
/// the first sub/super-diagonal. Lower-triangle CSC. Diagonally
/// dominant ⇒ strictly positive definite, factorises cleanly with
/// no delays. Used by Phase B tests that need enough structure for
/// the profiler to record at least one supernode timing.
fn tridiagonal_spd(n: usize) -> CscMatrix {
    let mut rows = Vec::with_capacity(2 * n);
    let mut cols = Vec::with_capacity(2 * n);
    let mut vals = Vec::with_capacity(2 * n);
    for j in 0..n {
        rows.push(j);
        cols.push(j);
        vals.push(4.0);
        if j + 1 < n {
            rows.push(j + 1);
            cols.push(j);
            vals.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("from_triplets")
}

/// A1 — no factor has run, so `last_factor_stats()` must report
/// `None`. Guards the `Option` contract before any state exists.
#[test]
fn a1_last_factor_stats_returns_none_before_factor() {
    let solver = Solver::new();
    let got: Option<FactorStats> = solver.last_factor_stats();
    assert!(
        got.is_none(),
        "expected None before first factor, got {:?}",
        got
    );
}

/// A2 — every `FactorStats` field is populated after one successful
/// `factor()` call. Field values are cross-checked against the
/// already-public per-field accessors so we are not re-asserting
/// numeric ground truth (that lives in dense_ldlt.rs etc.) — only
/// that the snapshot faithfully mirrors the per-field surface.
#[test]
fn a2_last_factor_stats_after_success_populates_all_fields() {
    let csc = diag2_spd();
    let mut solver = Solver::new().with_scaling(ScalingStrategy::Identity);

    let status = solver.factor(&csc, None);
    assert!(
        matches!(status, FactorStatus::Success),
        "factor failed: {:?}",
        status
    );

    let stats = solver
        .last_factor_stats()
        .expect("last_factor_stats() should be Some after success");

    // nnz_a is exactly the CscMatrix nnz (lower triangle stored).
    assert_eq!(stats.nnz_a, csc.nnz(), "nnz_a mirrors CscMatrix::nnz()");

    // nnz_l mirrors SparseFactors::factor_nnz().
    let factors = solver.factors().expect("factors stashed");
    assert_eq!(
        stats.nnz_l,
        factors.factor_nnz(),
        "nnz_l mirrors SparseFactors::factor_nnz()"
    );

    // fill_ratio is definitionally nnz_l / nnz_a.
    let expected_fill = stats.nnz_l as f64 / stats.nnz_a as f64;
    assert!(
        (stats.fill_ratio - expected_fill).abs() < 1e-15,
        "fill_ratio = {} expected {}",
        stats.fill_ratio,
        expected_fill
    );

    // Inertia mirrors the existing accessor. SPD ⇒ (2, 0, 0).
    let inertia = solver.inertia().expect("inertia stashed").clone();
    assert_eq!(stats.inertia, inertia, "inertia mirrors Solver::inertia()");

    // min/max_abs_pivot mirror the existing accessors.
    let min_pivot = solver
        .min_pivot_magnitude()
        .expect("min_pivot_magnitude stashed");
    let max_pivot = solver
        .max_pivot_magnitude()
        .expect("max_pivot_magnitude stashed");
    assert!(
        (stats.min_abs_pivot - min_pivot).abs() < 1e-15,
        "min_abs_pivot = {} expected {}",
        stats.min_abs_pivot,
        min_pivot
    );
    assert!(
        (stats.max_abs_pivot - max_pivot).abs() < 1e-15,
        "max_abs_pivot = {} expected {}",
        stats.max_abs_pivot,
        max_pivot
    );

    // pattern_reused is false on the very first factor of a Solver.
    assert!(
        !stats.pattern_reused,
        "first factor on a fresh Solver is never a cache hit"
    );

    // scaling_info mirrors Solver::scaling_info().
    let scaling = solver.scaling_info().expect("scaling_info stashed").clone();
    assert_eq!(
        stats.scaling_info, scaling,
        "scaling_info mirrors Solver::scaling_info()"
    );
}

/// A3 — bit-identical pattern replay must flip `pattern_reused` to
/// `true` on the second `factor()`. This is the symbolic-cache hit
/// signal pounce will key off when deciding whether the warm path
/// fired as expected.
#[test]
fn a3_pattern_reused_false_first_factor_true_second() {
    let csc = diag2_spd();
    let mut solver = Solver::new().with_scaling(ScalingStrategy::Identity);

    assert!(matches!(solver.factor(&csc, None), FactorStatus::Success));
    let s1 = solver.last_factor_stats().expect("stats after factor 1");
    assert!(!s1.pattern_reused, "factor 1 cannot be a cache hit");

    assert!(matches!(solver.factor(&csc, None), FactorStatus::Success));
    let s2 = solver.last_factor_stats().expect("stats after factor 2");
    assert!(
        s2.pattern_reused,
        "factor 2 on identical pattern must report cache hit"
    );

    // Sanity: symbolic_call_count agrees with pattern_reused.
    assert_eq!(
        solver.symbolic_call_count(),
        1,
        "symbolic should have run exactly once across two same-pattern factors"
    );
}

/// A4 — a structurally distinct matrix between two `factor()` calls
/// must report `pattern_reused = false` on the second call.
/// Complements A3 by exercising the cache-miss branch.
#[test]
fn a4_pattern_reused_false_after_pattern_change() {
    let small = diag2_spd();
    let bigger = tri3_spd_with_off_diag();

    let mut solver = Solver::new().with_scaling(ScalingStrategy::Identity);

    assert!(matches!(solver.factor(&small, None), FactorStatus::Success));
    let s1 = solver.last_factor_stats().expect("stats after factor 1");
    assert!(!s1.pattern_reused, "factor 1 is never a cache hit");

    assert!(matches!(
        solver.factor(&bigger, None),
        FactorStatus::Success
    ));
    let s2 = solver.last_factor_stats().expect("stats after factor 2");
    assert!(
        !s2.pattern_reused,
        "pattern change must invalidate the cache"
    );

    // The fingerprint mismatch should have re-run symbolic.
    assert_eq!(
        solver.symbolic_call_count(),
        2,
        "symbolic must rerun on pattern change"
    );
}

// ---------------------------------------------------------------------
// Phase B: opt-in profiler accessors
// ---------------------------------------------------------------------

/// B1 — default `Solver` (no `with_profiling`) must report `None`
/// from both profile-report accessors regardless of how many factor
/// calls have run. This is the contract that makes the "no
/// noticeable performance loss when not debugging" guarantee
/// observable from the API: if the caller hasn't opted in, the
/// profiler `Arc<Mutex<...>>` is never constructed and therefore
/// no report can be produced.
#[test]
fn b1_profile_report_none_when_profiling_disabled() {
    let csc = tridiagonal_spd(8);
    let mut solver = Solver::new().with_scaling(ScalingStrategy::Identity);

    assert!(matches!(solver.factor(&csc, None), FactorStatus::Success));

    let p: Option<ProfileReport> = solver.profile_report();
    let s: Option<SymbolicProfileReport> = solver.symbolic_profile_report();
    assert!(
        p.is_none(),
        "profile_report must be None when with_profiling not enabled"
    );
    assert!(
        s.is_none(),
        "symbolic_profile_report must be None when with_profiling not enabled"
    );
}

/// B2 — once `with_profiling(true)` is set, the numeric profile
/// report becomes available and records at least one supernode
/// timing. We assert structural invariants (n_supernodes > 0,
/// total_us > 0, no validation warnings) rather than exact numbers
/// to keep the test machine-independent.
#[test]
fn b2_profile_report_some_when_profiling_enabled() {
    // n=64 escapes both the tiny path (n ≤ 16) and the dense
    // fast-path density gate (tridiagonal nnz=127 vs cells=2080,
    // density 6% < 25%), so the multifrontal driver actually runs
    // and the per-supernode profiler records at least one timing.
    // The profiler is wired into
    // `factorize_multifrontal_supernodal_with_workspace` only;
    // matrices that route through the tiny / dense fast path will
    // produce a `Some(ProfileReport)` with `n_supernodes = 0`. That
    // is correct behavior — the per-supernode loop did not run.
    let csc = tridiagonal_spd(64);
    let mut solver = Solver::new()
        .with_scaling(ScalingStrategy::Identity)
        .with_profiling(true);

    assert!(matches!(solver.factor(&csc, None), FactorStatus::Success));

    let report = solver
        .profile_report()
        .expect("profile_report must be Some when with_profiling(true)");
    assert!(
        report.n_supernodes >= 1,
        "expected at least one supernode timing, got {}",
        report.n_supernodes
    );
    assert!(
        report.total_us > 0,
        "expected nonzero total wallclock, got {}",
        report.total_us
    );
    assert!(
        report.validation_warnings.is_empty(),
        "profile report flagged validation warnings: {:?}",
        report.validation_warnings
    );
}

/// B3 — symbolic profile is captured only on factor calls that
/// re-run the symbolic analysis (cache miss). On a cache hit the
/// symbolic phase is skipped entirely, and `symbolic_profile_report`
/// must reflect that by returning `None` (or by reporting the most
/// recent symbolic from the cache-miss factor — pinned here as
/// "None on cache hit" to keep the signal unambiguous for pounce).
#[test]
fn b3_symbolic_profile_report_only_on_cache_miss_factor() {
    let csc = tridiagonal_spd(8);
    let mut solver = Solver::new()
        .with_scaling(ScalingStrategy::Identity)
        .with_profiling(true);

    // Factor 1: cache miss — symbolic runs, report should be Some.
    assert!(matches!(solver.factor(&csc, None), FactorStatus::Success));
    let s1 = solver.symbolic_profile_report();
    assert!(
        s1.is_some(),
        "first factor must produce a symbolic profile report"
    );

    // Factor 2: cache hit — symbolic skipped, report should be None.
    assert!(matches!(solver.factor(&csc, None), FactorStatus::Success));
    let s2 = solver.symbolic_profile_report();
    assert!(
        s2.is_none(),
        "cache-hit factor must clear the symbolic profile report \
         (symbolic phase never ran), got Some(_)"
    );

    // Sanity: the numeric profile is always available on each factor.
    assert!(
        solver.profile_report().is_some(),
        "numeric profile_report should be Some on every factor when \
         with_profiling(true), independent of symbolic cache state"
    );
}
