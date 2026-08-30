//! The default refinement criterion must certify *componentwise* accuracy.
//!
//! Before 2026-08-21 `RefineOptions::default()` stopped on
//! `‖r‖₂ < ε·√n·‖b‖₂` alone. That test is normwise, so it is dominated by
//! the rows carrying the largest right-hand-side entries. On a RHS whose
//! entries span twelve orders of magnitude — the shape an interior-point
//! method produces near convergence, where the dual, primal and
//! complementarity blocks differ by many orders — it passes on the *raw*
//! solve and refinement never runs, while the rows carrying the small
//! entries are componentwise garbage.
//!
//! The oracle is canonical MUMPS 5.8.2 (`external_benchmarks/mumps_oracle`,
//! `ICNTL(10)=2`, `ICNTL(11)=1`), run on identical systems with identical
//! right-hand sides. It returns `ω` at machine precision where FERAL
//! returned as much as `9.5e-5`:
//!
//! | matrix | FERAL ω, old default | MUMPS ω1 |
//! |---|---:|---:|
//! | `r05_kkt` | 9.520e-5 | 3.608e-16 |
//! | `bratu3d` | 8.953e-6 | 2.844e-16 |
//! | `cont-201` | 3.860e-7 | 2.728e-16 |
//!
//! So `ω ≤ √ε` is attainable on these systems and FERAL was simply not
//! trying. Those three fixtures are gitignored (`tests/data/large/*`), so
//! this test runs the same experiment on the tracked parity corpus, where
//! the defect reproduces on 13 of 63 matrices.
//!
//! See `dev/research/issue-190-refine-target.md`, addendum 2026-08-21.

use feral::numeric::solve::{RefineOptions, StopCriterion};
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::{Path, PathBuf};

/// Arioli-Demmel-Duff componentwise backward error, LAPACK `dgerfs`'s
/// guarded form. `ω ≤ u` certifies that `x` solves `(A+δA)x = b+δb`
/// exactly with `|δA| ≤ ω|A|` and `|δb| ≤ ω|b|`, entry by entry.
fn omega(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0f64; n];
    a.symv(x, &mut ax);
    let r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
    let mut d = vec![0.0f64; n];
    a.abs_symv(x, &mut d);
    let safe1 = ((n + 1) as f64) * f64::MIN_POSITIVE;
    let safe2 = safe1 / f64::EPSILON;
    let mut om = 0.0f64;
    for i in 0..n {
        let den = d[i] + b[i].abs();
        if den == 0.0 {
            continue;
        }
        let v = if den > safe2 {
            r[i].abs() / den
        } else {
            (r[i].abs() + safe1) / (den + safe1)
        };
        if v > om {
            om = v;
        }
    }
    om
}

/// `b = A·v` with `v[i] = ±10^((i%13)-6)` — entries spanning `1e-6..1e6`.
fn badly_scaled_rhs(a: &CscMatrix) -> Vec<f64> {
    let n = a.n;
    let v: Vec<f64> = (0..n)
        .map(|i| {
            let mag = 10f64.powi(((i % 13) as i32) - 6);
            if i % 2 == 0 {
                mag
            } else {
                -mag
            }
        })
        .collect();
    let mut b = vec![0.0f64; n];
    a.symv(&v, &mut b);
    b
}

fn parity_matrices() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(fams) = std::fs::read_dir(Path::new("tests/data/parity")) else {
        return out;
    };
    for fam in fams.flatten() {
        let Ok(files) = std::fs::read_dir(fam.path()) else {
            continue;
        };
        for f in files.flatten() {
            let p = f.path();
            if p.extension().map(|e| e == "mtx").unwrap_or(false) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// `(ω under EpsSqrtN, ω under the shipped default)`.
fn omegas(path: &Path) -> Option<(f64, f64)> {
    let csc = read_mtx(path).ok()?.to_csc().ok()?;
    let n = csc.n;
    let b = badly_scaled_rhs(&csc);
    let mut s = Solver::new();
    if !matches!(
        s.factor(&csc, None),
        FactorStatus::Success | FactorStatus::WrongInertia { .. }
    ) {
        return None;
    }
    let mut xe = vec![0.0f64; n];
    s.solve_refined_into(
        &csc,
        &b,
        &mut xe,
        RefineOptions::default().and_stop(StopCriterion::EpsSqrtN),
    )
    .ok()?;
    let mut xd = vec![0.0f64; n];
    s.solve_refined_into(&csc, &b, &mut xd, RefineOptions::default())
        .ok()?;
    Some((omega(&csc, &xe, &b), omega(&csc, &xd, &b)))
}

/// Every matrix the corpus can factor must come back backward stable
/// under the shipped default, on a right-hand side the old normwise rule
/// declared converged at once.
#[test]
fn default_certifies_backward_stability_on_badly_scaled_rhs() {
    let target = f64::EPSILON.sqrt();
    let paths = parity_matrices();
    assert!(
        paths.len() >= 60,
        "parity corpus missing: found {} matrices",
        paths.len()
    );
    let mut bad: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for p in &paths {
        let Some((_, om_default)) = omegas(p) else {
            continue;
        };
        checked += 1;
        if om_default > target {
            bad.push(format!(
                "{}: omega = {:.3e} > sqrt(eps) = {:.3e}",
                p.display(),
                om_default,
                target
            ));
        }
    }
    assert!(checked >= 60, "only {checked} matrices factored");
    assert!(
        bad.is_empty(),
        "RefineOptions::default() left omega above sqrt(eps) on {} matrix/matrices:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// The defect this default exists to fix, pinned so it cannot silently
/// return: on these matrices the normwise rule stops at step 0 with a
/// componentwise error orders of magnitude above `√ε`, and the shipped
/// default drives it to machine precision.
///
/// Values are the `EpsSqrtN` omegas measured 2026-08-21; the assertions
/// are one-sided (`> 10·√ε`) so ordinary numerical drift cannot flip them.
#[test]
fn normwise_criterion_alone_is_not_enough() {
    let target = f64::EPSILON.sqrt();
    // (family, stem, omega under EpsSqrtN as measured)
    let cases = [
        ("degenlpb", "DEGENLPB_0046", 5.35e-6),
        ("degenlpb", "DEGENLPB_0045", 4.48e-6),
        ("degenlpb", "DEGENLPB_0047", 2.69e-6),
        ("degenlpa", "DEGENLPA_0065", 1.89e-6),
        ("acopp14", "ACOPP14_0003", 6.42e-7),
    ];
    let mut ran = 0usize;
    for (fam, stem, _measured) in cases {
        let p = PathBuf::from(format!("tests/data/parity/{fam}/{stem}.mtx"));
        if !p.exists() {
            continue;
        }
        ran += 1;
        let (om_eps, om_default) = omegas(&p).unwrap_or_else(|| panic!("{stem}: solve failed"));
        assert!(
            om_eps > 10.0 * target,
            "{stem}: EpsSqrtN omega = {om_eps:.3e} is no longer above 10*sqrt(eps) = {:.3e}; \
             the fixture no longer exercises the defect this default fixes",
            10.0 * target
        );
        assert!(
            om_default <= target,
            "{stem}: default omega = {om_default:.3e} > sqrt(eps) = {target:.3e} \
             (EpsSqrtN gave {om_eps:.3e})"
        );
    }
    // Without this the test passes vacuously if the `tests/data/parity`
    // layout moves: every fixture would `continue` and nothing would be
    // pinned. Its sibling guards the same way with `checked >= 60`.
    assert_eq!(
        ran,
        cases.len(),
        "only {ran} of {} pinned fixtures were found under tests/data/parity; \
         this test pins the #190 regression and must not pass vacuously",
        cases.len()
    );
}

/// The default is the conjunction, so it can only ever run *more* steps
/// than the historical rule -- never fewer. This is what lets it ship
/// without loosening any residual gate.
///
/// **This is an empirical corpus check, not an invariant the code
/// guarantees.** More steps does not imply a smaller `ω`: best-iterate
/// selection is still `min ‖r‖₂` (`solve.rs`, `if improved { best_omega
/// = omega; ... }`), so a step that lowers the residual is free to raise
/// the componentwise error. What the conjunction *does* guarantee is
/// that the returned iterate is never worse **normwise** -- the
/// `EpsSqrtN` half of the test still has to pass. The `ω` comparison
/// below is the stronger statement, and it holds on every matrix in the
/// tracked corpus today.
///
/// So a failure here is a finding to investigate, not necessarily a
/// regression: it would mean a real matrix on which the extra
/// componentwise steps land on a worse-`ω` iterate, which is the case
/// for revisiting best-iterate selection. Do not silence it by
/// weakening the comparison.
#[test]
fn default_is_never_weaker_than_the_historical_rule() {
    let paths = parity_matrices();
    let mut worse: Vec<String> = Vec::new();
    for p in &paths {
        let Some((om_eps, om_default)) = omegas(p) else {
            continue;
        };
        // Allow exact equality (the common case: the normwise rule was
        // already sufficient and no extra step ran).
        if om_default > om_eps {
            worse.push(format!(
                "{}: default {:.3e} > EpsSqrtN {:.3e}",
                p.display(),
                om_default,
                om_eps
            ));
        }
    }
    assert!(
        worse.is_empty(),
        "default produced a worse componentwise error than EpsSqrtN on:\n  {}",
        worse.join("\n  ")
    );
}
