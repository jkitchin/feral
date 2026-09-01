//! Regression test for https://github.com/jkitchin/feral/issues/18.
//!
//! Pounce-feral on Mittelmann `NARX_CFy.nl` hit a 110-iter MaxIter
//! stall with sustained WrongInertia status records — under the
//! pre-2026-05-16 pounce-feral default of
//! `Solver::new().with_cascade_break(0.5).with_cascade_break_eps(1e-10)`.
//!
//! KKT-dump diagnostic (feral journal 2026-05-16 21:30) showed
//! cascade-break perturbed borderline pivots into 100+ |D|<1e-10
//! values whose signs shifted reported inertia by +/-1 on the
//! mid-IPM solves (solve_001, solve_100, solve_400). With cb=off
//! (feral C-API default since da23d13, NumericParams default since
//! 585d739) the same matrices factored to the IPM-expected count
//! exactly. Pounce-feral was flipped to cb=off default on
//! 2026-05-16 to match.
//!
//! Corpus oracles (`NARX_CFy_*.json`) hold the IPM-side expected
//! inertia at the dumped iteration. This test locks: under default
//! `Solver` (cb=off), each corpus iter's factor reports inertia
//! matching the json oracle. Catches regressions where a future
//! default change re-introduces inertia drift on this problem.
//!
//! The corpus is gitignored, so the test skips gracefully on CI.

use std::fs;
use std::path::Path;

use feral::numeric::solver::{FactorStatus, Solver};
use feral::{read_mtx, CscMatrix, Inertia};

fn check_iter(iter: usize) {
    let mtx_path = format!("data/matrices/kkt-mittelmann/NARX_CFy/NARX_CFy_{iter:04}.mtx");
    let json_path = format!("data/matrices/kkt-mittelmann/NARX_CFy/NARX_CFy_{iter:04}.json");
    if !Path::new(&mtx_path).exists() {
        eprintln!("SKIP iter {iter}: {mtx_path} not present (corpus gitignored)");
        return;
    }

    let oracle_text = fs::read_to_string(&json_path).expect("read NARX_CFy oracle json");
    let (pos, neg) = parse_inertia(&oracle_text)
        .unwrap_or_else(|| panic!("oracle json missing inertia: {json_path}"));
    let expected = Inertia {
        positive: pos,
        negative: neg,
        zero: 0,
    };

    let mtx = read_mtx(Path::new(&mtx_path)).expect("read NARX_CFy mtx");
    let csc = mtx.to_csc().expect("to_csc");
    assert_eq!(csc.n, pos + neg, "iter {iter}: dim must equal pos+neg");

    let mut solver = Solver::new();
    let status = solver.factor(&csc, Some(expected.clone()));
    match status {
        FactorStatus::Success => {
            let got = solver.inertia().expect("inertia recorded on Success");
            assert_eq!(
                (got.positive, got.negative, got.zero),
                (expected.positive, expected.negative, expected.zero),
                "iter {iter}: inertia must match oracle"
            );
            // Phase B (issue #55): CB is now armed by default, but its
            // trigger is "delay budget exhausted at supernode", not the
            // legacy ratio heuristic. NARX_CFy's early-iter delay
            // catchment sits well within `delayed_capacity`, so CB
            // must not fire — `n_tiny == 0` mirrors MUMPS `INFO(25)`.
            let stats = solver
                .last_factor_stats()
                .expect("FactorStats present after Success");
            assert_eq!(
                stats.n_tiny, 0,
                "iter {iter}: Phase B budget-based CB trigger must not fire on \
                 NARX_CFy early-iter delay catchment (n_tiny={})",
                stats.n_tiny,
            );
        }
        FactorStatus::WrongInertia {
            actual,
            expected: exp,
        } => {
            panic!(
                "iter {iter}: inertia disagreement (issue #18 regression): \
                 got ({}, {}, {}), expected ({}, {}, {})",
                actual.positive, actual.negative, actual.zero, exp.positive, exp.negative, exp.zero,
            );
        }
        FactorStatus::Singular => panic!("iter {iter}: factor Singular"),
        FactorStatus::FatalError(e) => panic!("iter {iter}: factor FatalError: {e:?}"),
    }
}

fn parse_inertia(json: &str) -> Option<(usize, usize)> {
    let key = "\"inertia\":{";
    let i = json.find(key)? + key.len();
    let chunk = &json[i..i + 80];
    let pos = grab_int(chunk, "\"positive\":")?;
    let neg = grab_int(chunk, "\"negative\":")?;
    Some((pos, neg))
}

fn grab_int(s: &str, key: &str) -> Option<usize> {
    let i = s.find(key)? + key.len();
    let rest = &s[i..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

#[test]
fn narx_cfy_iter_0_matches_oracle_inertia() {
    check_iter(0);
}

#[test]
fn narx_cfy_iter_1_matches_oracle_inertia() {
    check_iter(1);
}

#[test]
fn narx_cfy_iter_2_matches_oracle_inertia() {
    check_iter(2);
}

// ---------------------------------------------------------------------
// Residual gate. Locks the second half of the issue #18 fix:
// `Solver::solve_many_refined` (which the C ABI invokes by default
// since commit 597a90a) must drive the relative residual below the
// IPM-relevant threshold on the captured NARX_CFy KKT snapshots.
//
// The original failure mode was the *unrefined* backsolve sitting at
// ~1e-5..1e-6 in late-iter IPM regimes, which exceeded the duality
// gap and stalled α in [0.05, 0.30]. Refined solve closes that gap.
//
// Threshold: 1e-10. MA57 routinely hits ≤ 1e-12 with one refinement
// step; 1e-10 is a generous gate that catches any regression where
// refinement silently becomes a no-op (e.g. if `solve_refined` were
// to revert to `solve` without us noticing).
//
// Test executes only if the corpus is present (gitignored). Uses the
// json's stored RHS — the same vector the IPM was solving against
// when this snapshot was captured.

const RESIDUAL_THRESHOLD: f64 = 1e-10;

fn matvec_lower_sym(csc: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for j in 0..csc.n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = csc.values[k];
            out[i] += v * x[j];
            if i != j {
                out[j] += v * x[i];
            }
        }
    }
}

fn rel_residual(csc: &CscMatrix, x: &[f64], rhs: &[f64]) -> f64 {
    let mut ax = vec![0.0; csc.n];
    matvec_lower_sym(csc, x, &mut ax);
    let mut num_sq = 0.0;
    let mut den_sq = 0.0;
    for i in 0..csc.n {
        let d = ax[i] - rhs[i];
        num_sq += d * d;
        den_sq += rhs[i] * rhs[i];
    }
    num_sq.sqrt() / den_sq.sqrt().max(1.0)
}

fn check_iter_residual(iter: usize) {
    let mtx_path = format!("data/matrices/kkt-mittelmann/NARX_CFy/NARX_CFy_{iter:04}.mtx");
    let json_path = format!("data/matrices/kkt-mittelmann/NARX_CFy/NARX_CFy_{iter:04}.json");
    if !Path::new(&mtx_path).exists() {
        eprintln!("SKIP iter {iter}: {mtx_path} not present (corpus gitignored)");
        return;
    }
    let oracle_text = fs::read_to_string(&json_path).expect("read NARX_CFy oracle json");
    let oracle: serde_json::Value =
        serde_json::from_str(&oracle_text).expect("parse NARX_CFy oracle json");
    let rhs: Vec<f64> = oracle["rhs"]
        .as_array()
        .expect("rhs must be array")
        .iter()
        .map(|v| v.as_f64().expect("rhs entry must be f64"))
        .collect();
    let mtx = read_mtx(Path::new(&mtx_path)).expect("read NARX_CFy mtx");
    let csc = mtx.to_csc().expect("to_csc");
    assert_eq!(rhs.len(), csc.n, "iter {iter}: rhs length must equal n");

    let mut solver = Solver::new();
    let expected_inertia = {
        let inertia = &oracle["inertia"];
        Inertia {
            positive: inertia["positive"].as_u64().expect("positive") as usize,
            negative: inertia["negative"].as_u64().expect("negative") as usize,
            zero: 0,
        }
    };
    let status = solver.factor(&csc, Some(expected_inertia));
    match status {
        FactorStatus::Success => {}
        other => panic!("iter {iter}: factor not Success: {other:?}"),
    }

    let x_refined = solver
        .solve_refined(&csc, &rhs)
        .expect("solve_refined must succeed when factor succeeded");
    let res = rel_residual(&csc, &x_refined, &rhs);
    assert!(
        res < RESIDUAL_THRESHOLD,
        "iter {iter}: refined solve residual {res:.3e} exceeds gate \
         {RESIDUAL_THRESHOLD:.0e} (issue #18 regression — refinement \
         path may have become a no-op)"
    );
}

#[test]
fn narx_cfy_iter_0_refined_residual_below_gate() {
    check_iter_residual(0);
}

#[test]
fn narx_cfy_iter_1_refined_residual_below_gate() {
    check_iter_residual(1);
}

#[test]
fn narx_cfy_iter_2_refined_residual_below_gate() {
    check_iter_residual(2);
}
