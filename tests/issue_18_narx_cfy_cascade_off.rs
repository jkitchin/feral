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
use feral::{read_mtx, Inertia};

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
