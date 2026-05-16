//! Regression test for https://github.com/jkitchin/feral/issues/17.
//!
//! IPM solver loops (ipopt-feral / pounce-feral) on the Mittelmann
//! `robot_1600` problem hit an inertia disagreement at iteration 3
//! when feral's `NumericParams::default()` had cascade-break armed
//! (`ratio=0.5, eps=1e-10`). 585d739 flipped the Rust-side default
//! to `None`; da23d13 carried the same default into the C API used
//! by ipopt-feral via `feral_new()`.
//!
//! This test locks both:
//!   1. `NumericParams::default()` has cascade-break disarmed.
//!   2. `robot_1600_0003` factors with default `Solver` settings
//!      and the reported inertia matches the MUMPS 5.8.2 reference
//!      oracle `(positive=14399, negative=9601, zero=0)`.
//!
//! Reference: data/matrices/kkt-mittelmann/robot_1600/robot_1600_0003.mumps.json
//!
//! The corpus is gitignored, so the test skips gracefully on CI.

use std::path::Path;

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus, Solver};
use feral::{read_mtx, Inertia};

#[test]
fn default_numeric_params_have_cascade_break_disarmed() {
    let p = NumericParams::default();
    assert!(
        p.cascade_break_ratio.is_none(),
        "cascade_break_ratio must default to None for ipopt-feral parity \
         (see issue #17, commit 585d739)"
    );
    assert!(
        p.cascade_break_eps.is_none(),
        "cascade_break_eps must default to None for ipopt-feral parity \
         (see issue #17, commit 585d739)"
    );
}

#[test]
fn robot_1600_iter_3_matches_mumps_inertia_with_defaults() {
    let path = Path::new("data/matrices/kkt-mittelmann/robot_1600/robot_1600_0003.mtx");
    if !path.exists() {
        eprintln!(
            "SKIP: {} not present (corpus is gitignored, not available in CI)",
            path.display()
        );
        return;
    }
    let mtx = read_mtx(path).expect("read robot_1600_0003.mtx");
    let csc = mtx.to_csc().expect("to_csc");
    assert_eq!(csc.n, 24000, "robot_1600 KKT must be n=24000");

    // MUMPS 5.8.2 reference from
    // data/matrices/kkt-mittelmann/robot_1600/robot_1600_0003.mumps.json
    let expected = Inertia {
        positive: 14399,
        negative: 9601,
        zero: 0,
    };

    let mut solver = Solver::new();
    let status = solver.factor(&csc, Some(expected.clone()));
    match status {
        FactorStatus::Success => {
            let got = solver.inertia().expect("inertia recorded on Success");
            assert_eq!(
                (got.positive, got.negative, got.zero),
                (expected.positive, expected.negative, expected.zero),
                "robot_1600_0003 inertia must match MUMPS reference"
            );
        }
        FactorStatus::WrongInertia {
            actual,
            expected: exp,
        } => {
            panic!(
                "robot_1600_0003 inertia disagreement (issue #17 regression): \
                 got ({}, {}, {}), expected ({}, {}, {})",
                actual.positive, actual.negative, actual.zero, exp.positive, exp.negative, exp.zero,
            );
        }
        FactorStatus::Singular => panic!("robot_1600_0003 reported Singular under defaults"),
        FactorStatus::FatalError(e) => panic!("robot_1600_0003 fatal error: {e:?}"),
    }
}
