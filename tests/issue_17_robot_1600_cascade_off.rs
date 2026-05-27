//! Regression test for https://github.com/jkitchin/feral/issues/17.
//!
//! History (Phase A → Phase B, issue #55):
//!   - 585d739 disarmed CB by default (`cascade_break_ratio = None`)
//!     after issue #17 showed IPM disagreement when the legacy
//!     ratio-based CB trigger fired on robot_1600 pivots that
//!     delay should have absorbed.
//!   - 2026-05-27 (Phase B) re-arms CB by default but changes the
//!     trigger from "delayed fraction ≥ ratio" to "symbolic
//!     `delayed_capacity` exhausted". On budgeted supernodes CB
//!     only fires when delay is structurally impossible, matching
//!     MUMPS's `dfac_front_aux.F:1251-1331` invariant. The
//!     numeric ratio value (`Some(0.5)`) is now used only on
//!     unbudgeted legacy paths.
//!
//! This test locks the Phase B contract on robot_1600:
//!   1. `NumericParams::default()` is the Phase B configuration —
//!      CB armed with `ratio = Some(0.5)`, `eps = Some(1e-10)`,
//!      budget-based trigger.
//!   2. `robot_1600_0003` factors with default `Solver` settings
//!      and the reported inertia matches the MUMPS 5.8.2 reference
//!      oracle `(positive=14399, negative=9601, zero=0)`.
//!   3. CB does *not* fire on robot_1600's delay catchment under
//!      the budget-based trigger — `n_tiny == 0` (the original
//!      issue-#17 regression no longer reproduces because the
//!      budget gates CB to delay-exhausted supernodes, not to
//!      the pivots that delay could have absorbed).
//!
//! Reference: data/matrices/kkt-mittelmann/robot_1600/robot_1600_0003.mumps.json
//!
//! The corpus is gitignored, so the test skips gracefully on CI.

use std::path::Path;

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus, Solver};
use feral::{read_mtx, Inertia};

#[test]
fn default_numeric_params_have_phase_b_cb_armed() {
    let p = NumericParams::default();
    assert_eq!(
        p.cascade_break_ratio,
        Some(0.5),
        "Phase B (issue #55): cascade_break_ratio must default to Some(0.5) — \
         CB armed with budget-based trigger; legacy ratio value retained for \
         unbudgeted paths"
    );
    assert_eq!(
        p.cascade_break_eps,
        Some(1e-10),
        "Phase B (issue #55): cascade_break_eps must default to Some(1e-10) — \
         per-pivot static perturbation floor"
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
            // Phase B (issue #55): CB armed by default with budget-based
            // trigger. robot_1600's delay catchment fits within
            // `delayed_capacity`, so CB must not fire. `n_tiny == 0`
            // mirrors MUMPS `INFO(25)` / NBTINYW.
            let stats = solver
                .last_factor_stats()
                .expect("FactorStats present after Success");
            assert_eq!(
                stats.n_tiny, 0,
                "Phase B budget-based CB trigger must not fire on \
                 robot_1600_0003 (n_tiny={})",
                stats.n_tiny,
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
