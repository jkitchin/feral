//! Issue #102 follow-up regression: `OrderingPreprocess::Auto` must escalate to
//! `LdltCompress` when the fill-preferred `None` ordering yields a numerically
//! catastrophic factor.
//!
//! Background: PR #92's Auto verify drops `LdltCompress` on symbolic fill. On
//! cont5_2_4_l's late IPM KKTs (μ→0, ~all diagonals near-singular), `None`
//! becomes 1×1 pivots of magnitude ~1e-16 with pivot growth ~4e32 — a garbage
//! factor that iterative refinement can't recover (refined resid ~1e-2),
//! breaking IPM convergence. `LdltCompress`'s MC64 matching pairs those into
//! stable 2×2 pivots (min-pivot ~2.7e-8, growth ~1e15, refined resid ~1e-16).
//! Fill can't see this; pivot growth can, so `factor()` escalates.
//!
//! Fixture `tests/data/large/cont5_late_kkt.mtx` is a late-iteration cont5 KKT
//! dumped from POUNCE (gitignored). Absent (e.g. CI) → SKIP.

use feral::symbolic::{OrderingPreprocess, SupernodeParams};
use feral::{read_mtx, NumericParams, Solver};
use std::path::Path;

#[test]
fn cont5_late_kkt_escalates_ordering_to_stable_pivots() {
    let path = Path::new("tests/data/large/cont5_late_kkt.mtx");
    if !path.is_file() {
        eprintln!("SKIP: {} not present.", path.display());
        return;
    }
    let csc = read_mtx(path)
        .and_then(|m| m.to_csc())
        .expect("read cont5 late KKT");

    // Default Auto: must escalate to LdltCompress on the catastrophic growth.
    let mut auto = Solver::new();
    assert!(matches!(
        auto.factor(&csc, None),
        feral::FactorStatus::Success
    ));
    let min_piv = auto.min_pivot_magnitude().expect("min pivot");
    let max_piv = auto.max_pivot_magnitude().expect("max pivot");
    let growth = max_piv / min_piv;
    assert!(
        growth < 1e24,
        "Auto did not escalate: pivot growth {growth:.1e} (min {min_piv:.2e}) — the \
         numerically-unstable None ordering was kept"
    );

    // Explicit None is respected (not escalated) → catastrophic growth remains.
    let mut none = Solver::with_params(
        NumericParams::default(),
        SupernodeParams {
            preprocess: OrderingPreprocess::None,
            ..SupernodeParams::default()
        },
    );
    assert!(matches!(
        none.factor(&csc, None),
        feral::FactorStatus::Success
    ));
    let none_growth = none.max_pivot_magnitude().unwrap() / none.min_pivot_magnitude().unwrap();
    assert!(
        none_growth > 1e24,
        "explicit None unexpectedly stable ({none_growth:.1e}); fixture may have drifted"
    );

    // Disabling escalation makes Auto behave like None (catastrophic).
    let mut auto_off = Solver::new().with_ordering_escalation(None);
    let _ = auto_off.factor(&csc, None);
    let off_growth =
        auto_off.max_pivot_magnitude().unwrap() / auto_off.min_pivot_magnitude().unwrap();
    assert!(
        off_growth > 1e24,
        "escalation-off Auto should match None ({off_growth:.1e})"
    );
}
