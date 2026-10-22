//! Issue #227: `ReanalysisPolicy` behaviour on the `Solver`.
//!
//! Inertia oracle: the synthetic KKT path of `tests/common227` has the
//! Sylvester/Haynsworth inertia `(n_x, m, 0)` on every iterate.

#[path = "common227/mod.rs"]
mod common227;

use common227::{synthetic_inertia, synthetic_kkt, N_ITER};
use feral::{FactorStatus, OnStaleParams, ReanalysisPolicy, Solver};

#[test]
fn builder_round_trip() {
    assert_eq!(*Solver::new().reanalysis_policy(), ReanalysisPolicy::Never);
    let p = ReanalysisPolicy::OnStale(OnStaleParams {
        margin: 0.5,
        ..OnStaleParams::default()
    });
    let s = Solver::new().with_reanalysis(p.clone());
    assert_eq!(*s.reanalysis_policy(), p);
    let s = Solver::new().with_reanalysis(ReanalysisPolicy::Always);
    assert_eq!(*s.reanalysis_policy(), ReanalysisPolicy::Always);
}

#[test]
fn never_reports_nothing() {
    let mut s = Solver::new();
    for k in 0..N_ITER {
        let st = s.factor(&synthetic_kkt(k), Some(synthetic_inertia()));
        assert!(matches!(st, FactorStatus::Success), "{k}: {st:?}");
        let fs = s.last_factor_stats().expect("stats");
        assert!(!fs.reanalysed);
        assert_eq!(fs.reanalysis_count, 0);
        assert!(s.last_reanalysis().is_none());
    }
}
