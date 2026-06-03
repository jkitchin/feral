//! Issue #65 regression: the default `Auto` scaling must not report a
//! wrong, rank-deficient inertia (spurious zero pivots) on an
//! ill-conditioned but effectively full-rank indefinite KKT. The
//! inertia-guided MC64 fallback re-runs the misfactored case with
//! `Mc64Symmetric` and recovers the true inertia.
//!
//! Fixtures (`tests/data/large/{sawpath,twirism1}_kkt.mtx`) are generated
//! IPM KKTs, not SuiteSparse downloads, so they are gitignored and
//! regenerated on demand via `dev/scripts/regen_issue65_kkts.sh`. When a
//! fixture is absent (e.g. CI), the test prints SKIP and passes — it is a
//! local/opt-in regression guard, not a CI gate.
//!
//! External oracle (the issue): dense `numpy.linalg.eigvalsh` and
//! Ipopt MA27/MA57 both give sawpath `(789,786,0)` and twirism1 iter-0
//! `(.,313,0)`.

use feral::scaling::ScalingStrategy;
use feral::{read_mtx, Inertia, Solver};
use std::path::Path;

fn load(name: &str) -> Option<feral::CscMatrix> {
    let path = format!("tests/data/large/{name}.mtx");
    let p = Path::new(&path);
    if !p.is_file() {
        eprintln!("SKIP: {path} not present. Regenerate with dev/scripts/regen_issue65_kkts.sh.");
        return None;
    }
    Some(read_mtx(p).and_then(|m| m.to_csc()).expect("read fixture"))
}

#[test]
fn sawpath_auto_recovers_full_rank_inertia_via_mc64_fallback() {
    let Some(m) = load("sawpath_kkt") else {
        return;
    };
    assert_eq!(m.n, 1575, "unexpected sawpath KKT dimension");

    // Default Auto: the picker routes this to InfNorm (diag_only=0), which
    // force-accepts ~116 zero pivots and reports (789,670,116). The
    // fallback must rescue it to the true (789,786,0).
    let mut s = Solver::new();
    let status = s.factor(&m, None);
    let inertia = s.inertia().cloned().expect("inertia");

    assert_eq!(
        inertia,
        Inertia::new(789, 786, 0),
        "issue #65: Auto must recover the true inertia (789,786,0); got {inertia:?} \
         (status {status:?})",
    );
    assert_eq!(
        s.mc64_scaling_fallback_count(),
        1,
        "the MC64 scaling fallback must have fired exactly once on sawpath",
    );
    assert!(
        s.min_pivot_magnitude().unwrap_or(0.0) > 0.0,
        "after the MC64 fallback the smallest pivot must be > 0 (was 0 under InfNorm)",
    );
}

#[test]
fn twirism1_iter0_auto_stays_infnorm_no_spurious_fallback() {
    let Some(m) = load("twirism1_kkt") else {
        return;
    };
    assert_eq!(m.n, 745, "unexpected twirism1 KKT dimension");

    // twirism1 iter-0 is the discriminating negative case: InfNorm gives
    // the CORRECT (432,313,0) with zero=0, so the fallback must NOT fire
    // (MC64 would give the WRONG (433,311,1) here). This guards against
    // the fallback regressing matrices the picker already handles.
    let mut s = Solver::new();
    let _ = s.factor(&m, None);
    let inertia = s.inertia().cloned().expect("inertia");

    assert_eq!(
        inertia,
        Inertia::new(432, 313, 0),
        "twirism1 iter-0 must keep InfNorm's correct inertia; got {inertia:?}",
    );
    assert_eq!(
        s.mc64_scaling_fallback_count(),
        0,
        "the MC64 fallback must NOT fire when Auto already gives zero=0",
    );
}

#[test]
fn explicit_infnorm_is_respected_no_fallback() {
    // The fallback is gated on `Auto`. An explicit InfNorm request must be
    // honored as-is even when it force-accepts zeros (sawpath), so callers
    // who deliberately pin a strategy get exactly what they asked for.
    let Some(m) = load("sawpath_kkt") else {
        return;
    };
    let mut s = Solver::new().with_scaling(ScalingStrategy::InfNorm);
    let _ = s.factor(&m, None);
    let inertia = s.inertia().cloned().expect("inertia");
    assert_eq!(
        inertia,
        Inertia::new(789, 670, 116),
        "explicit InfNorm must be respected (no Auto fallback); got {inertia:?}",
    );
    assert_eq!(s.mc64_scaling_fallback_count(), 0);
}
