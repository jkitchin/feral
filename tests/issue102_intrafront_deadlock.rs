//! Issue #102 regression: the parallel dense-front factorization must not
//! deadlock on the `dirichlet120` / `cont5_2_4_l` mittelmann KKTs.
//!
//! Root cause: the parallel multifrontal driver locked a per-thread workspace
//! `thread_ws[current_thread_index()]` and held it across `factor_one_supernode`
//! → `factor_frontal` → the intra-front `par_chunks_mut`. That nested rayon
//! let the blocked worker steal *another* `process_one_supernode` task onto the
//! **same** thread index, which re-locked the already-held mutex → self-deadlock
//! at 0 % CPU (converged → 300 s timeout in POUNCE). PR #92's ordering exposed
//! it by selecting a front that clears the intra-front area gate on these
//! problems; it is not Lever-B-specific (the old 256² floor deadlocks too).
//!
//! Fix: the driver `try_lock`s the per-thread workspace and falls back to a
//! throwaway workspace on `WouldBlock` (which uniquely means nested re-entry,
//! since each `thread_ws` slot is only ever locked by its own worker).
//!
//! The fixture `tests/data/large/dirichlet120_kkt.mtx` is a generated conic
//! iter-0 KKT (POUNCE on `dirichlet120.nl`), gitignored and regenerated via
//! `dev/scripts/regen_dirichlet120_kkt.sh`; absent (e.g. CI) → SKIP. The factor
//! runs on a spawned thread with a wall-clock guard so a *returning* deadlock
//! fails the test instead of hanging the suite.

use feral::{read_mtx, Solver};
use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

#[test]
fn dirichlet120_parallel_factor_does_not_deadlock() {
    let path = Path::new("tests/data/large/dirichlet120_kkt.mtx");
    if !path.is_file() {
        eprintln!(
            "SKIP: {} not present. Regenerate with dev/scripts/regen_dirichlet120_kkt.sh.",
            path.display()
        );
        return;
    }

    let csc = read_mtx(path)
        .and_then(|m| m.to_csc())
        .expect("read dirichlet120_kkt.mtx");
    assert_eq!(csc.n, 54363, "unexpected dirichlet120 KKT dimension");

    // Factor on a worker thread; guard with a generous wall-clock timeout. The
    // healthy factor is well under 1 s (10-core) / a few s single-threaded; the
    // pre-fix bug hung indefinitely at 0 % CPU. 120 s cleanly separates them.
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let mut solver = Solver::new(); // parallel driver (flops > PAR_MIN_FLOPS)
        let status = solver.factor(&csc, None);
        let inertia = solver.inertia().cloned();
        let _ = tx.send((format!("{status:?}"), inertia));
    });

    match rx.recv_timeout(Duration::from_secs(120)) {
        Ok((status, inertia)) => {
            handle.join().expect("factor thread panicked");
            assert_eq!(
                status, "Success",
                "dirichlet120 factor did not succeed: {status}"
            );
            // Quasi-definite KKT: no zero eigenvalues expected.
            let inertia = inertia.expect("inertia recorded");
            assert_eq!(inertia.zero, 0, "unexpected zero pivots: {inertia:?}");
            assert_eq!(
                inertia.positive + inertia.negative,
                54363,
                "inertia partition does not cover n: {inertia:?}"
            );
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!(
                "issue #102 regression: parallel dense-front factor of dirichlet120 \
                 did not finish in 120 s — the intra-front re-entrant deadlock is back"
            );
        }
        Err(e) => panic!("factor thread channel error: {e:?}"),
    }
}
