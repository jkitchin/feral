//! Issue #194: `Solver::factor` must be cooperatively cancellable.
//!
//! A host enforcing a wall-clock budget cannot enforce it across a single
//! `factor` call: it never regains control to check its deadline. Measured in
//! pounce on `emfl050_5_5`, a `max_wall_time=5` solve returned `TIME_LIMIT`
//! after 48.8 s because one factorization ran ~44 s uninterrupted.
//!
//! The fix is a caller-owned `Arc<AtomicBool>` polled at supernode boundaries
//! and, within a supernode, at dense panel boundaries. These tests pin the
//! contract the reporter said they would rely on:
//!
//! * unarmed → byte-identical behaviour;
//! * armed and set → `FactorStatus::Interrupted`, no factor stored;
//! * clearing the flag → the next `factor` re-runs cleanly, same inertia;
//! * armed and clear → `Success` (the flag is polled, not merely present);
//! * both the sequential and parallel drivers honour it;
//! * a flag set from another thread mid-factorization stops it early.

use feral::{CscMatrix, FactorStatus, FeralError, Inertia, Solver};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Symmetric tridiagonal SPD matrix, lower triangle. Cheap to factor —
/// used for the contract tests, where the point is the control flow.
fn tridiag_spd(n: usize) -> CscMatrix {
    let mut rows = Vec::with_capacity(2 * n);
    let mut cols = Vec::with_capacity(2 * n);
    let mut vals = Vec::with_capacity(2 * n);
    for c in 0..n {
        rows.push(c);
        cols.push(c);
        vals.push(4.0);
        if c + 1 < n {
            rows.push(c + 1);
            cols.push(c);
            vals.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("tridiagonal triplets")
}

/// 5-point Laplacian on a `g x g` grid, lower triangle. `n = g^2`. Chosen
/// for the mid-flight test because its elimination tree is genuinely
/// two-dimensional: fill grows like `n log n` and the factor is dominated
/// by wide separator fronts, so it takes long enough to interrupt.
fn grid_laplacian(g: usize) -> CscMatrix {
    let n = g * g;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let at = |i: usize, j: usize| i * g + j;
    for i in 0..g {
        for j in 0..g {
            let c = at(i, j);
            rows.push(c);
            cols.push(c);
            vals.push(4.0);
            if i + 1 < g {
                rows.push(at(i + 1, j));
                cols.push(c);
                vals.push(-1.0);
            }
            if j + 1 < g {
                rows.push(at(i, j + 1));
                cols.push(c);
                vals.push(-1.0);
            }
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("grid laplacian triplets")
}

fn baseline_inertia(a: &CscMatrix, parallel: bool) -> Inertia {
    let mut s = Solver::new().with_parallel(parallel);
    let status = s.factor(a, None);
    assert!(
        matches!(status, FactorStatus::Success),
        "baseline factor must succeed, got {status:?}"
    );
    s.inertia().cloned().expect("baseline inertia")
}

#[test]
fn unarmed_solver_is_unaffected() {
    let a = tridiag_spd(500);
    for parallel in [false, true] {
        let mut s = Solver::new().with_parallel(parallel);
        assert!(
            s.interrupt().is_none(),
            "a fresh Solver must not be armed (parallel={parallel})"
        );
        let status = s.factor(&a, None);
        assert!(
            matches!(status, FactorStatus::Success),
            "unarmed factor must succeed (parallel={parallel}), got {status:?}"
        );
        assert_eq!(
            s.inertia().cloned(),
            Some(baseline_inertia(&a, parallel)),
            "unarmed factor must report the baseline inertia (parallel={parallel})"
        );
    }
}

#[test]
fn armed_but_clear_flag_factors_normally() {
    let a = tridiag_spd(500);
    for parallel in [false, true] {
        let flag = Arc::new(AtomicBool::new(false));
        let mut s = Solver::new()
            .with_parallel(parallel)
            .with_interrupt(Arc::clone(&flag));
        assert!(
            s.interrupt().is_some(),
            "with_interrupt must arm the Solver (parallel={parallel})"
        );
        let status = s.factor(&a, None);
        assert!(
            matches!(status, FactorStatus::Success),
            "a clear flag must not interrupt (parallel={parallel}), got {status:?}"
        );
        assert_eq!(
            s.inertia().cloned(),
            Some(baseline_inertia(&a, parallel)),
            "arming with a clear flag must not perturb the factor (parallel={parallel})"
        );
        assert!(
            !flag.load(Ordering::Relaxed),
            "feral must never write the caller's flag (parallel={parallel})"
        );
    }
}

#[test]
fn set_flag_interrupts_and_leaves_no_factor() {
    let a = tridiag_spd(500);
    for parallel in [false, true] {
        let flag = Arc::new(AtomicBool::new(true));
        let mut s = Solver::new()
            .with_parallel(parallel)
            .with_interrupt(Arc::clone(&flag));
        let status = s.factor(&a, None);
        assert!(
            matches!(status, FactorStatus::Interrupted),
            "a set flag must interrupt (parallel={parallel}), got {status:?}"
        );
        // Contract: factors are left invalid, exactly as after any failed
        // factor. The caller must not solve against them.
        assert!(
            s.inertia().is_none(),
            "interrupted factor must not leave an inertia (parallel={parallel})"
        );
        match s.solve(&vec![1.0; a.n]) {
            Err(FeralError::NoFactor) => {}
            other => panic!(
                "solve after an interrupted factor must fail with NoFactor \
                 (parallel={parallel}), got {:?}",
                other.map(|v| v.len())
            ),
        }
        assert!(
            flag.load(Ordering::Relaxed),
            "feral must not clear the caller's flag (parallel={parallel})"
        );
    }
}

#[test]
fn clearing_the_flag_lets_the_next_factor_run_clean() {
    let a = tridiag_spd(500);
    for parallel in [false, true] {
        let expected = baseline_inertia(&a, parallel);
        let flag = Arc::new(AtomicBool::new(true));
        let mut s = Solver::new()
            .with_parallel(parallel)
            .with_interrupt(Arc::clone(&flag));

        let status = s.factor(&a, None);
        assert!(
            matches!(status, FactorStatus::Interrupted),
            "setup: expected Interrupted (parallel={parallel}), got {status:?}"
        );

        // The half of the contract that a naive implementation breaks: an
        // interrupted factorization must not poison the reused workspace,
        // symbolic cache or permute cache.
        flag.store(false, Ordering::Relaxed);
        let status = s.factor(&a, None);
        assert!(
            matches!(status, FactorStatus::Success),
            "re-factor after clearing the flag must succeed (parallel={parallel}), \
             got {status:?}"
        );
        assert_eq!(
            s.inertia().cloned(),
            Some(expected),
            "re-factor must reproduce the un-interrupted inertia (parallel={parallel})"
        );

        // And a solve against the recovered factor must work.
        let rhs = vec![1.0; a.n];
        let x = s.solve(&rhs).expect("solve after clean re-factor");
        assert_eq!(x.len(), a.n);
        assert!(
            x.iter().all(|v| v.is_finite()),
            "re-factor solve produced non-finite values (parallel={parallel})"
        );
    }
}

#[test]
fn set_interrupt_arms_and_disarms_through_a_mut_borrow() {
    // The consumer in issue #194 holds the Solver behind `&mut self`
    // (`SparseSymLinearSolverInterface::set_interrupt`), where a consuming
    // builder cannot be called. `set_interrupt` is that path, and is also
    // the only way to disarm.
    let a = tridiag_spd(200);
    let flag = Arc::new(AtomicBool::new(true));
    let mut s = Solver::new().with_parallel(false);

    s.set_interrupt(Some(Arc::clone(&flag)));
    assert!(s.interrupt().is_some(), "set_interrupt(Some) must arm");
    assert!(
        matches!(s.factor(&a, None), FactorStatus::Interrupted),
        "armed + set flag must interrupt"
    );

    s.set_interrupt(None);
    assert!(s.interrupt().is_none(), "set_interrupt(None) must disarm");
    assert!(
        matches!(s.factor(&a, None), FactorStatus::Success),
        "a disarmed Solver must ignore a still-set flag"
    );
    assert!(
        flag.load(Ordering::Relaxed),
        "disarming must not touch the caller's flag"
    );
}

/// The reported scenario: the flag is clear when `factor` starts and is set
/// by a watchdog while the factorization is running. This is what proves the
/// poll fires *inside* the factorization rather than only at entry.
///
/// Shaped like the IPM case in the issue: the solver is warm (symbolic
/// analysis already cached from a prior factor on the same pattern), so what
/// is being timed and interrupted is the numeric phase alone — which is the
/// only phase the flag covers.
fn mid_factorization_interrupt(parallel: bool) {
    // 5-point Laplacian, n = 122_500. Big enough that the numeric factor is
    // comfortably longer than a thread wake-up even in `--release`.
    let a = grid_laplacian(350);
    let flag = Arc::new(AtomicBool::new(false));
    let mut s = Solver::new()
        .with_parallel(parallel)
        .with_interrupt(Arc::clone(&flag));

    // First factor: warms the symbolic + permute caches.
    assert!(
        matches!(s.factor(&a, None), FactorStatus::Success),
        "warm-up factor must succeed (parallel={parallel})"
    );
    let expected = s.inertia().cloned().expect("warm-up inertia");

    // Second factor, un-interrupted: this is the cost the watchdog below is
    // calibrated against, measured on *this* machine rather than assumed.
    let t0 = Instant::now();
    assert!(matches!(s.factor(&a, None), FactorStatus::Success));
    let baseline = t0.elapsed();
    eprintln!(
        "warm baseline factor (parallel={parallel}): {baseline:?} (n = {})",
        a.n
    );

    // If the baseline is short enough that a watchdog cannot reliably fire
    // inside it, this test cannot distinguish "interrupted mid-flight" from
    // "finished first". Report and pass rather than flake.
    if baseline < Duration::from_millis(100) {
        eprintln!(
            "SKIP: warm baseline {baseline:?} is too short to interrupt reliably; \
             the mid-flight assertion needs >= 100 ms."
        );
        return;
    }

    let delay = baseline / 10;
    let watchdog_flag = Arc::clone(&flag);
    let watchdog = std::thread::spawn(move || {
        std::thread::sleep(delay);
        watchdog_flag.store(true, Ordering::Relaxed);
    });

    let t0 = Instant::now();
    let status = s.factor(&a, None);
    let interrupted_in = t0.elapsed();
    watchdog.join().expect("watchdog thread panicked");

    eprintln!(
        "parallel={parallel}: watchdog fired at {delay:?}; factor returned \
         {status:?} after {interrupted_in:?}"
    );
    assert!(
        matches!(status, FactorStatus::Interrupted),
        "a flag set {delay:?} into a {baseline:?} factorization must interrupt it \
         (parallel={parallel}), got {status:?}"
    );
    // The whole point of the issue: the overshoot is bounded, not "the
    // factorization runs to completion and the caller finds out afterwards".
    assert!(
        interrupted_in < baseline,
        "interrupted factor took {interrupted_in:?}, which is not shorter than the \
         un-interrupted baseline {baseline:?} (parallel={parallel}) — the poll is \
         not bounding the overshoot"
    );

    // And the contract still holds after a mid-flight abort: no factor left
    // behind, and clearing the flag gives a clean re-factor with the same
    // inertia as before the interrupt.
    assert!(
        s.inertia().is_none(),
        "mid-flight interrupt must leave no inertia (parallel={parallel})"
    );
    flag.store(false, Ordering::Relaxed);
    assert!(
        matches!(s.factor(&a, None), FactorStatus::Success),
        "re-factor after a mid-flight interrupt must succeed (parallel={parallel})"
    );
    assert_eq!(
        s.inertia().cloned(),
        Some(expected),
        "re-factor after a mid-flight interrupt must reproduce the inertia \
         (parallel={parallel})"
    );
}

#[test]
fn sequential_driver_interrupts_mid_factorization() {
    mid_factorization_interrupt(false);
}

#[test]
fn parallel_driver_interrupts_mid_factorization() {
    mid_factorization_interrupt(true);
}

/// The free-function driver surface gets the same mechanism, since the flag
/// rides on the params struct rather than on `Solver`.
#[test]
fn free_function_driver_reports_interrupted_error() {
    use feral::dense::factor::BunchKaufmanParams;
    use feral::numeric::factorize::factorize_multifrontal;
    use feral::symbolic::{symbolic_factorize, SupernodeParams};
    use feral::NumericParams;

    let a = tridiag_spd(300);
    let sym = symbolic_factorize(&a, &SupernodeParams::default()).expect("symbolic");
    let flag = Arc::new(AtomicBool::new(true));
    let params = NumericParams {
        bk: BunchKaufmanParams {
            interrupt: Some(Arc::clone(&flag)),
            ..NumericParams::default().bk
        },
        ..NumericParams::default()
    };
    match factorize_multifrontal(&a, &sym, &params) {
        Err(FeralError::Interrupted) => {}
        Ok(_) => panic!("a set flag must abort the free-function driver"),
        Err(e) => panic!("expected FeralError::Interrupted, got {e:?}"),
    }
}
