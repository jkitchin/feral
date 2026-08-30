//! Issue #194 x issue #65: an interrupt observed during the MC64 *retry*
//! factorization must be reported as `Interrupted`, and must not arm the
//! per-pattern non-adoption latch.
//!
//! `Solver::factor` can run *two* factorizations. When the first one
//! reports the singular signature (`inertia.zero > 0`) under a non-MC64
//! `Auto` scaling, the issue-#65 rescue re-factors the whole matrix with
//! `Mc64Symmetric` and adopts the result iff it strictly reduces the zero
//! count (`src/numeric/solver.rs`, the `mc64_fallback_adopted` block).
//!
//! The non-adoption arm was written as a bare `_`, so it also caught
//! `Err(_)` -- including the `Err(FeralError::Interrupted)` that issue
//! #194's polling now produces. Two things went wrong:
//!
//!   1. `factor` returned `Success`/`Singular` rather than `Interrupted`,
//!      contradicting the published contract that the first observation of
//!      the flag aborts the factorization.
//!   2. `mc64_retry_not_adopted` latched. That latch is keyed on the
//!      *pattern* and cleared only on a pattern change, so an
//!      interior-point host -- fixed pattern for a whole solve -- would
//!      have the issue-#65 rescue suppressed for every remaining iterate
//!      by a single unrelated cancellation, reporting unrescued inertia
//!      where it would otherwise have recovered. The field's own doc
//!      flags exactly that interaction with the inertia hard rule.
//!
//! This is not a narrow race: the vulnerable window is the entire duration
//! of the retry, which is a second *full* factorization of the same
//! matrix. It is precisely the long-factorization case issue #194 exists
//! for.
//!
//! RED (bare `_` arm): the interrupted call returns non-`Interrupted`, and
//! the follow-up call finds the retry suppressed (count stays 1).
//! GREEN (explicit `Err(Interrupted)` arm): the call returns `Interrupted`
//! and the follow-up call re-runs the retry (count reaches 2).

use feral::{CscMatrix, FactorStatus, Solver};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A 2D 5-point grid Laplacian on `k x k`, block-diagonal with a rank-1
/// `[[1,1],[1,1]]` tail.
///
/// The grid block supplies enough work that one factorization is
/// milliseconds rather than microseconds, which is what makes the retry
/// window wide enough to aim at. The 2x2 tail is the same genuinely
/// rank-deficient construction `tests/n4_mc64_retry_latch.rs` uses: pivot
/// on the leading 1, and the Schur complement is `1 - 1 = 0`, so BK
/// force-accepts exactly one zero pivot and `inertia.zero == 1` fires the
/// issue-#65 gate. MC64 cannot change rank, so the retry never adopts --
/// which is the non-adoption arm this test is about.
fn grid_plus_rank_deficient_tail(k: usize) -> CscMatrix {
    let g = k * k;
    let n = g + 2;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    // Lower triangle of the grid Laplacian.
    for j in 0..k {
        for i in 0..k {
            let idx = j * k + i;
            rows.push(idx);
            cols.push(idx);
            vals.push(4.0);
            if i + 1 < k {
                rows.push(idx + 1);
                cols.push(idx);
                vals.push(-1.0);
            }
            if j + 1 < k {
                rows.push(idx + k);
                cols.push(idx);
                vals.push(-1.0);
            }
        }
    }
    // Rank-deficient 2x2 tail: [[1,1],[1,1]].
    rows.push(g);
    cols.push(g);
    vals.push(1.0);
    rows.push(g + 1);
    cols.push(g);
    vals.push(1.0);
    rows.push(g + 1);
    cols.push(g + 1);
    vals.push(1.0);
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("grid triplets")
}

#[test]
fn an_interrupt_during_the_mc64_retry_is_reported_and_does_not_latch() {
    let a = grid_plus_rank_deficient_tail(90);

    // Calibrate on a fresh Solver, timing the *first* factor() -- symbolic
    // analysis plus factorization #1 plus the retry -- so the fractions
    // below are relative to the same shape of work the armed run does.
    // A fresh Solver is required: this call leaves `mc64_retry_not_adopted`
    // armed, which would suppress the retry on any reuse.
    let mut cal = Solver::new();
    let t0 = Instant::now();
    let cal_status = cal.factor(&a, None);
    let t_total = t0.elapsed();

    assert_eq!(
        cal.mc64_retry_attempt_count(),
        1,
        "the fixture must actually drive the issue-#65 retry, otherwise \
         this test exercises nothing; got status {cal_status:?}",
    );

    // Aim the watchdog inside the retry, and sweep the back half of the
    // call at fine resolution.
    //
    // The watchdog spins rather than sleeping. `thread::sleep` has ~1 ms
    // granularity and overshoots, which on a call this short is coarser
    // than the whole window -- the first version of this test used sleep
    // and never landed, failing its own vacuity guard.
    //
    // The "did this attempt land inside the retry?" test is deliberately
    // independent of the behaviour under test, so that it reports the same
    // answer before and after the fix. Two observables together pin it:
    //
    //   * `mc64_retry_attempt_count() == 1` proves factorization #1
    //     returned `Ok(..)` and the issue-#65 gate was entered -- had the
    //     flag been seen during factorization #1, that call would have
    //     returned `Err`, the gate (which keys on `Ok(..)`) would never
    //     have been entered, and the count would be 0. So the flag was set
    //     *after* factorization #1 finished.
    //   * `delay < call_elapsed` proves the watchdog fired before `factor`
    //     returned. So the flag was set *before* the call ended.
    //
    // Together: set after factorization #1 and before the call returned,
    // i.e. during the retry. Note this says nothing about what `factor`
    // returned, which is the thing being asserted -- an earlier version
    // keyed the detector on `status == Interrupted` and could therefore
    // never land pre-fix, reporting the bug as "never exercised".
    let mut landed = false;
    let mut attempts = 0usize;
    for step in 0..=60u32 {
        let frac = 0.40 + f64::from(step) * 0.01;
        attempts += 1;
        let flag = Arc::new(AtomicBool::new(false));
        let mut s = Solver::new().with_interrupt(Arc::clone(&flag));
        let delay = t_total.mul_f64(frac);

        let watchdog = {
            let flag = Arc::clone(&flag);
            std::thread::spawn(move || {
                let t = Instant::now();
                while t.elapsed() < delay {
                    std::hint::spin_loop();
                }
                flag.store(true, Ordering::Relaxed);
            })
        };

        let t_call = Instant::now();
        let status = s.factor(&a, None);
        let call_elapsed = t_call.elapsed();
        watchdog.join().expect("watchdog thread");

        if s.mc64_retry_attempt_count() != 1 || delay >= call_elapsed {
            // Too early (the flag was seen by the uninterruptible symbolic
            // phase or by factorization #1), or too late (the call had
            // already finished). Try the next offset.
            continue;
        }
        landed = true;
        eprintln!(
            "landed inside the retry at frac={frac:.2} (delay {delay:?} of \
             call {call_elapsed:?}) after {attempts} attempts; status {status:?}"
        );

        // 1. The cancellation must be reported. Pre-fix the bare `_` arm
        //    mapped Err(Interrupted) to Ok(original factor), so `factor`
        //    returned Success/Singular and the host's cancellation
        //    silently did not take effect.
        assert!(
            matches!(status, FactorStatus::Interrupted),
            "an interrupt observed during the MC64 retry must be reported \
             as Interrupted, not swallowed into the original factor's \
             status; got {status:?}",
        );

        // 2. And the latch must be left disarmed, so once the caller
        //    clears its flag the issue-#65 rescue is still available on
        //    this pattern.
        flag.store(false, Ordering::Relaxed);
        let after = s.factor(&a, None);
        assert_eq!(
            s.mc64_retry_attempt_count(),
            2,
            "the MC64 rescue must still be available after a cancellation. \
             A cancellation is not evidence that MC64 does not help -- the \
             retry never finished -- so `mc64_retry_not_adopted` must stay \
             disarmed. Pre-fix the bare `_` arm armed it, and being keyed \
             on the pattern and cleared only on a pattern change, that \
             suppressed the rescue for every later factor of this pattern. \
             Follow-up status: {after:?}",
        );
        assert!(
            !matches!(after, FactorStatus::Interrupted),
            "with the flag cleared the follow-up factor must run to \
             completion; got {after:?}",
        );
        break;
    }

    assert!(
        landed,
        "none of {attempts} attempts landed inside the retry window, so the \
         regression this test guards was never exercised. Calibrated full \
         call = {t_total:?}. Fail rather than pass vacuously.",
    );
}

#[test]
fn an_interrupt_during_the_first_factorization_still_reports_interrupted() {
    // The companion case, and the one that was already correct: when the
    // flag is seen by factorization #1 the issue-#65 gate is never
    // entered (it keys on `Ok(..)`), so no retry is counted and the error
    // reaches the caller through the ordinary path. Pinned so that a
    // future edit to the gate cannot start swallowing this one too.
    let a = grid_plus_rank_deficient_tail(90);
    let flag = Arc::new(AtomicBool::new(false));
    let mut s = Solver::new().with_interrupt(Arc::clone(&flag));

    let watchdog = {
        let flag = Arc::clone(&flag);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_micros(200));
            flag.store(true, Ordering::Relaxed);
        })
    };
    let status = s.factor(&a, None);
    watchdog.join().expect("watchdog thread");

    assert!(
        matches!(status, FactorStatus::Interrupted),
        "a flag set during the first factorization must abort it; got {status:?}",
    );
    assert_eq!(
        s.mc64_retry_attempt_count(),
        0,
        "the issue-#65 gate keys on Ok(..), so an aborted factorization \
         #1 must never reach the retry",
    );
    assert!(
        s.inertia().is_none(),
        "an interrupted factor stores no factor",
    );
}
