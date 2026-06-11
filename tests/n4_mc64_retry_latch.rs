//! N4 (`dev/research/repo-review-2026-06-09.md`): the issue-#65 MC64
//! scaling retry has no "tried and not adopted" latch. On *adoption* the
//! sticky-Auto pick pins `Mc64Symmetric`, so the retry gate's
//! `!matches!(..Mc64Symmetric)` clause suppresses the retry on every
//! subsequent same-pattern `factor()`. But on *non-adoption* — the case the
//! issue-#65 comment explicitly anticipates, "a GENUINELY singular matrix
//! [where] the retry also force-accepts zeros, the strict-improvement gate
//! fails, and the original factor is kept (cost: one wasted
//! factorization)" — nothing was recorded. The retry gate keys on
//! `self.numeric_params.scaling == Auto` (the user config, which never
//! changes) and on the resolved scaling staying non-MC64 (it does, since
//! the picker re-pins InfNorm), so every subsequent `factor()` on the same
//! pattern re-pays a full Hungarian + a complete second factorization. The
//! comment's "one wasted factorization" is actually one *per call*,
//! indefinitely.
//!
//! Reproduction (no fixture needed, unlike `issue65_mc64_fallback.rs`): a
//! tiny, well-scaled, genuinely rank-deficient symmetric matrix.
//!
//!   A = [[1, 1, 0],
//!        [1, 1, 0],
//!        [0, 0, 1]]   (stored as the lower triangle)
//!
//! Rows 0 and 1 are identical ⇒ rank 2, inertia (2, 0, 1). It is small, so
//! `pick_scaling_strategy` routes `Auto` to `InfNorm` (no arrow head); the
//! BK factorization force-accepts the one zero pivot and reports
//! `zero == 1`. That fires the issue-#65 retry, but MC64 cannot change rank
//! — the retry also reports `zero == 1`, the strict-improvement gate fails,
//! and the original factor is kept (non-adoption).
//!
//! `mc64_retry_attempt_count()` counts every retry that *ran*:
//!   * Pre-fix (no latch): call 1 runs the retry (count 1); call 2 on the
//!     SAME pattern re-runs it (count 2) — RED.
//!   * Post-fix (per-pattern latch): call 1 runs the retry (count 1); call 2
//!     is suppressed by the latch (count stays 1) — GREEN.

use feral::scaling::ScalingStrategy;
use feral::{CscMatrix, Inertia, Solver};

/// `[[1,1,0],[1,1,0],[0,0,1]]` as a lower-triangle CSC: genuinely rank-2.
fn rank_deficient_3x3() -> CscMatrix {
    // Lower-triangle entries: (0,0)=1, (1,0)=1, (1,1)=1, (2,2)=1.
    let rows = [0usize, 1, 1, 2];
    let cols = [0usize, 0, 1, 2];
    let vals = [1.0f64, 1.0, 1.0, 1.0];
    CscMatrix::from_triplets(3, &rows, &cols, &vals).expect("3x3 triplets")
}

#[test]
fn n4_singular_pattern_runs_mc64_retry_at_most_once() {
    let a = rank_deficient_3x3();

    // Default Solver: `Auto` scaling. The picker routes this small matrix to
    // InfNorm, which force-accepts the zero pivot.
    let mut s = Solver::new();

    let _ = s.factor(&a, None);
    let inertia = s.inertia().cloned().expect("inertia after first factor");
    // Sanity: confirm we actually hit the singular signature that drives the
    // retry. If this changes, the reproduction no longer exercises N4.
    assert_eq!(
        inertia,
        Inertia::new(2, 0, 1),
        "the reproduction matrix must be genuinely rank-2 (zero=1); got {inertia:?}",
    );
    assert!(
        matches!(s.scaling_strategy(), ScalingStrategy::Auto),
        "the user config must stay Auto across factor() calls",
    );
    assert_eq!(
        s.mc64_retry_attempt_count(),
        1,
        "the MC64 retry must run exactly once on the first factor of a \
         singular Auto pattern",
    );

    // Re-factor the SAME pattern with the SAME values, exactly as an IPM
    // driver that never regularizes a singular KKT would.
    let _ = s.factor(&a, None);

    assert_eq!(
        s.mc64_retry_attempt_count(),
        1,
        "N4: a genuinely-singular pattern must NOT re-pay the MC64 retry on \
         every subsequent factor(). Pre-fix the non-adoption arm set no \
         latch, so the retry re-ran every call (count climbs to 2+); the \
         per-pattern latch caps it at 1.",
    );

    // A third call must still not re-pay it.
    let _ = s.factor(&a, None);
    assert_eq!(
        s.mc64_retry_attempt_count(),
        1,
        "the latch must hold for the lifetime of the pattern, not just one call",
    );
}
