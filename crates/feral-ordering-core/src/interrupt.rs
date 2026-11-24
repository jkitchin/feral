//! Cooperative interruption of long-running orderings (feral issue #246).
//!
//! An [`Interrupt`] carries an optional wall-clock deadline and an optional
//! cancel flag. [`scope`] installs one for the duration of a closure on the
//! current thread. Long-running loops in the ordering crates call [`poll`]
//! or [`check`] at safe points (between pivots, levels, trials and passes)
//! with no change to their signatures.
//!
//! **Single-thread assumption.** A thread-local is sound here because every
//! ordering in this family (`feral-amd`, `-amf`, `-metis`, `-scotch`,
//! `-kahip`), and feral's symbolic pipeline and MC64 matching, run
//! entirely on the thread that called [`scope`]. If any of them ever moves
//! work onto rayon or spawned threads, polls on those workers see *no*
//! active interrupt, and the deadline silently stops working there. Code
//! that goes parallel must capture the interrupt with [`current`] and
//! re-install it on each worker with [`scope`]. Each ordering crate has a
//! `tests/interrupt_threads.rs` that uses the [`census`] to fail if an
//! ordering polls from a thread with no scope while its caller has one, so
//! such a change cannot slip in unnoticed.
//!
//! **Sticky.** Once a poll inside a scope has seen the interrupt trigger,
//! every later poll in that scope reports it, even if the flag is cleared.
//! A loop that breaks out early on [`poll`] can therefore rely on the next
//! [`check`] up the stack returning [`OrderingError::Interrupted`], so a
//! partially computed intermediate is never used as a result.
//!
//! With no interrupt installed, a poll costs one thread-local read.

use crate::OrderingError;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A deadline and/or a cancel flag. `Interrupt::default()` never triggers.
#[derive(Debug, Clone, Default)]
pub struct Interrupt {
    deadline: Option<Instant>,
    flag: Option<Arc<AtomicBool>>,
}

impl Interrupt {
    /// An interrupt that never triggers.
    pub fn none() -> Self {
        Self::default()
    }

    /// Trigger once `deadline` has passed.
    pub fn at(deadline: Instant) -> Self {
        Self {
            deadline: Some(deadline),
            flag: None,
        }
    }

    /// Trigger once `budget` has elapsed from now.
    pub fn after(budget: Duration) -> Self {
        Self::at(Instant::now() + budget)
    }

    /// Also trigger when `flag` is set (from any thread).
    pub fn with_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.flag = Some(flag);
        self
    }

    /// Also trigger at `deadline` (replacing any earlier deadline).
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// The deadline, if any.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// True when a deadline is set and has passed. Lets a host that set
    /// both a deadline and a flag tell which one fired.
    pub fn deadline_passed(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// True when the cancel flag is set.
    pub fn flag_set(&self) -> bool {
        self.flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed))
    }

    /// True when either the deadline has passed or the flag is set.
    pub fn is_triggered(&self) -> bool {
        self.flag_set() || self.deadline_passed()
    }

    /// True when this interrupt can never trigger.
    pub fn is_inert(&self) -> bool {
        self.deadline.is_none() && self.flag.is_none()
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Interrupt>> = const { RefCell::new(None) };
    static TRIPPED: Cell<bool> = const { Cell::new(false) };
}

/// Restores the previous scope state on drop, including during unwinding.
struct ScopeGuard {
    prev: Option<Interrupt>,
    prev_tripped: bool,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        ACTIVE.with(|a| *a.borrow_mut() = prev);
        TRIPPED.with(|t| t.set(self.prev_tripped));
    }
}

/// Run `f` with `interrupt` active on this thread, then restore whatever
/// was active before (also on panic). An inert interrupt still opens a
/// scope, which shields `f` from an outer interrupt; use [`scope_if`] to
/// keep the outer one.
pub fn scope<R>(interrupt: &Interrupt, f: impl FnOnce() -> R) -> R {
    let prev = ACTIVE.with(|a| a.borrow_mut().replace(interrupt.clone()));
    let prev_tripped = TRIPPED.with(|t| t.replace(false));
    let _guard = ScopeGuard { prev, prev_tripped };
    f()
}

/// [`scope`] when `interrupt` is `Some`, otherwise run `f` under whatever
/// is already active.
pub fn scope_if<R>(interrupt: Option<&Interrupt>, f: impl FnOnce() -> R) -> R {
    match interrupt {
        Some(i) => scope(i, f),
        None => f(),
    }
}

/// True when the active interrupt has triggered (sticky within its scope).
/// With no active interrupt, returns false.
pub fn poll() -> bool {
    if TRIPPED.with(|t| t.get()) {
        census::record(true);
        return true;
    }
    let (present, hit) = ACTIVE.with(|a| {
        let a = a.borrow();
        (a.is_some(), a.as_ref().is_some_and(Interrupt::is_triggered))
    });
    census::record(present);
    if hit {
        TRIPPED.with(|t| t.set(true));
    }
    hit
}

/// The interrupt active on this thread, if any. Parallel code captures it
/// here and re-installs it on each worker with [`scope`] (see the module
/// docs on the single-thread assumption).
pub fn current() -> Option<Interrupt> {
    ACTIVE.with(|a| a.borrow().clone())
}

/// Opt-in poll census for tests that guard the single-thread assumption
/// (PR #247 review).
///
/// While armed, every [`poll`]/[`check`] on any thread is counted as either
/// *scoped* (its thread had a [`scope`] installed) or *unscoped*. A test arms
/// it, runs an ordering inside a scope, and asserts `unscoped == 0` and
/// `scoped > 0`. A non-zero `unscoped` means some poll ran on a thread the
/// interrupt never reached. Disarmed (always, outside such tests), the cost
/// is one relaxed atomic load per poll. Run each census test in its own
/// test binary, so that no unrelated thread polls while it is armed.
pub mod census {
    use super::{AtomicBool, AtomicUsize, Ordering};

    static ARMED: AtomicBool = AtomicBool::new(false);
    static SCOPED: AtomicUsize = AtomicUsize::new(0);
    static UNSCOPED: AtomicUsize = AtomicUsize::new(0);

    #[inline]
    pub(super) fn record(scoped: bool) {
        if ARMED.load(Ordering::Relaxed) {
            let c = if scoped { &SCOPED } else { &UNSCOPED };
            c.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Poll counts gathered by [`take`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Counts {
        /// Polls from a thread with a scope installed.
        pub scoped: usize,
        /// Polls from a thread with no scope installed.
        pub unscoped: usize,
    }

    /// Zero the counters and start counting.
    pub fn arm() {
        SCOPED.store(0, Ordering::Relaxed);
        UNSCOPED.store(0, Ordering::Relaxed);
        ARMED.store(true, Ordering::SeqCst);
    }

    /// Stop counting and return what was counted since [`arm`].
    pub fn take() -> Counts {
        ARMED.store(false, Ordering::SeqCst);
        Counts {
            scoped: SCOPED.load(Ordering::Relaxed),
            unscoped: UNSCOPED.load(Ordering::Relaxed),
        }
    }
}

/// `Err(OrderingError::Interrupted)` when [`poll`] is true.
pub fn check() -> Result<(), OrderingError> {
    if poll() {
        Err(OrderingError::Interrupted)
    } else {
        Ok(())
    }
}

/// True when an interrupt that can trigger is active on this thread.
pub fn is_active() -> bool {
    ACTIVE.with(|a| a.borrow().as_ref().is_some_and(|i| !i.is_inert()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expired() -> Interrupt {
        Interrupt::at(Instant::now() - Duration::from_secs(1))
    }

    #[test]
    fn none_never_triggers_and_poll_without_scope_is_false() {
        assert!(!Interrupt::none().is_triggered());
        assert!(Interrupt::none().is_inert());
        assert!(!poll());
        assert!(check().is_ok());
        assert!(!is_active());
        assert!(!scope(&Interrupt::none(), poll));
    }

    #[test]
    fn deadline_and_flag_trigger() {
        assert!(expired().is_triggered());
        assert!(expired().deadline_passed());
        let far = Interrupt::after(Duration::from_secs(3600));
        assert!(!far.is_triggered());
        let flag = Arc::new(AtomicBool::new(false));
        let i = far.with_flag(flag.clone());
        assert!(!i.is_triggered());
        flag.store(true, Ordering::Relaxed);
        assert!(i.is_triggered());
        assert!(i.flag_set());
        assert!(!i.deadline_passed());
    }

    #[test]
    fn scope_nests_and_restores() {
        scope(&expired(), || {
            assert!(poll());
            assert_eq!(check(), Err(OrderingError::Interrupted));
            // An inner inert scope shields from the outer interrupt...
            scope(&Interrupt::none(), || assert!(!poll()));
            // ...and the outer one (still tripped) is back afterwards.
            assert!(poll());
            // scope_if(None) keeps the outer interrupt.
            scope_if(None, || assert!(poll()));
        });
        assert!(!poll());
        assert!(!is_active());
    }

    #[test]
    fn scope_restores_on_panic() {
        let r = std::panic::catch_unwind(|| {
            scope(&expired(), || {
                assert!(poll());
                panic!("boom");
            })
        });
        assert!(r.is_err());
        assert!(!poll(), "the panicking scope must not leak its interrupt");
    }

    #[test]
    fn trigger_is_sticky_within_a_scope() {
        let flag = Arc::new(AtomicBool::new(true));
        let i = Interrupt::none().with_flag(flag.clone());
        scope(&i, || {
            assert!(poll());
            flag.store(false, Ordering::Relaxed);
            assert!(poll(), "a seen trigger must stay seen in its scope");
        });
        // A fresh scope with the cleared flag starts untripped.
        scope(&i, || assert!(!poll()));
    }

    /// A path graph 0-1-...-(n-1) as a full symmetric CSC pattern.
    fn path(n: usize) -> (Vec<i32>, Vec<i32>) {
        let mut cp = vec![0i32];
        let mut ri = Vec::new();
        for j in 0..n {
            if j > 0 {
                ri.push((j - 1) as i32);
            }
            ri.push(j as i32);
            if j + 1 < n {
                ri.push((j + 1) as i32);
            }
            cp.push(ri.len() as i32);
        }
        (cp, ri)
    }

    #[test]
    fn elimination_loops_stop_on_an_expired_interrupt() {
        use crate::quotient_graph::{order, MinDegree, MinFill, WorkspaceOptions};
        let (cp, ri) = path(500);
        let pat = crate::CscPattern::new(500, &cp, &ri).expect("pattern");
        let opts = WorkspaceOptions::default();
        let amd = scope(&expired(), || order::<MinDegree>(&pat, &opts, true));
        assert_eq!(amd.err(), Some(OrderingError::Interrupted));
        let amf = scope(&expired(), || order::<MinFill>(&pat, &opts, true));
        assert_eq!(amf.err(), Some(OrderingError::Interrupted));
        // No interrupt: unchanged result.
        let base = order::<MinDegree>(&pat, &opts, true).expect("amd").0;
        let inert = scope(&Interrupt::none(), || order::<MinDegree>(&pat, &opts, true))
            .expect("amd")
            .0;
        assert_eq!(base, inert);
    }

    /// The census catches exactly the failure the single-thread assumption
    /// guards against: a worker thread polling while its parent has a scope.
    /// (Other tests may poll concurrently while it is armed; counts only grow,
    /// so `unscoped >= 1` stays a sound assertion. The `== 0` direction lives
    /// in each ordering crate's own `tests/interrupt_threads.rs` binary.)
    #[test]
    fn census_detects_a_worker_thread_without_a_scope() {
        census::arm();
        scope(&Interrupt::after(Duration::from_secs(3600)), || {
            assert!(current().is_some());
            std::thread::spawn(|| {
                assert!(current().is_none(), "scopes do not cross threads");
                let _ = poll();
            })
            .join()
            .expect("worker");
            let _ = poll();
        });
        let c = census::take();
        assert!(c.unscoped >= 1, "{c:?}");
        assert!(c.scoped >= 1, "{c:?}");
    }
}
