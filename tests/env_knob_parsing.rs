//! Issue #176: a numeric `FERAL_*` knob must never be silently replaced
//! by its built-in default.
//!
//! The reported case, verbatim:
//!
//!     $ FERAL_PAR_TASK_MIN_FLOPS=1e18 pounce NARX_CFy.nl --no-sol max_iter=1
//!     task_plan: n_snodes=45736 n_tasks=21 seeds=11 cutoff=1000000 min_seeds=2
//!
//! `cutoff=1000000` is `PAR_TASK_MIN_FLOPS`, the default — the `1e18`
//! failed `parse::<u64>()` and was thrown away by `.ok()`. Two perf
//! measurements were taken from runs whose knobs had never taken effect.
//!
//! This binary holds exactly one `#[test]`, on purpose. libtest runs the
//! tests in a binary on concurrent threads, and `std::env::set_var` is
//! undefined behaviour if any other thread reads the environment while
//! it runs — including a `getenv` from inside std or the harness. One
//! test means there is no other thread to race. The source-scan half of
//! this issue's coverage is in `env_knob_scan.rs` for that reason.

use feral::numeric::factorize::{par_min_seeds, par_task_min_flops, PAR_TASK_MIN_FLOPS};

/// One test, one process: every env mutation in this binary happens here,
/// in order, so nothing observes a half-set environment.
///
/// SAFETY (every `set_var`/`remove_var` below): `std::env::set_var` is
/// sound only when no other thread touches the environment
/// concurrently. This is the only `#[test]` in this integration binary
/// (see the module comment), so libtest has no second test thread to
/// run against it, and the code under test reads the environment on
/// this thread.
#[test]
fn numeric_knobs_take_scientific_notation_and_never_default_silently() {
    // --- The issue's case: `1e18` on the flop-count knob.
    unsafe { std::env::set_var("FERAL_PAR_TASK_MIN_FLOPS", "1e18") };
    assert_eq!(
        par_task_min_flops(),
        1_000_000_000_000_000_000,
        "FERAL_PAR_TASK_MIN_FLOPS=1e18 must mean 1e18, not the default"
    );

    // --- The two spellings of the same number must agree.
    unsafe { std::env::set_var("FERAL_PAR_TASK_MIN_FLOPS", "1000000000000000000") };
    assert_eq!(par_task_min_flops(), 1_000_000_000_000_000_000);

    // --- Plain integers keep working exactly (no f64 round-trip).
    unsafe { std::env::set_var("FERAL_PAR_TASK_MIN_FLOPS", "18446744073709551615") };
    assert_eq!(par_task_min_flops(), u64::MAX);

    // --- A magnitude past u64 clamps to "off", it does not fall back to
    // the default: the operator asked for a cutoff nothing can reach.
    unsafe { std::env::set_var("FERAL_PAR_TASK_MIN_FLOPS", "1e30") };
    assert_eq!(par_task_min_flops(), u64::MAX);

    // --- Past f64 range too: `1e400` parses to +inf, and an operator
    // who escalates past `1e30` to be extra sure must not land back on
    // the default (which would be more parallel, not less).
    unsafe { std::env::set_var("FERAL_PAR_TASK_MIN_FLOPS", "1e400") };
    assert_eq!(par_task_min_flops(), u64::MAX);

    // --- A genuinely unusable value falls back (loudly — the warning
    // goes to stderr, which is the part this test cannot assert) rather
    // than being interpreted as something else.
    unsafe { std::env::set_var("FERAL_PAR_TASK_MIN_FLOPS", "1e18x") };
    assert_eq!(par_task_min_flops(), PAR_TASK_MIN_FLOPS);

    unsafe { std::env::remove_var("FERAL_PAR_TASK_MIN_FLOPS") };
    assert_eq!(par_task_min_flops(), PAR_TASK_MIN_FLOPS);

    // --- Fractional input rounds instead of truncating: 0.9 seeds must
    // not become "0" (= always parallel), the opposite of the request.
    unsafe { std::env::set_var("FERAL_PAR_MIN_SEEDS", "0.9") };
    assert_eq!(par_min_seeds(), 1);
    unsafe { std::env::set_var("FERAL_PAR_MIN_SEEDS", "4") };
    assert_eq!(par_min_seeds(), 4);
    unsafe { std::env::remove_var("FERAL_PAR_MIN_SEEDS") };

    // --- The other knob named in the issue, through the shared reader
    // it now goes through (`CbThreshold::resolve` is private).
    unsafe { std::env::set_var("FERAL_CB_THRESH", "1e18") };
    assert_eq!(
        feral::env::u64_var("FERAL_CB_THRESH"),
        Some(1_000_000_000_000_000_000)
    );
    unsafe { std::env::set_var("FERAL_CB_THRESH", "wat") };
    assert_eq!(feral::env::u64_var("FERAL_CB_THRESH"), None);
    unsafe { std::env::remove_var("FERAL_CB_THRESH") };
    assert_eq!(feral::env::u64_var("FERAL_CB_THRESH"), None);

    // --- Float knobs: the pivot-threshold spellings, and the validity
    // check that used to swallow an out-of-range value in silence.
    unsafe { std::env::set_var("FERAL_PIVTOL", "1e-8") };
    assert_eq!(
        feral::env::f64_var_where("FERAL_PIVTOL", ">= 0", |v| v >= 0.0),
        Some(1e-8)
    );
    unsafe { std::env::set_var("FERAL_PIVTOL", "-1") };
    assert_eq!(
        feral::env::f64_var_where("FERAL_PIVTOL", ">= 0", |v| v >= 0.0),
        None
    );
    unsafe { std::env::set_var("FERAL_PIVTOL", "nan") };
    assert_eq!(
        feral::env::f64_var_where("FERAL_PIVTOL", ">= 0", |v| v >= 0.0),
        None
    );
    unsafe { std::env::remove_var("FERAL_PIVTOL") };
}
