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
//! Lives in its own integration-test binary because the first test sets
//! process-global env vars: it cannot race tests in the same process.

use feral::numeric::factorize::{par_min_seeds, par_task_min_flops, PAR_TASK_MIN_FLOPS};

/// One test, one process: every env mutation in this binary happens here,
/// in order, so nothing observes a half-set environment.
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

/// Source scan: no `FERAL_*` env read may parse its own value again.
///
/// The bug in #176 was not one call site but one *shape*, copied to
/// sixteen places. `feral::env` is now the single place that decides
/// what a numeric knob accepts and what happens to a value it cannot
/// use; this test fails if a new knob reintroduces the local
/// `.parse().ok()` that discards the error.
#[test]
fn no_feral_env_read_parses_its_own_value() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_rs(&root.join("src"), &mut files);
    collect_rs(&root.join("crates"), &mut files);

    let mut offenders = Vec::new();
    for path in files {
        // The one file allowed to parse: the shared policy itself.
        if path.ends_with("src/env.rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (start, _) in text.match_indices("env::var(\"FERAL_") {
            // Window = the rest of that statement, capped so a `;`-less
            // construct cannot swallow the whole file.
            let rest = &text[start..];
            let end = rest.find(';').unwrap_or(rest.len()).min(400);
            let window = &rest[..end];
            // Comma-separated *list* knobs (`FERAL_NEMIN_LIST`,
            // `FERAL_MERGE_BUDGET_LIST`, both diagnostics-only) parse
            // per token after a split; they are not single-value knobs
            // and have no `feral::env` shape to use.
            if window.contains(".parse") && !window.contains(".split(") {
                let line = text[..start].matches('\n').count() + 1;
                offenders.push(format!("{}:{}", path.display(), line));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "these FERAL_* env reads parse locally instead of going through \
         feral::env, which is how issue #176 happened: {offenders:?}"
    );
}

fn collect_rs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}
