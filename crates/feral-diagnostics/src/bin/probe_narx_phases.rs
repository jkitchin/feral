//! Issue #44 phase-probe — decompose the NARX_CFy numeric loop.
//!
//! `probe_narx_factor` and `diag_narx_kernel_gflops` localized the cost
//! to the dense frontal factor, and the Schur micro-kernel widening
//! (commit 5f1661c) bought only ~3-7% end-to-end despite a 2.2-2.5×
//! micro-bench win — so the Schur kernel is *not* the whole loop. This
//! probe attributes the ~800 ms NARX_CFy numeric loop to its phases:
//!
//!   numeric loop = assembly + densefactor + loop-overhead
//!   densefactor  = panelfactor + schur + scalartail + densefactor-overhead
//!
//! `assembly`   = `build_row_indices` + original-entry scatter + child
//!                extend-add (`factor_one_supernode` Steps 1-2).
//! `densefactor`= the whole dense frontal factor (Step 3).
//! `panelfactor`= panel/diagonal Bunch-Kaufman (`lblt_panel_frontal`).
//! `schur`      = deferred Schur trailing update (`apply_blocked_schur`).
//! `scalartail` = the scalar pivot tail (`scalar_pivot_step`).
//!
//! It uses the global `dense::factor::phase_timing` counters (gated by
//! `PHASE_TIMING_ENABLED`) for loop-wide totals and the per-supernode
//! `SupernodeTiming` phase deltas for the top-N breakdown. Sequential
//! mode so per-supernode timings line up 1:1.
//!
//! Usage: cargo run --release --bin probe_narx_phases

use feral::dense::factor::{phase_timing, PHASE_TIMING_ENABLED};
use feral::numeric::factorize::{Profiler, SupernodeTiming};
use feral::symbolic::supernode::SupernodeParams;
use feral::{read_mtx, NumericParams, Solver};
use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const DIR: &str = "data/matrices/kkt-mittelmann/NARX_CFy";

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

/// One snapshot tuple of the global phase counters, in ns:
/// `(assembly, densefactor, panelfactor, schur, scalartail)`.
type Phases = (u64, u64, u64, u64, u64);
/// Sub-phase drill-down, in ns:
/// `(buildrow, scatter, extendadd, lextract, contribextract)`.
type Detail = (u64, u64, u64, u64, u64);

fn report_loop(label: &str, loop_us: u64, p: Phases, d: Detail, contrib_zerofill: u64) {
    let (asm, df, panel, schur, tail) = p;
    let (buildrow, scatter, extendadd, lextract, contribextract) = d;
    let loop_ns = loop_us * 1000;
    let pct = |ns: u64| {
        if loop_ns > 0 {
            ns as f64 * 100.0 / loop_ns as f64
        } else {
            0.0
        }
    };
    let line = |indent: &str, name: &str, ns: u64| {
        println!("{indent}{name:<18}{:>9.1} ms  {:>5.1}%", ms(ns), pct(ns));
    };
    // Loop-overhead = whatever in the supernode loop is neither assembly
    // nor dense factor: frontal-buffer build, Step-4 contrib deposit,
    // row_map populate/clear, profiler bookkeeping.
    let loop_oh = loop_ns.saturating_sub(asm + df);
    // assembly residual = assembly minus its three timed sub-phases:
    // row_map populate/clear + the F3.2b Schur-layout swap.
    let asm_res = asm.saturating_sub(buildrow + scatter + extendadd);
    // df residual = dense factor minus its five timed sub-phases:
    // entry setup, the panel-loop control, perm_inv, growth flag,
    // result-struct build.
    let df_res = df.saturating_sub(panel + schur + tail + lextract + contribextract);
    println!("  {label}  loop={:.1} ms", ms(loop_ns));
    line("    ", "assembly", asm);
    line("      ├─ ", "build_row_indices", buildrow);
    line("      ├─ ", "scatter (D·A·D)", scatter);
    line("      ├─ ", "extend_add", extendadd);
    line("      └─ ", "asm-residual", asm_res);
    line("    ", "densefactor", df);
    line("      ├─ ", "panelfactor", panel);
    line("      ├─ ", "schur", schur);
    line("      ├─ ", "scalartail", tail);
    line("      ├─ ", "L-extract", lextract);
    line("      ├─ ", "contrib-extract", contribextract);
    line("      │  └ ", "zerofill (dead)", contrib_zerofill);
    line("      └─ ", "df-residual", df_res);
    line("    ", "loop-overhead", loop_oh);
}

/// Bucket supernodes by `ncol` (number of eliminated columns). The
/// per-call overhead of `factor_one_supernode` (assembly + dense-factor
/// bookkeeping) is fixed per front, so a front distribution skewed to
/// small `ncol` pays that overhead many times — the amalgamation lever.
fn size_distribution(timings: &[SupernodeTiming]) {
    const RANGES: &[(&str, usize, usize)] = &[
        ("1", 1, 1),
        ("2-4", 2, 4),
        ("5-16", 5, 16),
        ("17-64", 17, 64),
        ("65-256", 65, 256),
        (">256", 257, usize::MAX),
    ];
    let total_us: u64 = timings.iter().map(|t| t.us).sum();
    println!(
        "    {:>8} {:>7} {:>10} {:>7} {:>9} {:>9}",
        "ncol", "count", "sum_us", "pct", "asm_us", "df_us"
    );
    for &(label, lo, hi) in RANGES {
        let mut count = 0usize;
        let (mut sum, mut asm, mut df) = (0u64, 0u64, 0u64);
        for t in timings {
            if t.ncol >= lo && t.ncol <= hi {
                count += 1;
                sum += t.us;
                asm += t.assembly_us;
                df += t.densefactor_us;
            }
        }
        let pct = if total_us > 0 {
            sum as f64 * 100.0 / total_us as f64
        } else {
            0.0
        };
        println!(
            "    {:>8} {:>7} {:>10} {:>6.1}% {:>9} {:>9}",
            label, count, sum, pct, asm, df
        );
    }
}

fn top_supernodes(timings: &[SupernodeTiming]) {
    let mut v: Vec<&SupernodeTiming> = timings.iter().collect();
    v.sort_by_key(|t| std::cmp::Reverse(t.us));
    println!(
        "    {:>6} {:>6} {:>6} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "snode", "nrow", "ncol", "us", "asm_us", "df_us", "panel", "schur", "tail"
    );
    for t in v.into_iter().take(10) {
        println!(
            "    {:>6} {:>6} {:>6} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8}",
            t.snode_idx,
            t.nrow,
            t.ncol,
            t.us,
            t.assembly_us,
            t.densefactor_us,
            t.panelfactor_us,
            t.schur_us,
            t.scalartail_us,
        );
    }
}

fn run(iter: usize) {
    let mtx_path = format!("{DIR}/NARX_CFy_{iter:04}.mtx");
    let path = Path::new(&mtx_path);
    if !path.exists() {
        eprintln!("SKIP iter {iter}: {mtx_path} not present (corpus gitignored)");
        return;
    }
    let csc = match read_mtx(path).and_then(|m| m.to_csc()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP iter {iter}: load failed: {e:?}");
            return;
        }
    };
    println!(
        "\nNARX_CFy_{iter:04}  n={}  nnz={}",
        csc.n, csc.col_ptr[csc.n]
    );

    let prof = Arc::new(Mutex::new(Profiler::new()));
    let np = NumericParams {
        profiler: Some(prof.clone()),
        ..NumericParams::default()
    };
    let mut solver = Solver::with_params(np, SupernodeParams::default()).with_parallel(false);

    // Cold factor: includes symbolic. Reset phase counters first so the
    // snapshot reflects only this factorization.
    phase_timing::reset();
    let t0 = Instant::now();
    let st_cold = solver.factor(&csc, None);
    let cold_ms = t0.elapsed().as_secs_f64() * 1e3;
    let cold_phases = phase_timing::snapshot();
    let cold_detail = phase_timing::snapshot_detail();
    let cold_zerofill = phase_timing::snapshot_contrib_zerofill();
    let cold_loop_us = prof.lock().map(|p| p.report().loop_us).unwrap_or(0);

    // Warm factor: numeric only (symbolic reused). Re-arm profiler and
    // reset counters.
    if let Ok(mut p) = prof.lock() {
        *p = Profiler::new();
    }
    phase_timing::reset();
    let t0 = Instant::now();
    let st_warm = solver.factor(&csc, None);
    let warm_ms = t0.elapsed().as_secs_f64() * 1e3;
    let warm_phases = phase_timing::snapshot();
    let warm_detail = phase_timing::snapshot_detail();
    let warm_zerofill = phase_timing::snapshot_contrib_zerofill();

    let prof = match prof.lock() {
        Ok(p) => p.clone(),
        Err(_) => {
            eprintln!("profiler mutex poisoned");
            return;
        }
    };
    let report = prof.report();

    println!(
        "  cold={cold_ms:.1} ms ({st_cold:?})  warm={warm_ms:.1} ms ({st_warm:?})  \
         n_snodes={}",
        report.n_supernodes
    );
    report_loop(
        "COLD",
        cold_loop_us,
        cold_phases,
        cold_detail,
        cold_zerofill,
    );
    report_loop(
        "WARM",
        report.loop_us,
        warm_phases,
        warm_detail,
        warm_zerofill,
    );
    println!("  supernode distribution by ncol (warm):");
    size_distribution(prof.timings());
    println!("  top 10 supernodes by time (warm):");
    top_supernodes(prof.timings());
}

fn main() {
    println!("Issue #44 phase-probe — NARX_CFy numeric-loop phase breakdown");
    PHASE_TIMING_ENABLED.store(true, Relaxed);
    for iter in 0..3 {
        run(iter);
    }
    PHASE_TIMING_ENABLED.store(false, Relaxed);
}
