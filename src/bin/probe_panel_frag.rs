//! Issue #129: measure blocked-panel fragmentation on the dense frontal
//! kernel. Every swap-1×1 / mid-panel swap-2×2 terminates the blocked panel
//! (`ScalarFallback`), flushing the deferred Schur update at a truncated
//! width. This probe factors a set of (strongly-indefinite) matrices with
//! the panel-diagnostic + phase-timing counters on and reports, per matrix
//! and in aggregate:
//!   - PANEL_FULL vs PANEL_PARTIAL vs PANEL_DELAYED (fragmentation rate)
//!   - the FALLBACK_2X2_* breakdown (why panels bail)
//!   - pivots handled inline vs pushed to the scalar path
//!   - the dense Schur trailing update as a fraction of dense-front time
//!
//! The 2026-05-13 decision requires measure-before-kernel-work. If the
//! Schur update is a small fraction of dense-front time and/or panels rarely
//! fragment, #129's in-panel-interchange work is not justified.
//!
//! Sequential driver only (clean phase attribution; the counters are atomic
//! but SCHUR_NS/DENSEFACTOR_NS overlap under parallelism).

use std::path::PathBuf;
use std::sync::atomic::Ordering::Relaxed;

use feral::dense::factor::{panel_diag, phase_timing, PANEL_DIAG_ENABLED, PHASE_TIMING_ENABLED};
use feral::{read_mtx, NumericParams, Solver};

fn main() {
    PANEL_DIAG_ENABLED.store(true, Relaxed);
    PHASE_TIMING_ENABLED.store(true, Relaxed);

    let paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        eprintln!("usage: probe_panel_frag <matrix.mtx>...");
        std::process::exit(2);
    }

    // Aggregate accumulators.
    let mut agg_full = 0u64;
    let mut agg_partial = 0u64;
    let mut agg_delayed = 0u64;
    let mut agg_pin = 0u64;
    let mut agg_psc = 0u64;
    let mut agg_densef = 0u64;
    let mut agg_schur = 0u64;
    let mut agg_panelf = 0u64;
    let mut agg_scalartail = 0u64;

    println!(
        "{:<24} {:>8} {:>6} {:>6} {:>7} {:>7} {:>8} {:>8} {:>9}",
        "matrix", "n", "full", "part", "frag%", "scal%", "schur%", "panelf%", "densef_us"
    );
    for p in &paths {
        let csc = match read_mtx(p).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {}: {e}", p.display());
                continue;
            }
        };
        panel_diag::reset();
        phase_timing::reset();
        let mut solver =
            Solver::with_params(NumericParams::default(), Default::default()).with_parallel(false);
        let _ = solver.factor(&csc, None);

        let s = panel_diag::snapshot();
        let get = |name: &str| {
            s.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| *v)
                .unwrap_or(0)
        };
        let full = get("panel_full");
        let partial = get("panel_partial");
        let delayed = get("panel_delayed");
        let pin = get("pivots_inline");
        let psc = get("pivots_scalar");
        let (_assembly, densef, panelf, schur, scalartail) = phase_timing::snapshot();

        let panels = (full + partial + delayed).max(1);
        let pivots = (pin + psc).max(1);
        let frag = 100.0 * partial as f64 / panels as f64;
        let scal = 100.0 * psc as f64 / pivots as f64;
        let schur_pct = if densef > 0 {
            100.0 * schur as f64 / densef as f64
        } else {
            0.0
        };
        let panelf_pct = if densef > 0 {
            100.0 * panelf as f64 / densef as f64
        } else {
            0.0
        };
        println!(
            "{:<24} {:>8} {:>6} {:>6} {:>6.1} {:>6.1} {:>7.1} {:>7.1} {:>9}",
            p.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            csc.n,
            full,
            partial,
            frag,
            scal,
            schur_pct,
            panelf_pct,
            densef / 1000,
        );

        agg_full += full;
        agg_partial += partial;
        agg_delayed += delayed;
        agg_pin += pin;
        agg_psc += psc;
        agg_densef += densef;
        agg_schur += schur;
        agg_panelf += panelf;
        agg_scalartail += scalartail;
    }

    let panels = (agg_full + agg_partial + agg_delayed).max(1);
    let pivots = (agg_pin + agg_psc).max(1);
    println!("\n--- aggregate ---");
    println!(
        "panels: full={agg_full} partial={agg_partial} delayed={agg_delayed} \
         (frag={:.1}%)",
        100.0 * agg_partial as f64 / panels as f64
    );
    println!(
        "pivots: inline={agg_pin} scalar={agg_psc} (scalar={:.1}%)",
        100.0 * agg_psc as f64 / pivots as f64
    );
    if agg_densef > 0 {
        println!(
            "dense-front time: schur={:.1}%  panel-factor={:.1}%  scalar-tail={:.1}%  (densefactor={} us)",
            100.0 * agg_schur as f64 / agg_densef as f64,
            100.0 * agg_panelf as f64 / agg_densef as f64,
            100.0 * agg_scalartail as f64 / agg_densef as f64,
            agg_densef / 1000,
        );
    }
}
