//! diag_factor_phases — where does feral's numeric factorization time go?
//!
//! Splits the warm **numeric driver** wall (not the `Solver::factor`
//! wall — that folds in symbolic reuse checks and scaling-cache lookups
//! the phase counters do not cover) into:
//!
//!   driver total = prologue + per-supernode loop + epilogue
//!   loop         = assembly + dense factor + per-node remainder
//!   dense factor = panel + Schur + scalar tail + L-extract
//!                  + contrib-extract + dense remainder
//!
//! The two remainder columns are the point of the probe: an
//! optimization item aimed at a named sub-phase cannot pay for itself
//! if the unnamed remainder is larger than the phase it targets.
//!
//! Usage: diag_factor_phases <a.mtx> [b.mtx ...]

use feral::dense::factor::{phase_timing, PHASE_TIMING_ENABLED};
use feral::numeric::factorize::{
    factorize_multifrontal_supernodal_with_workspace, FactorWorkspace, Profiler,
};
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, NumericParams};
use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const N_REPS: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: diag_factor_phases <a.mtx> [b.mtx ...]");
        std::process::exit(2);
    }
    println!(
        "{:<22}{:>9}{:>7}{:>9}{:>8}{:>8}{:>8}{:>8}{:>9}{:>10}",
        "matrix",
        "drv_us",
        "prol%",
        "permute%",
        "scaling%",
        "schur%",
        "cbextr%",
        "zerofil%",
        "dense_o%",
        "buildrow%"
    );
    for a in &args {
        let csc = match read_mtx(Path::new(a)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {a}: {e:?}");
                continue;
            }
        };
        let sp = SupernodeParams::default();
        let symbolic = match symbolic_factorize_with_method(&csc, &sp, OrderingMethod::Auto) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{a}: symbolic failed: {e:?}");
                continue;
            }
        };
        let nparams = NumericParams::default();
        let mut ws = FactorWorkspace::new();
        // Warm the workspace and branch predictors; not timed.
        if factorize_multifrontal_supernodal_with_workspace(&csc, &symbolic, &nparams, &mut ws)
            .is_err()
        {
            eprintln!("{a}: warm-up factorization failed");
            continue;
        }

        // Best-of-N by driver wall, per the `min` convention in
        // dev/decisions.md.
        let mut best = u64::MAX;
        let mut keep = [0u64; 15];
        for _ in 0..N_REPS {
            PHASE_TIMING_ENABLED.store(true, Relaxed);
            phase_timing::reset();
            let prof = Arc::new(Mutex::new(Profiler::new()));
            let mut np = nparams.clone();
            np.profiler = Some(prof.clone());
            np.pattern_reused_hint = true;

            let t0 = Instant::now();
            let res =
                factorize_multifrontal_supernodal_with_workspace(&csc, &symbolic, &np, &mut ws);
            let wall_ns = t0.elapsed().as_nanos() as u64;
            PHASE_TIMING_ENABLED.store(false, Relaxed);
            if res.is_err() {
                eprintln!("{a}: timed factorization failed");
                break;
            }
            let guard = match prof.lock() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("{a}: profiler poisoned: {e}");
                    break;
                }
            };
            let report = guard.report();
            let timings = guard.timings();
            let asm_ns: u64 = timings.iter().map(|t| t.assembly_ns).sum();
            let dense_ns: u64 = timings.iter().map(|t| t.densefactor_ns).sum();
            let panel_ns: u64 = timings.iter().map(|t| t.panelfactor_ns).sum();
            let schur_ns: u64 = timings.iter().map(|t| t.schur_ns).sum();
            let tail_ns: u64 = timings.iter().map(|t| t.scalartail_ns).sum();
            let (br_ns, _sc, _xa, lex_ns, cbex_ns) = phase_timing::snapshot_detail();
            let zerofill_ns = phase_timing::CONTRIBZEROFILL_NS.load(Relaxed);
            let pb = report.prologue_breakdown;
            drop(guard);

            if wall_ns < best {
                best = wall_ns;
                keep = [
                    report.prologue_us * 1000,
                    report.epilogue_us * 1000,
                    report.loop_ns,
                    asm_ns,
                    dense_ns,
                    panel_ns,
                    schur_ns,
                    lex_ns + tail_ns,
                    cbex_ns,
                    zerofill_ns,
                    pb.permute_us * 1000,
                    pb.scaling_us * 1000,
                    pb.symmetric_pattern_us * 1000,
                    pb.setup_us * 1000,
                    br_ns,
                ];
            }
        }
        if best == u64::MAX {
            continue;
        }
        let [prol, epil, loop_ns, asm, dense, panel, schur, lex_tail, cbex, zerofill, permute, scaling, sympat, setup, buildrow] =
            keep;
        let _ = (sympat, setup);
        // Per-node remainder: loop time not inside assembly or the dense
        // factor. Dense remainder: dense time outside its five sub-phases.
        let node_other = loop_ns.saturating_sub(asm + dense);
        let dense_other = dense.saturating_sub(panel + schur + lex_tail + cbex);
        let pct = |v: u64| 100.0 * v as f64 / best.max(1) as f64;
        let name = Path::new(a)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(a);
        println!(
            "{:<22}{:>9}{:>6.1}%{:>8.1}%{:>7.1}%{:>7.1}%{:>7.1}%{:>7.1}%{:>8.1}%{:>9.1}%",
            name,
            best / 1000,
            pct(prol),
            pct(permute),
            pct(scaling),
            pct(schur),
            pct(cbex),
            pct(zerofill),
            pct(dense_other),
            pct(buildrow)
        );
        let _ = (epil, loop_ns, asm, dense, panel, lex_tail, node_other);
    }
    Ok(())
}
