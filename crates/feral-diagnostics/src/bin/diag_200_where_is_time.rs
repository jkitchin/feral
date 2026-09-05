//! Issue #200 step 1: name the "UNATTRIBUTED 33-39%".
//!
//! The issue computes `UNATTRIBUTED = total - ASSEMBLY - DENSEFACTOR` where
//! `total` is the whole `factor()` wallclock. But `factor()` is not only the
//! per-supernode loop: it also runs a prologue (scaling, pivot-order scaling
//! vector, `P·A·Pᵀ` permute, symmetric pattern, workspace setup) and an
//! epilogue (`SparseFactors` construction, two perm clones, `build_node_parents`).
//! Neither is per-supernode, and neither is covered by the `phase_timing`
//! counters the issue read. So the residual it measured is an upper bound on
//! per-node overhead, not a measurement of it.
//!
//! This probe splits the same wallclock the honest way, using the in-tree
//! `Profiler` (`prologue_us` / `loop_ns` / `epilogue_us` + `PrologueBreakdown`)
//! together with the `phase_timing` counters, and reports:
//!
//!   PROLOGUE (with sub-phases) | LOOP (assembly / densefactor / node tail) | EPILOGUE
//!
//! It also reports the supernode-size distribution, so a matrix can be checked
//! against the issue's regime claim (median front `nrow` = 4, 63-70% of
//! supernodes with `nrow <= 8`) before its numbers are used as evidence.
//!
//! Instrumentation perturbs the thing it measures, so the probe reports BOTH
//! an uninstrumented `min`-of-N factor time and the instrumented one; the
//! ratio is the probe tax and is printed, not hidden.
//!
//! Sequential driver only: the phase counters are process-global atomics and
//! overlap under parallelism.
//!
//! Usage: `diag_200_where_is_time [--reps N] <matrix.mtx>...`

use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use feral::dense::factor::{phase_timing, PHASE_TIMING_ENABLED};
use feral::symbolic::supernode::SupernodeParams;
use feral::symbolic::symbolic_factorize;
use feral::{read_mtx, CscMatrix, NumericParams, Solver};

fn pct(part: f64, whole: f64) -> f64 {
    if whole > 0.0 {
        100.0 * part / whole
    } else {
        0.0
    }
}

/// Uninstrumented *numeric* factor time, min over `reps`, in microseconds.
///
/// `min` per the 2026-08-09 decision: it is the least-interfered per-sample
/// statistic.
///
/// The solver is **warmed first and reused**, so the symbolic analysis is
/// paid once outside the timed region. This is deliberate and it is what
/// makes the number comparable to the profiler's `total_us`, which covers
/// only `factorize_multifrontal_supernodal_with_workspace`. Rebuilding the
/// solver per rep instead measures symbolic + numeric and reads ~2-4x larger
/// — it is also not the quantity issue #200 is about, since an interior-point
/// host factors the same pattern every iteration and pays symbolic once.
fn min_factor_us(csc: &CscMatrix, reps: usize) -> f64 {
    let mut solver = Solver::with_params(NumericParams::default(), SupernodeParams::default())
        .with_parallel(false);
    // Warm-up: two calls, so the timed reps are steady state (the second
    // call is the one that builds the permute cache; see the instrumented
    // path below for why).
    let _ = solver.factor(csc, None);
    let _ = solver.factor(csc, None);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let _ = solver.factor(csc, None);
        let us = t.elapsed().as_secs_f64() * 1e6;
        if us < best {
            best = us;
        }
    }
    best
}

fn main() {
    let mut reps = 5usize;
    let mut paths: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--reps" {
            reps = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                eprintln!("--reps needs a positive integer");
                std::process::exit(2)
            });
        } else {
            paths.push(a);
        }
    }
    if paths.is_empty() {
        eprintln!("usage: diag_200_where_is_time [--reps N] <matrix.mtx>...");
        std::process::exit(2);
    }

    for p in &paths {
        let csc = match read_mtx(std::path::Path::new(p)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {p}: {e}");
                continue;
            }
        };
        let name = std::path::Path::new(p)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.clone());

        // --- regime check: supernode size distribution -------------------
        let sp = SupernodeParams::default();
        let (n_sn, med_nrow, frac_le8, max_front) = match symbolic_factorize(&csc, &sp) {
            Ok(sym) => {
                let mut rows: Vec<usize> = sym.supernodes.iter().map(|s| s.nrow).collect();
                rows.sort_unstable();
                let n_sn = rows.len();
                let med = if n_sn == 0 { 0 } else { rows[n_sn / 2] };
                let le8 = rows.iter().filter(|&&r| r <= 8).count();
                let mx = rows.last().copied().unwrap_or(0);
                (n_sn, med, pct(le8 as f64, n_sn as f64), mx)
            }
            Err(e) => {
                eprintln!("skip {name}: symbolic failed: {e}");
                continue;
            }
        };

        // --- uninstrumented baseline -------------------------------------
        PHASE_TIMING_ENABLED.store(false, Relaxed);
        let raw_us = min_factor_us(&csc, reps);

        // --- instrumented run --------------------------------------------
        let mut solver = Solver::with_params(NumericParams::default(), SupernodeParams::default())
            .with_parallel(false)
            .with_profiling(true);
        // Warm with TWO calls, not one. On a fresh `Solver` the first
        // factor runs with `pattern_reused_hint == false` (no symbolic
        // cache yet) and deliberately does not build the permute-value-map
        // cache (issue #56 N7: a one-shot caller must not pay for a cache
        // it will never read). The second runs with the hint true but the
        // cache still empty, so it takes the cold `from_triplets` path AND
        // pays the cache build. Only the third call onward is steady state.
        // Measuring call 2 reports the cache-build cost as if it were the
        // per-factorization cost and overstates `permute` several-fold.
        let _ = solver.factor(&csc, None);
        let _ = solver.factor(&csc, None);
        PHASE_TIMING_ENABLED.store(true, Relaxed);
        phase_timing::reset();
        let _ = solver.factor(&csc, None);
        let rep = match solver.profile_report() {
            Some(r) => r,
            None => {
                eprintln!("skip {name}: no profile report");
                continue;
            }
        };
        PHASE_TIMING_ENABLED.store(false, Relaxed);

        let total = rep.total_us as f64;
        let prologue = rep.prologue_us as f64;
        let epilogue = rep.epilogue_us as f64;
        let loop_us = rep.loop_ns as f64 / 1e3;
        // Driver time inside `factor()` that the profiler attributes to
        // neither the prologue, the loop, nor the epilogue.
        let driver_other = (total - prologue - loop_us - epilogue).max(0.0);

        let (asm_ns, df_ns, panel_ns, schur_ns, tail_ns) = phase_timing::snapshot();
        let asm_us = asm_ns as f64 / 1e3;
        let df_us = df_ns as f64 / 1e3;
        let arith_us = (panel_ns + schur_ns + tail_ns) as f64 / 1e3;
        // Per-supernode work the counters do NOT cover: the contribution-block
        // deposit, the `row_map` mirror-clear and the `NodeFactors` build.
        let node_tail_us = (loop_us - asm_us - df_us).max(0.0);
        // Dense-factor time that is not panel/Schur/scalar-tail arithmetic.
        // On small fronts this is the single largest block of factor time and
        // is what issue #200 saw as part of its "UNATTRIBUTED".
        let bookkeep_us = (df_us - arith_us).max(0.0);
        let ld = |c: &std::sync::atomic::AtomicU64| c.load(Relaxed) as f64 / 1e3;
        let lextract_us = ld(&phase_timing::LEXTRACT_NS);
        let contribx_us = ld(&phase_timing::CONTRIBEXTRACT_NS);
        let zerofill_us = ld(&phase_timing::CONTRIBZEROFILL_NS);
        let buildrow_us = ld(&phase_timing::BUILDROW_NS);
        let scatter_us = ld(&phase_timing::SCATTER_NS);
        let extendadd_us = ld(&phase_timing::EXTENDADD_NS);

        let b = &rep.prologue_breakdown;

        println!("=== {name} ===");
        println!(
            "  n={} nnz={} nnz/n={:.2} | supernodes={} median_nrow={} nrow<=8={:.1}% max_front={}",
            csc.n,
            csc.values.len(),
            csc.values.len() as f64 / csc.n as f64,
            n_sn,
            med_nrow,
            frac_le8,
            max_front
        );
        println!(
            "  factor: {raw_us:.0} us uninstrumented (min of {reps}) | {total:.0} us instrumented \
             (probe tax {:.2}x)",
            if raw_us > 0.0 { total / raw_us } else { 0.0 }
        );
        println!("  --- split of the instrumented {total:.0} us ---");
        println!(
            "  PROLOGUE      {prologue:>9.0} us  {:>5.1}%",
            pct(prologue, total)
        );
        println!(
            "      scaling   {:>9} us  {:>5.1}%   permute {:>7} us  {:>5.1}%   (from_triplets {:>6} us => {})",
            b.scaling_us,
            pct(b.scaling_us as f64, total),
            b.permute_us,
            pct(b.permute_us as f64, total),
            b.permute_from_triplets_us,
            if b.permute_from_triplets_us > 0 {
                "COLD: cache MISS"
            } else {
                "warm: cache hit"
            }
        );
        println!(
            "      scal_pivot{:>9} us  {:>5.1}%   infnorm {:>7} us  {:>5.1}%   row_map/setup {:>5} us  {:>5.1}%",
            b.scaling_pivot_order_us,
            pct(b.scaling_pivot_order_us as f64, total),
            b.infnorm_tol_us,
            pct(b.infnorm_tol_us as f64, total),
            b.row_map_us + b.setup_us,
            pct((b.row_map_us + b.setup_us) as f64, total)
        );
        println!(
            "  LOOP          {loop_us:>9.0} us  {:>5.1}%",
            pct(loop_us, total)
        );
        println!(
            "      assembly  {asm_us:>9.0} us  {:>5.1}%   densefactor {df_us:>7.0} us  {:>5.1}%   node_tail {:>7.0} us  {:>5.1}%",
            pct(asm_us, total),
            pct(df_us, total),
            node_tail_us,
            pct(node_tail_us, total)
        );
        println!(
            "      of which arithmetic (panel+schur+scalar-tail) {arith_us:.0} us  {:.1}% of total",
            pct(arith_us, total)
        );
        println!(
            "      densefactor BOOKKEEPING (densefactor - arithmetic) {bookkeep_us:>7.0} us  {:>5.1}% of total",
            pct(bookkeep_us, total)
        );
        println!(
            "         L/D extract {lextract_us:>7.0} us {:>5.1}%   contrib extract {contribx_us:>7.0} us {:>5.1}%   (of which zero-fill {zerofill_us:>6.0} us {:>4.1}%)   unnamed {:>7.0} us {:>5.1}%",
            pct(lextract_us, total),
            pct(contribx_us, total),
            pct(zerofill_us, total),
            (bookkeep_us - lextract_us - contribx_us).max(0.0),
            pct((bookkeep_us - lextract_us - contribx_us).max(0.0), total)
        );
        println!(
            "      assembly breakdown: build_row {buildrow_us:>7.0} us {:>5.1}%   scatter {scatter_us:>7.0} us {:>5.1}%   extend_add {extendadd_us:>7.0} us {:>5.1}%",
            pct(buildrow_us, total),
            pct(scatter_us, total),
            pct(extendadd_us, total)
        );
        println!(
            "  EPILOGUE      {epilogue:>9.0} us  {:>5.1}%",
            pct(epilogue, total)
        );
        println!(
            "  DRIVER_OTHER  {driver_other:>9.0} us  {:>5.1}%",
            pct(driver_other, total)
        );
        println!(
            "  => per-supernode loop is {:.1}% of factor; non-loop is {:.1}%",
            pct(loop_us, total),
            pct(prologue + epilogue + driver_other, total)
        );
        println!();
    }
}
