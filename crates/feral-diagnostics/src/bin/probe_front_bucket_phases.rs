//! probe_front_bucket_phases — front-size histogram weighted by **loop
//! time**, with the phase breakdown computed inside each size bucket.
//!
//! Issue #153 item 2. The existing instruments answer half of this each:
//! `profile_supernode_distribution` buckets loop time by `nrow` but on a
//! hardcoded matrix list and with no phase split, and `diag_factor_phases`
//! splits phases but aggregates over every front. Neither can say *which
//! size class* a phase's time sits in, which is the question.
//!
//! What it found, and what that killed: dtoc1nd's cost is 148 fronts of
//! mean shape `ncol = 62`, `nrow = 88` — 91% of loop time — and inside
//! them the panel factor is 53% while every other #153 fixture is panel
//! 6-15% / Schur 17-30%. The hypothesis this probe was written to test,
//! that the regime falls below the packed-SIMD work gate
//! (`PACKED_SIMD_MIN_WORK = 1024`, `src/dense/factor.rs`), is **false**:
//! `n_elim · (nrow − col_start) · ncol ≈ 3.4e5` clears it by 330×, and
//! `FERAL_PACKED_SIMD_MIN_WORK=0` changes nothing while `=1e12` costs
//! 15.7%. See `dev/research/issue-153-dtoc1nd-dense-front-2026-08-19.md`.
//!
//! Method: sequential supernodal driver (the per-supernode phase deltas
//! are read from process-global counters, so a parallel driver would
//! interleave them), warm workspace, `N_REPS` runs, the run with median
//! `loop_ns` reported. Buckets are the `Profiler::report()` ranges so the
//! numbers are comparable with every prior probe: <=8, 9-16, 17-32,
//! 33-64, 65-128, >128.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin probe_front_bucket_phases \
//!       -- <a.mtx> [b.mtx ...]
//!
//! Env:
//!   FERAL_BUCKET_REPS=5     repetitions per matrix (median run reported)
//!   FERAL_BUCKET_BY=ncol    bucket by `ncol` instead of `nrow` (default nrow)
//!   FERAL_BUCKET_BS=64      `BunchKaufmanParams::block_size` (default 64)
//!
//! Phase columns: `panel%`, `schur%` and `tail%` are the three parts
//! of `dense%` — the left-looking BLAS-2 panel factor, the blocked
//! BLAS-3 trailing update, and the scalar tail that finishes columns
//! the panel could not eliminate. They are reported separately because
//! `block_size` trades the first two against each other without
//! changing the front's total work, so only the wall clock next to
//! them says whether a trade paid.
//!
//! Note on totals: the phase counters are sampled per supernode, so
//! `assembly + densefactor` is less than the bucket's wall by the
//! per-node driver remainder. That gap is reported as `other%` rather
//! than hidden — an optimization aimed at a named phase cannot pay for
//! itself if the unnamed remainder is bigger than the phase.

use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};

use feral::dense::factor::PHASE_TIMING_ENABLED;
use feral::numeric::factorize::{
    factorize_multifrontal_supernodal_with_workspace, FactorWorkspace, NumericParams, Profiler,
    SupernodeTiming,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, ZeroPivotAction};

const RANGES: &[(&str, usize, usize)] = &[
    ("<=8", 0, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    (">128", 129, usize::MAX),
];

#[derive(Default, Clone)]
struct Bucket {
    count: usize,
    ns: u64,
    assembly_ns: u64,
    densefactor_ns: u64,
    panel_ns: u64,
    schur_ns: u64,
    scalartail_ns: u64,
    /// Summed `ncol * nrow * nrow`, the per-front dense-cost proxy used
    /// by `probe_front_concentration`. Reported next to measured time so
    /// a bucket whose share of flops and share of time disagree is
    /// visible rather than inferred.
    flops: f64,
    /// Summed `ncol` and `nrow`, for the bucket's mean front shape. A
    /// nearly-square front spends its time in the panel; a wide-and-thin
    /// one spends it in the trailing Schur update, so the shape is what
    /// makes a phase share interpretable.
    sum_ncol: usize,
    sum_nrow: usize,
}

fn bucket_of(t: &SupernodeTiming, by_ncol: bool) -> usize {
    let key = if by_ncol { t.ncol } else { t.nrow };
    RANGES
        .iter()
        .position(|&(_, lo, hi)| key >= lo && key <= hi)
        .unwrap_or(RANGES.len() - 1)
}

fn ldlt_params(profiler: Arc<Mutex<Profiler>>, block_size: usize) -> NumericParams {
    NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            block_size,
            ..BunchKaufmanParams::default()
        },
        profiler: Some(profiler),
        ..NumericParams::default()
    }
}

fn load_csc(path: &str) -> Option<CscMatrix> {
    match read_mtx(Path::new(path)).and_then(|m| m.to_csc()) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("{path}: load failed: {e}");
            None
        }
    }
}

/// One run's per-supernode timings, or `None` if the factor failed.
fn run_once(
    csc: &CscMatrix,
    sym: &feral::symbolic::SymbolicFactorization,
    ws: &mut FactorWorkspace,
    block_size: usize,
) -> Option<Vec<SupernodeTiming>> {
    let prof = Arc::new(Mutex::new(Profiler::new()));
    let params = ldlt_params(Arc::clone(&prof), block_size);
    if let Err(e) = factorize_multifrontal_supernodal_with_workspace(csc, sym, &params, ws) {
        eprintln!("factor failed: {e}");
        return None;
    }
    // Bind the lock result before the match so the temporary guard is
    // dropped before `prof` is, rather than after it (E0597).
    let timings = prof.lock().map(|p| p.timings().to_vec());
    match timings {
        Ok(t) => Some(t),
        Err(_) => {
            eprintln!("profiler mutex poisoned");
            None
        }
    }
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

fn report(label: &str, timings: &[SupernodeTiming], by_ncol: bool, block_size: usize) {
    let mut buckets = vec![Bucket::default(); RANGES.len()];
    for t in timings {
        let b = &mut buckets[bucket_of(t, by_ncol)];
        b.count += 1;
        b.ns += t.ns;
        b.assembly_ns += t.assembly_ns;
        b.densefactor_ns += t.densefactor_ns;
        b.panel_ns += t.panelfactor_ns;
        b.schur_ns += t.schur_ns;
        b.scalartail_ns += t.scalartail_ns;
        b.flops += t.ncol as f64 * t.nrow as f64 * t.nrow as f64;
        b.sum_ncol += t.ncol;
        b.sum_nrow += t.nrow;
    }
    let loop_ns: u64 = buckets.iter().map(|b| b.ns).sum();
    let total_flops: f64 = buckets.iter().map(|b| b.flops).sum();

    println!(
        "=== {label} (fronts={}, loop={:.3} ms, bucketed by {}, bs={block_size}) ===",
        timings.len(),
        loop_ns as f64 / 1e6,
        if by_ncol { "ncol" } else { "nrow" }
    );
    println!(
        "{:>8}{:>9}{:>11}{:>9}{:>9}{:>11}{:>8}{:>8}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}",
        "bucket",
        "count",
        "time_ms",
        "time%",
        "flop%",
        "avg_ns",
        "ncol",
        "nrow",
        "asm%",
        "dense%",
        "panel%",
        "schur%",
        "tail%",
        "other%"
    );
    for (i, &(name, _, _)) in RANGES.iter().enumerate() {
        let b = &buckets[i];
        if b.count == 0 {
            continue;
        }
        // `other` is the per-node driver remainder: the bucket's wall
        // minus the two phases that partition it.
        let named = b.assembly_ns.saturating_add(b.densefactor_ns);
        let other = b.ns.saturating_sub(named);
        println!(
            "{:>8}{:>9}{:>11.3}{:>9.1}{:>9.1}{:>11.0}{:>8.0}{:>8.0}{:>9.1}{:>9.1}{:>9.1}{:>9.1}{:>9.1}{:>9.1}",
            name,
            b.count,
            b.ns as f64 / 1e6,
            pct(b.ns, loop_ns),
            if total_flops > 0.0 {
                100.0 * b.flops / total_flops
            } else {
                0.0
            },
            b.ns as f64 / b.count as f64,
            b.sum_ncol as f64 / b.count as f64,
            b.sum_nrow as f64 / b.count as f64,
            pct(b.assembly_ns, b.ns),
            pct(b.densefactor_ns, b.ns),
            pct(b.panel_ns, b.ns),
            pct(b.schur_ns, b.ns),
            pct(b.scalartail_ns, b.ns),
            pct(other, b.ns),
        );
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: probe_front_bucket_phases <a.mtx> [b.mtx ...]");
        std::process::exit(2);
    }
    let reps: usize = feral::env::usize_var("FERAL_BUCKET_REPS")
        .unwrap_or(5)
        .max(1);
    let by_ncol = matches!(std::env::var("FERAL_BUCKET_BY").as_deref(), Ok("ncol"));
    // Panel width, so the phase shares can be read *as a function of*
    // the lever. A `panel%` that does not move when `block_size` is
    // cut from 64 to 8 says the phase attribution, not the panel
    // kernel, is what needs explaining.
    let block_size: usize = feral::env::usize_var("FERAL_BUCKET_BS")
        .unwrap_or(BunchKaufmanParams::default().block_size)
        .max(1);

    PHASE_TIMING_ENABLED.store(true, Relaxed);

    for path in &args {
        let Some(csc) = load_csc(path) else { continue };
        let sym = match symbolic_factorize(&csc, &SupernodeParams::default()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{path}: symbolic failed: {e}");
                continue;
            }
        };
        let mut ws = FactorWorkspace::default();
        // Warm the workspace and the caches: the IPM host's steady state
        // is a warm re-factor, and a cold first call would put its
        // allocation cost in whichever bucket ran first.
        if run_once(&csc, &sym, &mut ws, block_size).is_none() {
            continue;
        }

        let mut runs: Vec<Vec<SupernodeTiming>> = Vec::with_capacity(reps);
        for _ in 0..reps {
            if let Some(t) = run_once(&csc, &sym, &mut ws, block_size) {
                runs.push(t);
            }
        }
        if runs.is_empty() {
            continue;
        }
        runs.sort_by_key(|r| r.iter().map(|t| t.ns).sum::<u64>());
        let median = &runs[runs.len() / 2];

        let label = Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        report(label, median, by_ncol, block_size);
    }
}
