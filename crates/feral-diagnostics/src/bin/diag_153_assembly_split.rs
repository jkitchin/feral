//! Issue #153 — which passes over the frontal matrix make up the
//! non-kernel part of the supernode loop?
//!
//! `diag_153_kernel_headroom` shows the loop runs 1.3-1.9x slower than
//! feral's own dense kernel on the identical shape sequence, and that
//! the excess is proportional to **front area** (a flat ~0.5 ns per
//! `nrow^2` element on the large-front matrices, 2-3 ns on the 17-32
//! row fronts) rather than to the front *count*. Area-proportional
//! means passes over the frontal matrix. This binary names them.
//!
//! It reports the assembly phase counters in **absolute ns per front
//! and per front element**, never as a percentage of the driver wall.
//! That distinction is the whole point: `PHASE_TIMING_ENABLED` costs
//! 350-430 ns per front (`diag_200_probe_tax`), which inflates the
//! *denominator* badly enough that `diag_factor_phases`' percentage
//! columns cannot be compared against a probe-free measurement. Each
//! individual counter, though, is inflated only by its own
//! `Instant::now()` pair — tens of ns — so the absolute per-front
//! numbers here are usable, and the per-counter floor is printed
//! alongside so the reader can see how much of a small number is probe.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin diag_153_assembly_split \
//!     -- [--reps N] <matrix.mtx>...
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

/// Cost of one `Instant::now()` pair on this machine, in ns — the floor
/// under every per-front counter below.
fn probe_floor() -> f64 {
    let n = 200_000;
    let t0 = Instant::now();
    let mut acc = 0u64;
    for _ in 0..n {
        let a = Instant::now();
        acc = acc.wrapping_add(a.elapsed().as_nanos() as u64);
    }
    std::hint::black_box(acc);
    t0.elapsed().as_nanos() as f64 / n as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut reps = 5usize;
    let mut paths: Vec<String> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--reps" {
            if let Some(v) = it.next() {
                reps = v.parse()?;
            }
        } else {
            paths.push(a.clone());
        }
    }
    println!(
        "probe floor = {:.0} ns per Instant::now() pair",
        probe_floor()
    );
    println!(
        "{:<22}{:>8}{:>12}{:>10}{:>10}{:>10}{:>10}{:>10}",
        "matrix", "snodes", "sum nrow^2", "buildrow", "scatter", "extendadd", "zerofill", "cbextr"
    );
    println!("{:<22}{:>8}{:>12}{:>50}", "", "", "", "-- ns per front --");

    for p in &paths {
        let csc = match read_mtx(Path::new(p)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {p}: {e:?}");
                continue;
            }
        };
        let label = Path::new(p)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let sp = SupernodeParams::default();
        let symbolic = match symbolic_factorize_with_method(&csc, &sp, OrderingMethod::Auto) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{label}: symbolic failed: {e:?}");
                continue;
            }
        };
        let nparams = NumericParams::default();
        let mut ws = FactorWorkspace::new();
        if factorize_multifrontal_supernodal_with_workspace(&csc, &symbolic, &nparams, &mut ws)
            .is_err()
        {
            eprintln!("{label}: warm-up failed");
            continue;
        }

        let mut best = u64::MAX;
        let mut keep = [0f64; 6];
        let mut nsn = 0usize;
        let mut area = 0f64;
        for _ in 0..reps {
            PHASE_TIMING_ENABLED.store(true, Relaxed);
            phase_timing::reset();
            let prof = Arc::new(Mutex::new(Profiler::new()));
            let mut np = nparams.clone();
            np.profiler = Some(prof.clone());
            np.pattern_reused_hint = true;
            let t0 = Instant::now();
            let res =
                factorize_multifrontal_supernodal_with_workspace(&csc, &symbolic, &np, &mut ws);
            let wall = t0.elapsed().as_nanos() as u64;
            PHASE_TIMING_ENABLED.store(false, Relaxed);
            if res.is_err() {
                eprintln!("{label}: timed factorization failed");
                break;
            }
            let (br, sc, xa, _lex, cbex) = phase_timing::snapshot_detail();
            let zf = phase_timing::snapshot_contrib_zerofill();
            let guard = match prof.lock() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("{label}: profiler poisoned: {e}");
                    break;
                }
            };
            if wall < best {
                best = wall;
                nsn = guard.len();
                area = guard
                    .timings()
                    .iter()
                    .map(|t| (t.nrow * t.nrow) as f64)
                    .sum();
                keep = [br as f64, sc as f64, xa as f64, zf as f64, cbex as f64, 0.0];
            }
        }
        if best == u64::MAX || nsn == 0 {
            continue;
        }
        let f = nsn as f64;
        println!(
            "{:<22}{:>8}{:>12.3e}{:>10.0}{:>10.0}{:>10.0}{:>10.0}{:>10.0}",
            label,
            nsn,
            area,
            keep[0] / f,
            keep[1] / f,
            keep[2] / f,
            keep[3] / f,
            keep[4] / f
        );
        println!(
            "{:<22}{:>8}{:>12}{:>10.3}{:>10.3}{:>10.3}{:>10.3}{:>10.3}   ns/element",
            "",
            "",
            "",
            keep[0] / area,
            keep[1] / area,
            keep[2] / area,
            keep[3] / area,
            keep[4] / area
        );
    }
    Ok(())
}
