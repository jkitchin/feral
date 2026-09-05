//! Issue #200 — how much of the "unattributed" per-supernode time is the
//! probe itself?
//!
//! Issue #200 attributes 33–39% of factor time to an UNATTRIBUTED bucket
//! computed as `total - ASSEMBLY - DENSEFACTOR`, using numbers gathered
//! with `phase_timing` enabled. Every one of those counters costs two
//! `Instant::now()` calls per bracketed region, and the multifrontal
//! driver brackets ~10 regions per supernode. On a KKT matrix whose
//! median front is 3×3 that is a per-front constant of the same order as
//! the residual being attributed.
//!
//! This binary measures the tax directly: the same `Solver`, the same
//! warm matrix, factored alternately with `PHASE_TIMING_ENABLED` off and
//! on, paired, reporting `min_us` per arm per the measurement protocol in
//! `dev/decisions.md` (2026-08-09).
use feral::dense::factor::{phase_timing, PHASE_TIMING_ENABLED};
use feral::symbolic::{supernode::SupernodeParams, symbolic_factorize};
use feral::{read_mtx, Solver};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut reps = 9usize;
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
    if paths.is_empty() {
        eprintln!("usage: diag_200_probe_tax [--reps N] <matrix.mtx>...");
        std::process::exit(2);
    }

    println!(
        "{:<22}{:>10}{:>10}{:>9}{:>9}{:>12}",
        "matrix", "OFF us", "ON us", "ratio", "fronts", "tax ns/front"
    );
    for p in &paths {
        let csc = read_mtx(std::path::Path::new(p)).and_then(|m| m.to_csc())?;
        let mut solver = Solver::new();

        // Warm: call 1 has no symbolic cache and (issue #56 N7) builds no
        // permute-value map; call 2 builds it. Steady state starts at 3.
        let _ = solver.factor(&csc, None);
        let _ = solver.factor(&csc, None);
        let _ = solver.factor(&csc, None);

        let (mut off, mut on) = (f64::MAX, f64::MAX);
        for _ in 0..reps {
            // OFF arm
            PHASE_TIMING_ENABLED.store(false, Relaxed);
            let t = Instant::now();
            let _ = solver.factor(&csc, None);
            off = off.min(t.elapsed().as_secs_f64() * 1e6);
            // ON arm
            PHASE_TIMING_ENABLED.store(true, Relaxed);
            let t = Instant::now();
            let _ = solver.factor(&csc, None);
            on = on.min(t.elapsed().as_secs_f64() * 1e6);
            PHASE_TIMING_ENABLED.store(false, Relaxed);
        }

        // One extra instrumented pass to read the front count.
        phase_timing::reset();
        PHASE_TIMING_ENABLED.store(true, Relaxed);
        let _ = solver.factor(&csc, None);
        PHASE_TIMING_ENABLED.store(false, Relaxed);
        // Front count comes from the symbolic structure, not a runtime
        // counter: the whole point of this probe is that runtime counters
        // in the per-front path are not free.
        let fronts = symbolic_factorize(&csc, &SupernodeParams::default())?
            .supernodes
            .len() as u64;

        let name = std::path::Path::new(p)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        println!(
            "{:<22}{:>10.0}{:>10.0}{:>9.2}{:>9}{:>12.0}",
            name,
            off,
            on,
            on / off,
            fronts,
            (on - off) * 1e3 / fronts.max(1) as f64
        );
    }
    Ok(())
}
