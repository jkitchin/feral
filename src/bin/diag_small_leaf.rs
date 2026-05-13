//! Phase 2.9 SmallLeafSubtree batching diagnostic (Step E of
//! `dev/plans/phase-2.9-small-leaf-subtree.md`).
//!
//! For a curated set of long-tail IPM matrices, runs the multifrontal
//! numeric phase with `small_leaf: Off` (current default) and
//! `small_leaf: On`, plus symbolic grouping stats, and prints
//! side-by-side timings with the computed speedup. Also shows the
//! MUMPS oracle factor_us for context.
//!
//! Usage: `cargo run --release --bin diag_small_leaf`
//!
//! Run order: each config is executed `N_REPEAT` times per matrix
//! and the minimum is reported. Symbolic factorization is done once
//! per matrix (both paths share it). Timings are for
//! `factorize_multifrontal` only — i.e. numeric phase.
//!
//! Success criterion for flipping `SmallLeafBatch::default()` to
//! `On` (Step F of the plan): geomean speedup ≥ 3× on the archetype
//! matrices, no bulk regressions > 5% in the full bench.
//!
//! This binary alone is not sufficient to flip the default — the
//! 154k-matrix bench in `cargo run --release --bin bench` is the
//! authoritative signal.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams, SmallLeafBatch};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, ZeroPivotAction};
use std::path::Path;
use std::time::Instant;

const N_REPEAT: usize = 5;

fn load_csc(path: &str) -> Option<CscMatrix> {
    let mtx = read_mtx(Path::new(path)).ok()?;
    mtx.to_csc().ok()
}

fn read_mumps_factor_us(path: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let data: serde_json::Value = serde_json::from_str(&text).ok()?;
    if data["factorization_status"].as_str() != Some("ok") {
        return None;
    }
    Some(data["factor_us"].as_u64().unwrap_or(0))
}

fn bench_path(
    csc: &CscMatrix,
    params: &NumericParams,
    sym: &feral::symbolic::SymbolicFactorization,
) -> u128 {
    // Warm-up
    let _ = factorize_multifrontal(csc, sym, params);
    let mut best = u128::MAX;
    for _ in 0..N_REPEAT {
        let t = Instant::now();
        match factorize_multifrontal(csc, sym, params) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("factor error: {}", e);
                return 0;
            }
        }
        let us = t.elapsed().as_micros();
        if us < best {
            best = us;
        }
    }
    best
}

fn params_off() -> NumericParams {
    NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        },
        scaling: Default::default(),
        small_leaf: SmallLeafBatch::Off,
        profiler: None,
        parallel_telemetry: None,
        fma: false,
        allow_delayed_pivots: true,
        cascade_break_ratio: None,
    }
}

fn params_on() -> NumericParams {
    NumericParams {
        small_leaf: SmallLeafBatch::On,
        ..params_off()
    }
}

fn report_matrix(label: &str, mtx_path: &str, mumps_path: Option<&str>) {
    let csc = match load_csc(mtx_path) {
        Some(c) => c,
        None => {
            println!("{:28} SKIP (load failed: {})", label, mtx_path);
            return;
        }
    };
    let sym = match symbolic_factorize(&csc, &SupernodeParams::default()) {
        Ok(s) => s,
        Err(e) => {
            println!("{:28} SKIP (symbolic: {})", label, e);
            return;
        }
    };

    let n_snodes = sym.supernodes.len();
    let n_groups = sym.small_leaf_groups.len();
    let n_grouped: usize = sym.snode_group.iter().filter(|g| g.is_some()).count();
    let avg_members = if n_groups > 0 {
        n_grouped as f64 / n_groups as f64
    } else {
        0.0
    };

    let off_us = bench_path(&csc, &params_off(), &sym);
    let on_us = bench_path(&csc, &params_on(), &sym);
    let mumps_us = mumps_path
        .and_then(|p| read_mumps_factor_us(Path::new(p)))
        .unwrap_or(0);

    let speedup = if on_us > 0 {
        off_us as f64 / on_us as f64
    } else {
        0.0
    };
    let vs_mumps_off = if mumps_us > 0 {
        off_us as f64 / mumps_us as f64
    } else {
        0.0
    };
    let vs_mumps_on = if mumps_us > 0 {
        on_us as f64 / mumps_us as f64
    } else {
        0.0
    };

    println!(
        "{:28} snodes={:>5} groups={:>4} grouped={:>5} avg={:>4.1}  off_us={:>8} on_us={:>8} speedup={:>4.2}x  vs_mumps_off={:>5.2}x on={:>5.2}x",
        label,
        n_snodes,
        n_groups,
        n_grouped,
        avg_members,
        off_us,
        on_us,
        speedup,
        vs_mumps_off,
        vs_mumps_on
    );
}

fn main() {
    println!("=== Phase 2.9 SmallLeafSubtree diagnostic ===");
    println!(
        "(reporting min of {} runs per config; warm-up not counted)\n",
        N_REPEAT
    );

    let targets: &[(&str, &str)] = &[
        // Archetype long-tail IPM matrices from the research note.
        ("ACOPR30_0067", "data/matrices/kkt/ACOPR30/ACOPR30_0067"),
        ("ACOPR30_0000", "data/matrices/kkt/ACOPR30/ACOPR30_0000"),
        ("CRESC100_0000", "data/matrices/kkt/CRESC100/CRESC100_0000"),
        ("HAIFAM_0082", "data/matrices/kkt/HAIFAM/HAIFAM_0082"),
        ("HAIFAM_0000", "data/matrices/kkt/HAIFAM/HAIFAM_0000"),
        // Bulk canaries (should show neutral effect, not a regression).
        ("VESUVIO_0000", "data/matrices/kkt/VESUVIO/VESUVIO_0000"),
        ("HAHN1_0000", "data/matrices/kkt/HAHN1/HAHN1_0000"),
        ("BATCH_0000", "data/matrices/kkt/BATCH/BATCH_0000"),
        ("AVION2_0000", "data/matrices/kkt/AVION2/AVION2_0000"),
    ];

    for (label, base) in targets {
        let mtx = format!("{}.mtx", base);
        let mumps = format!("{}.mumps.json", base);
        report_matrix(label, &mtx, Some(&mumps));
    }
}
