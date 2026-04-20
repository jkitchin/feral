//! HS85_0022 diagnostic — decompose where the ~80× factor-ratio
//! recorded against MUMPS in the post-D.3 stage-3 bench
//! (`dev/results/lever-d3/bench-post-d3-2026-04-19.txt`) comes from.
//!
//! Reports:
//!   1. gate decision and input shape (n, nnz_lower, density ρ, threshold)
//!   2. symbolic structure (num supernodes, etree height, peak contrib bytes)
//!   3. phase breakdown of the multifrontal path
//!   4. cold single-shot timing (analogous to `bench.rs`) vs warm median
//!
//! Single-matrix probe. Run with `cargo run --release --bin hs85_diag`.

use feral::numeric::factorize::{
    factorize_multifrontal, factorize_multifrontal_supernodal_with_workspace,
    factorize_multifrontal_with_workspace, should_use_dense_fast_path, FactorWorkspace,
};
use feral::scaling::compute_scaling;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, NumericParams, ZeroPivotAction};
use std::path::PathBuf;
use std::time::Instant;

const N_ITERS: u32 = 2000;
const MATRIX: &str = "data/matrices/kkt/HS85/HS85_0022.mtx";

fn params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    })
}

fn median_ns<F: FnMut() -> u128>(mut f: F) -> u128 {
    let mut samples: Vec<u128> = (0..N_ITERS).map(|_| f()).collect();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn etree_height(parent: &[Option<usize>]) -> usize {
    let n = parent.len();
    let mut max_h = 0usize;
    for i in 0..n {
        let mut d = 0usize;
        let mut cur = i;
        for _ in 0..n {
            match parent[cur] {
                Some(p) => {
                    d += 1;
                    cur = p;
                }
                None => break,
            }
        }
        if d > max_h {
            max_h = d;
        }
    }
    max_h
}

fn main() {
    println!("HS85_0022 diagnostic (median of {} iters)", N_ITERS);
    println!();

    let path = PathBuf::from(MATRIX);
    let mtx = read_mtx(&path).expect("read_mtx");
    let csc: CscMatrix = mtx.to_csc().expect("to_csc");
    let n = csc.n;
    let nnz_lower = csc.row_idx.len();
    let lower_cells = n * (n + 1) / 2;
    let density = nnz_lower as f64 / lower_cells as f64;
    let threshold_nnz = lower_cells.div_ceil(4); // ρ_MIN = 1/4
    let gate = should_use_dense_fast_path(n, nnz_lower);

    println!("Input shape");
    println!("  n                 = {}", n);
    println!("  nnz_lower         = {}", nnz_lower);
    println!("  lower_cells       = {}", lower_cells);
    println!("  density ρ         = {:.4}", density);
    println!("  gate threshold    = {} (ρ_MIN = 0.25)", threshold_nnz);
    println!(
        "  should_use_dense  = {}  ({})",
        gate,
        if gate {
            "DENSE FAST-PATH"
        } else {
            "MULTIFRONTAL"
        }
    );
    println!();

    let p = params();
    let sn = SupernodeParams::default();

    // Symbolic structure.
    let sym = symbolic_factorize(&csc, &sn).expect("symbolic");
    let n_supernodes = sym.supernodes.len();
    let etree_h = etree_height(&sym.etree.parent);
    let peak_contrib_kb = sym.peak_contrib_bytes as f64 / 1024.0;
    println!("Symbolic structure");
    println!("  supernodes        = {}", n_supernodes);
    println!("  etree height      = {}", etree_h);
    println!("  factor_nnz_est    = {}", sym.factor_nnz_estimate);
    println!("  peak_contrib      = {:.1} KiB", peak_contrib_kb);
    println!();

    // Phase timings (warm, median).
    let sym_ns = median_ns(|| {
        let t = Instant::now();
        let _s = symbolic_factorize(&csc, &sn).expect("symbolic");
        t.elapsed().as_nanos()
    });

    let scale_ns = median_ns(|| {
        let t = Instant::now();
        let _ = compute_scaling(&csc, &p.scaling).expect("scaling");
        t.elapsed().as_nanos()
    });

    // Cold workspace per call (matches `factorize_multifrontal` semantics).
    let numeric_cold_ws_ns = median_ns(|| {
        let t = Instant::now();
        let _ = factorize_multifrontal(&csc, &sym, &p).expect("numeric");
        t.elapsed().as_nanos()
    });

    // Warm workspace (post-D.1 amortised mode).
    let mut ws = FactorWorkspace::new();
    // Prime the workspace once so its internal buffers are sized.
    let _ = factorize_multifrontal_with_workspace(&csc, &sym, &p, &mut ws).expect("numeric prime");
    let numeric_warm_ws_ns = median_ns(|| {
        let t = Instant::now();
        let _ = factorize_multifrontal_supernodal_with_workspace(&csc, &sym, &p, &mut ws)
            .expect("numeric warm");
        t.elapsed().as_nanos()
    });

    // Full cold pipeline (symbolic + scaling + numeric), analogous to
    // what `bench.rs` measures on a first call.
    let full_cold_ns = median_ns(|| {
        let t = Instant::now();
        let sym2 = symbolic_factorize(&csc, &sn).expect("symbolic");
        let _ = factorize_multifrontal(&csc, &sym2, &p).expect("numeric");
        t.elapsed().as_nanos()
    });

    println!("Multifrontal phase timings (warm, median)");
    let fmt = |ns: u128| format!("{:>7.2} μs", ns as f64 / 1000.0);
    println!("  symbolic                   = {}", fmt(sym_ns));
    println!("  compute_scaling            = {}", fmt(scale_ns));
    println!("  numeric (cold workspace)   = {}", fmt(numeric_cold_ws_ns));
    println!("  numeric (warm workspace)   = {}", fmt(numeric_warm_ws_ns));
    println!("  full cold (sym + numeric)  = {}", fmt(full_cold_ns));
    println!();

    // Phase share against full cold pipeline.
    let total = full_cold_ns.max(1);
    let pct = |ns: u128| 100.0 * ns as f64 / total as f64;
    println!("Phase share (vs full cold, %)");
    println!("  symbolic                   = {:5.1} %", pct(sym_ns));
    println!(
        "  numeric (cold ws)          = {:5.1} %",
        pct(numeric_cold_ws_ns)
    );
    // Numeric may double-count scaling if it runs compute_scaling internally;
    // print scale as a standalone reference only.
    println!("  compute_scaling (ref only) = {:5.1} %", pct(scale_ns));
    println!();

    // Cold single-shot timings (analogous to bench harness's per-matrix measurement).
    // Do several independent cold runs and print min/median/max to bracket the 1845 μs.
    const COLD_REPS: usize = 50;
    let mut cold_samples: Vec<u128> = Vec::with_capacity(COLD_REPS);
    for _ in 0..COLD_REPS {
        let t = Instant::now();
        let sym2 = symbolic_factorize(&csc, &sn).expect("symbolic");
        let _ = factorize_multifrontal(&csc, &sym2, &p).expect("numeric");
        cold_samples.push(t.elapsed().as_nanos());
    }
    cold_samples.sort_unstable();
    let cold_min = cold_samples[0];
    let cold_p50 = cold_samples[COLD_REPS / 2];
    let cold_p90 = cold_samples[(COLD_REPS * 9) / 10];
    let cold_max = *cold_samples.last().expect("cold_samples non-empty");
    println!(
        "Cold single-shot wall time (full pipeline, {} reps)",
        COLD_REPS
    );
    println!("  min    = {}", fmt(cold_min));
    println!("  p50    = {}", fmt(cold_p50));
    println!("  p90    = {}", fmt(cold_p90));
    println!("  max    = {}", fmt(cold_max));
    println!();
    println!("Recorded bench.rs result for HS85_0022: 1845 μs feral vs 23 μs MUMPS (80×).");
}
