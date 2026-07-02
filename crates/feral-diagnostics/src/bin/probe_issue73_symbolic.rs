//! Issue #73 step 1: is the n>100k thin AMF-vs-MetisND question a *symbolic*
//! (ordering/fill) blowup or a *numeric* (factorization) cost difference?
//!
//! #67 bounded the AMF reroute at `n <= 100_000` because, just above the band,
//! `pinene_3200` (n≈128k) still favored AMF but `RDW2D51U` (n≈195k) did not
//! finish a single full factor+solve A/B in ~10 min. That timeout came from
//! the *full* `Solver` path (ordering + symbolic + numeric + solve), so it
//! cannot tell us whether the cost lives in the ordering/fill or in the dense
//! numeric kernels.
//!
//! This probe runs **symbolic only** — no numeric factorization, no solve —
//! for AMF and MetisND, and reports the cheap predictors that drive numeric
//! cost:
//!
//! - `sym_ms`: wall time of `symbolic_factorize_with_method` (ordering +
//!   analysis). Cheap; isolates the ordering cost itself.
//! - `nnz_L_est`: `factor_nnz_estimate` (predicted L nonzeros = fill).
//! - `max_front`: largest frontal `nrow` (dimension of the biggest dense
//!   block the numeric phase must factor).
//! - `flop_proxy`: Σ ncol·nrow² over supernodes — a dense-multifrontal work
//!   proxy (panel factor + Schur update), the dominant term in numeric
//!   wall-time.
//! - `peak_MB`: `peak_contrib_bytes` / 1e6 (peak contribution pool).
//!
//! Interpretation: if AMF's symbolic finishes in seconds with smaller
//! nnz_L/flop_proxy than MetisND, the #67 RDW2D51U timeout was numeric, and
//! AMF is the cheaper ordering there too (supporting a band extension). If
//! AMF's flop_proxy is *larger* than MetisND's, the bound is doing real work
//! and should stay.
//!
//! Run:
//!   cargo run -p feral-diagnostics --release --bin probe_issue73_symbolic -- <matrix.mtx>...
//!
//! Throwaway diagnostic, not a test.

use feral::read_mtx;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use std::time::Instant;

/// Cheap symbolic predictors for one (matrix, method) pair.
struct SymStats {
    sym_ms: f64,
    nnz_l_est: usize,
    max_front: usize,
    flop_proxy: f64,
    peak_mb: f64,
}

fn measure(
    m: &feral::sparse::csc::CscMatrix,
    method: OrderingMethod,
) -> Result<SymStats, feral::FeralError> {
    let params = SupernodeParams::default();
    let t = Instant::now();
    let sym = symbolic_factorize_with_method(m, &params, method.clone())?;
    let sym_ms = t.elapsed().as_secs_f64() * 1e3;

    let mut max_front = 0usize;
    let mut flop_proxy = 0f64;
    for s in &sym.supernodes {
        if s.nrow > max_front {
            max_front = s.nrow;
        }
        // Dense-multifrontal work proxy: panel factor + Schur update on an
        // nrow × ncol front is ~ ncol · nrow². f64 to avoid usize overflow on
        // large fronts (nrow² alone can exceed u32, and the sum exceeds u64
        // headroom only in extreme cases — f64 is the safe accumulator here).
        flop_proxy += s.ncol as f64 * (s.nrow as f64) * (s.nrow as f64);
    }

    Ok(SymStats {
        sym_ms,
        nnz_l_est: sym.factor_nnz_estimate,
        max_front,
        flop_proxy,
        peak_mb: sym.peak_contrib_bytes as f64 / 1e6,
    })
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: probe_issue73_symbolic <matrix.mtx>...");
        eprintln!("  symbolic-only AMF vs MetisND predictors for n>100k thin (#73)");
        return;
    }

    println!(
        "{:<22} {:>9} {:>7} {:>12} {:>10} {:>13} {:>10}",
        "matrix/method", "n", "sym_ms", "nnz_L_est", "max_front", "flop_proxy", "peak_MB"
    );

    for p in &paths {
        let pb = std::path::Path::new(p);
        let name = pb.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let m = match read_mtx(pb).and_then(|x| x.to_csc()) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("skip {name}: read/convert failed: {e:?}");
                continue;
            }
        };

        let mut amf_nnz = 0f64;
        let mut metis_nnz = 0f64;
        let mut amf_flop = 0f64;
        let mut metis_flop = 0f64;
        for (label, method) in [
            ("AMF", OrderingMethod::Amf),
            ("MetisND", OrderingMethod::MetisND),
        ] {
            match measure(&m, method) {
                Ok(s) => {
                    println!(
                        "{:<22} {:>9} {:>7.1} {:>12} {:>10} {:>13.3e} {:>10.1}",
                        format!("{name}/{label}"),
                        m.n,
                        s.sym_ms,
                        s.nnz_l_est,
                        s.max_front,
                        s.flop_proxy,
                        s.peak_mb,
                    );
                    if label == "AMF" {
                        amf_nnz = s.nnz_l_est as f64;
                        amf_flop = s.flop_proxy;
                    } else {
                        metis_nnz = s.nnz_l_est as f64;
                        metis_flop = s.flop_proxy;
                    }
                }
                Err(e) => {
                    eprintln!("{name}/{label}: symbolic failed: {e:?}");
                }
            }
        }
        // Ratios MetisND/AMF: >1 means AMF is the cheaper ordering (the #67
        // hypothesis above the band). <1 means MetisND wins and the bound earns
        // its keep.
        if amf_nnz > 0.0 && metis_nnz > 0.0 {
            println!(
                "  -> MetisND/AMF  nnz_L {:.3}x  flop_proxy {:.3}x",
                metis_nnz / amf_nnz,
                metis_flop / amf_flop,
            );
        }
    }
}
