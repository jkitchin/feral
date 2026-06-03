//! Issue #64 probe: confirm the arrow/bordered fill ranking on the real
//! r05 KKT and gather degree-distribution statistics for the candidate
//! arrow detector, across r05 and the other large/*.mtx fixtures.
//!
//! Run: `cargo run --release --bin probe_issue64_arrow`
//!
//! Throwaway diagnostic — not a test. Reads gitignored
//! `tests/data/large/*.mtx`.

use feral::sparse::csc::CscPattern;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, CscMatrix};
use std::path::{Path, PathBuf};

/// Candidate predicate stats on the full symmetric pattern.
struct ArrowStats {
    n: usize,
    full_nnz: usize,
    avg_deg: f64,
    max_deg: usize,
    threshold: usize,
    heavy_count: usize,
    heavy_nnz: usize,
}

fn arrow_stats(pat: &CscPattern) -> ArrowStats {
    let n = pat.n;
    let full_nnz = pat.row_idx.len();
    let avg_deg = full_nnz as f64 / n.max(1) as f64;
    // deg > max(64, 8*avg_deg)
    let threshold = ((8.0 * avg_deg).ceil() as usize).max(64);
    let mut max_deg = 0usize;
    let mut heavy_count = 0usize;
    let mut heavy_nnz = 0usize;
    for j in 0..n {
        let deg = pat.col_ptr[j + 1] - pat.col_ptr[j];
        max_deg = max_deg.max(deg);
        if deg > threshold {
            heavy_count += 1;
            heavy_nnz += deg;
        }
    }
    ArrowStats {
        n,
        full_nnz,
        avg_deg,
        max_deg,
        threshold,
        heavy_count,
        heavy_nnz,
    }
}

fn nnz_l(m: &CscMatrix, method: OrderingMethod) -> usize {
    let sp = SupernodeParams::default();
    match symbolic_factorize_with_method(m, &sp, method) {
        Ok(s) => s.col_counts.iter().sum(),
        Err(e) => {
            eprintln!("  {:?} FAILED: {:?}", method, e);
            0
        }
    }
}

fn report(label: &str, m: &CscMatrix, with_fill: bool) {
    let pat = m.symmetric_pattern();
    let s = arrow_stats(&pat);
    let hc_frac = s.heavy_count as f64 / s.n as f64;
    let hn_frac = s.heavy_nnz as f64 / s.full_nnz.max(1) as f64;
    println!("=== {label} ===");
    println!(
        "  n={} full_nnz={} avg_deg={:.2} max_deg={} thr={}",
        s.n, s.full_nnz, s.avg_deg, s.max_deg, s.threshold
    );
    println!(
        "  heavy_count={} ({:.3}% of n)  heavy_nnz={} ({:.1}% of full_nnz)",
        s.heavy_count,
        hc_frac * 100.0,
        s.heavy_nnz,
        hn_frac * 100.0
    );
    if with_fill {
        let amf = nnz_l(m, OrderingMethod::Amf);
        let amd = nnz_l(m, OrderingMethod::Amd);
        let metis = nnz_l(m, OrderingMethod::MetisND);
        println!(
            "  nnz_L: Amf={} Amd={} MetisND={}  (MetisND/Amf={:.2}x)",
            amf,
            amd,
            metis,
            metis as f64 / amf.max(1) as f64
        );
    }
}

fn main() {
    let dir = PathBuf::from("tests/data/large");

    // r05: the target arrow matrix. Full fill ranking.
    let r05 = dir.join("r05_kkt.mtx");
    if r05.is_file() {
        match read_mtx(&r05).and_then(|m| m.to_csc()) {
            Ok(m) => report("r05_kkt (ARROW target)", &m, true),
            Err(e) => eprintln!("r05_kkt read failed: {:?}", e),
        }
    } else {
        eprintln!("SKIP r05_kkt.mtx (not present)");
    }

    // The other large fixtures — these must NOT look like arrows
    // (false-positive check). Fill ranking too, where cheap.
    for name in ["bratu3d", "cont-201", "bcsstk38"] {
        let p = dir.join(format!("{name}.mtx"));
        if p.is_file() {
            match read_mtx(&p).and_then(|m| m.to_csc()) {
                Ok(m) => report(name, &m, true),
                Err(e) => eprintln!("{name} read failed: {:?}", e),
            }
        } else {
            eprintln!("SKIP {name}.mtx (not present)");
        }
    }

    // PoissonControl K=58 (n=10092, avg_deg~2.67): the matrix that the
    // issue_3 test pins to MetisND. Build it via the same generator the
    // tests use, if available as a binary helper — otherwise skipped.
    let _ = Path::new("");
}
