//! Phase 2.6.5 corpus bench: compare symbolic + numeric factorization
//! with `OrderingPreprocess::None` vs `OrderingPreprocess::LdltCompress`
//! on the high-leverage tail of the KKT corpus (worst-10 ratio matrices
//! from the `diag_mc64_cycles` survey) plus a random sample across the
//! full histogram to detect regressions outside the target bucket.
//!
//! Reports per-matrix deltas on:
//!   - `symbolic_us` (compression adds MC64 + graph contract cost)
//!   - `factor_nnz_estimate` (the fill-prediction output of symbolic)
//!   - `factor_us` (the headline metric — wins here are what justify
//!     flipping the default)
//!   - inertia parity (MUST be 0 mismatches)
//!
//! Decision criterion (from `dev/plans/phase-2.6.5-ldlt-compressed-graph.md`
//! Step 8): geomean factor_us improvement >= 5% on the compRat <= 0.7
//! bucket AND 0 inertia regressions AND no residual regressions worse
//! than the existing `PartialSingular` tolerance.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, OrderingPreprocess, SupernodeParams};
use feral::{read_mtx, CscMatrix};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn collect(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().map(|e| e == "mtx").unwrap_or(false) {
            out.push(p);
        }
    }
}

struct Row {
    matrix: String,
    n: usize,
    sym_base_us: f64,
    sym_cmp_us: f64,
    nnz_base: usize,
    nnz_cmp: usize,
    fac_base_us: f64,
    fac_cmp_us: f64,
    inertia_match: bool,
}

fn bench_one(name: &str, csc: &CscMatrix) -> Option<Row> {
    let np = NumericParams::default();

    let base_params = SupernodeParams {
        preprocess: OrderingPreprocess::None,
        small_leaf: Default::default(),
        ..Default::default()
    };
    let cmp_params = SupernodeParams {
        preprocess: OrderingPreprocess::LdltCompress,
        small_leaf: Default::default(),
        ..Default::default()
    };

    // Warmup — one factor of each to absorb allocator cost.
    let _ = symbolic_factorize(csc, &base_params).ok()?;

    // Symbolic timing.
    let t = Instant::now();
    let sym_base = symbolic_factorize(csc, &base_params).ok()?;
    let sym_base_us = t.elapsed().as_secs_f64() * 1e6;

    let t = Instant::now();
    let sym_cmp = symbolic_factorize(csc, &cmp_params).ok()?;
    let sym_cmp_us = t.elapsed().as_secs_f64() * 1e6;

    // Numeric timing.
    let t = Instant::now();
    let (_fb, ib) = factorize_multifrontal(csc, &sym_base, &np).ok()?;
    let fac_base_us = t.elapsed().as_secs_f64() * 1e6;

    let t = Instant::now();
    let (_fc, ic) = factorize_multifrontal(csc, &sym_cmp, &np).ok()?;
    let fac_cmp_us = t.elapsed().as_secs_f64() * 1e6;

    Some(Row {
        matrix: name.to_string(),
        n: csc.n,
        sym_base_us,
        sym_cmp_us,
        nnz_base: sym_base.factor_nnz_estimate,
        nnz_cmp: sym_cmp.factor_nnz_estimate,
        fac_base_us,
        fac_cmp_us,
        inertia_match: ib == ic,
    })
}

fn print_row(r: &Row) {
    let sym_delta = (r.sym_cmp_us - r.sym_base_us) / r.sym_base_us * 100.0;
    let nnz_delta = (r.nnz_cmp as f64 - r.nnz_base as f64) / (r.nnz_base as f64) * 100.0;
    let fac_delta = (r.fac_cmp_us - r.fac_base_us) / r.fac_base_us * 100.0;
    println!(
        "{:28} n={:>6}  sym Δ{:>+6.1}%  nnz Δ{:>+6.1}%  fac Δ{:>+6.1}%  inertia={}",
        r.matrix,
        r.n,
        sym_delta,
        nnz_delta,
        fac_delta,
        if r.inertia_match { "MATCH" } else { "MISMATCH" }
    );
}

fn geomean_ratio(pairs: &[(f64, f64)]) -> f64 {
    // Geometric mean of (cmp / base) ratios. < 1.0 means compression wins.
    if pairs.is_empty() {
        return 1.0;
    }
    let mut log_sum = 0.0f64;
    let mut count = 0;
    for &(base, cmp) in pairs {
        if base > 0.0 && cmp > 0.0 {
            log_sum += (cmp / base).ln();
            count += 1;
        }
    }
    if count == 0 {
        return 1.0;
    }
    (log_sum / count as f64).exp()
}

fn main() {
    let worst_ten = [
        "MUONSINE/MUONSINE_0000",
        "CRESC100/CRESC100_0000",
        "KIRBY2/KIRBY2_0007",
        "HAHN1/HAHN1_0259",
        "KIRBY2/KIRBY2_0006",
        "KIRBY2/KIRBY2_0008",
        "GAUSS2/GAUSS2_0000",
        "VESUVIO/VESUVIO_0011",
        "VESUVIO/VESUVIO_0019",
        "VESUVIO/VESUVIO_0013",
    ];

    let mut rows: Vec<Row> = Vec::new();

    println!("=== High-leverage tail (compRat 0.66-0.75) ===");
    for name in worst_ten.iter() {
        let path = format!("data/matrices/kkt/{}.mtx", name);
        let Ok(mtx) = read_mtx(Path::new(&path)) else {
            continue;
        };
        let Ok(csc) = mtx.to_csc() else { continue };
        if let Some(row) = bench_one(name, &csc) {
            print_row(&row);
            rows.push(row);
        }
    }

    // Stratified sample: walk a deterministic stride through the corpus
    // and keep every Nth matrix, so the sample covers the full histogram
    // (including the 53% compRat=1.0 bucket where compression is a no-op
    // and we need to prove we don't regress).
    let mut matrices = Vec::new();
    collect(Path::new("data/matrices/kkt"), &mut matrices);
    matrices.sort();
    let sample_stride = std::env::var("SAMPLE_STRIDE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(500);

    println!();
    println!(
        "=== Random-stride sample (stride={}, {} matrices) ===",
        sample_stride,
        matrices.len() / sample_stride.max(1)
    );
    for (k, path) in matrices.iter().enumerate() {
        if k % sample_stride != 0 {
            continue;
        }
        let Ok(mtx) = read_mtx(path) else { continue };
        let Ok(csc) = mtx.to_csc() else { continue };
        if csc.n == 0 {
            continue;
        }
        let name = path
            .strip_prefix("data/matrices/kkt/")
            .unwrap_or(path)
            .with_extension("")
            .to_string_lossy()
            .to_string();
        if let Some(row) = bench_one(&name, &csc) {
            print_row(&row);
            rows.push(row);
        }
    }

    // Summary.
    let inertia_mismatches = rows.iter().filter(|r| !r.inertia_match).count();
    let sym_pairs: Vec<(f64, f64)> = rows.iter().map(|r| (r.sym_base_us, r.sym_cmp_us)).collect();
    let nnz_pairs: Vec<(f64, f64)> = rows
        .iter()
        .map(|r| (r.nnz_base as f64, r.nnz_cmp as f64))
        .collect();
    let fac_pairs: Vec<(f64, f64)> = rows.iter().map(|r| (r.fac_base_us, r.fac_cmp_us)).collect();

    let gm_sym = geomean_ratio(&sym_pairs);
    let gm_nnz = geomean_ratio(&nnz_pairs);
    let gm_fac = geomean_ratio(&fac_pairs);

    println!();
    println!("=== Summary ({} matrices) ===", rows.len());
    println!("inertia mismatches: {}", inertia_mismatches);
    println!(
        "geomean cmp/base: sym={:.3} nnz={:.3} fac={:.3}",
        gm_sym, gm_nnz, gm_fac
    );
    println!(
        "(ratio < 1.0 = compression wins; >= 0.95 = <5% change; \
         decision threshold: factor geomean <= 0.95 on compRat<=0.7 bucket)"
    );
}
