//! Issue #10 final pre-floor lever: forced supernode amalgamation
//! sweep on the 1D-banded Mittelmann panel.
//!
//! Hypothesis from `issue-10-ordering-supernode-shape.md` §"joint
//! implication": all four levers tried so far (SLB, MAXFROMM, axpy
//! SIMD, ordering swap) come up within noise on the panel. The
//! remaining symbolic-side lever is to forcibly widen the bottom-of-
//! tree supernodes by raising `SupernodeParams::nemin` above the
//! Phase 2.13a default of 16. This sweeps {16, 32, 64, 128, 256,
//! usize::MAX} and reports per-matrix shape stats (ncol_mean / p90),
//! fill (factor_nnz), and factor time. If a higher nemin widens
//! fronts AND reduces factor time, MAXFROMM should be re-A/B'd on
//! top of that nemin. If even nemin=MAX leaves factor time flat or
//! worse, the corpus hits the rank-1 axpy hardware floor.
//!
//! Usage: `cargo run --release --bin diag_nemin_amalgamation_panel`

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams, SymbolicFactorization};
use feral::{read_mtx, CscMatrix};

const N_REPEAT: usize = 3;
const CORPUS: &str = "data/matrices/kkt-mittelmann";
const FAMILIES: &[&str] = &["clnlbeam", "henon120", "lane_emden120", "dirichlet120"];
// Capped at 128: pilot run on clnlbeam_0000 showed nemin=MAX did not
// finish a single factor in 30+ min (fronts collapse into a single
// near-dense block of order >n/2). 256 ran but cost 4.6× the nemin=16
// baseline. 128 is the largest value that still completes in seconds
// while preserving the monotonic-regression signal.
const NEMINS: &[usize] = &[16, 32, 64, 128];

fn load_csc(path: &Path) -> Option<CscMatrix> {
    let mtx = read_mtx(path).ok()?;
    mtx.to_csc().ok()
}

fn percentile(sorted: &[usize], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)] as f64
}

struct Shape {
    n_snodes: usize,
    ncol_mean: f64,
    ncol_p90: f64,
    nrow_mean: f64,
}

fn shape_of(sym: &SymbolicFactorization) -> Shape {
    let mut ncols: Vec<usize> = sym.supernodes.iter().map(|s| s.ncol).collect();
    let nrows: Vec<usize> = sym.supernodes.iter().map(|s| s.nrow).collect();
    ncols.sort_unstable();
    let n = ncols.len();
    let ncol_sum: usize = ncols.iter().sum();
    let nrow_sum: usize = nrows.iter().sum();
    Shape {
        n_snodes: n,
        ncol_mean: ncol_sum as f64 / n.max(1) as f64,
        ncol_p90: percentile(&ncols, 0.90),
        nrow_mean: nrow_sum as f64 / n.max(1) as f64,
    }
}

fn factor_time_and_nnz(
    csc: &CscMatrix,
    sym: &SymbolicFactorization,
    np: &NumericParams,
) -> Option<(u128, usize)> {
    let (factors, _) = factorize_multifrontal(csc, sym, np).ok()?;
    let nnz = factors.factor_nnz();
    let mut best = u128::MAX;
    for _ in 0..N_REPEAT {
        let t = Instant::now();
        if factorize_multifrontal(csc, sym, np).is_err() {
            return None;
        }
        let us = t.elapsed().as_micros();
        if us < best {
            best = us;
        }
    }
    Some((best, nnz))
}

fn enumerate(family: &str) -> Vec<(String, std::path::PathBuf)> {
    let dir = Path::new(CORPUS).join(family);
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<_> = entries
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.extension()?.to_str()? != "mtx" {
                return None;
            }
            let stem = p.file_stem()?.to_str()?.to_string();
            Some((stem, p))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn geomean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let s: f64 = xs.iter().map(|v| v.ln()).sum();
    (s / xs.len() as f64).exp()
}

fn fmt_nemin(n: usize) -> String {
    if n == usize::MAX {
        "MAX".into()
    } else {
        n.to_string()
    }
}

fn process_family(family: &str) {
    let matrices = enumerate(family);
    if matrices.is_empty() {
        println!("\n[{family}] MISSING corpus");
        return;
    }
    println!("\n=== {family} ({} matrices) ===", matrices.len());
    println!(
        "{:>20}  {:>5}  {:>7}  {:>9}  {:>9}  {:>9}  {:>12}  {:>11}",
        "label", "nemin", "snodes", "ncol_mean", "ncol_p90", "nrow_mean", "factor_nnz", "factor_us"
    );

    // (nemin_index, vec<factor_us>), (nemin_index, vec<factor_nnz>), etc.
    let mut t_by_nemin: Vec<Vec<f64>> = vec![Vec::new(); NEMINS.len()];
    let mut nnz_by_nemin: Vec<Vec<f64>> = vec![Vec::new(); NEMINS.len()];
    let mut ncol_by_nemin: Vec<Vec<f64>> = vec![Vec::new(); NEMINS.len()];
    let mut nrow_by_nemin: Vec<Vec<f64>> = vec![Vec::new(); NEMINS.len()];

    let np = NumericParams::default();

    for (label, path) in &matrices {
        let Some(csc) = load_csc(path) else {
            println!("{label:>20}  load FAIL");
            continue;
        };
        for (ni, &nemin) in NEMINS.iter().enumerate() {
            let sp = SupernodeParams {
                nemin,
                ..SupernodeParams::default()
            };
            let sym = match symbolic_factorize(&csc, &sp) {
                Ok(s) => s,
                Err(e) => {
                    println!("{label:>20}  {:>5}  symbolic FAIL: {e}", fmt_nemin(nemin));
                    continue;
                }
            };
            let sh = shape_of(&sym);
            let (us, nnz) = match factor_time_and_nnz(&csc, &sym, &np) {
                Some(v) => v,
                None => {
                    println!("{label:>20}  {:>5}  factor FAILED", fmt_nemin(nemin));
                    continue;
                }
            };
            println!(
                "{label:>20}  {:>5}  {:>7}  {:>9.2}  {:>9.0}  {:>9.2}  {:>12}  {:>11}",
                fmt_nemin(nemin),
                sh.n_snodes,
                sh.ncol_mean,
                sh.ncol_p90,
                sh.nrow_mean,
                nnz,
                us,
            );
            t_by_nemin[ni].push(us as f64);
            nnz_by_nemin[ni].push(nnz as f64);
            ncol_by_nemin[ni].push(sh.ncol_mean);
            nrow_by_nemin[ni].push(sh.nrow_mean);
        }
    }

    println!("  --- {family} summary (paired vs nemin=16, geomean) ---");
    println!(
        "  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}",
        "nemin", "factor_us", "factor_nnz", "ncol_mean", "nrow_mean"
    );
    let base_t = &t_by_nemin[0].clone();
    let base_nnz = &nnz_by_nemin[0].clone();
    let base_nc = &ncol_by_nemin[0].clone();
    let base_nr = &nrow_by_nemin[0].clone();
    for ni in 0..NEMINS.len() {
        let t = &t_by_nemin[ni];
        let nnz = &nnz_by_nemin[ni];
        let nc = &ncol_by_nemin[ni];
        let nr = &nrow_by_nemin[ni];
        if t.len() != base_t.len() {
            continue;
        }
        let rt = geomean(&t.iter().zip(base_t).map(|(x, b)| x / b).collect::<Vec<_>>());
        let rnnz = geomean(
            &nnz.iter()
                .zip(base_nnz)
                .map(|(x, b)| x / b)
                .collect::<Vec<_>>(),
        );
        let rnc = geomean(
            &nc.iter()
                .zip(base_nc)
                .map(|(x, b)| x / b)
                .collect::<Vec<_>>(),
        );
        let rnr = geomean(
            &nr.iter()
                .zip(base_nr)
                .map(|(x, b)| x / b)
                .collect::<Vec<_>>(),
        );
        println!(
            "  {:>5}  {:>10.3}  {:>10.3}  {:>10.3}  {:>10.3}",
            fmt_nemin(NEMINS[ni]),
            rt,
            rnnz,
            rnc,
            rnr
        );
    }
}

fn main() {
    println!("=== Issue #10 nemin amalgamation sweep (1D-banded Mittelmann) ===");
    println!(
        "nemin values: {:?}",
        NEMINS.iter().map(|n| fmt_nemin(*n)).collect::<Vec<_>>()
    );
    println!("ordering: default (Amd via symbolic_factorize)");
    println!("min-of-{N_REPEAT} factor timings; warm-up uncounted");

    for fam in FAMILIES {
        process_family(fam);
    }

    println!("\nInterpretation guide:");
    println!("  - factor_us  / nemin16 < 0.9  → forced amalgamation gives a direct win");
    println!("  - ncol_mean  / nemin16 > 1.5  → fronts are meaningfully wider");
    println!("  - factor_nnz / nemin16 > 1.3  → fill inflated (cost of forcing merges)");
    println!("  - if no nemin reduces factor_us, the rank-1 axpy floor is reached");
}
