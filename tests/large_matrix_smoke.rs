//! Manual smoke test for the full sparse pipeline at n ≈ 10^4–10^5.
//!
//! Symbolic ordering has been validated up to n = 345,241 (c-big) but
//! the full ordering → scaling → factor → solve path is otherwise
//! exercised only on the parity corpus (median n = 77) and the KKT
//! corpus (largest n = 5,314). This test closes that gap by running
//! the SuiteSparse large-matrix corpus end-to-end.
//!
//! `#[ignore]` so it does not run on `cargo test`. Invoke with:
//!
//!     cargo test --release --test large_matrix_smoke -- --ignored --nocapture
//!
//! Two buckets, both gated `#[ignore]`:
//!
//!   - `large_matrix_full_pipeline_smoke` — matrices with `n <= 100_000`
//!     after sorting smallest-first by file size. Default smoke run.
//!   - `xl_matrix_full_pipeline_smoke` — matrices with `n > 100_000`.
//!     Run with `--ignored xl_matrix_full_pipeline_smoke` (or just
//!     `--ignored` to run both).
//!
//! Memory budget: each test estimates `16 * sym.factor_nnz_estimate`
//! bytes for the L factor before factoring and skips with a clear
//! message if it exceeds `FERAL_SMOKE_MEM_GB` (default 16 GB). Raise
//! it via `FERAL_SMOKE_MEM_GB=48 cargo test ...`.
//!
//! Output is flushed line-by-line to stderr so partial results survive
//! a SIGKILL (e.g. macOS jetsam OOM).
//!
//! Requires `tests/data/large/*.mtx` (gitignored). Fetch with
//! `dev/scripts/fetch_large_matrices.sh` first; the test prints a
//! SKIP message and exits cleanly if the directory is empty.
//!
//! Per matrix: builds `b = A * 1`, factors with default `Solver`,
//! runs one plain `solve` and one `solve_refined`, reports relative
//! residual and wall-clock for each phase. Fails only on factor
//! error, dimension mismatch, or refined residual > `RES_TOL`.

use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, FactorStatus, Solver};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const RES_TOL: f64 = 1e-3;
const XL_THRESHOLD: usize = 100_000;
const DEFAULT_BUDGET_GB: f64 = 16.0;
/// L factor bytes per nonzero: 8 bytes data + ~8 bytes index.
const BYTES_PER_NNZ: usize = 16;

#[test]
#[ignore]
fn large_matrix_full_pipeline_smoke() {
    run_smoke(SizeBucket::Regular);
}

#[test]
#[ignore]
fn xl_matrix_full_pipeline_smoke() {
    run_smoke(SizeBucket::ExtraLarge);
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum SizeBucket {
    /// `n <= XL_THRESHOLD`
    Regular,
    /// `n > XL_THRESHOLD`
    ExtraLarge,
}

fn run_smoke(bucket: SizeBucket) {
    let dir = PathBuf::from("tests/data/large");
    if !dir.is_dir() {
        emit(format_args!(
            "SKIP: {} not found. Run dev/scripts/fetch_large_matrices.sh.",
            dir.display()
        ));
        return;
    }

    // Sort smallest-first by file size so a per-file failure (OOM, kill)
    // takes out the largest matrix only after the smaller ones have
    // already produced output.
    let mut entries: Vec<(PathBuf, u64)> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
        .filter_map(|p| {
            let size = std::fs::metadata(&p).ok()?.len();
            Some((p, size))
        })
        .collect();
    entries.sort_by_key(|(_, size)| *size);
    let paths: Vec<PathBuf> = entries.into_iter().map(|(p, _)| p).collect();

    if paths.is_empty() {
        emit(format_args!(
            "SKIP: no .mtx files in {}. Run dev/scripts/fetch_large_matrices.sh.",
            dir.display()
        ));
        return;
    }

    let budget_gb: f64 = std::env::var("FERAL_SMOKE_MEM_GB")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(DEFAULT_BUDGET_GB);
    let budget_bytes = (budget_gb * 1024.0 * 1024.0 * 1024.0) as u64;

    emit(format_args!(
        "bucket: {}  budget: {:.1} GB  ({} matrix file(s) in {})",
        match bucket {
            SizeBucket::Regular => "regular (n <= 100_000)",
            SizeBucket::ExtraLarge => "xl (n > 100_000)",
        },
        budget_gb,
        paths.len(),
        dir.display()
    ));
    emit(format_args!(
        "{:<14} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "matrix", "n", "nnz", "read(s)", "factor(s)", "solve(s)", "rel_res", "rel_res_r"
    ));

    let mut failures: Vec<String> = Vec::new();
    let mut tested = 0usize;
    let mut skipped = 0usize;

    for path in &paths {
        match smoke_one(path, bucket, budget_bytes, &mut failures) {
            SmokeOutcome::Tested => tested += 1,
            SmokeOutcome::Skipped => skipped += 1,
            SmokeOutcome::Failed => {}
            SmokeOutcome::OutOfBucket => {}
        }
    }

    emit(format_args!(""));
    emit(format_args!(
        "smoke: {} matrix(es) tested, {} skipped, {} failure(s)",
        tested,
        skipped,
        failures.len()
    ));
    if !failures.is_empty() {
        for f in &failures {
            emit(format_args!("  {f}"));
        }
        panic!("{} failure(s)", failures.len());
    }
}

enum SmokeOutcome {
    Tested,
    Skipped,
    Failed,
    OutOfBucket,
}

fn smoke_one(
    path: &Path,
    bucket: SizeBucket,
    budget_bytes: u64,
    failures: &mut Vec<String>,
) -> SmokeOutcome {
    let name = path.file_stem().unwrap().to_string_lossy().to_string();

    let t = Instant::now();
    let mtx = match read_mtx(path) {
        Ok(m) => m,
        Err(e) => {
            failures.push(format!("{name}: read_mtx failed: {e}"));
            return SmokeOutcome::Failed;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            failures.push(format!("{name}: to_csc failed: {e}"));
            return SmokeOutcome::Failed;
        }
    };
    let read_s = t.elapsed().as_secs_f64();
    let n = csc.n;
    let nnz = csc.row_idx.len();

    // Bucket selection by `n` so c-big-class matrices land in the XL test
    // even if a future fetch script reorders sizes.
    match bucket {
        SizeBucket::Regular if n > XL_THRESHOLD => return SmokeOutcome::OutOfBucket,
        SizeBucket::ExtraLarge if n <= XL_THRESHOLD => return SmokeOutcome::OutOfBucket,
        _ => {}
    }

    // Memory pre-check: factor_nnz_estimate × 16 bytes must fit in budget.
    // Symbolic factorisation on n=300k runs in seconds, so this is cheap
    // insurance against jetsam.
    let sym = match symbolic_factorize(&csc, &SupernodeParams::default()) {
        Ok(s) => s,
        Err(e) => {
            failures.push(format!("{name}: symbolic_factorize failed: {e}"));
            return SmokeOutcome::Failed;
        }
    };
    let est_bytes = (sym.factor_nnz_estimate as u64).saturating_mul(BYTES_PER_NNZ as u64);
    if est_bytes > budget_bytes {
        let est_gb = est_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let bud_gb = budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        emit(format_args!(
            "{:<14} {:>8} {:>10} {:>10.3} {:>10} {:>10} {:>10} {:>10}  SKIP: est {:.1} GB > budget {:.1} GB (raise FERAL_SMOKE_MEM_GB)",
            name, n, nnz, read_s, "-", "-", "-", "-", est_gb, bud_gb
        ));
        return SmokeOutcome::Skipped;
    }

    // b = A * 1 ⇒ exact solution x = 1, so any residual is solver
    // error. (Scale does not match a real RHS, but it isolates the
    // pipeline from sidecar dependencies.)
    let ones = vec![1.0f64; n];
    let mut rhs = vec![0.0f64; n];
    csc.symv(&ones, &mut rhs);

    let mut solver = Solver::new();

    let t = Instant::now();
    let status = solver.factor(&csc, None);
    let factor_s = t.elapsed().as_secs_f64();

    match &status {
        FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
        FactorStatus::Singular => {
            emit(format_args!(
                "{:<14} {:>8} {:>10} {:>10.3} {:>10.3} {:>10} {:>10} {:>10}  SINGULAR",
                name, n, nnz, read_s, factor_s, "-", "-", "-"
            ));
            return SmokeOutcome::Tested;
        }
        FactorStatus::FatalError(e) => {
            failures.push(format!("{name}: factor fatal: {e}"));
            return SmokeOutcome::Failed;
        }
    }

    let t = Instant::now();
    let x = match solver.solve(&rhs) {
        Ok(x) => x,
        Err(e) => {
            failures.push(format!("{name}: solve failed: {e}"));
            return SmokeOutcome::Failed;
        }
    };
    let solve_s = t.elapsed().as_secs_f64();
    let rel_res = relative_residual(&csc, &x, &rhs);

    let x_r = match solver.solve_refined(&csc, &rhs) {
        Ok(x) => x,
        Err(e) => {
            failures.push(format!("{name}: solve_refined failed: {e}"));
            return SmokeOutcome::Failed;
        }
    };
    let rel_res_r = relative_residual(&csc, &x_r, &rhs);

    emit(format_args!(
        "{:<14} {:>8} {:>10} {:>10.3} {:>10.3} {:>10.3} {:>10.2e} {:>10.2e}",
        name, n, nnz, read_s, factor_s, solve_s, rel_res, rel_res_r
    ));

    if !rel_res_r.is_finite() || rel_res_r > RES_TOL {
        failures.push(format!(
            "{name}: refined relative residual {rel_res_r:.2e} > tol {RES_TOL:.0e}"
        ));
        return SmokeOutcome::Failed;
    }
    SmokeOutcome::Tested
}

fn emit(args: std::fmt::Arguments<'_>) {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "{args}");
    let _ = err.flush();
}

fn relative_residual(a: &feral::CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0f64; n];
    a.symv(x, &mut ax);
    let mut res_sq = 0.0;
    let mut b_sq = 0.0;
    for i in 0..n {
        let r = ax[i] - b[i];
        res_sq += r * r;
        b_sq += b[i] * b[i];
    }
    if b_sq > 0.0 {
        (res_sq / b_sq).sqrt()
    } else {
        res_sq.sqrt()
    }
}
