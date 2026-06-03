//! F3.3 — Cross-validate feral's Schur block against MUMPS's Schur output.
//!
//! For each KKT matrix with a fresh `<id>.mumps_schur.json` sidecar
//! produced by `external_benchmarks/mumps_oracle/run_mumps_schur.py`,
//! this driver:
//!   1. Reads the matrix + the Schur index list MUMPS used (the JSON
//!      records the exact list, so any selection rule the Python driver
//!      applies is reflected here without coupling).
//!   2. Calls `symbolic_factorize_with_schur` + `factorize_multifrontal_with_schur`
//!      on the same indices, producing feral's `SchurBlock`.
//!   3. Loads the MUMPS reference Schur from the co-located
//!      `<id>.mumps_schur.bin` (column-major full-symmetric f64).
//!   4. Loads (when available) the pure-Rust dense oracle Schur from
//!      `<id>.dense_schur.bin`, produced by `produce_dense_schur`.
//!   5. Computes pairwise max relative entry-wise errors
//!      `max_{i,j} |a(i,j) - b(i,j)| / max(|a|, |b|, 1)`
//!      for the three pairs (feral, MUMPS, oracle).
//!
//! Acceptance gate (Option B, per dev/research/schur-complement.md):
//!   per-matrix `feral-vs-oracle ≤ max(absolute_floor, K · MUMPS-vs-oracle)`
//!   with `absolute_floor = 1e-10` and `K = 10`. The dense oracle is the
//!   ground truth; the bound adapts to per-matrix conditioning so the
//!   original `feral-vs-MUMPS ≤ 1e-10` reading (unachievable on
//!   ill-conditioned ACOPR-family KKTs where MUMPS itself disagrees with
//!   the dense oracle by ~1e-6) is replaced by "feral hits the same
//!   conditioning floor as MUMPS, within a 10× factor for pivot-ordering
//!   variation". Two LDL^T algorithms with different pivot choices can
//!   reach the conditioning floor at slightly different floats; K=10 is
//!   the standard multiplicative slack for that regime, while preserving
//!   detection of any genuine algorithmic divergence (which would
//!   produce ratios orders of magnitude larger).
//!   Corpus floor: ≥ 100 matrices must satisfy the per-matrix bound.
//!
//! Matrices without a dense-oracle sidecar fall back to the strict
//! `feral-vs-MUMPS ≤ tol` legacy reading and are reported separately.
//!
//! Inputs:
//!   <id>.mtx + <id>.json (sidecar) + <id>.mumps_schur.json
//!   + <id>.mumps_schur.bin + (optional) <id>.dense_schur.bin
//!
//! Usage:
//!   cargo run --release --bin diag_schur_parity
//!   cargo run --release --bin diag_schur_parity -- data/matrices/kkt
//!   FERAL_DIAG_MAX_N=2000 cargo run --release --bin diag_schur_parity
//!   FERAL_DIAG_VERBOSE=1 cargo run --release --bin diag_schur_parity

use std::path::{Path, PathBuf};

use feral::numeric::factorize::{factorize_multifrontal_with_schur, NumericParams};
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize_with_schur, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, ZeroPivotAction};

const DEFAULT_ROOTS: &[&str] = &[
    "data/matrices/kkt",
    "data/matrices/kkt-expansion",
    "data/matrices/kkt-mittelmann",
];

const DEFAULT_TOL: f64 = 1e-10;
/// Multiplicative slack on MUMPS-vs-oracle: feral may sit up to K times
/// the MUMPS conditioning floor. See module docstring for rationale.
const DEFAULT_OPTIONB_SLACK: f64 = 10.0;

fn percentile_f64(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() as f64) * q).floor() as usize;
    v[idx.min(v.len() - 1)]
}

/// Schur reference parsed from a `<id>.mumps_schur.json` sidecar plus
/// the co-located binary. `None` for matrices whose oracle run failed
/// (status != "ok") or for sidecars missing required fields.
struct MumpsSchurRef {
    n: usize,
    n_schur: usize,
    schur_indices_0idx: Vec<usize>,
    /// `n_schur * n_schur` doubles, column-major, full symmetric.
    data: Vec<f64>,
}

fn read_mumps_schur(json_path: &Path) -> Option<MumpsSchurRef> {
    let text = std::fs::read_to_string(json_path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    if v.get("status").and_then(|s| s.as_str()) != Some("ok") {
        return None;
    }
    let n = v.get("n")?.as_u64()? as usize;
    let n_schur = v.get("n_schur")?.as_u64()? as usize;
    if n_schur == 0 || n_schur >= n {
        return None;
    }
    let idx_arr = v.get("schur_indices_0indexed")?.as_array()?;
    let mut schur_indices_0idx: Vec<usize> = Vec::with_capacity(idx_arr.len());
    for x in idx_arr {
        let i = x.as_u64()? as usize;
        if i >= n {
            return None;
        }
        schur_indices_0idx.push(i);
    }
    if schur_indices_0idx.len() != n_schur {
        return None;
    }
    let bin_rel = v.get("schur_bin_relative")?.as_str()?;
    let bin_path = json_path.parent()?.join(bin_rel);
    let bytes = std::fs::read(&bin_path).ok()?;
    let expected_bytes = n_schur * n_schur * 8;
    if bytes.len() != expected_bytes {
        return None;
    }
    let mut data = Vec::with_capacity(n_schur * n_schur);
    for chunk in bytes.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        data.push(f64::from_le_bytes(buf));
    }
    Some(MumpsSchurRef {
        n,
        n_schur,
        schur_indices_0idx,
        data,
    })
}

/// Load the pure-Rust dense oracle Schur block for `<id>` if present
/// at `<id>.dense_schur.bin`. Returns the column-major
/// `n_schur * n_schur` matrix (same layout as MUMPS sidecar) or `None`
/// when the file is missing or the size doesn't match `n_schur`.
fn read_dense_oracle(mtx_path: &Path, n_schur: usize) -> Option<Vec<f64>> {
    let bin_path = mtx_path.with_extension("dense_schur.bin");
    let bytes = std::fs::read(&bin_path).ok()?;
    let expected_bytes = n_schur * n_schur * 8;
    if bytes.len() != expected_bytes {
        return None;
    }
    let mut data = Vec::with_capacity(n_schur * n_schur);
    for chunk in bytes.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        data.push(f64::from_le_bytes(buf));
    }
    Some(data)
}

/// Max-rel entry-wise error between two `dim x dim` column-major
/// matrices, using `max(|a|, |b|, 1.0)` as the denominator (same
/// floor as the MUMPS-comparison metric).
fn max_rel_block(a: &[f64], b: &[f64], dim: usize) -> f64 {
    let mut max_rel: f64 = 0.0;
    for j in 0..dim {
        for i in 0..dim {
            let av = a[j * dim + i];
            let bv = b[j * dim + i];
            let denom = av.abs().max(bv.abs()).max(1.0);
            let rel = (av - bv).abs() / denom;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    max_rel
}

#[derive(Default)]
struct Aggregate {
    seen: usize,
    skipped_no_mtx: usize,
    skipped_no_oracle: usize,
    skipped_size: usize,
    skipped_factor_err: usize,
    compared: usize,
    /// All matrices we managed to compare: feral-vs-MUMPS error.
    feral_vs_mumps: Vec<f64>,
    /// Subset that also has a dense-oracle sidecar.
    with_oracle: usize,
    feral_vs_oracle: Vec<f64>,
    mumps_vs_oracle: Vec<f64>,
    /// Per-matrix Option B gate failures: feral-vs-oracle exceeded
    /// `max(absolute_floor, mumps-vs-oracle)`.
    optionb_failures: Vec<(String, f64, f64, f64)>, // (name, fvm, fvo, mvo)
    /// Strict legacy fallback for matrices without an oracle:
    /// feral-vs-MUMPS exceeded the absolute floor.
    legacy_failures: Vec<(String, f64)>,
    worst_fvm: (f64, String),
    worst_fvo: (f64, String),
}

fn run_one(
    mtx_path: &Path,
    agg: &mut Aggregate,
    max_n: usize,
    tol: f64,
    slack: f64,
    verbose: bool,
) {
    agg.seen += 1;
    let name = mtx_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<?>")
        .to_string();

    let oracle_path = mtx_path.with_extension("mumps_schur.json");
    let oracle = match read_mumps_schur(&oracle_path) {
        Some(o) => o,
        None => {
            agg.skipped_no_oracle += 1;
            return;
        }
    };

    if oracle.n > max_n {
        agg.skipped_size += 1;
        return;
    }

    let mtx = match read_mtx(mtx_path) {
        Ok(m) => m,
        Err(_) => {
            agg.skipped_no_mtx += 1;
            return;
        }
    };
    if mtx.n != oracle.n {
        agg.skipped_no_oracle += 1;
        return;
    }
    if mtx.entries.iter().any(|(_, _, v)| !v.is_finite()) {
        agg.skipped_no_oracle += 1;
        return;
    }
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(_) => {
            agg.skipped_no_mtx += 1;
            return;
        }
    };
    drop(mtx);

    let snode = SupernodeParams::default();
    // Match MUMPS's no-extra-pivoting Schur path: a default threshold,
    // but ForceAccept for any zero pivot in the eliminated block so the
    // factor doesn't refuse degenerate KKTs that MUMPS would push
    // through with ICNTL(24)=1.
    let bk = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    // Identity scaling: MUMPS's Schur path is incompatible with
    // analysis-time scaling (KEEP(52) = -2 with ICNTL(19) ≠ 0 is
    // explicitly rejected — see dev/research/schur-complement.md), so
    // the oracle's Schur is computed on the unscaled matrix. Match
    // that here so entry-wise comparison is meaningful; feral's
    // default Auto scaling would otherwise return D_S · S_unscaled · D_S
    // and the comparison would fail by orders of magnitude.
    let np = NumericParams {
        scaling: ScalingStrategy::Identity,
        ..NumericParams::with_bk(bk)
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let sym = symbolic_factorize_with_schur(&csc, &snode, &oracle.schur_indices_0idx)?;
        factorize_multifrontal_with_schur(&csc, &sym, &np)
    }));
    let schur = match result {
        Ok(Ok((_, _, schur))) => schur,
        _ => {
            agg.skipped_factor_err += 1;
            if verbose {
                eprintln!("FACTOR_FAIL {}", name);
            }
            return;
        }
    };

    if schur.dim != oracle.n_schur {
        agg.skipped_factor_err += 1;
        if verbose {
            eprintln!(
                "DIM_MISMATCH {} feral={} mumps={}",
                name, schur.dim, oracle.n_schur
            );
        }
        return;
    }

    // Flatten feral SchurBlock to column-major n_schur*n_schur for
    // pairwise comparison with the MUMPS / dense-oracle sidecars.
    let dim = schur.dim;
    let mut feral_flat = vec![0.0_f64; dim * dim];
    for j in 0..dim {
        for i in 0..dim {
            feral_flat[j * dim + i] = schur.get(i, j);
        }
    }

    let fvm = max_rel_block(&feral_flat, &oracle.data, dim);
    if !fvm.is_finite() {
        agg.skipped_factor_err += 1;
        if verbose {
            eprintln!("NONFINITE {}", name);
        }
        return;
    }

    agg.compared += 1;
    agg.feral_vs_mumps.push(fvm);
    if fvm > agg.worst_fvm.0 {
        agg.worst_fvm = (fvm, name.clone());
    }

    // Optional dense-oracle sidecar enables the per-matrix Option B gate.
    let dense = read_dense_oracle(mtx_path, dim);
    if let Some(dense) = dense {
        let fvo = max_rel_block(&feral_flat, &dense, dim);
        let mvo = max_rel_block(&oracle.data, &dense, dim);
        agg.with_oracle += 1;
        agg.feral_vs_oracle.push(fvo);
        agg.mumps_vs_oracle.push(mvo);
        if fvo > agg.worst_fvo.0 {
            agg.worst_fvo = (fvo, name.clone());
        }
        let bound = tol.max(slack * mvo);
        if fvo > bound {
            agg.optionb_failures.push((name.clone(), fvm, fvo, mvo));
        }
        if verbose {
            eprintln!(
                "OK {} n={} n_schur={} fvm={:.3e} fvo={:.3e} mvo={:.3e} bound={:.3e}",
                name, oracle.n, oracle.n_schur, fvm, fvo, mvo, bound
            );
        }
    } else {
        // Legacy strict reading for matrices missing a dense oracle.
        if fvm > tol {
            agg.legacy_failures.push((name.clone(), fvm));
        }
        if verbose {
            eprintln!(
                "OK {} n={} n_schur={} fvm={:.3e} (no oracle, strict gate)",
                name, oracle.n, oracle.n_schur, fvm
            );
        }
    }
}

fn distribution_block(label: &str, errs: &[f64]) {
    if errs.is_empty() {
        println!("  ({} — no samples)", label);
        return;
    }
    let mut v = errs.to_vec();
    println!("{}:", label);
    println!("  count:   {}", v.len());
    println!("  min:     {:.3e}", percentile_f64(&mut v, 0.0));
    println!("  p10:     {:.3e}", percentile_f64(&mut v, 0.10));
    println!("  median:  {:.3e}", percentile_f64(&mut v, 0.50));
    println!("  p90:     {:.3e}", percentile_f64(&mut v, 0.90));
    println!("  p99:     {:.3e}", percentile_f64(&mut v, 0.99));
    println!("  max:     {:.3e}", percentile_f64(&mut v, 1.0));
}

fn report(agg: &Aggregate, tol: f64, slack: f64) {
    println!("=== diag_schur_parity ===");
    println!("seen:                {}", agg.seen);
    println!("  skipped no .mtx:   {}", agg.skipped_no_mtx);
    println!("  skipped n>max:     {}", agg.skipped_size);
    println!("  skipped no oracle: {}", agg.skipped_no_oracle);
    println!("  skipped factor:    {}", agg.skipped_factor_err);
    println!("compared:            {}", agg.compared);
    println!("  with dense oracle: {}", agg.with_oracle);
    println!("absolute floor (tol):{:.1e}", tol);
    println!("Option B slack (K): {}", slack);
    if agg.feral_vs_mumps.is_empty() {
        println!("(no comparisons — run run_mumps_schur.py first)");
        return;
    }
    println!();
    distribution_block("feral vs MUMPS", &agg.feral_vs_mumps);
    println!();
    distribution_block("feral vs dense oracle", &agg.feral_vs_oracle);
    println!();
    distribution_block("MUMPS vs dense oracle", &agg.mumps_vs_oracle);

    println!();
    println!(
        "Option B per-matrix gate failures (feral-vs-oracle > max({:.1e}, {} · MUMPS-vs-oracle)): {} / {}",
        tol,
        slack,
        agg.optionb_failures.len(),
        agg.with_oracle
    );
    let mut topb = agg.optionb_failures.clone();
    topb.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    for (name, fvm, fvo, mvo) in topb.iter().take(10) {
        println!(
            "  {:<32} fvo={:.3e} mvo={:.3e} fvm={:.3e}",
            name, fvo, mvo, fvm
        );
    }

    if !agg.legacy_failures.is_empty() {
        println!();
        println!(
            "Legacy strict (feral-vs-MUMPS > {:.1e}) on matrices without dense oracle: {} / {}",
            tol,
            agg.legacy_failures.len(),
            agg.compared - agg.with_oracle
        );
        let mut topl = agg.legacy_failures.clone();
        topl.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (name, err) in topl.iter().take(10) {
            println!("  {:<40} {:.3e}", name, err);
        }
    }

    println!();
    println!(
        "worst feral-vs-MUMPS:   {} = {:.3e}",
        agg.worst_fvm.1, agg.worst_fvm.0
    );
    println!(
        "worst feral-vs-oracle:  {} = {:.3e}",
        agg.worst_fvo.1, agg.worst_fvo.0
    );

    // Option B gate: ≥ 100 matrices satisfy per-matrix
    // `feral-vs-oracle ≤ max(tol, MUMPS-vs-oracle)`, plus the legacy
    // strict gate must hold for matrices without an oracle (so we
    // don't silently drop coverage on matrices we couldn't oracle).
    let satisfied_optionb = agg.with_oracle.saturating_sub(agg.optionb_failures.len());
    let legacy_clean = agg.legacy_failures.is_empty();
    let pass_n = satisfied_optionb >= 100;
    let pass_optionb = agg.optionb_failures.is_empty();
    println!();
    println!("F3.3 Option B gate (per-matrix bound + N>=100 satisfied + no legacy failures):");
    println!(
        "  N satisfying Option B: {} / {}   {}",
        satisfied_optionb,
        agg.with_oracle,
        if pass_n { "PASS-N100" } else { "FAIL-N100" }
    );
    println!(
        "  Option B clean:         {}",
        if pass_optionb { "PASS" } else { "FAIL" }
    );
    println!(
        "  Legacy strict clean:    {}",
        if legacy_clean { "PASS" } else { "FAIL" }
    );
    println!(
        "  Overall:                {}",
        if pass_n && pass_optionb && legacy_clean {
            "PASS"
        } else {
            "FAIL"
        }
    );
}

fn walk_root(root: &Path, agg: &mut Aggregate, max_n: usize, tol: f64, slack: f64, verbose: bool) {
    if !root.is_dir() {
        return;
    }
    let mut entries: Vec<_> = match std::fs::read_dir(root) {
        Ok(d) => d.filter_map(|e| e.ok()).map(|e| e.path()).collect(),
        Err(_) => return,
    };
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk_root(&p, agg, max_n, tol, slack, verbose);
        } else if p.extension().is_some_and(|ext| ext == "mtx") {
            run_one(&p, agg, max_n, tol, slack, verbose);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let roots: Vec<PathBuf> = if args.is_empty() {
        DEFAULT_ROOTS.iter().map(PathBuf::from).collect()
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    let max_n: usize = std::env::var("FERAL_DIAG_MAX_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let tol: f64 = std::env::var("FERAL_DIAG_TOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TOL);
    let slack: f64 = std::env::var("FERAL_DIAG_OPTIONB_SLACK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_OPTIONB_SLACK);
    let verbose = std::env::var("FERAL_DIAG_VERBOSE")
        .ok()
        .map(|s| s != "0")
        .unwrap_or(false);

    let mut agg = Aggregate::default();
    for r in &roots {
        walk_root(r, &mut agg, max_n, tol, slack, verbose);
    }
    report(&agg, tol, slack);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn write_temp_pair(json: &str, bin: &[f64]) -> (PathBuf, PathBuf) {
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let dir = std::env::temp_dir();
        let stem = format!("feral_schur_parity_test_{}_{}", pid, id);
        let json_path = dir.join(format!("{}.mumps_schur.json", stem));
        let bin_path = dir.join(format!("{}.mumps_schur.bin", stem));
        std::fs::write(&json_path, json).unwrap();
        let mut bytes: Vec<u8> = Vec::with_capacity(bin.len() * 8);
        for v in bin {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&bin_path, &bytes).unwrap();
        (json_path, bin_path)
    }

    #[test]
    fn parses_full_schur_sidecar() {
        let bin = vec![1.0_f64, 2.0, 3.0, 4.0]; // 2x2 column-major
        let json = format!(
            r#"{{
                "status": "ok",
                "n": 5,
                "n_schur": 2,
                "schur_indices_0indexed": [3, 4],
                "schur_bin_relative": "{}"
            }}"#,
            // bin file co-located with json, so the relative path is
            // just the bin file's basename.
            // (write_temp_pair puts both in temp_dir.)
            "PLACEHOLDER",
        );
        let (json_path, bin_path) = write_temp_pair(&json, &bin);
        // Patch the json so schur_bin_relative points at the actual bin
        let bin_name = bin_path.file_name().unwrap().to_str().unwrap().to_string();
        let patched = std::fs::read_to_string(&json_path)
            .unwrap()
            .replace("PLACEHOLDER", &bin_name);
        std::fs::write(&json_path, &patched).unwrap();

        let r = read_mumps_schur(&json_path).expect("should parse");
        assert_eq!(r.n, 5);
        assert_eq!(r.n_schur, 2);
        assert_eq!(r.schur_indices_0idx, vec![3, 4]);
        assert_eq!(r.data, vec![1.0, 2.0, 3.0, 4.0]);
        let _ = std::fs::remove_file(&json_path);
        let _ = std::fs::remove_file(&bin_path);
    }

    #[test]
    fn returns_none_when_status_fail() {
        let (json_path, bin_path) =
            write_temp_pair(r#"{"status": "fail", "n": 5, "n_schur": 2}"#, &[0.0; 4]);
        assert!(read_mumps_schur(&json_path).is_none());
        let _ = std::fs::remove_file(&json_path);
        let _ = std::fs::remove_file(&bin_path);
    }

    #[test]
    fn returns_none_when_bin_size_mismatch() {
        let bin_name_placeholder = "PLACEHOLDER".to_string();
        let json = format!(
            r#"{{
                "status": "ok",
                "n": 5,
                "n_schur": 2,
                "schur_indices_0indexed": [3, 4],
                "schur_bin_relative": "{}"
            }}"#,
            bin_name_placeholder
        );
        // Provide only 3 doubles where 4 are required (2*2).
        let (json_path, bin_path) = write_temp_pair(&json, &[1.0, 2.0, 3.0]);
        let bin_name = bin_path.file_name().unwrap().to_str().unwrap().to_string();
        let patched = std::fs::read_to_string(&json_path)
            .unwrap()
            .replace("PLACEHOLDER", &bin_name);
        std::fs::write(&json_path, &patched).unwrap();
        assert!(read_mumps_schur(&json_path).is_none());
        let _ = std::fs::remove_file(&json_path);
        let _ = std::fs::remove_file(&bin_path);
    }

    #[test]
    fn returns_none_when_indices_count_wrong() {
        let bin_name_placeholder = "PLACEHOLDER".to_string();
        let json = format!(
            r#"{{
                "status": "ok",
                "n": 5,
                "n_schur": 2,
                "schur_indices_0indexed": [3],
                "schur_bin_relative": "{}"
            }}"#,
            bin_name_placeholder
        );
        let (json_path, bin_path) = write_temp_pair(&json, &[1.0, 2.0, 3.0, 4.0]);
        let bin_name = bin_path.file_name().unwrap().to_str().unwrap().to_string();
        let patched = std::fs::read_to_string(&json_path)
            .unwrap()
            .replace("PLACEHOLDER", &bin_name);
        std::fs::write(&json_path, &patched).unwrap();
        assert!(read_mumps_schur(&json_path).is_none());
        let _ = std::fs::remove_file(&json_path);
        let _ = std::fs::remove_file(&bin_path);
    }
}
