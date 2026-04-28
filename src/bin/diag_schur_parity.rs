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
//!   4. Computes max relative entry-wise error
//!      `max_{i,j} |feral(i,j) - mumps(i,j)| / max(|feral|, |mumps|, 1)`.
//!
//! Acceptance gate (per dev/research/schur-complement.md D7):
//!   max relative error <= 1e-10 on N >= 100 corpus matrices.
//!
//! Inputs:
//!   <id>.mtx + <id>.json (sidecar) + <id>.mumps_schur.json + <id>.mumps_schur.bin
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

#[derive(Default)]
struct Aggregate {
    seen: usize,
    skipped_no_mtx: usize,
    skipped_no_oracle: usize,
    skipped_size: usize,
    skipped_factor_err: usize,
    compared: usize,
    max_rel_errs: Vec<f64>,
    above_tol: Vec<(String, f64)>,
    worst: (f64, String),
}

impl Aggregate {
    fn record(&mut self, name: &str, max_rel: f64, tol: f64) {
        if !max_rel.is_finite() {
            self.skipped_factor_err += 1;
            return;
        }
        self.compared += 1;
        self.max_rel_errs.push(max_rel);
        if max_rel > tol {
            self.above_tol.push((name.to_string(), max_rel));
        }
        if max_rel > self.worst.0 {
            self.worst = (max_rel, name.to_string());
        }
    }
}

fn run_one(mtx_path: &Path, agg: &mut Aggregate, max_n: usize, tol: f64, verbose: bool) {
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

    // Column-major entry-wise comparison with relative-error denominator
    // max(|a|, |b|, 1.0). The +1.0 floor stops near-zero entries from
    // dominating; max-rel-err over the full block is the F3.3 acceptance
    // metric.
    let dim = schur.dim;
    let mut max_rel: f64 = 0.0;
    for j in 0..dim {
        for i in 0..dim {
            let a = schur.get(i, j);
            let b = oracle.data[j * dim + i];
            let denom = a.abs().max(b.abs()).max(1.0);
            let rel = (a - b).abs() / denom;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }

    if verbose {
        eprintln!(
            "OK {} n={} n_schur={} max_rel_err={:.3e}",
            name, oracle.n, oracle.n_schur, max_rel
        );
    }
    agg.record(&name, max_rel, tol);
}

fn report(agg: &Aggregate, tol: f64) {
    println!("=== diag_schur_parity ===");
    println!("seen:                {}", agg.seen);
    println!("  skipped no .mtx:   {}", agg.skipped_no_mtx);
    println!("  skipped n>max:     {}", agg.skipped_size);
    println!("  skipped no oracle: {}", agg.skipped_no_oracle);
    println!("  skipped factor:    {}", agg.skipped_factor_err);
    println!("compared:            {}", agg.compared);
    println!("acceptance tol:      {:.1e}", tol);
    if agg.max_rel_errs.is_empty() {
        println!("(no comparisons — run run_mumps_schur.py first)");
        return;
    }
    let mut v = agg.max_rel_errs.clone();
    println!();
    println!("max relative entry-wise error (feral vs MUMPS Schur):");
    println!("  count:   {}", v.len());
    println!("  min:     {:.3e}", percentile_f64(&mut v, 0.0));
    println!("  p10:     {:.3e}", percentile_f64(&mut v, 0.10));
    println!("  median:  {:.3e}", percentile_f64(&mut v, 0.50));
    println!("  p90:     {:.3e}", percentile_f64(&mut v, 0.90));
    println!("  p99:     {:.3e}", percentile_f64(&mut v, 0.99));
    println!("  max:     {:.3e}", percentile_f64(&mut v, 1.0));
    println!();
    println!(
        "above tol ({:.1e}): {} / {}",
        tol,
        agg.above_tol.len(),
        agg.compared
    );
    let mut top = agg.above_tol.clone();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (name, err) in top.iter().take(20) {
        println!("  {:<40} {:.3e}", name, err);
    }
    println!();
    println!("worst overall: {} max_rel={:.3e}", agg.worst.1, agg.worst.0);

    let n100 = agg.compared >= 100;
    let pass_tol = agg.above_tol.is_empty();
    println!();
    println!(
        "F3.3 gate (N >= 100 and max_rel <= {:.1e} on all): {}",
        tol,
        if n100 && pass_tol { "PASS" } else { "FAIL" }
    );
}

fn walk_root(root: &Path, agg: &mut Aggregate, max_n: usize, tol: f64, verbose: bool) {
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
            walk_root(&p, agg, max_n, tol, verbose);
        } else if p.extension().is_some_and(|ext| ext == "mtx") {
            run_one(&p, agg, max_n, tol, verbose);
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
    let verbose = std::env::var("FERAL_DIAG_VERBOSE")
        .ok()
        .map(|s| s != "0")
        .unwrap_or(false);

    let mut agg = Aggregate::default();
    for r in &roots {
        walk_root(r, &mut agg, max_n, tol, verbose);
    }
    report(&agg, tol);
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
