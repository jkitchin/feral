//! F2.2 — Cross-validate feral's `estimate_condition_1norm` against
//! MUMPS RINFOG(11) (COND2) on the corpus.
//!
//! For each KKT matrix with a fresh `<id>.mumps.json` sidecar that
//! includes the `conditioning` block (mumps_bench.F with ICNTL(11)=1),
//! we factor with feral, call `estimate_condition_1norm`, and report the
//! ratio `kappa_feral / max(cond1, cond2)`.
//!
//! Caveat: MUMPS RINFOG(10)/(11) are componentwise condition numbers in
//! the ∞-norm (Arioli-Demmel-Duff; see dsol_aux.F:935 in the MUMPS
//! source), not `||A||_1·||A^{-1}||_1`. The two estimators share the
//! Hager–Higham 1-norm power iteration but apply it to different
//! operators, so apples-to-apples agreement is not expected. We compare
//! orders of magnitude. The acceptance gate from
//! dev/plans/kkt-feature-gaps.md F2.2 says "geomean ratio within
//! [0.5, 5.0]" — that target is calibrated against a 1-norm oracle, so
//! we report a wider [0.1, 10.0] band against COND2 and treat the
//! geomean as a directional check.
//!
//! Inputs: `<id>.mtx` + `<id>.json` (RHS sidecar) + `<id>.mumps.json`
//! (oracle conditioning).
//!
//! Usage:
//!   cargo run --release --bin diag_cond_parity
//!   cargo run --release --bin diag_cond_parity -- data/matrices/kkt
//!   FERAL_DIAG_MAX_N=20000 cargo run --release --bin diag_cond_parity
//!   FERAL_DIAG_VERBOSE=1 cargo run --release --bin diag_cond_parity

use std::path::{Path, PathBuf};

use feral::numeric::condition::estimate_condition_1norm;
use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, ZeroPivotAction};

const DEFAULT_ROOTS: &[&str] = &[
    "data/matrices/kkt",
    "data/matrices/kkt-expansion",
    "data/matrices/kkt-mittelmann",
];

fn percentile_f64(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() as f64) * q).floor() as usize;
    v[idx.min(v.len() - 1)]
}

fn geomean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let s: f64 = v.iter().map(|x| x.max(1e-300).ln()).sum();
    (s / v.len() as f64).exp()
}

/// Pull the `conditioning.cond1` / `cond2` floats out of a MUMPS sidecar.
/// Returns `None` if the file is missing, the run failed, or the
/// conditioning block is absent (older sidecar written before F2.2).
/// Zero-valued fields are treated as "MUMPS could not estimate" and
/// dropped (the Python runner's `_opt_float` already maps 0.0 to null,
/// but a stale sidecar may carry literal 0.0).
fn read_mumps_conditioning(path: &Path) -> Option<(Option<f64>, Option<f64>)> {
    let text = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    if v.get("factorization_status").and_then(|s| s.as_str()) != Some("ok") {
        return None;
    }
    let c = v.get("conditioning")?;
    let pull = |key: &str| -> Option<f64> {
        c.get(key)
            .and_then(|x| x.as_f64())
            .filter(|x| x.is_finite() && *x > 0.0)
    };
    let cond1 = pull("cond1");
    let cond2 = pull("cond2");
    if cond1.is_none() && cond2.is_none() {
        return None;
    }
    Some((cond1, cond2))
}

#[derive(Default)]
struct Aggregate {
    seen: usize,
    skipped_no_mtx: usize,
    skipped_filter: usize,
    skipped_size: usize,
    skipped_factor_err: usize,
    skipped_no_oracle: usize,
    compared: usize,
    ratios_vs_cond2: Vec<f64>,
    ratios_vs_max_cond: Vec<f64>,
    worst_ratio: (f64, String),
    best_ratio: (f64, String),
}

impl Aggregate {
    fn record(&mut self, name: &str, kappa_feral: f64, cond1: Option<f64>, cond2: Option<f64>) {
        let max_cond = match (cond1, cond2) {
            (Some(a), Some(b)) => a.max(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => return,
        };
        if max_cond <= 0.0 || !max_cond.is_finite() {
            return;
        }
        if !kappa_feral.is_finite() || kappa_feral <= 0.0 {
            return;
        }
        self.compared += 1;
        if let Some(c2) = cond2 {
            self.ratios_vs_cond2.push(kappa_feral / c2);
        }
        let ratio = kappa_feral / max_cond;
        self.ratios_vs_max_cond.push(ratio);
        let dev = (ratio.ln()).abs();
        if dev > self.worst_ratio.0 {
            self.worst_ratio = (dev, format!("{} ratio={:.3e}", name, ratio));
        }
        if self.best_ratio.0 == 0.0 || dev < self.best_ratio.0 {
            self.best_ratio = (dev, format!("{} ratio={:.3e}", name, ratio));
        }
    }
}

fn run_one(mtx_path: &Path, agg: &mut Aggregate, max_n: usize, verbose: bool) {
    agg.seen += 1;
    let name = mtx_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<?>")
        .to_string();

    let mtx = match read_mtx(mtx_path) {
        Ok(m) => m,
        Err(_) => {
            agg.skipped_no_mtx += 1;
            return;
        }
    };

    let sidecar_path = mtx_path.with_extension("json");
    let sidecar = match read_sidecar(&sidecar_path) {
        Ok(s) => s,
        Err(_) => {
            agg.skipped_filter += 1;
            return;
        }
    };
    if sidecar.finite_rhs().is_none() {
        agg.skipped_filter += 1;
        return;
    }
    if mtx.entries.iter().any(|(_, _, v)| !v.is_finite()) {
        agg.skipped_filter += 1;
        return;
    }
    if mtx.n != sidecar.n + sidecar.m {
        agg.skipped_filter += 1;
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

    if csc.n > max_n {
        agg.skipped_size += 1;
        return;
    }

    let oracle = match read_mumps_conditioning(&mtx_path.with_extension("mumps.json")) {
        Some(c) => c,
        None => {
            agg.skipped_no_oracle += 1;
            return;
        }
    };

    let snode = SupernodeParams::default();
    let bk = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    let np = NumericParams::with_bk(bk);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let sym = symbolic_factorize_with_method(&csc, &snode, OrderingMethod::Auto)?;
        let (factors, _) = factorize_multifrontal(&csc, &sym, &np)?;
        estimate_condition_1norm(&csc, &factors)
    }));
    let kappa = match result {
        Ok(Ok(k)) => k,
        _ => {
            agg.skipped_factor_err += 1;
            if verbose {
                eprintln!("FACTOR_FAIL {}", name);
            }
            return;
        }
    };

    if verbose {
        eprintln!(
            "OK {} kappa_feral={:.3e} cond1={:?} cond2={:?}",
            name, kappa, oracle.0, oracle.1
        );
    }
    agg.record(&name, kappa, oracle.0, oracle.1);
}

fn report(agg: &Aggregate) {
    println!("=== diag_cond_parity ===");
    println!("seen:                {}", agg.seen);
    println!("  skipped no .mtx:   {}", agg.skipped_no_mtx);
    println!("  skipped filter:    {}", agg.skipped_filter);
    println!("  skipped n>max:     {}", agg.skipped_size);
    println!("  skipped factor:    {}", agg.skipped_factor_err);
    println!("  skipped no oracle: {}", agg.skipped_no_oracle);
    println!("compared:            {}", agg.compared);
    if agg.ratios_vs_cond2.is_empty() && agg.ratios_vs_max_cond.is_empty() {
        println!("(no comparisons — regenerate sidecars with ICNTL(11)=1)");
        return;
    }
    println!();
    println!("kappa_feral / cond2  (RINFOG(11), MUMPS COND2 ∞-norm):");
    let mut r2 = agg.ratios_vs_cond2.clone();
    if !r2.is_empty() {
        println!("  count:   {}", r2.len());
        println!("  geomean: {:.3e}", geomean(&r2));
        println!("  min:     {:.3e}", percentile_f64(&mut r2, 0.0));
        println!("  p10:     {:.3e}", percentile_f64(&mut r2, 0.10));
        println!("  median:  {:.3e}", percentile_f64(&mut r2, 0.50));
        println!("  p90:     {:.3e}", percentile_f64(&mut r2, 0.90));
        println!("  max:     {:.3e}", percentile_f64(&mut r2, 1.0));
    }
    println!();
    println!("kappa_feral / max(cond1, cond2):");
    let mut rm = agg.ratios_vs_max_cond.clone();
    if !rm.is_empty() {
        println!("  count:   {}", rm.len());
        println!("  geomean: {:.3e}", geomean(&rm));
        println!("  median:  {:.3e}", percentile_f64(&mut rm, 0.50));
        println!("  p10:     {:.3e}", percentile_f64(&mut rm, 0.10));
        println!("  p90:     {:.3e}", percentile_f64(&mut rm, 0.90));
    }
    println!();
    println!("extremes (largest |ln(ratio)|):");
    println!("  worst: {}", agg.worst_ratio.1);
    println!("  best:  {}", agg.best_ratio.1);
}

fn walk_root(root: &Path, agg: &mut Aggregate, max_n: usize, verbose: bool) {
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
            walk_root(&p, agg, max_n, verbose);
        } else if p.extension().is_some_and(|ext| ext == "mtx") {
            run_one(&p, agg, max_n, verbose);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn write_temp_sidecar(contents: &str) -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("feral_cond_parity_test_{}_{}.json", pid, id));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn parses_full_conditioning_block() {
        let p = write_temp_sidecar(
            r#"{
                "factorization_status": "ok",
                "conditioning": {
                    "cond1": 1.5e3,
                    "cond2": 2.0e1
                }
            }"#,
        );
        let got = read_mumps_conditioning(&p).expect("should parse");
        assert!((got.0.unwrap() - 1500.0).abs() < 1e-9);
        assert!((got.1.unwrap() - 20.0).abs() < 1e-9);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn drops_zero_cond_fields() {
        let p = write_temp_sidecar(
            r#"{
                "factorization_status": "ok",
                "conditioning": {
                    "cond1": 0.0,
                    "cond2": 1.0e2
                }
            }"#,
        );
        let got = read_mumps_conditioning(&p).expect("should parse");
        assert!(got.0.is_none());
        assert!((got.1.unwrap() - 100.0).abs() < 1e-9);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn returns_none_when_status_not_ok() {
        let p = write_temp_sidecar(
            r#"{
                "factorization_status": "fail",
                "conditioning": {"cond1": 1.5, "cond2": 2.0}
            }"#,
        );
        assert!(read_mumps_conditioning(&p).is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn returns_none_when_conditioning_missing() {
        let p = write_temp_sidecar(
            r#"{
                "factorization_status": "ok",
                "factor_us": 100
            }"#,
        );
        assert!(read_mumps_conditioning(&p).is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn returns_none_when_both_cond_missing() {
        let p = write_temp_sidecar(
            r#"{
                "factorization_status": "ok",
                "conditioning": {"cond1": null, "cond2": null}
            }"#,
        );
        assert!(read_mumps_conditioning(&p).is_none());
        let _ = std::fs::remove_file(&p);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let roots: Vec<PathBuf> = if args.is_empty() {
        DEFAULT_ROOTS.iter().map(PathBuf::from).collect()
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    let max_n = std::env::var("FERAL_DIAG_MAX_N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50_000);
    let verbose = std::env::var("FERAL_DIAG_VERBOSE")
        .map(|v| v == "1")
        .unwrap_or(false);

    eprintln!("FERAL_DIAG_MAX_N = {}", max_n);

    let mut agg = Aggregate::default();
    for root in &roots {
        walk_root(root, &mut agg, max_n, verbose);
    }
    report(&agg);
}
