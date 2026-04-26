//! Run feral default settings over the Mittelmann KKT harvest
//! (`data/matrices/kkt-mittelmann/<problem>/*.mtx`) and report
//! per-matrix correctness gates plus an aggregate breakdown.
//!
//! Correctness gates per matrix (the same the bench harness uses on
//! the canonical kkt corpus):
//!   1. `inertia` matches the IPM-emitted sidecar inertia exactly.
//!   2. ‖Ax − b‖ / ‖b‖ ≤ n · ε · 1e6 (b from sidecar `rhs`).
//!
//! Plus a fill diagnostic (no reference yet, just feral's own ratio):
//!   3. `nnz_L / nnz(A)` for ordering-quality monitoring.
//!
//! A factorization that succeeds but fails inertia or residual is
//! counted in `n_inertia_fail` / `n_residual_fail`, NOT in `n_ok`.
//! Numeric breakdowns are still `n_num_err`.
//!
//! Output sections:
//!   1. Per-problem progress line: ok/total, skip count, residual fail
//!      count, inertia fail count.
//!   2. Aggregate: counts, ordering / amalgamation / preprocess /
//!      scaling distributions, p50/p95/p99 of factor_us, residual
//!      max, fill-ratio p50/p95/p99, total n_2x2, total n_delayed.
//!
//! Usage:
//!   cargo run --release --bin diag_mittelmann
//!   cargo run --release --bin diag_mittelmann -- arki0003 ex8_2_2
//!   FERAL_DIAG_MAX_N=20000 cargo run --release --bin diag_mittelmann
//!   DIAG_VERBOSE=1 cargo run --release --bin diag_mittelmann -- arki0003

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, solve_sparse, BunchKaufmanParams, Inertia};

const ROOT: &str = "data/matrices/kkt-mittelmann";

fn percentile_u128(v: &mut [u128], q: f64) -> u128 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    let idx = ((v.len() as f64) * q).floor() as usize;
    v[idx.min(v.len() - 1)]
}

fn percentile_f64(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() as f64) * q).floor() as usize;
    v[idx.min(v.len() - 1)]
}

#[derive(Default)]
struct Aggregate {
    n_seen: usize,
    n_ok: usize,
    n_mtx_err: usize,
    n_csc_err: usize,
    n_sym_err: usize,
    n_num_err: usize,
    n_inertia_fail: usize,
    n_residual_fail: usize,
    n_no_sidecar: usize,
    n_skipped_size: usize,
    ordering: BTreeMap<String, usize>,
    amalgamation: BTreeMap<String, usize>,
    preprocess: BTreeMap<String, usize>,
    scaling: BTreeMap<String, usize>,
    total_n_2x2: usize,
    total_n_delayed: usize,
    factor_us: Vec<u128>,
    fill_ratio: Vec<f64>,
    residuals: Vec<f64>,
    worst_residual: f64,
    worst_residual_name: String,
    worst_fill: f64,
    worst_fill_name: String,
    largest_n: usize,
    largest_name: String,
}

#[derive(Default)]
struct ProbeResult {
    inertia_fail: bool,
    residual_fail: bool,
    no_sidecar: bool,
}

fn run_one(mtx_path: &Path, agg: &mut Aggregate, verbose: bool, max_n: usize) -> ProbeResult {
    let mut pr = ProbeResult::default();
    agg.n_seen += 1;
    let name = mtx_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<?>")
        .to_string();

    let mtx = match read_mtx(mtx_path) {
        Ok(m) => m,
        Err(e) => {
            agg.n_mtx_err += 1;
            if verbose {
                println!("    {:30}  MTX_ERR  {}", name, e);
            }
            return pr;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            agg.n_csc_err += 1;
            if verbose {
                println!("    {:30}  CSC_ERR  {}", name, e);
            }
            return pr;
        }
    };
    if csc.n > agg.largest_n {
        agg.largest_n = csc.n;
        agg.largest_name = name.clone();
    }
    if csc.n > max_n {
        agg.n_skipped_size += 1;
        if verbose {
            println!("    {:30}  SKIP_BIG  n={} > {}", name, csc.n, max_n);
        }
        return pr;
    }

    // Sidecar is the source of truth for expected inertia and rhs.
    let sidecar_path = mtx_path.with_extension("json");
    let sidecar = match read_sidecar(&sidecar_path) {
        Ok(s) => Some(s),
        Err(_) => {
            agg.n_no_sidecar += 1;
            pr.no_sidecar = true;
            None
        }
    };

    let snode = SupernodeParams::default();
    let np = NumericParams::with_bk(BunchKaufmanParams::default());

    let sym = match symbolic_factorize(&csc, &snode) {
        Ok(s) => s,
        Err(e) => {
            agg.n_sym_err += 1;
            if verbose {
                println!("    {:30}  SYM_ERR  {}", name, e);
            }
            return pr;
        }
    };

    let t = Instant::now();
    let (factors, computed_inertia) = match factorize_multifrontal(&csc, &sym, &np) {
        Ok(pair) => pair,
        Err(e) => {
            agg.n_num_err += 1;
            if verbose {
                println!("    {:30}  NUM_ERR  {}", name, e);
            }
            return pr;
        }
    };
    let factor_us = t.elapsed().as_micros();

    *agg.ordering
        .entry(format!("{:?}", factors.resolved_method))
        .or_insert(0) += 1;
    *agg.amalgamation
        .entry(format!("{:?}", factors.resolved_amalgamation))
        .or_insert(0) += 1;
    *agg.preprocess
        .entry(format!("{:?}", factors.resolved_preprocess))
        .or_insert(0) += 1;
    *agg.scaling
        .entry(format!("{:?}", factors.scaling_info))
        .or_insert(0) += 1;

    // Pivot/delay tally (mirrors summary() but kept here for the
    // aggregate counters; cheap, O(supernodes)).
    let mut n_2x2 = 0usize;
    let mut n_delayed = 0usize;
    for nf in &factors.node_factors {
        let ff = &nf.frontal_factors;
        n_delayed += ff.n_delayed;
        let nelim = ff.nelim;
        let mut k = 0;
        while k < nelim {
            let two = k + 1 < nelim && ff.d_subdiag[k] != 0.0;
            if two {
                n_2x2 += 1;
                k += 2;
            } else {
                k += 1;
            }
        }
    }
    agg.total_n_2x2 += n_2x2;
    agg.total_n_delayed += n_delayed;
    agg.factor_us.push(factor_us);

    // Fill ratio: nnz_L / nnz(A). Lower-triangle nnz from CSC.
    let nnz_a = csc.values.len().max(1);
    let nnz_l = factors.factor_nnz();
    let fill_ratio = nnz_l as f64 / nnz_a as f64;
    agg.fill_ratio.push(fill_ratio);
    if fill_ratio > agg.worst_fill {
        agg.worst_fill = fill_ratio;
        agg.worst_fill_name = name.clone();
    }

    // Inertia gate.
    if let Some(ref sc) = sidecar {
        let expected = Inertia::new(sc.inertia.positive, sc.inertia.negative, sc.inertia.zero);
        if computed_inertia != expected {
            agg.n_inertia_fail += 1;
            pr.inertia_fail = true;
            if verbose {
                println!(
                    "    {:30}  INERTIA_FAIL  got=({},{},{}) want=({},{},{})",
                    name,
                    computed_inertia.positive,
                    computed_inertia.negative,
                    computed_inertia.zero,
                    expected.positive,
                    expected.negative,
                    expected.zero,
                );
            }
        }
    }

    // Residual gate (only when the sidecar provides a finite RHS).
    let mut residual_rel = f64::NAN;
    if let Some(ref sc) = sidecar {
        if let Some(rhs) = sc.finite_rhs() {
            if rhs.len() == csc.n {
                match solve_sparse(&factors, &rhs) {
                    Ok(x) => {
                        let mut ax = vec![0.0; csc.n];
                        csc.symv(&x, &mut ax);
                        let mut r2 = 0.0;
                        let mut b2 = 0.0;
                        for i in 0..csc.n {
                            let r = ax[i] - rhs[i];
                            r2 += r * r;
                            b2 += rhs[i] * rhs[i];
                        }
                        residual_rel = if b2 > 0.0 {
                            (r2 / b2).sqrt()
                        } else {
                            r2.sqrt()
                        };
                        agg.residuals.push(residual_rel);
                        if residual_rel > agg.worst_residual {
                            agg.worst_residual = residual_rel;
                            agg.worst_residual_name = name.clone();
                        }
                        let tol = (csc.n as f64) * f64::EPSILON * 1e6;
                        if !(residual_rel.is_finite() && residual_rel <= tol) {
                            agg.n_residual_fail += 1;
                            pr.residual_fail = true;
                            if verbose {
                                println!(
                                    "    {:30}  RES_FAIL  rel_res={:.3e} tol={:.3e}",
                                    name, residual_rel, tol
                                );
                            }
                        }
                    }
                    Err(e) => {
                        agg.n_num_err += 1;
                        if verbose {
                            println!("    {:30}  SOLVE_ERR  {}", name, e);
                        }
                        return pr;
                    }
                }
            }
        }
    }

    // Promote to OK only if neither gate failed (and a sidecar existed).
    if !pr.inertia_fail && !pr.residual_fail && !pr.no_sidecar {
        agg.n_ok += 1;
    }

    if verbose {
        let summary = factors.summary();
        let res_str = if residual_rel.is_finite() {
            format!("{:.2e}", residual_rel)
        } else {
            "n/a".to_string()
        };
        println!(
            "    {:30}  {:>7} μs  fill={:5.2}  res={}  {}",
            name, factor_us, fill_ratio, res_str, summary
        );
    }

    pr
}

fn list_problems(filter: &[String]) -> Vec<PathBuf> {
    let root = Path::new(ROOT);
    let mut out = Vec::new();
    if !root.is_dir() {
        eprintln!(
            "ERROR: {} not found; run scripts/harvest-mittelmann-kkt.sh",
            ROOT
        );
        return out;
    }
    let entries: Vec<_> = match std::fs::read_dir(root) {
        Ok(d) => d.filter_map(|e| e.ok()).collect(),
        Err(_) => return out,
    };
    let mut paths: Vec<_> = entries.into_iter().map(|e| e.path()).collect();
    paths.sort();
    for p in paths {
        if !p.is_dir() {
            continue;
        }
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if !filter.is_empty() && !filter.contains(&name) {
            continue;
        }
        out.push(p);
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verbose = std::env::var("DIAG_VERBOSE").is_ok();
    let max_n: usize = std::env::var("FERAL_DIAG_MAX_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    let problems = list_problems(&args);
    if problems.is_empty() {
        eprintln!("no problem directories matched");
        std::process::exit(1);
    }

    println!("=== feral diag_mittelmann ===");
    println!("root:    {}", ROOT);
    println!("verbose: {} (DIAG_VERBOSE=1 for per-matrix lines)", verbose);
    println!(
        "max_n  : {} (FERAL_DIAG_MAX_N to override; matrices above this skip factor)",
        max_n
    );
    println!("problems: {}", problems.len());
    println!();
    let _ = std::io::stdout().flush();

    let mut agg = Aggregate::default();

    for prob_dir in &problems {
        let prob_name = prob_dir.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        let mut mtxs: Vec<_> = match std::fs::read_dir(prob_dir) {
            Ok(d) => d
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "mtx"))
                .collect(),
            Err(_) => continue,
        };
        mtxs.sort();
        if verbose {
            println!("[{}]  {} mtx", prob_name, mtxs.len());
        } else {
            print!("[{:25}] {:3} mtx ... ", prob_name, mtxs.len());
        }
        let _ = std::io::stdout().flush();
        let before_ok = agg.n_ok;
        let before_total = agg.n_seen;
        let before_skipped = agg.n_skipped_size;
        let before_inertia = agg.n_inertia_fail;
        let before_resid = agg.n_residual_fail;

        for mtx in &mtxs {
            run_one(mtx, &mut agg, verbose, max_n);
        }

        if !verbose {
            let ok = agg.n_ok - before_ok;
            let total = agg.n_seen - before_total;
            let sk = agg.n_skipped_size - before_skipped;
            let ifl = agg.n_inertia_fail - before_inertia;
            let rfl = agg.n_residual_fail - before_resid;
            println!(
                "OK {:3}/{:3} | skip {:2} | inertia_fail {:2} | res_fail {:2}",
                ok, total, sk, ifl, rfl
            );
            let _ = std::io::stdout().flush();
        }
    }

    println!();
    println!("=== aggregate ===");
    println!("n_seen          : {}", agg.n_seen);
    println!("n_ok            : {}", agg.n_ok);
    println!("n_mtx_err       : {}", agg.n_mtx_err);
    println!("n_csc_err       : {}", agg.n_csc_err);
    println!("n_sym_err       : {}", agg.n_sym_err);
    println!("n_num_err       : {}", agg.n_num_err);
    println!("n_inertia_fail  : {}", agg.n_inertia_fail);
    println!("n_residual_fail : {}", agg.n_residual_fail);
    println!("n_no_sidecar    : {}", agg.n_no_sidecar);
    println!("n_skipped       : {} (n > {})", agg.n_skipped_size, max_n);
    println!("largest_n       : {} ({})", agg.largest_n, agg.largest_name);
    println!("total_n_2x2     : {}", agg.total_n_2x2);
    println!("total_n_delayed : {}", agg.total_n_delayed);

    if !agg.factor_us.is_empty() {
        let p50 = percentile_u128(&mut agg.factor_us.clone(), 0.50);
        let p95 = percentile_u128(&mut agg.factor_us.clone(), 0.95);
        let p99 = percentile_u128(&mut agg.factor_us.clone(), 0.99);
        println!("factor_us p50   : {} μs", p50);
        println!("factor_us p95   : {} μs", p95);
        println!("factor_us p99   : {} μs", p99);
    }

    if !agg.fill_ratio.is_empty() {
        let p50 = percentile_f64(&mut agg.fill_ratio.clone(), 0.50);
        let p95 = percentile_f64(&mut agg.fill_ratio.clone(), 0.95);
        let p99 = percentile_f64(&mut agg.fill_ratio.clone(), 0.99);
        println!("fill_ratio p50  : {:.2}× nnz(A)", p50);
        println!("fill_ratio p95  : {:.2}× nnz(A)", p95);
        println!("fill_ratio p99  : {:.2}× nnz(A)", p99);
        println!(
            "fill_ratio max  : {:.2}× nnz(A)  ({})",
            agg.worst_fill, agg.worst_fill_name
        );
    }

    if !agg.residuals.is_empty() {
        let p50 = percentile_f64(&mut agg.residuals.clone(), 0.50);
        let p95 = percentile_f64(&mut agg.residuals.clone(), 0.95);
        let p99 = percentile_f64(&mut agg.residuals.clone(), 0.99);
        println!("residual p50    : {:.2e}", p50);
        println!("residual p95    : {:.2e}", p95);
        println!("residual p99    : {:.2e}", p99);
        println!(
            "residual max    : {:.2e}  ({})",
            agg.worst_residual, agg.worst_residual_name
        );
    }

    fn print_dist(label: &str, m: &BTreeMap<String, usize>) {
        println!("{}", label);
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        for (k, n) in v {
            println!("    {:30} {:>6}", k, n);
        }
    }
    print_dist("ordering distribution:", &agg.ordering);
    print_dist("amalgamation distribution:", &agg.amalgamation);
    print_dist("preprocess distribution:", &agg.preprocess);
    print_dist("scaling distribution:", &agg.scaling);
}
