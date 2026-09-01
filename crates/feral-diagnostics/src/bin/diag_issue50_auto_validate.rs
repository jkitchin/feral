//! Issue #50 — Fix A validation probe.
//!
//! Runs `OrderingMethod::Auto` (which exercises the post-Fix-A
//! dispatch in `pick_default_method` + `choose_adaptive`) on every
//! chain-catch-class representative in the IPM corpus, and emits the
//! resolved method, factor status, factor time, and `num_nnz_l`. Pair
//! with `dev/research/issue-50-numeric-inventory.csv` (AMD/MetisND/
//! ScotchND reference) to detect regressions.
//!
//! Output: CSV row per matrix on stdout with columns:
//!
//! ```text
//! dir,file,n,stored_nnz,full_avg_deg,pdm_branch,ca_branch,
//! auto_resolved,auto_num_nnz_l,auto_us,auto_status,err
//! ```
//!
//! Usage:
//!
//! ```ignore
//! cargo run --release --bin diag_issue50_auto_validate \
//!     > /tmp/issue50_auto_validate.csv
//! ```
//!
//! Env vars (same semantics as `diag_issue50_numeric_inventory`):
//! `MAX_N` (default 200_000), `LIMIT`, `ROOTS`, `CHAIN_CATCH_ONLY`
//! (default 1).
//!
//! See `dev/research/issue-50-metisnd-symbolic-cost.md` §F8.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use feral::symbolic::OrderingMethod;
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};

fn classify_pdm_pre_fix(n: usize, stored_nnz: usize) -> &'static str {
    // Mirrors the pre-Fix-A `pick_default_method`, used only to filter
    // the corpus down to chain-catch-class matrices — the population
    // whose routing actually changed.
    if n == 0 {
        return "empty_amd";
    }
    let avg_deg = stored_nnz as f64 / n as f64;
    if n >= 5000 && avg_deg < 6.0 {
        return "chain_metis_5k_6";
    }
    if n >= 2000 && avg_deg < 4.0 {
        return "chain_metis_2k_4";
    }
    if n <= 10_000 {
        return "small_amf";
    }
    "large_metis"
}

fn classify_ca_pre_fix(n: usize, full_avg_deg: f64, pdm: &'static str) -> &'static str {
    if n == 0 {
        return "empty_amd";
    }
    if n > 100_000 && full_avg_deg < 5.0 {
        return "ca_big_scotch";
    }
    if n < 10_000 && full_avg_deg < 15.0 {
        return "ca_small_kahip";
    }
    pdm
}

fn is_chain_catch(pdm: &str) -> bool {
    pdm == "chain_metis_5k_6" || pdm == "chain_metis_2k_4"
}

const NA: &str = "NA";

fn collect_representatives(roots: &[PathBuf]) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for root in roots {
        let Ok(entries) = std::fs::read_dir(root) else {
            eprintln!("[skip] cannot read {}", root.display());
            continue;
        };
        let mut subdirs: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        subdirs.sort();
        for sub in subdirs {
            let dir_name = sub
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string();
            let Ok(files) = std::fs::read_dir(&sub) else {
                continue;
            };
            let mut mtxs: Vec<PathBuf> = files
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mtx"))
                .collect();
            mtxs.sort();
            let pick = mtxs
                .iter()
                .find(|p| {
                    p.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.ends_with("_0000"))
                        .unwrap_or(false)
                })
                .cloned()
                .or_else(|| mtxs.first().cloned());
            if let Some(p) = pick {
                out.push((dir_name, p));
            }
        }
    }
    out
}

fn run_auto(matrix: &CscMatrix) -> (Option<usize>, Option<u64>, String, String) {
    let mut solver = Solver::new().with_ordering(OrderingMethod::Auto);
    let t = Instant::now();
    let status = solver.factor(matrix, None);
    let us = t.elapsed().as_micros() as u64;
    let label = match &status {
        FactorStatus::Success => "ok",
        FactorStatus::Singular => "singular",
        FactorStatus::WrongInertia { .. } => "wrong_inertia",
        FactorStatus::FatalError(_) => "fatal",
    };
    let (resolved, nnz_l) = match solver.factors() {
        Some(f) => (format!("{:?}", f.resolved_method), Some(f.factor_nnz())),
        None => (NA.to_string(), None),
    };
    let factor_us = if matches!(status, FactorStatus::Success) {
        Some(us)
    } else {
        None
    };
    (nnz_l, factor_us, label.to_string(), resolved)
}

fn main() {
    let max_n: usize = feral::env::usize_var("MAX_N").unwrap_or(200_000);
    let limit: Option<usize> = feral::env::usize_var("LIMIT");
    let chain_only: bool = feral::env::usize_var("CHAIN_CATCH_ONLY").unwrap_or(1) != 0;
    let roots: Vec<PathBuf> = env::var("ROOTS")
        .unwrap_or_else(|_| "data/matrices/kkt,data/matrices/kkt-expansion".to_string())
        .split(',')
        .map(|s| PathBuf::from(s.trim()))
        .collect();

    eprintln!("[auto-validate] roots = {:?}", roots);
    eprintln!("[auto-validate] MAX_N = {max_n}");
    eprintln!("[auto-validate] CHAIN_CATCH_ONLY = {chain_only}");
    if let Some(l) = limit {
        eprintln!("[auto-validate] LIMIT = {l}");
    }

    let reps = collect_representatives(&roots);
    eprintln!("[auto-validate] {} problem dirs found", reps.len());

    println!(
        "dir,file,n,stored_nnz,full_avg_deg,pdm_branch,ca_branch,\
         auto_resolved,auto_num_nnz_l,auto_us,auto_status,err"
    );

    let t_total = Instant::now();
    let mut last_log = Instant::now();
    let mut processed = 0usize;
    let cap = limit.unwrap_or(usize::MAX);

    for (i, (dir, path)) in reps.iter().enumerate() {
        if processed >= cap {
            break;
        }

        let mtx = match read_mtx(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[skip read_mtx] {}: {e:?}", path.display());
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[skip to_csc] {}: {e:?}", path.display());
                continue;
            }
        };

        let n = csc.n;
        let stored_nnz = csc.row_idx.len();
        let pdm = classify_pdm_pre_fix(n, stored_nnz);

        if chain_only && !is_chain_catch(pdm) {
            continue;
        }

        let full = csc.symmetric_pattern();
        let full_avg_deg = full.row_idx.len() as f64 / n.max(1) as f64;
        let ca = classify_ca_pre_fix(n, full_avg_deg, pdm);

        let file = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();

        let mut err = String::new();
        let (nnz, us, st, resolved) = if n > max_n {
            err = format!("skipped: n={} > MAX_N={}", n, max_n);
            (None, None, NA.to_string(), NA.to_string())
        } else {
            run_auto(&csc)
        };

        let cu = |v: Option<usize>| match v {
            Some(x) => x.to_string(),
            None => NA.to_string(),
        };
        let ct = |v: Option<u64>| match v {
            Some(x) => x.to_string(),
            None => NA.to_string(),
        };

        println!(
            "{},{},{},{},{:.4},{},{},{},{},{},{},{}",
            dir,
            file,
            n,
            stored_nnz,
            full_avg_deg,
            pdm,
            ca,
            resolved,
            cu(nnz),
            ct(us),
            st,
            err,
        );
        processed += 1;

        if last_log.elapsed().as_secs() >= 5 {
            eprintln!(
                "[auto-validate] {}/{} dirs scanned, {} chain-catch rows emitted ({:.1} s)",
                i + 1,
                reps.len(),
                processed,
                t_total.elapsed().as_secs_f64(),
            );
            last_log = Instant::now();
        }
    }
    eprintln!(
        "[auto-validate] done: {} rows in {:.1} s",
        processed,
        t_total.elapsed().as_secs_f64(),
    );
}
