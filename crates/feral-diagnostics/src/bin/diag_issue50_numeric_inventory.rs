//! Issue #50 — numeric inventory probe for `CHAIN_CATCH_N_CAP`
//! calibration.
//!
//! `diag_issue50_inventory` recorded `factor_nnz_estimate` per
//! ordering and concluded AMD wins 78% of chain-catch matrices.
//! That conclusion was wrong: the chain catch was calibrated
//! against numeric `num_nnz_l` after BK pivoting cascade
//! (`dev/journal/2026-04-27-02.org`), and CHAINWOO_0000 shows the
//! gap is entirely numeric (AMD sym_est 68k ≈ MetisND 70k; AMD
//! num_nnz_l 2.10M vs MetisND 282k). This probe runs the *numeric*
//! factorization and records actual `factor_nnz()` per ordering.
//!
//! Output: CSV row per chain-catch-firing matrix on stdout with
//! columns:
//!
//! ```text
//! dir,file,n,stored_nnz,full_avg_deg,pdm_branch,ca_branch,
//! amd_num_nnz_l,metis_num_nnz_l,scotch_num_nnz_l,
//! amd_us,metis_us,scotch_us,
//! amd_status,metis_status,scotch_status,err
//! ```
//!
//! Usage:
//!
//! ```ignore
//! cargo run --release --bin diag_issue50_numeric_inventory \
//!     > /tmp/issue50_numeric_inventory.csv
//! ```
//!
//! Env vars:
//!
//! - `MAX_N` (default 200_000): skip numeric factorization for
//!   matrices larger than this. The numeric phase is ~10x the
//!   symbolic, and the chain catch's load-bearing beneficiaries
//!   all sit below n=10_000, so MAX_N is a safety stop.
//! - `LIMIT` (default unset): cap on number of matrices processed,
//!   for smoke runs.
//! - `ROOTS` (default `data/matrices/kkt,data/matrices/kkt-expansion`):
//!   comma-separated corpus roots.
//! - `CHAIN_CATCH_ONLY` (default `1`): if `1`, skip matrices that
//!   do not hit the chain catch through `pick_default_method`.
//!
//! See `dev/research/issue-50-metisnd-symbolic-cost.md` §F6.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use feral::symbolic::OrderingMethod;
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};

/// Same heuristic as `src/symbolic/mod.rs:291` and the symbolic
/// inventory probe — kept in lock-step.
fn classify_pdm(n: usize, stored_nnz: usize) -> &'static str {
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

fn classify_ca(n: usize, full_avg_deg: f64, pdm: &'static str) -> &'static str {
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

struct Row {
    dir: String,
    file: String,
    n: usize,
    stored_nnz: usize,
    full_avg_deg: f64,
    pdm: &'static str,
    ca: &'static str,
    amd_num_nnz_l: Option<usize>,
    metis_num_nnz_l: Option<usize>,
    scotch_num_nnz_l: Option<usize>,
    amd_us: Option<u64>,
    metis_us: Option<u64>,
    scotch_us: Option<u64>,
    amd_status: String,
    metis_status: String,
    scotch_status: String,
    err: String,
}

impl Row {
    fn header() -> &'static str {
        "dir,file,n,stored_nnz,full_avg_deg,pdm_branch,ca_branch,\
         amd_num_nnz_l,metis_num_nnz_l,scotch_num_nnz_l,\
         amd_us,metis_us,scotch_us,\
         amd_status,metis_status,scotch_status,err"
    }

    fn emit(&self) {
        let cu = |v: Option<usize>| match v {
            Some(x) => x.to_string(),
            None => NA.to_string(),
        };
        let ct = |v: Option<u64>| match v {
            Some(x) => x.to_string(),
            None => NA.to_string(),
        };
        println!(
            "{},{},{},{},{:.4},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.dir,
            self.file,
            self.n,
            self.stored_nnz,
            self.full_avg_deg,
            self.pdm,
            self.ca,
            cu(self.amd_num_nnz_l),
            cu(self.metis_num_nnz_l),
            cu(self.scotch_num_nnz_l),
            ct(self.amd_us),
            ct(self.metis_us),
            ct(self.scotch_us),
            self.amd_status,
            self.metis_status,
            self.scotch_status,
            self.err,
        );
    }
}

/// Same representative-picker as `diag_issue50_inventory`. Prefer
/// `*_0000.mtx` (iter-0 KKT), else first alphabetically.
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

/// Run one numeric factorization under `method`. Returns
/// `(num_nnz_l, factor_us, status_label)`. A label of `"ok"`
/// means the factorization succeeded; otherwise the label
/// encodes the `FactorStatus` variant for downstream filtering.
fn run_one(matrix: &CscMatrix, method: OrderingMethod) -> (Option<usize>, Option<u64>, String) {
    let mut solver = Solver::new().with_ordering(method);
    let t = Instant::now();
    let status = solver.factor(matrix, None);
    let us = t.elapsed().as_micros() as u64;
    let label = match &status {
        FactorStatus::Success => "ok",
        FactorStatus::Singular => "singular",
        FactorStatus::WrongInertia { .. } => "wrong_inertia",
        FactorStatus::FatalError(_) => "fatal",
    };
    let nnz_l = solver.factors().map(|f| f.factor_nnz());
    let factor_us = if matches!(status, FactorStatus::Success) {
        Some(us)
    } else {
        None
    };
    (nnz_l, factor_us, label.to_string())
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

    eprintln!("[numeric-inventory] roots = {:?}", roots);
    eprintln!("[numeric-inventory] MAX_N = {max_n}");
    eprintln!("[numeric-inventory] CHAIN_CATCH_ONLY = {chain_only}");
    if let Some(l) = limit {
        eprintln!("[numeric-inventory] LIMIT = {l}");
    }

    let reps = collect_representatives(&roots);
    eprintln!("[numeric-inventory] {} problem dirs found", reps.len());

    println!("{}", Row::header());

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
        let pdm = classify_pdm(n, stored_nnz);

        if chain_only && !is_chain_catch(pdm) {
            continue;
        }

        let full = csc.symmetric_pattern();
        let full_avg_deg = full.row_idx.len() as f64 / n.max(1) as f64;
        let ca = classify_ca(n, full_avg_deg, pdm);

        let mut row = Row {
            dir: dir.clone(),
            file: path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string(),
            n,
            stored_nnz,
            full_avg_deg,
            pdm,
            ca,
            amd_num_nnz_l: None,
            metis_num_nnz_l: None,
            scotch_num_nnz_l: None,
            amd_us: None,
            metis_us: None,
            scotch_us: None,
            amd_status: NA.to_string(),
            metis_status: NA.to_string(),
            scotch_status: NA.to_string(),
            err: String::new(),
        };

        if n > max_n {
            row.err = format!("skipped: n={} > MAX_N={}", n, max_n);
            row.emit();
            processed += 1;
            continue;
        }

        let (nnz, us, st) = run_one(&csc, OrderingMethod::Amd);
        row.amd_num_nnz_l = nnz;
        row.amd_us = us;
        row.amd_status = st;

        let (nnz, us, st) = run_one(&csc, OrderingMethod::MetisND);
        row.metis_num_nnz_l = nnz;
        row.metis_us = us;
        row.metis_status = st;

        let (nnz, us, st) = run_one(&csc, OrderingMethod::ScotchND);
        row.scotch_num_nnz_l = nnz;
        row.scotch_us = us;
        row.scotch_status = st;

        row.emit();
        processed += 1;

        if last_log.elapsed().as_secs() >= 5 {
            eprintln!(
                "[numeric-inventory] {}/{} dirs scanned, {} chain-catch rows emitted ({:.1} s)",
                i + 1,
                reps.len(),
                processed,
                t_total.elapsed().as_secs_f64(),
            );
            last_log = Instant::now();
        }
    }
    eprintln!(
        "[numeric-inventory] done: {} rows in {:.1} s",
        processed,
        t_total.elapsed().as_secs_f64(),
    );
}
