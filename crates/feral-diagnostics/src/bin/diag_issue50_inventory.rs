//! Issue #50 — corpus inventory for `pick_default_method` /
//! `choose_adaptive` re-calibration.
//!
//! Walks `data/matrices/kkt` and `data/matrices/kkt-expansion` (the
//! IPM corpus), takes one representative `.mtx` per problem
//! directory, and records:
//!
//! - matrix shape: `n`, stored `nnz`, stored `avg_deg`, full
//!   `avg_deg` (post `symmetric_pattern`),
//! - dispatcher classification: which branch of `pick_default_method`
//!   fires; which branch of `choose_adaptive` fires (the path that
//!   actually runs through `Solver::Auto`),
//! - symbolic outcome: `factor_nnz_estimate` and total symbolic wall
//!   for `Amd`, `MetisND`, and `ScotchND`.
//!
//! The output is a CSV row per matrix on stdout. Run as:
//!
//! ```ignore
//! cargo run --release --bin diag_issue50_inventory \
//!     > /tmp/issue50_inventory.csv
//! ```
//!
//! Env vars:
//!
//! - `MAX_N` (default 200_000): skip the three-way symbolic
//!   comparison for matrices larger than this. Still emits the row
//!   with dispatcher classification so the size distribution
//!   stays complete.
//! - `LIMIT` (default unset): cap on number of matrices processed,
//!   for smoke runs.
//! - `ROOTS` (default `data/matrices/kkt,data/matrices/kkt-expansion`):
//!   comma-separated corpus roots.
//!
//! This is the "inventory probe" prerequisite from issue #50's Fix A
//! validation plan — see `dev/research/issue-50-metisnd-symbolic-cost.md`.

use std::env;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use feral::symbolic::{
    symbolic_factorize_with_method, OrderingMethod, SupernodeParams, SymbolicProfiler,
};
use feral::{read_mtx, CscMatrix};

/// Classify a matrix under `pick_default_method`'s rules. Returns a
/// short string that matches one of the documented branches.
///
/// Kept in lock-step with `src/symbolic/mod.rs:291` — update both
/// together if the heuristic changes.
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

/// Classify under `choose_adaptive`'s rules. Mirrors
/// `src/symbolic/mod.rs:142`. `full_avg_deg` is from the symmetric
/// pattern (not the stored lower triangle).
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
    // Falls through to pick_default_method.
    pdm
}

/// Sentinel value emitted in CSV cells when a measurement is
/// suppressed by the `MAX_N` budget. Chosen so it's easy to filter
/// out downstream (`awk '$N != "NA"'`).
const NA: &str = "NA";

#[derive(Default)]
struct Row {
    dir: String,
    file: String,
    n: usize,
    stored_nnz: usize,
    full_nnz: usize,
    pdm: &'static str,
    ca: &'static str,
    amd_nnz: Option<usize>,
    metis_nnz: Option<usize>,
    scotch_nnz: Option<usize>,
    amd_us: Option<u64>,
    metis_us: Option<u64>,
    scotch_us: Option<u64>,
    err: Option<String>,
}

impl Row {
    fn header() -> &'static str {
        "dir,file,n,stored_nnz,stored_avg_deg,full_nnz,full_avg_deg,\
         pdm_branch,ca_branch,\
         amd_nnz,metis_nnz,scotch_nnz,\
         amd_us,metis_us,scotch_us,err"
    }

    fn emit(&self) {
        let stored_avg = self.stored_nnz as f64 / self.n.max(1) as f64;
        let full_avg = self.full_nnz as f64 / self.n.max(1) as f64;
        let cell_u = |v: Option<usize>| match v {
            Some(x) => x.to_string(),
            None => NA.to_string(),
        };
        let cell_t = |v: Option<u64>| match v {
            Some(x) => x.to_string(),
            None => NA.to_string(),
        };
        let err = self.err.as_deref().unwrap_or("");
        println!(
            "{},{},{},{},{:.4},{},{:.4},{},{},{},{},{},{},{},{},{}",
            self.dir,
            self.file,
            self.n,
            self.stored_nnz,
            stored_avg,
            self.full_nnz,
            full_avg,
            self.pdm,
            self.ca,
            cell_u(self.amd_nnz),
            cell_u(self.metis_nnz),
            cell_u(self.scotch_nnz),
            cell_t(self.amd_us),
            cell_t(self.metis_us),
            cell_t(self.scotch_us),
            err,
        );
    }
}

/// Pick one representative `.mtx` per problem directory. Prefer
/// `*_0000.mtx` (the IPM-loop-iter-0 snapshot), else the first
/// alphabetically. Returns `(problem_dir_relative, mtx_path)`.
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

fn run_one_method(matrix: &CscMatrix, method: OrderingMethod) -> Result<(usize, u64), String> {
    let prof = Arc::new(Mutex::new(SymbolicProfiler::new()));
    let params = SupernodeParams {
        symbolic_profiler: Some(Arc::clone(&prof)),
        ..SupernodeParams::default()
    };
    let t = Instant::now();
    let sym =
        symbolic_factorize_with_method(matrix, &params, method).map_err(|e| format!("{e:?}"))?;
    let us = t.elapsed().as_micros() as u64;
    Ok((sym.factor_nnz_estimate, us))
}

fn main() {
    let max_n: usize = env::var("MAX_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let limit: Option<usize> = env::var("LIMIT").ok().and_then(|s| s.parse().ok());
    let roots: Vec<PathBuf> = env::var("ROOTS")
        .unwrap_or_else(|_| "data/matrices/kkt,data/matrices/kkt-expansion".to_string())
        .split(',')
        .map(|s| PathBuf::from(s.trim()))
        .collect();

    eprintln!("[inventory] roots = {:?}", roots);
    eprintln!("[inventory] MAX_N = {max_n}");
    if let Some(l) = limit {
        eprintln!("[inventory] LIMIT = {l}");
    }

    let reps = collect_representatives(&roots);
    eprintln!("[inventory] {} problem dirs found", reps.len());

    println!("{}", Row::header());

    let take = limit.unwrap_or(reps.len());
    let t_total = Instant::now();
    let mut last_log = Instant::now();
    for (i, (dir, path)) in reps.iter().take(take).enumerate() {
        let mut row = Row {
            dir: dir.clone(),
            file: path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string(),
            ..Default::default()
        };

        let mtx = match read_mtx(path) {
            Ok(m) => m,
            Err(e) => {
                row.err = Some(format!("read_mtx: {e:?}"));
                row.emit();
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                row.err = Some(format!("to_csc: {e:?}"));
                row.emit();
                continue;
            }
        };

        row.n = csc.n;
        row.stored_nnz = csc.row_idx.len();
        let full = csc.symmetric_pattern();
        row.full_nnz = full.row_idx.len();
        row.pdm = classify_pdm(row.n, row.stored_nnz);
        let full_avg = row.full_nnz as f64 / row.n.max(1) as f64;
        row.ca = classify_ca(row.n, full_avg, row.pdm);

        if row.n <= max_n {
            match run_one_method(&csc, OrderingMethod::Amd) {
                Ok((nnz, us)) => {
                    row.amd_nnz = Some(nnz);
                    row.amd_us = Some(us);
                }
                Err(e) => row.err = Some(format!("amd: {e}")),
            }
            match run_one_method(&csc, OrderingMethod::MetisND) {
                Ok((nnz, us)) => {
                    row.metis_nnz = Some(nnz);
                    row.metis_us = Some(us);
                }
                Err(e) => row.err = Some(format!("metis: {e}")),
            }
            match run_one_method(&csc, OrderingMethod::ScotchND) {
                Ok((nnz, us)) => {
                    row.scotch_nnz = Some(nnz);
                    row.scotch_us = Some(us);
                }
                Err(e) => row.err = Some(format!("scotch: {e}")),
            }
        }

        row.emit();

        if last_log.elapsed().as_secs() >= 5 {
            eprintln!(
                "[inventory] {}/{} done ({:.1} s elapsed)",
                i + 1,
                take,
                t_total.elapsed().as_secs_f64()
            );
            last_log = Instant::now();
        }
    }
    eprintln!(
        "[inventory] done in {:.1} s",
        t_total.elapsed().as_secs_f64()
    );
}
