//! Small-and-sparse inventory probe (issue #50 F11 follow-up).
//!
//! Walks the IPM corpus, filters to matrices that hit
//! `choose_adaptive`'s (now-deleted) small-and-sparse predicate
//! (`n < 10_000 && full_avg_deg < 15.0`), runs numeric
//! factorization under AMD / AMF / MetisND / KahipND, and emits
//! per-ordering `num_nnz_l` and `factor_us`. Output is a CSV used
//! to decide whether `Auto`'s small-and-sparse branch should keep
//! routing to KahipND, switch to AMD, or fall through to AMF.
//!
//! The chain-catch validation in issue #50 (F11) noted that
//! KahipND produces 1.10–1.72× larger `num_nnz_l` than the best of
//! AMD/MetisND on small chain-catch matrices. KahipND's original
//! justification (K1 reductions finding short cycles AMD misses)
//! came from a 41-matrix bake-off; this probe extends to the full
//! corpus population that the branch actually catches. See
//! `dev/research/issue-50-metisnd-symbolic-cost.md` §F12 for the
//! resulting decision (delete the branch, fall through to AMF).
//!
//! Output CSV columns:
//!
//! ```text
//! dir,file,n,stored_nnz,full_avg_deg,
//! amd_num_nnz_l,amf_num_nnz_l,metis_num_nnz_l,kahip_num_nnz_l,
//! amd_us,amf_us,metis_us,kahip_us,
//! amd_status,amf_status,metis_status,kahip_status,err
//! ```
//!
//! Usage:
//!
//! ```ignore
//! cargo run --release --bin diag_small_sparse_inventory \
//!     > /tmp/small_sparse_inventory.csv
//! ```
//!
//! Env vars: `LIMIT`, `ROOTS`.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use feral::symbolic::OrderingMethod;
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};

const NA: &str = "NA";

struct Row {
    dir: String,
    file: String,
    n: usize,
    stored_nnz: usize,
    full_avg_deg: f64,
    amd_num_nnz_l: Option<usize>,
    amf_num_nnz_l: Option<usize>,
    metis_num_nnz_l: Option<usize>,
    kahip_num_nnz_l: Option<usize>,
    amd_us: Option<u64>,
    amf_us: Option<u64>,
    metis_us: Option<u64>,
    kahip_us: Option<u64>,
    amd_status: String,
    amf_status: String,
    metis_status: String,
    kahip_status: String,
    err: String,
}

impl Row {
    fn header() -> &'static str {
        "dir,file,n,stored_nnz,full_avg_deg,\
         amd_num_nnz_l,amf_num_nnz_l,metis_num_nnz_l,kahip_num_nnz_l,\
         amd_us,amf_us,metis_us,kahip_us,\
         amd_status,amf_status,metis_status,kahip_status,err"
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
            "{},{},{},{},{:.4},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.dir,
            self.file,
            self.n,
            self.stored_nnz,
            self.full_avg_deg,
            cu(self.amd_num_nnz_l),
            cu(self.amf_num_nnz_l),
            cu(self.metis_num_nnz_l),
            cu(self.kahip_num_nnz_l),
            ct(self.amd_us),
            ct(self.amf_us),
            ct(self.metis_us),
            ct(self.kahip_us),
            self.amd_status,
            self.amf_status,
            self.metis_status,
            self.kahip_status,
            self.err,
        );
    }
}

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
    let limit: Option<usize> = feral::env::usize_var("LIMIT");
    let roots: Vec<PathBuf> = env::var("ROOTS")
        .unwrap_or_else(|_| "data/matrices/kkt,data/matrices/kkt-expansion".to_string())
        .split(',')
        .map(|s| PathBuf::from(s.trim()))
        .collect();

    eprintln!("[small-sparse-inventory] roots = {:?}", roots);
    if let Some(l) = limit {
        eprintln!("[small-sparse-inventory] LIMIT = {l}");
    }

    let reps = collect_representatives(&roots);
    eprintln!("[small-sparse-inventory] {} problem dirs found", reps.len());

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
        if n == 0 || n >= 10_000 {
            continue;
        }
        let full = csc.symmetric_pattern();
        let full_avg_deg = full.row_idx.len() as f64 / n.max(1) as f64;
        if !full_avg_deg.is_finite() || full_avg_deg >= 15.0 {
            continue;
        }
        let stored_nnz = csc.row_idx.len();

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
            amd_num_nnz_l: None,
            amf_num_nnz_l: None,
            metis_num_nnz_l: None,
            kahip_num_nnz_l: None,
            amd_us: None,
            amf_us: None,
            metis_us: None,
            kahip_us: None,
            amd_status: NA.to_string(),
            amf_status: NA.to_string(),
            metis_status: NA.to_string(),
            kahip_status: NA.to_string(),
            err: String::new(),
        };

        let (nnz, us, st) = run_one(&csc, OrderingMethod::Amd);
        row.amd_num_nnz_l = nnz;
        row.amd_us = us;
        row.amd_status = st;

        let (nnz, us, st) = run_one(&csc, OrderingMethod::Amf);
        row.amf_num_nnz_l = nnz;
        row.amf_us = us;
        row.amf_status = st;

        let (nnz, us, st) = run_one(&csc, OrderingMethod::MetisND);
        row.metis_num_nnz_l = nnz;
        row.metis_us = us;
        row.metis_status = st;

        let (nnz, us, st) = run_one(&csc, OrderingMethod::KahipND);
        row.kahip_num_nnz_l = nnz;
        row.kahip_us = us;
        row.kahip_status = st;

        row.emit();
        processed += 1;

        if last_log.elapsed().as_secs() >= 5 {
            eprintln!(
                "[small-sparse-inventory] {}/{} dirs scanned, {} rows emitted ({:.1} s)",
                i + 1,
                reps.len(),
                processed,
                t_total.elapsed().as_secs_f64(),
            );
            last_log = Instant::now();
        }
    }
    eprintln!(
        "[small-sparse-inventory] done: {} rows in {:.1} s",
        processed,
        t_total.elapsed().as_secs_f64(),
    );
}
