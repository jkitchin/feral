//! Stream the corpus and compare feral's factor nnz against the
//! MUMPS / SSIDS oracle sidecars produced by
//! `external_benchmarks/{mumps,ssids}_oracle/`. Memory stays at one
//! matrix — no `Vec<KktEntry>`, no `to_dense()` — so this is the safe
//! way to get a fill-parity readout on the full ~164k corpus that
//! `cargo run --bin bench` cannot load without OOM-killing itself.
//!
//! Methodology — what makes the comparison apples-to-apples:
//!
//!   - **MUMPS oracle** (`mumps_bench.F`) leaves `ICNTL(7)` at default,
//!     which on our build is an AMD-flavored automatic. So the meaningful
//!     parity readout for MUMPS is feral with `OrderingMethod::Amd`.
//!   - **SSIDS oracle** (`ssids_bench.f90`) leaves `options%ordering` at
//!     default = METIS. So the meaningful parity readout for SSIDS is
//!     feral with `OrderingMethod::MetisND`.
//!   - **Auto** is what feral *ships* — `pick_default_method` adaptively
//!     dispatches AMD or MetisND. Reported alongside as a sanity check
//!     against MUMPS (since auto is AMD-biased on this corpus).
//!
//! Numeric params on all three passes match `params_kkt_sparse` in
//! `src/bin/bench.rs` (`pivot_threshold=0.01`, `ForceAccept`).
//!
//! Inputs: `<id>.mtx` + `<id>.json` (RHS sidecar) + optional
//! `<id>.mumps.json` / `<id>.ssids.json` (oracle factor_nnz).
//!
//! Usage:
//!   cargo run --release --bin diag_fill_parity
//!   cargo run --release --bin diag_fill_parity -- data/matrices/kkt
//!   FERAL_DIAG_MAX_N=20000 cargo run --release --bin diag_fill_parity
//!   FERAL_DIAG_VERBOSE=1 cargo run --release --bin diag_fill_parity

use std::path::{Path, PathBuf};

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

/// Pull `factor_nnz` out of a canonical oracle sidecar. Tolerant of the
/// file being absent or written before the field existed (returns None).
fn read_oracle_factor_nnz(path: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    if v.get("factorization_status").and_then(|s| s.as_str()) != Some("ok") {
        return None;
    }
    v.get("factor_nnz")
        .and_then(|x| x.as_u64())
        .filter(|n| *n > 0)
}

#[derive(Default, Clone)]
struct Aggregate {
    label: String,
    factored: usize,
    skipped_factor_err: usize,
    skipped_panic: usize,
    fill_self: Vec<f64>,           // nnz_L(feral) / nnz(A)
    fill_vs_mumps: Vec<f64>,       // nnz_L(feral) / nnz_L(MUMPS)
    fill_vs_ssids: Vec<f64>,       // nnz_L(feral) / nnz_L(SSIDS)
    worst_vs_mumps: (f64, String), // (ratio, name)
    worst_vs_ssids: (f64, String),
}

impl Aggregate {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            ..Default::default()
        }
    }

    fn merge_from(&mut self, other: &Aggregate) {
        self.factored += other.factored;
        self.skipped_factor_err += other.skipped_factor_err;
        self.skipped_panic += other.skipped_panic;
        self.fill_self.extend_from_slice(&other.fill_self);
        self.fill_vs_mumps.extend_from_slice(&other.fill_vs_mumps);
        self.fill_vs_ssids.extend_from_slice(&other.fill_vs_ssids);
        if other.worst_vs_mumps.0 > self.worst_vs_mumps.0 {
            self.worst_vs_mumps = other.worst_vs_mumps.clone();
        }
        if other.worst_vs_ssids.0 > self.worst_vs_ssids.0 {
            self.worst_vs_ssids = other.worst_vs_ssids.clone();
        }
    }
}

/// Three per-ordering aggregates rolled up at one corpus-or-combined level.
#[derive(Clone)]
struct PerOrdering {
    label: String,
    seen: usize,
    skipped_size: usize,
    skipped_no_mtx: usize,
    skipped_filter: usize,
    auto: Aggregate,
    amd: Aggregate,
    metis: Aggregate,
}

impl PerOrdering {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            seen: 0,
            skipped_size: 0,
            skipped_no_mtx: 0,
            skipped_filter: 0,
            auto: Aggregate::new("auto"),
            amd: Aggregate::new("amd"),
            metis: Aggregate::new("metis"),
        }
    }

    fn merge_from(&mut self, other: &PerOrdering) {
        self.seen += other.seen;
        self.skipped_size += other.skipped_size;
        self.skipped_no_mtx += other.skipped_no_mtx;
        self.skipped_filter += other.skipped_filter;
        self.auto.merge_from(&other.auto);
        self.amd.merge_from(&other.amd);
        self.metis.merge_from(&other.metis);
    }
}

fn factor_under(
    csc: &feral::CscMatrix,
    method: OrderingMethod,
    snode: &SupernodeParams,
    np: &NumericParams,
) -> Option<u64> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let sym = symbolic_factorize_with_method(csc, snode, method)?;
        factorize_multifrontal(csc, &sym, np)
    }));
    match result {
        Ok(Ok((factors, _))) => Some(factors.factor_nnz() as u64),
        _ => None,
    }
}

fn record(
    agg: &mut Aggregate,
    name: &str,
    nnz_a: u64,
    nnz_l_feral: u64,
    nnz_l_mumps: Option<u64>,
    nnz_l_ssids: Option<u64>,
) {
    agg.factored += 1;
    agg.fill_self.push(nnz_l_feral as f64 / nnz_a.max(1) as f64);
    if let Some(m) = nnz_l_mumps {
        let r = nnz_l_feral as f64 / m.max(1) as f64;
        agg.fill_vs_mumps.push(r);
        if r > agg.worst_vs_mumps.0 {
            agg.worst_vs_mumps = (r, name.to_string());
        }
    }
    if let Some(s) = nnz_l_ssids {
        let r = nnz_l_feral as f64 / s.max(1) as f64;
        agg.fill_vs_ssids.push(r);
        if r > agg.worst_vs_ssids.0 {
            agg.worst_vs_ssids = (r, name.to_string());
        }
    }
}

fn run_one(mtx_path: &Path, po: &mut PerOrdering, max_n: usize, verbose: bool) {
    po.seen += 1;
    let name = mtx_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<?>")
        .to_string();

    let mtx = match read_mtx(mtx_path) {
        Ok(m) => m,
        Err(_) => {
            po.skipped_no_mtx += 1;
            return;
        }
    };

    let sidecar_path = mtx_path.with_extension("json");
    let sidecar = match read_sidecar(&sidecar_path) {
        Ok(s) => s,
        Err(_) => {
            po.skipped_filter += 1;
            return;
        }
    };
    if sidecar.finite_rhs().is_none() {
        po.skipped_filter += 1;
        return;
    }
    if mtx.entries.iter().any(|(_, _, v)| !v.is_finite()) {
        po.skipped_filter += 1;
        return;
    }
    if mtx.n != sidecar.n + sidecar.m {
        po.skipped_filter += 1;
        return;
    }

    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(_) => {
            po.skipped_no_mtx += 1;
            return;
        }
    };
    drop(mtx);

    if csc.n > max_n {
        po.skipped_size += 1;
        return;
    }

    let snode = SupernodeParams::default();
    // Mirror src/bin/bench.rs `params_kkt_sparse`.
    let bk = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    let np = NumericParams::with_bk(bk);

    if verbose {
        eprintln!("BEGIN {} (n={} nnz={})", name, csc.n, csc.values.len());
    }

    let nnz_a = csc.values.len() as u64;
    let nnz_l_mumps = read_oracle_factor_nnz(&mtx_path.with_extension("mumps.json"));
    let nnz_l_ssids = read_oracle_factor_nnz(&mtx_path.with_extension("ssids.json"));

    for (method, agg) in [
        (OrderingMethod::Auto, &mut po.auto),
        (OrderingMethod::Amd, &mut po.amd),
        (OrderingMethod::MetisND, &mut po.metis),
    ] {
        match factor_under(&csc, method, &snode, &np) {
            Some(nnz_l) => record(agg, &name, nnz_a, nnz_l, nnz_l_mumps, nnz_l_ssids),
            None => {
                // Cannot distinguish panic vs Err without a second match,
                // but the corpus signal is "this ordering+matrix combo
                // failed feral factor"; bug isolation goes to the panic
                // log when verbose.
                if verbose {
                    eprintln!("FAIL {} ordering={:?} (n={})", name, method, csc.n);
                }
                agg.skipped_factor_err += 1;
            }
        }
    }
}

fn report_aggregate(agg: &Aggregate) {
    let report =
        |label: &str, vals: &[f64], worst: Option<&(f64, String)>| {
            if vals.is_empty() {
                println!("    {:<24}  (no data)", label);
                return;
            }
            let mut v = vals.to_vec();
            let g = geomean(&v);
            let p50 = percentile_f64(&mut v, 0.50);
            let p90 = percentile_f64(&mut v, 0.90);
            let p99 = percentile_f64(&mut v, 0.99);
            let max = *v.last().unwrap_or(&0.0);
            let worst_str = worst
                .map(|(_, name)| format!("  worst={}", name))
                .unwrap_or_default();
            println!(
            "    {:<24}  n={:>6}  geomean={:7.2}  p50={:7.2}  p90={:7.2}  p99={:7.2}  max={:7.2}{}",
            label, vals.len(), g, p50, p90, p99, max, worst_str,
        );
        };

    println!(
        "  [{}]  factored={}  factor_fail={}",
        agg.label, agg.factored, agg.skipped_factor_err
    );
    report("nnzL(feral)/nnz(A)", &agg.fill_self, None);
    report(
        "nnzL(feral)/nnzL(MUMPS)",
        &agg.fill_vs_mumps,
        Some(&agg.worst_vs_mumps),
    );
    report(
        "nnzL(feral)/nnzL(SSIDS)",
        &agg.fill_vs_ssids,
        Some(&agg.worst_vs_ssids),
    );
}

fn print_section(po: &PerOrdering) {
    println!(
        "\n=== {} ===  seen={}  skipped(size)={}  skipped(read)={}  skipped(filter)={}",
        po.label, po.seen, po.skipped_size, po.skipped_no_mtx, po.skipped_filter,
    );
    println!("  Methodology: feral-amd vs MUMPS, feral-metis vs SSIDS, feral-auto = production dispatcher.");
    report_aggregate(&po.auto);
    report_aggregate(&po.amd);
    report_aggregate(&po.metis);
}

fn walk_root(root: &Path, po: &mut PerOrdering, max_n: usize, verbose: bool) {
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
            walk_root(&p, po, max_n, verbose);
        } else if p.extension().is_some_and(|ext| ext == "mtx") {
            run_one(&p, po, max_n, verbose);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let max_n: usize = std::env::var("FERAL_DIAG_MAX_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50_000);
    let verbose = std::env::var("FERAL_DIAG_VERBOSE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let roots: Vec<PathBuf> = if args.is_empty() {
        DEFAULT_ROOTS.iter().map(PathBuf::from).collect()
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    println!("diag_fill_parity: max_n={}", max_n);

    let mut combined = PerOrdering::new("combined");
    for root in &roots {
        let mut per = PerOrdering::new(&root.display().to_string());
        walk_root(root, &mut per, max_n, verbose);
        combined.merge_from(&per);
        print_section(&per);
    }
    print_section(&combined);
}
