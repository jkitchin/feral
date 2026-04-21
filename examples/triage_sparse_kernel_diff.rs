//! Phase 2.4.1c triage — sparse path under scalar vs blocked frontal.
//!
//! Walks the full KKT corpus at `data/matrices/kkt/`, factors each
//! matrix once with the blocked kernel (`factor_frontal_blocked`) and
//! once with the scalar kernel (`factor_frontal`, forced via the
//! `FORCE_SCALAR_FRONTAL` atomic in `dense::factor`), and reports the
//! matrices whose inertia or residual pass/fail classification differs
//! between the two runs.
//!
//! Expected drift from session 2026-04-20-08 validation: 1 sparse
//! inertia match, 67 sparse residual passes. This binary identifies
//! exactly which matrices they are so we can classify each as pure
//! rounding-order noise vs an algorithmic bug in the panel path.
//!
//! Usage:
//!
//!     cargo run --example triage_sparse_kernel_diff --release
//!
//! The run takes ~10-20 min (factors each of 154588 matrices twice).

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use feral::dense::factor::FORCE_SCALAR_FRONTAL;
use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    read_mtx, read_sidecar, solve_sparse_refined, BunchKaufmanParams, CscMatrix, Inertia,
    KktSidecar, ZeroPivotAction,
};

struct KktEntry {
    name: String,
    csc: CscMatrix,
    sidecar: KktSidecar,
}

fn load_kkt_dir(dir: &Path) -> Vec<KktEntry> {
    if !dir.is_dir() {
        return Vec::new();
    }
    let mut entries = Vec::new();
    let mut subdirs: Vec<_> = match std::fs::read_dir(dir) {
        Ok(d) => d.filter_map(|e| e.ok()).collect(),
        Err(_) => return Vec::new(),
    };
    subdirs.sort_by_key(|e| e.file_name());

    for subdir in subdirs {
        let subdir_path = subdir.path();
        if !subdir_path.is_dir() {
            continue;
        }
        let mut mtx_files: Vec<_> = match std::fs::read_dir(&subdir_path) {
            Ok(d) => d
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "mtx"))
                .collect(),
            Err(_) => continue,
        };
        mtx_files.sort_by_key(|e| e.file_name());

        for mtx_entry in mtx_files {
            let mtx_path: PathBuf = mtx_entry.path();
            let stem = mtx_path.file_stem().unwrap().to_string_lossy().to_string();
            let json_path = mtx_path.with_extension("json");
            if !json_path.exists() {
                continue;
            }
            let mtx = match read_mtx(&mtx_path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let csc = match mtx.to_csc() {
                Ok(c) => c,
                Err(_) => continue,
            };
            let sidecar = match read_sidecar(&json_path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            entries.push(KktEntry {
                name: stem,
                csc,
                sidecar,
            });
        }
    }
    entries
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Outcome {
    inertia_ok: bool,
    residual_ok: bool,
    residual: f64,
    // (pos, neg, zero) — the actual inertia produced by feral.
    pos: usize,
    neg: usize,
    zero: usize,
    // Error classes (mutually exclusive with ok):
    factor_err: bool,
    solve_err: bool,
    symbolic_err: bool,
    no_rhs: bool,
}

const ZERO: Outcome = Outcome {
    inertia_ok: false,
    residual_ok: false,
    residual: 0.0,
    pos: 0,
    neg: 0,
    zero: 0,
    factor_err: false,
    solve_err: false,
    symbolic_err: false,
    no_rhs: false,
};

/// Residual pass threshold — matches `sp_residual_ok` in `src/bin/bench.rs`.
const RESIDUAL_TOL: f64 = 1.0e-6;

fn run_one(entry: &KktEntry, params: &NumericParams) -> Outcome {
    let mut out = ZERO;
    let snode_params = SupernodeParams::default();

    let sym = match symbolic_factorize(&entry.csc, &snode_params) {
        Ok(s) => s,
        Err(_) => {
            out.symbolic_err = true;
            return out;
        }
    };

    let (factors, inertia) = match factorize_multifrontal(&entry.csc, &sym, params) {
        Ok(r) => r,
        Err(_) => {
            out.factor_err = true;
            return out;
        }
    };

    let expected = Inertia {
        positive: entry.sidecar.inertia.positive,
        negative: entry.sidecar.inertia.negative,
        zero: entry.sidecar.inertia.zero,
    };
    out.pos = inertia.positive;
    out.neg = inertia.negative;
    out.zero = inertia.zero;
    out.inertia_ok = inertia == expected;

    let rhs = match entry.sidecar.finite_rhs() {
        Some(r) => r,
        None => {
            out.no_rhs = true;
            return out;
        }
    };

    let x = match solve_sparse_refined(&entry.csc, &factors, &rhs) {
        Ok(x) => x,
        Err(_) => {
            out.solve_err = true;
            return out;
        }
    };

    let n = entry.csc.n;
    let mut ax = vec![0.0; n];
    entry.csc.symv(&x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        let r = ax[i] - rhs[i];
        rs += r * r;
        bs += rhs[i] * rhs[i];
    }
    out.residual = if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    };
    out.residual_ok = out.residual <= RESIDUAL_TOL;
    out
}

fn main() {
    let dir = Path::new("data/matrices/kkt");
    let entries = load_kkt_dir(dir);
    eprintln!("Loaded {} KKT matrices", entries.len());

    // Mirror the sparse-path defaults used by `src/bin/bench.rs`:
    // ZeroPivotAction::RejectPivot at the root is fine for sparse (it'll
    // just force-accept via the may_delay=false flag); scaling defaults.
    // Mirror the sparse-path defaults used by `src/bin/bench.rs:1014`.
    // `on_zero_pivot = ForceAccept` + `pivot_threshold = 0.01` is the
    // sparse-KKT production configuration.
    let bk = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    let mut params = NumericParams::with_bk(bk);
    params.scaling = ScalingStrategy::Auto;

    // Phase 1: blocked (production default).
    FORCE_SCALAR_FRONTAL.store(false, Ordering::Relaxed);
    eprintln!("Phase 1/2: factoring with factor_frontal_blocked ...");
    let blocked: Vec<Outcome> = entries.iter().map(|e| run_one(e, &params)).collect();

    // Phase 2: scalar.
    FORCE_SCALAR_FRONTAL.store(true, Ordering::Relaxed);
    eprintln!("Phase 2/2: factoring with factor_frontal (forced) ...");
    let scalar: Vec<Outcome> = entries.iter().map(|e| run_one(e, &params)).collect();

    // Reset so any downstream code sees production defaults.
    FORCE_SCALAR_FRONTAL.store(false, Ordering::Relaxed);

    // Aggregate counts.
    let mut b_inertia = 0usize;
    let mut s_inertia = 0usize;
    let mut b_residual = 0usize;
    let mut s_residual = 0usize;
    for i in 0..entries.len() {
        if blocked[i].inertia_ok {
            b_inertia += 1;
        }
        if scalar[i].inertia_ok {
            s_inertia += 1;
        }
        if blocked[i].residual_ok {
            b_residual += 1;
        }
        if scalar[i].residual_ok {
            s_residual += 1;
        }
    }

    println!("=== Aggregate counts ===");
    println!(
        "blocked: inertia={}/{}  residual={}/{}",
        b_inertia,
        entries.len(),
        b_residual,
        entries.len()
    );
    println!(
        "scalar:  inertia={}/{}  residual={}/{}",
        s_inertia,
        entries.len(),
        s_residual,
        entries.len()
    );
    println!(
        "delta (blocked - scalar):  inertia={}  residual={}",
        b_inertia as i64 - s_inertia as i64,
        b_residual as i64 - s_residual as i64
    );

    // Per-matrix diff: any matrix where the two outcomes disagree on
    // inertia_ok or residual_ok. Lists each with both residuals.
    println!("\n=== Per-matrix differences ===");
    println!(
        "{:<28} {:>4} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "matrix", "n", "blk_inertia", "scl_inertia", "blk_residual", "scl_residual", "class"
    );

    let mut diff_count = 0usize;
    let mut class_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();

    for i in 0..entries.len() {
        let name = &entries[i].name;
        let n = entries[i].csc.n;
        let b = &blocked[i];
        let s = &scalar[i];
        let inertia_flip = b.inertia_ok != s.inertia_ok;
        let residual_flip = b.residual_ok != s.residual_ok;
        if !inertia_flip && !residual_flip {
            continue;
        }
        diff_count += 1;

        let class = if inertia_flip && b.inertia_ok && !s.inertia_ok {
            "IB_SF" // blocked inertia pass, scalar fail  (blocked improves)
        } else if inertia_flip && !b.inertia_ok && s.inertia_ok {
            "IS_BF" // scalar inertia pass, blocked fail  (blocked regresses)
        } else if residual_flip && b.residual_ok && !s.residual_ok {
            "RB_SF"
        } else if residual_flip && !b.residual_ok && s.residual_ok {
            "RS_BF"
        } else {
            "OTHER"
        };
        *class_counts.entry(class).or_insert(0) += 1;

        println!(
            "{:<28} {:>4} {:>12} {:>12} {:>12.2e} {:>12.2e} {:>12}",
            name,
            n,
            if b.inertia_ok { "pass" } else { "fail" },
            if s.inertia_ok { "pass" } else { "fail" },
            b.residual,
            s.residual,
            class
        );
    }

    println!("\n=== Summary ===");
    println!("Total matrices with any diff: {}", diff_count);
    let mut classes: Vec<_> = class_counts.iter().collect();
    classes.sort_by_key(|(k, _)| k.to_string());
    for (k, v) in classes {
        let desc = match *k {
            "IB_SF" => "inertia: blocked pass, scalar fail (blocked improves)",
            "IS_BF" => "inertia: scalar pass, blocked fail (blocked regresses)",
            "RB_SF" => "residual: blocked pass, scalar fail",
            "RS_BF" => "residual: scalar pass, blocked fail (blocked regresses)",
            _ => "other",
        };
        println!("  {:<6} = {:>6}   {}", k, v, desc);
    }
}
