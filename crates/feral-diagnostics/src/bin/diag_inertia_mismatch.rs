//! Audit feral's reported inertia vs sidecar (MUMPS-derived) inertia for
//! a sample of matrices in a given family. Prints (n, expected, feral,
//! delta_w, delta_c) per matrix and aggregates the mismatch pattern so
//! we can tell whether the discrepancy is a near-zero-pivot threshold
//! issue, a 2x2-pivot accounting issue, or something else.
//!
//! Usage: `cargo run --release --bin diag_inertia_mismatch -- FAMILY [LIMIT]`
//! (e.g. `diag_inertia_mismatch -- RAYBENDS 5`)

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, ZeroPivotAction};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn read_sidecar_inertia(json_path: &Path) -> Option<(usize, usize, usize, f64, f64)> {
    let txt = std::fs::read_to_string(json_path).ok()?;
    let v: Value = serde_json::from_str(&txt).ok()?;
    let inertia = v.get("inertia")?;
    let pos = inertia.get("positive")?.as_u64()? as usize;
    let neg = inertia.get("negative")?.as_u64()? as usize;
    let zer = inertia.get("zero")?.as_u64()? as usize;
    let dw = v.get("delta_w").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let dc = v.get("delta_c").and_then(|x| x.as_f64()).unwrap_or(0.0);
    Some((pos, neg, zer, dw, dc))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: diag_inertia_mismatch FAMILY [LIMIT]");
        std::process::exit(2);
    }
    let family = &args[1];
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    let roots = [
        "data/matrices/kkt-expansion",
        "data/matrices/kkt-mittelmann",
        "data/matrices/kkt",
    ];
    let mut family_dir: Option<PathBuf> = None;
    for r in &roots {
        let p = Path::new(r).join(family);
        if p.exists() {
            family_dir = Some(p);
            break;
        }
    }
    let family_dir = match family_dir {
        Some(d) => d,
        None => {
            eprintln!("family {} not found in any corpus root", family);
            std::process::exit(1);
        }
    };

    let mut entries: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&family_dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("mtx") {
                entries.push(p);
            }
        }
    }
    entries.sort();
    entries.truncate(limit);

    let bk = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..Default::default()
    };
    let snode_params = SupernodeParams::default();
    let nparams = NumericParams::with_bk(bk);

    println!("family={}, dir={}", family, family_dir.display());
    println!(
        "{:32}  {:>6}  expected             feral                delta_w     delta_c",
        "name", "n"
    );

    let mut n_total = 0usize;
    let mut n_match = 0usize;
    let mut n_diff_pos1 = 0usize;
    let mut n_other = 0usize;

    for mtx_path in &entries {
        let json_path = mtx_path.with_extension("json");
        let (epos, eneg, ezer, dw, dc) = match read_sidecar_inertia(&json_path) {
            Some(t) => t,
            None => continue,
        };
        let mtx = match read_mtx(mtx_path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(_) => continue,
        };
        drop(mtx);

        let sym = match symbolic_factorize(&csc, &snode_params) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let (_factors, fer_inertia) = match factorize_multifrontal(&csc, &sym, &nparams) {
            Ok(r) => r,
            Err(_) => continue,
        };

        n_total += 1;
        let name = mtx_path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let matched = fer_inertia.positive == epos
            && fer_inertia.negative == eneg
            && fer_inertia.zero == ezer;
        if matched {
            n_match += 1;
        } else if fer_inertia.positive + 1 == epos
            && fer_inertia.negative == eneg
            && fer_inertia.zero == ezer + 1
        {
            // feral reports one extra zero pivot where MUMPS reports positive
            n_diff_pos1 += 1;
        } else {
            n_other += 1;
        }
        println!(
            "{:32}  {:>6}  ({:>5},{:>5},{:>5})  ({:>5},{:>5},{:>5})  {:>10.3e}  {:>10.3e}",
            name,
            csc.n,
            epos,
            eneg,
            ezer,
            fer_inertia.positive,
            fer_inertia.negative,
            fer_inertia.zero,
            dw,
            dc
        );
    }
    println!(
        "\n{} runs, {} match, {} pos->zero off-by-1, {} other",
        n_total, n_match, n_diff_pos1, n_other
    );
}
