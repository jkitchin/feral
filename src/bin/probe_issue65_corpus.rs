//! Issue #65 corpus validation: confirm the inertia-guided MC64 scaling
//! fallback does not regress inertia on the KKT consensus corpus, and
//! report where it fires / fixes.
//!
//! Walks `data/matrices/kkt/**/*.mtx` that have a `.verdict.json`,
//! factors each under default `Auto` scaling (fallback ON), and compares
//! FERAL's inertia to the oracle `consensus_inertia` on matrices with a
//! `definitive` verdict and `strong` inertia agreement. Reports
//! match/mismatch, how often the fallback fired, and lists mismatches.
//!
//! Run: `cargo run --release --bin probe_issue65_corpus -- [max] [seed_stride]`
//!   max         — stop after this many checked matrices (0 = all)
//!   seed_stride — sample every Nth matrix (1 = all; default 1)
//!
//! Throwaway diagnostic, not a test.

use feral::{read_mtx, Solver};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Verdict {
    verdict: Option<String>,
    inertia_agreement: Option<String>,
    consensus_inertia: Option<ConsInertia>,
}

#[derive(Deserialize)]
struct ConsInertia {
    positive: usize,
    negative: usize,
    zero: usize,
}

fn collect_mtx(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(root) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_mtx(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("mtx") {
            out.push(p);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let stride: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1).max(1);

    let root = PathBuf::from("data/matrices/kkt");
    if !root.is_dir() {
        eprintln!("SKIP: {} not present", root.display());
        return;
    }
    let mut mtxs = Vec::new();
    collect_mtx(&root, &mut mtxs);
    mtxs.sort();
    eprintln!("found {} .mtx files", mtxs.len());

    let (mut checked, mut matched, mut fired, mut fired_changed) = (0usize, 0usize, 0usize, 0usize);
    let mut mismatches: Vec<String> = Vec::new();

    for (idx, mtx) in mtxs.iter().enumerate() {
        if idx % stride != 0 {
            continue;
        }
        if max != 0 && checked >= max {
            break;
        }
        let verdict_path = mtx.with_extension("verdict.json");
        let Ok(vtext) = std::fs::read_to_string(&verdict_path) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Verdict>(&vtext) else {
            continue;
        };
        // Only gate on matrices where the oracles agree definitively.
        if v.verdict.as_deref() != Some("definitive")
            || v.inertia_agreement.as_deref() != Some("strong")
        {
            continue;
        }
        let Some(ci) = v.consensus_inertia else {
            continue;
        };
        let Ok(m) = read_mtx(mtx).and_then(|x| x.to_csc()) else {
            continue;
        };

        let mut s = Solver::new();
        let _ = s.factor(&m, None);
        let Some(inertia) = s.inertia().cloned() else {
            continue;
        };
        checked += 1;
        let fb = s.mc64_scaling_fallback_count();
        if fb > 0 {
            fired += 1;
        }
        let ok = inertia.positive == ci.positive
            && inertia.negative == ci.negative
            && inertia.zero == ci.zero;
        if ok {
            matched += 1;
        } else {
            if fb > 0 {
                fired_changed += 1;
            }
            if mismatches.len() < 60 {
                mismatches.push(format!(
                    "{}: feral=({},{},{}) consensus=({},{},{}) fallback={}",
                    mtx.file_stem().and_then(|s| s.to_str()).unwrap_or("?"),
                    inertia.positive,
                    inertia.negative,
                    inertia.zero,
                    ci.positive,
                    ci.negative,
                    ci.zero,
                    fb,
                ));
            }
        }
        if checked % 5000 == 0 {
            eprintln!(
                "  checked={checked} matched={matched} fired={fired} mismatch={}",
                checked - matched
            );
        }
    }

    println!("=== issue #65 corpus inertia validation (fallback ON) ===");
    println!("  definitive+strong checked: {checked}");
    println!(
        "  matched consensus:         {matched}  ({:.4}%)",
        100.0 * matched as f64 / checked.max(1) as f64
    );
    println!("  mismatched:                {}", checked - matched);
    println!("  fallback fired:            {fired}");
    println!("  fallback fired AND mismatch: {fired_changed}  (potential regressions to inspect)");
    if !mismatches.is_empty() {
        println!("  --- mismatches (first {}): ---", mismatches.len());
        for m in &mismatches {
            println!("    {m}");
        }
    }
}
