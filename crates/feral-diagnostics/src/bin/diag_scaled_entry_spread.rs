//! diag_scaled_entry_spread — is there an O(nnz) statistic of
//! `D0 · A_N · D0` that stays put while the diagonal-dominance ratio
//! runs away with the IPM barrier trajectory?
//!
//! The 2026-05-21 Track-B2 rejection
//! (`dev/tried-and-rejected.md`, "gate metric confounded by IPM
//! delta") found condition 1 of the value bound -- growth of
//! `max_j off_max/diag` against its iter-0 baseline -- rejecting
//! every warm iterate on pinene_3200, because the KKT (2,2) block
//! carries a delta-regularized diagonal against O(1) off-diagonals,
//! so the ratio is ~1/delta and explodes as delta->0. That entry
//! concluded "diagonal dominance of D*A*D is the wrong proxy".
//!
//! The complaint is specifically about the DIAGONAL. MC64's own
//! objective never mentions it: the matching maximises the product
//! of matched magnitudes, and the resulting scaling drives matched
//! entries to ~1 with every other entry <= 1. So the question this
//! probe asks is whether the entry magnitudes alone -- no diagonal
//! anywhere in the statistic -- stay bounded along the same
//! trajectory that blows up the ratio.
//!
//! Prints, per iterate, both statistics computed on the SAME scaled
//! matrix so they can be compared directly:
//!   max_ratio  -- what condition 1 measures today (diagonal-based)
//!   max_ent    -- max |(D0 A_N D0)_ij| over all stored entries
//!   p999_ent   -- 99.9th percentile of the same, by magnitude
//!   frac_gt1   -- fraction of entries with magnitude > 1
//!
//! `D0` is iterate 0's own MC64 vector, so iterate 0 is the baseline
//! and every later row is drift against it.
//!
//! Usage: diag_scaled_entry_spread <iter0.mtx> <iter1.mtx> ...

use feral::symbolic::SupernodeParams;
use feral::{read_mtx, CscMatrix, FactorStatus, NumericParams, Solver};
use std::path::Path;

/// Both statistics in one O(nnz) sweep of the lower triangle.
/// Returns (max_ratio over rows with a nonzero diagonal, max entry,
/// 99.9th-percentile entry, fraction of entries > 1).
fn stats(a: &CscMatrix, d: &[f64]) -> (f64, f64, f64, f64) {
    let n = a.n;
    let mut diag = vec![0.0f64; n];
    let mut offmax = vec![0.0f64; n];
    let mut mags: Vec<f64> = Vec::with_capacity(a.nnz());
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let v = (a.values[k] * d[i] * d[j]).abs();
            mags.push(v);
            if i == j {
                diag[j] = v;
            } else {
                // Symmetric: the stored (i,j) is also (j,i).
                if v > offmax[i] {
                    offmax[i] = v;
                }
                if v > offmax[j] {
                    offmax[j] = v;
                }
            }
        }
    }
    let mut max_ratio = 0.0f64;
    for j in 0..n {
        if diag[j] > 0.0 {
            let r = offmax[j] / diag[j];
            if r > max_ratio {
                max_ratio = r;
            }
        }
    }
    mags.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let max_ent = mags.last().copied().unwrap_or(0.0);
    let idx = ((mags.len() as f64) * 0.999) as usize;
    let p999 = mags
        .get(idx.min(mags.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0);
    let n_gt1 = mags.iter().filter(|v| **v > 1.0).count();
    let frac = n_gt1 as f64 / mags.len().max(1) as f64;
    (max_ratio, max_ent, p999, frac)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: diag_scaled_entry_spread <iter0.mtx> <iter1.mtx> ...");
        std::process::exit(2);
    }
    let first = read_mtx(Path::new(&args[0])).and_then(|m| m.to_csc())?;
    let mut seed = Solver::with_params(NumericParams::default(), SupernodeParams::default())
        .with_parallel(false);
    if !matches!(
        seed.factor(&first, None),
        FactorStatus::Success | FactorStatus::WrongInertia { .. }
    ) {
        eprintln!("seed factorization failed");
        std::process::exit(1);
    }
    let d0 = match seed.factors() {
        Some(f) => f.scaling.clone(),
        None => {
            eprintln!("seed produced no factors");
            std::process::exit(1);
        }
    };
    println!("D0 from {} (n={})", args[0], d0.len());
    println!(
        "{:<24}{:>12}{:>10}{:>12}{:>12}{:>12}",
        "iterate", "max_ratio", "vs iter0", "max_ent", "p999_ent", "frac_gt1"
    );
    let mut base_ratio = 0.0f64;
    let mut base_ent = 0.0f64;
    for (idx, a) in args.iter().enumerate() {
        let csc = match read_mtx(Path::new(a)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {a}: {e:?}");
                continue;
            }
        };
        if csc.n != d0.len() {
            println!("{a} n changed -- skipped");
            continue;
        }
        let (r, me, p999, frac) = stats(&csc, &d0);
        if idx == 0 {
            base_ratio = r.max(1.0);
            base_ent = me.max(1e-300);
        }
        let name = Path::new(a)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(a);
        println!(
            "{:<24}{:>12.3e}{:>9.1}x{:>12.3e}{:>12.3e}{:>12.3e}   ent {:.1}x",
            name,
            r,
            r / base_ratio,
            me,
            p999,
            frac,
            me / base_ent
        );
    }
    Ok(())
}
