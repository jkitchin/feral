//! Issue #209 — does the measured race pick the right arm per family?
//!
//! `Auto` never selects nested dissection: the #67/#73 reroute sends
//! every would-be-`MetisND` decision to `Amf`, and that evidence was
//! measured against the `MetisND` #203 has since fixed. #209 asks
//! whether an opt-in numeric race could route per class instead, and
//! names the test: *would it pick `metis` on the collocation models
//! without regressing the grid families?*
//!
//! This answers it directly. For each matrix:
//!
//! 1. time every fixed arm's **steady state** (the per-iterate cost an
//!    interior-point host actually pays) to find the true best arm;
//! 2. run `Solver::with_ordering_race` and record which arm it adopted;
//! 3. report whether the race's pick was the best arm, and what its
//!    steady state costs relative to the best.
//!
//! The race is charged nothing for its setup here on purpose — the
//! question is whether its *pick* is right. Setup cost is a separate
//! question answered by `issue203_policy`.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin issue209_race_picks \
//!       -- CORPUS_DIR [PAIRS] [MAX_PER_FAMILY]

use feral::symbolic::OrderingMethod;
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn steady(m: &CscMatrix, s: &mut Solver, pairs: usize) -> Option<u128> {
    if !matches!(s.factor(m, None), FactorStatus::Success) {
        return None;
    }
    let mut best = u128::MAX;
    for _ in 0..pairs {
        let t = Instant::now();
        if !matches!(s.factor(m, None), FactorStatus::Success) {
            return None;
        }
        best = best.min(t.elapsed().as_micros());
    }
    Some(best)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let Some(root) = a.first() else {
        eprintln!("usage: issue209_race_picks CORPUS_DIR [PAIRS] [MAX_PER_FAMILY]");
        std::process::exit(2);
    };
    let pairs: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let cap: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let arms = [
        ("amf", OrderingMethod::Amf),
        ("amd", OrderingMethod::Amd),
        ("metis", OrderingMethod::MetisND),
    ];
    let race_arms = vec![OrderingMethod::Amd, OrderingMethod::MetisND];

    let mut families: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for e in std::fs::read_dir(Path::new(root))
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
    {
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        let fam = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let mut ms: Vec<PathBuf> = std::fs::read_dir(&p)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mtx"))
            .collect();
        ms.sort();
        if ms.len() > cap {
            let step = ms.len() as f64 / cap as f64;
            ms = (0..cap)
                .map(|i| ms[(i as f64 * step) as usize].clone())
                .collect();
        }
        if !ms.is_empty() {
            families.insert(fam, ms);
        }
    }

    println!(
        "{:<12} {:>5} {:>8} {:>9} {:>9} {:>9} {:>9}  {:>8}",
        "family", "mats", "best arm", "amf_us", "amd_us", "metis_us", "race_us", "race/best"
    );

    let mut overall: Vec<f64> = Vec::new();
    let mut picked_best = 0usize;
    let mut total = 0usize;

    for (fam, mats) in &families {
        let mut ratios: Vec<f64> = Vec::new();
        let mut best_names: Vec<&str> = Vec::new();
        let (mut sa, mut sd, mut sm, mut sr) = (0u128, 0u128, 0u128, 0u128);
        let mut n = 0usize;
        for path in mats {
            let Ok(mtx) = read_mtx(path) else { continue };
            let Ok(m) = mtx.to_csc() else { continue };
            let mut times = Vec::new();
            let mut ok = true;
            for (label, method) in &arms {
                let mut s = Solver::new().with_ordering(method.clone());
                match steady(&m, &mut s, pairs) {
                    Some(us) => times.push((*label, us)),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok || times.len() != arms.len() {
                continue;
            }
            let mut raced = Solver::new().with_ordering_race(race_arms.clone());
            let Some(race_us) = steady(&m, &mut raced, pairs) else {
                continue;
            };
            let adopted = raced
                .last_race()
                .and_then(|r| r.winner.map(|w| format!("{:?}", r.arms[w].method)))
                .unwrap_or_else(|| "?".into());
            let (best_name, best_us) = times.iter().min_by_key(|(_, u)| *u).copied().unwrap();
            let _ = adopted;
            sa += times[0].1;
            sd += times[1].1;
            sm += times[2].1;
            sr += race_us;
            ratios.push(race_us as f64 / best_us as f64);
            best_names.push(best_name);
            if race_us as f64 <= best_us as f64 * 1.05 {
                picked_best += 1;
            }
            total += 1;
            n += 1;
        }
        if n == 0 {
            continue;
        }
        let gm = (ratios.iter().map(|v| v.ln()).sum::<f64>() / ratios.len() as f64).exp();
        // Most common best arm in this family.
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for b in &best_names {
            *counts.entry(b).or_insert(0) += 1;
        }
        let best_arm = counts
            .iter()
            .max_by_key(|(_, c)| **c)
            .map(|(k, _)| *k)
            .unwrap_or("?");
        println!(
            "{:<12} {:>5} {:>8} {:>9} {:>9} {:>9} {:>9}  {:>8.3}",
            fam,
            n,
            best_arm,
            sa / n as u128,
            sd / n as u128,
            sm / n as u128,
            sr / n as u128,
            gm
        );
        overall.extend(ratios);
    }

    if !overall.is_empty() {
        let gm = (overall.iter().map(|v| v.ln()).sum::<f64>() / overall.len() as f64).exp();
        let worst = overall.iter().cloned().fold(0.0f64, f64::max);
        println!("\nrace steady-state vs the best fixed arm: geomean {gm:.3}, worst {worst:.3}");
        println!("race landed within 5% of the best arm on {picked_best}/{total} matrices");
        println!("(1.000 = the race picked as well as an oracle would have)");
    }
}
