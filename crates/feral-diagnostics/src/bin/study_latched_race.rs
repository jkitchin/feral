//! Issue #110 end-to-end prototype: the **latched actual-time ordering race**.
//!
//! §5.7 (`study_ordering_timemem`) showed no cheap symbolic proxy reaches the
//! time-oracle, so the deterministic way to capture the §5.6 factor-time headroom is
//! to *measure* candidate orderings on a real factor and reuse the fastest. This
//! binary prototypes that end-to-end on real IPM iterate sequences and asks: does
//! racing candidate orderings on the FIRST iterate, latching the fastest, and reusing
//! it for the rest of the sequence beat `Auto` / `AutoRace` — *including* the race
//! overhead?
//!
//! It drives the public `Solver` API only (no core-solver change): a latched race is
//! just "factor iterate 0 with each candidate ordering, keep the Solver whose factor
//! was fastest, and continue factoring the sequence with it." Ordering choice is
//! inertia-neutral, so this carries no correctness risk.
//!
//! Real IPM sequences change sparsity pattern at the first iterate or two (the active
//! set settling), then hold a stable pattern; only the pattern-*reused* iterates are the
//! long-run steady state, so all steady-state metrics are geomeans over reused iterates.
//!
//! For each family under the corpus root (default `data/matrices/kkt-mittelmann`)
//! with ≥ `MIN_ITERS` iterates and a first-iterate factor above `FLOOR_US` (small
//! matrices don't matter — §5.6), it runs the full grid (each of 5 explicit orderings
//! × each iterate, capped at `MAX_ITERS`) plus `Auto` and `AutoRace`, then reports:
//!   * winner ordering (argmin factor time on the first iterate — what a "race the first
//!     factor, latch it" strategy picks);
//!   * steady-state per-iterate cost of the latched winner vs `Auto` / `AutoRace`;
//!   * representativeness: how often the first-iterate winner equals the
//!     steady-state-optimal ordering (does racing the transient first factor pick right?);
//!   * race-overhead amortization: the break-even iterate count where the k−1 extra
//!     first-factors pay back at the per-iterate steady-state saving vs `Auto`.
//!
//! Usage:
//!   cargo run --release --bin study_latched_race
//!   cargo run --release --bin study_latched_race -- data/matrices/kkt-mittelmann --max-iters 12

use feral::symbolic::OrderingMethod;
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::{Path, PathBuf};
use std::time::Instant;

const FLOOR_US: u128 = 200;
const MIN_ITERS: usize = 3;

const CANDIDATES: &[(&str, OrderingMethod)] = &[
    ("amd", OrderingMethod::Amd),
    ("amf", OrderingMethod::Amf),
    ("metis", OrderingMethod::MetisND),
    ("scotch", OrderingMethod::ScotchND),
    ("kahip", OrderingMethod::KahipND),
];

/// Per-iterate measurement: wall-clock (us) and whether this factor reused the cached
/// symbolic (pattern unchanged from the previous iterate). IPM sequences typically see
/// a pattern change at the first iterate or two (active set settling), then a stable
/// pattern — only the reused iterates are the true steady state.
struct IterMeas {
    us: u128,
    reused: bool,
}

/// Factor every matrix in `seq` with one fixed `method` on a single reusing `Solver`.
/// The first iterate's factor includes the one-off symbolic build; later iterates
/// reuse the cached symbolic when the pattern is unchanged (numeric only) and rebuild
/// it when the pattern shifts. Single-pass timing — a real sequence is factored once.
/// `None` if any factor fails.
fn factor_sequence(seq: &[CscMatrix], method: OrderingMethod) -> Option<Vec<IterMeas>> {
    let mut solver = Solver::new().with_ordering(method);
    let mut out = Vec::with_capacity(seq.len());
    for m in seq {
        let t = Instant::now();
        if !matches!(solver.factor(m, None), FactorStatus::Success) {
            return None;
        }
        let us = t.elapsed().as_micros();
        let reused = solver
            .last_factor_stats()
            .map(|s| s.pattern_reused)
            .unwrap_or(false);
        out.push(IterMeas { us, reused });
    }
    Some(out)
}

fn collect_families(root: &Path) -> Vec<(String, Vec<PathBuf>)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        eprintln!("error: cannot read {}", root.display());
        return out;
    };
    let mut subs: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    subs.sort();
    for sub in subs {
        let Ok(inner) = std::fs::read_dir(&sub) else {
            continue;
        };
        let mut mtxs: Vec<PathBuf> = inner
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mtx"))
            .collect();
        mtxs.sort();
        let name = sub
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        if mtxs.len() >= MIN_ITERS {
            out.push((name, mtxs));
        }
    }
    out
}

fn geomean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp()
}

/// Geomean of the `us` field over iterates that reused the cached symbolic — the true
/// steady-state per-iterate cost of a long run. Falls back to all iterates if none
/// reused (e.g. every iterate changed the pattern).
fn steady_state(meas: &[IterMeas]) -> f64 {
    let reused: Vec<f64> = meas
        .iter()
        .filter(|m| m.reused)
        .map(|m| m.us as f64)
        .collect();
    if reused.is_empty() {
        geomean(&meas.iter().map(|m| m.us as f64).collect::<Vec<_>>())
    } else {
        geomean(&reused)
    }
}

struct FamilyResult {
    family: String,
    n: usize,
    iters: usize,
    n_reused: usize,
    winner: String,
    // iterate-0 race winner == the steady-state-optimal ordering?
    winner_is_ss_optimal: bool,
    // Steady-state per-iterate cost (geomean over reused iterates).
    ss_latched: f64,
    ss_auto: f64,
    ss_autorace: f64,
    // Race overhead expressed as iterations-to-amortize vs the Auto baseline.
    breakeven_iters_vs_auto: Option<f64>,
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut max_iters = 12usize;
    let mut positional = Vec::new();
    let mut it = raw.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--max-iters" => {
                if let Some(v) = it.next().and_then(|s| s.parse().ok()) {
                    max_iters = v;
                }
            }
            _ => positional.push(a),
        }
    }
    let root = PathBuf::from(
        positional
            .into_iter()
            .next()
            .unwrap_or_else(|| "data/matrices/kkt-mittelmann".to_string()),
    );

    eprintln!("scanning {} ...", root.display());
    let fams = collect_families(&root);
    eprintln!("{} families with >= {} iterates", fams.len(), MIN_ITERS);

    let mut results: Vec<FamilyResult> = Vec::new();
    let mut skipped = 0usize;

    for (fi, (family, paths)) in fams.iter().enumerate() {
        eprintln!(
            "  [{}/{}] {} ({} iterates)",
            fi,
            fams.len(),
            family,
            paths.len()
        );
        // Load the (capped) iterate sequence.
        let take = paths.len().min(max_iters);
        let mut seq: Vec<CscMatrix> = Vec::with_capacity(take);
        for p in paths.iter().take(take) {
            match read_mtx(p).and_then(|m| m.to_csc()) {
                Ok(c) if c.n > 0 => seq.push(c),
                _ => {}
            }
        }
        if seq.len() < MIN_ITERS {
            skipped += 1;
            continue;
        }
        let n = seq[0].n;

        // Full grid: each candidate ordering over the whole sequence.
        let mut grid: Vec<Vec<IterMeas>> = Vec::new();
        let mut failed = false;
        for (_, method) in CANDIDATES.iter() {
            match factor_sequence(&seq, method.clone()) {
                Some(times) => grid.push(times),
                None => {
                    failed = true;
                    break;
                }
            }
        }
        if failed || grid.len() != CANDIDATES.len() {
            skipped += 1;
            continue;
        }
        // Size gate: skip if the first iterate factors too fast to matter.
        let iter0_min: u128 = grid.iter().map(|t| t[0].us).min().unwrap_or(0);
        if iter0_min < FLOOR_US {
            skipped += 1;
            continue;
        }

        let (Some(auto), Some(autorace)) = (
            factor_sequence(&seq, OrderingMethod::Auto),
            factor_sequence(&seq, OrderingMethod::AutoRace),
        ) else {
            skipped += 1;
            continue;
        };

        let ni = seq.len();
        let n_reused = grid[0].iter().filter(|m| m.reused).count();
        // The latched race picks the ordering fastest on iterate 0 (the first factor).
        let winner_ci = (0..CANDIDATES.len())
            .min_by(|&a, &b| grid[a][0].us.cmp(&grid[b][0].us))
            .unwrap();
        // The steady-state-optimal ordering (best geomean over reused iterates) — the
        // ideal latch. Representativeness = does racing on iterate 0 find it?
        let ss_opt_ci = (0..CANDIDATES.len())
            .min_by(|&a, &b| {
                steady_state(&grid[a])
                    .partial_cmp(&steady_state(&grid[b]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        // Steady-state per-iterate cost (geomean over reused iterates). After the
        // latch, every steady iterate uses the winner ordering, so ss_latched is the
        // winner candidate's steady state.
        let ss_latched = steady_state(&grid[winner_ci]);
        let ss_auto = steady_state(&auto);
        let ss_autorace = steady_state(&autorace);

        // Race overhead as iterations-to-amortize vs Auto: the k−1 extra first-factors
        // (race_cost minus the one first-factor Auto also pays), divided by the
        // steady-state per-iterate saving of the latched winner over Auto.
        let race_cost: u128 = grid.iter().map(|t| t[0].us).sum();
        let extra_overhead = race_cost.saturating_sub(auto[0].us) as f64;
        let per_iter_saving = ss_auto - ss_latched;
        let breakeven_iters_vs_auto = if per_iter_saving > 0.0 {
            Some(extra_overhead / per_iter_saving)
        } else {
            None // latched not faster than Auto at steady state → never amortizes
        };

        results.push(FamilyResult {
            family: family.clone(),
            n,
            iters: ni,
            n_reused,
            winner: CANDIDATES[winner_ci].0.to_string(),
            winner_is_ss_optimal: winner_ci == ss_opt_ci,
            ss_latched,
            ss_auto,
            ss_autorace,
            breakeven_iters_vs_auto,
        });
    }

    // --- report ---
    println!();
    println!(
        "=== issue #110 latched actual-time race prototype ({} families, {} skipped) ===",
        results.len(),
        skipped
    );
    println!(
        "root: {}   (iterates capped at {}; steady state = geomean over pattern-reused iterates)",
        root.display(),
        max_iters
    );
    println!();
    println!(
        "{:<16}{:>8}{:>4}{:>5}  {:>7}  {:>20}  {:>12}",
        "family", "n", "N", "reus", "winner", "steady/iter vs", "breakeven"
    );
    println!(
        "{:<16}{:>8}{:>4}{:>5}  {:>7}  {:>20}  {:>12}",
        "", "", "", "", "(ssopt?)", "auto     race", "iters(vAuto)"
    );
    println!("{}", "-".repeat(80));
    for r in &results {
        let vs_auto = r.ss_latched / r.ss_auto.max(1.0);
        let vs_race = r.ss_latched / r.ss_autorace.max(1.0);
        println!(
            "{:<16}{:>8}{:>4}{:>5}  {:>3}{:<4}  {:>8.2} {:>8.2}  {:>12}",
            r.family,
            r.n,
            r.iters,
            r.n_reused,
            r.winner,
            if r.winner_is_ss_optimal {
                " ✓"
            } else {
                " ✗"
            },
            vs_auto,
            vs_race,
            r.breakeven_iters_vs_auto
                .map(|b| format!("{b:.0}"))
                .unwrap_or_else(|| "never".to_string()),
        );
    }

    // --- aggregate ---
    let n = results.len().max(1);
    let vs_auto: Vec<f64> = results
        .iter()
        .map(|r| r.ss_latched / r.ss_auto.max(1.0))
        .collect();
    let vs_race: Vec<f64> = results
        .iter()
        .map(|r| r.ss_latched / r.ss_autorace.max(1.0))
        .collect();
    let repr = results.iter().filter(|r| r.winner_is_ss_optimal).count();
    let mut breakevens: Vec<f64> = results
        .iter()
        .filter_map(|r| r.breakeven_iters_vs_auto)
        .collect();
    breakevens.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_be = if breakevens.is_empty() {
        f64::NAN
    } else {
        breakevens[breakevens.len() / 2]
    };
    let never = results
        .iter()
        .filter(|r| r.breakeven_iters_vs_auto.is_none())
        .count();

    println!();
    println!("STEADY STATE (per-iterate, pattern-reused iterates — the long-run asymptote):");
    println!(
        "  latched winner vs Auto:     geomean {:.3}  ({:.1}% faster)",
        geomean(&vs_auto),
        100.0 * (1.0 - geomean(&vs_auto))
    );
    println!(
        "  latched winner vs AutoRace: geomean {:.3}  ({:.1}% faster)",
        geomean(&vs_race),
        100.0 * (1.0 - geomean(&vs_race))
    );
    println!();
    println!("REPRESENTATIVENESS: iterate-0 race winner == steady-state-optimal ordering");
    println!(
        "  {}/{} ({:.0}%) — how often racing on the first factor picks the right latch.",
        repr,
        n,
        100.0 * repr as f64 / n as f64
    );
    println!();
    println!("RACE OVERHEAD amortization vs Auto (the good baseline):");
    println!("  break-even iterate count (k−1 extra first-factors / per-iterate saving):");
    println!(
        "    median {:.0} iterations   |   {} of {} families never amortize (latched not faster)",
        median_be, never, n
    );
    println!("  => latched race pays off once the pattern is reused past ~the break-even count.");
}
