//! Issue #110 measurement study: does the cheap `Auto` ordering heuristic
//! (`choose_adaptive`) leave material fill on the table vs the measure-then-choose
//! `AutoRace`, and how expensive is `AutoRace` by comparison?
//!
//! This is the gating measurement for the issue #110 research note
//! (`dev/research/issue-110-learned-tuner.md`): a learned ordering surrogate only
//! earns its keep if the deterministic cheap pick is materially worse than the
//! race optimum on a real corpus. If the gap is small, the hand-written heuristic
//! already captures the value and a model is not worth the maintenance surface.
//!
//! For each family in a corpus root (default `data/matrices/kkt`), one
//! representative matrix (the first `.mtx`; ordering fill depends only on the
//! sparsity pattern, which is shared across a family's IPM-iteration dumps) is
//! factored symbolically under seven ordering choices:
//!   AMD, AMF, MetisND, ScotchND, KahipND, Auto (cheap heuristic), AutoRace.
//! All use `SupernodeParams::default()` (so preprocess = Auto for every one —
//! apples to apples, the only variable is the ordering method).
//!
//! Reported per matrix (CSV to `--csv <path>`, default none) and aggregated to
//! stdout:
//! * `oracle_best`  = min fill over the 5 explicit methods (incl. AMF).
//! * `ratio_auto`   = fill_auto / oracle_best (>= 1; cheap pick's loss).
//! * `ratio_race`   = fill_autorace / oracle_best (race's loss; AutoRace does not
//!   race AMF, so this can exceed 1).
//! * `auto_vs_race` = fill_auto / fill_autorace (cheap pick vs the race).
//! * time_auto vs time_autorace (the cost the race pays for its pick).
//!
//! Usage:
//!   cargo run --release --bin study_ordering_gap
//!   cargo run --release --bin study_ordering_gap -- data/matrices/kkt --csv /tmp/gap.csv
//!   cargo run --release --bin study_ordering_gap -- <root> --limit 200

use feral::read_mtx;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Rec {
    family: String,
    n: usize,
    nnz: usize,
    fill_amd: u64,
    fill_amf: u64,
    fill_metis: u64,
    fill_scotch: u64,
    fill_kahip: u64,
    fill_auto: u64,
    fill_autorace: u64,
    t_auto_us: u128,
    t_autorace_us: u128,
}

fn measure(
    matrix: &feral::sparse::csc::CscMatrix,
    params: &SupernodeParams,
    method: OrderingMethod,
) -> Option<(u64, u128)> {
    let t = Instant::now();
    let sym = symbolic_factorize_with_method(matrix, params, method).ok()?;
    let us = t.elapsed().as_micros();
    Some((sym.factor_nnz_estimate as u64, us))
}

/// One representative matrix per family subdirectory of `root`.
fn collect_representatives(root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let rd = match std::fs::read_dir(root) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", root.display(), e);
            return out;
        }
    };
    let mut subs: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    subs.sort();
    for sub in subs {
        let inner = match std::fs::read_dir(&sub) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let mut mtxs: Vec<PathBuf> = inner
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mtx"))
            .collect();
        mtxs.sort();
        if let Some(first) = mtxs.into_iter().next() {
            out.push(first);
        }
    }
    out
}

fn geomean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    let n = vals.len() as f64;
    (vals.iter().map(|v| v.ln()).sum::<f64>() / n).exp()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut csv_path: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut it = raw.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--csv" => csv_path = it.next().map(PathBuf::from),
            "--limit" => limit = it.next().and_then(|s| s.parse().ok()),
            _ => positional.push(arg),
        }
    }
    let root = PathBuf::from(
        positional
            .into_iter()
            .next()
            .unwrap_or_else(|| "data/matrices/kkt".to_string()),
    );

    eprintln!("scanning {} for family representatives ...", root.display());
    let mut files = collect_representatives(&root);
    if let Some(l) = limit {
        files.truncate(l);
    }
    if files.is_empty() {
        eprintln!("no family representatives found under {}", root.display());
        std::process::exit(1);
    }
    eprintln!("{} family representatives", files.len());

    let params = SupernodeParams::default();
    let mut recs: Vec<Rec> = Vec::new();
    let mut skipped = 0usize;

    for (idx, path) in files.iter().enumerate() {
        let family = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        if idx % 50 == 0 {
            eprintln!("  [{}/{}] {}", idx, files.len(), family);
        }
        let mtx = match read_mtx(path) {
            Ok(m) => m,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        if csc.n == 0 {
            skipped += 1;
            continue;
        }
        let n = csc.n;
        let nnz = csc.row_idx.len();

        let (
            Some((fill_amd, _)),
            Some((fill_amf, _)),
            Some((fill_metis, _)),
            Some((fill_scotch, _)),
            Some((fill_kahip, _)),
            Some((fill_auto, t_auto_us)),
            Some((fill_autorace, t_autorace_us)),
        ) = (
            measure(&csc, &params, OrderingMethod::Amd),
            measure(&csc, &params, OrderingMethod::Amf),
            measure(&csc, &params, OrderingMethod::MetisND),
            measure(&csc, &params, OrderingMethod::ScotchND),
            measure(&csc, &params, OrderingMethod::KahipND),
            measure(&csc, &params, OrderingMethod::Auto),
            measure(&csc, &params, OrderingMethod::AutoRace),
        )
        else {
            skipped += 1;
            continue;
        };

        recs.push(Rec {
            family,
            n,
            nnz,
            fill_amd,
            fill_amf,
            fill_metis,
            fill_scotch,
            fill_kahip,
            fill_auto,
            fill_autorace,
            t_auto_us,
            t_autorace_us,
        });
    }

    if let Some(p) = &csv_path {
        match std::fs::File::create(p) {
            Ok(mut f) => {
                let _ = writeln!(
                    f,
                    "family,n,nnz,fill_amd,fill_amf,fill_metis,fill_scotch,fill_kahip,fill_auto,fill_autorace,oracle_best,ratio_auto,ratio_race,auto_vs_race,t_auto_us,t_autorace_us"
                );
                for r in &recs {
                    let oracle = r
                        .fill_amd
                        .min(r.fill_amf)
                        .min(r.fill_metis)
                        .min(r.fill_scotch)
                        .min(r.fill_kahip)
                        .max(1);
                    let ratio_auto = r.fill_auto as f64 / oracle as f64;
                    let ratio_race = r.fill_autorace as f64 / oracle as f64;
                    let auto_vs_race = r.fill_auto as f64 / r.fill_autorace.max(1) as f64;
                    let _ = writeln!(
                        f,
                        "{},{},{},{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{},{}",
                        r.family,
                        r.n,
                        r.nnz,
                        r.fill_amd,
                        r.fill_amf,
                        r.fill_metis,
                        r.fill_scotch,
                        r.fill_kahip,
                        r.fill_auto,
                        r.fill_autorace,
                        oracle,
                        ratio_auto,
                        ratio_race,
                        auto_vs_race,
                        r.t_auto_us,
                        r.t_autorace_us,
                    );
                }
                eprintln!("wrote per-matrix CSV to {}", p.display());
            }
            Err(e) => eprintln!("could not write CSV {}: {}", p.display(), e),
        }
    }

    // --- aggregate ---
    let mut ratio_auto: Vec<f64> = Vec::new();
    let mut ratio_race: Vec<f64> = Vec::new();
    let mut auto_vs_race: Vec<f64> = Vec::new();
    let (mut auto_optimal, mut auto_ties_race, mut auto_beats_race, mut race_beats_auto) =
        (0usize, 0usize, 0usize, 0usize);
    let (mut loss_gt_2, mut loss_gt_10, mut loss_gt_50) = (0usize, 0usize, 0usize);
    let (mut t_auto_total, mut t_race_total) = (0u128, 0u128);

    for r in &recs {
        let oracle = r
            .fill_amd
            .min(r.fill_amf)
            .min(r.fill_metis)
            .min(r.fill_scotch)
            .min(r.fill_kahip)
            .max(1);
        let ra = r.fill_auto as f64 / oracle as f64;
        let rr = r.fill_autorace as f64 / oracle as f64;
        let avr = r.fill_auto as f64 / r.fill_autorace.max(1) as f64;
        ratio_auto.push(ra);
        ratio_race.push(rr);
        auto_vs_race.push(avr);
        if r.fill_auto <= oracle {
            auto_optimal += 1;
        }
        if r.fill_auto == r.fill_autorace {
            auto_ties_race += 1;
        } else if r.fill_auto < r.fill_autorace {
            auto_beats_race += 1;
        } else {
            race_beats_auto += 1;
        }
        if ra > 1.02 {
            loss_gt_2 += 1;
        }
        if ra > 1.10 {
            loss_gt_10 += 1;
        }
        if ra > 1.50 {
            loss_gt_50 += 1;
        }
        t_auto_total += r.t_auto_us;
        t_race_total += r.t_autorace_us;
    }

    let mut sa = ratio_auto.clone();
    sa.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut sr = ratio_race.clone();
    sr.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let m = recs.len().max(1);
    println!();
    println!(
        "=== issue #110 ordering-selection gap study ({} matrices, {} skipped) ===",
        recs.len(),
        skipped
    );
    println!("root: {}", root.display());
    println!();
    println!("fill of cheap `Auto` vs the 5-method oracle (min incl. AMF):");
    println!(
        "  Auto == oracle optimum : {}/{}  ({:.1}%)",
        auto_optimal,
        recs.len(),
        100.0 * auto_optimal as f64 / m as f64
    );
    println!("  ratio_auto = fill_auto / oracle_best   (1.000 = optimal):");
    println!(
        "    geomean {:.4}   p50 {:.4}   p90 {:.4}   p99 {:.4}   max {:.4}",
        geomean(&ratio_auto),
        percentile(&sa, 0.50),
        percentile(&sa, 0.90),
        percentile(&sa, 0.99),
        percentile(&sa, 1.0),
    );
    println!(
        "  matrices where Auto loses >2% / >10% / >50% fill: {} / {} / {}",
        loss_gt_2, loss_gt_10, loss_gt_50
    );
    println!();
    println!("fill of `AutoRace` vs the 5-method oracle (AutoRace does not race AMF):");
    println!(
        "  ratio_race geomean {:.4}   p50 {:.4}   p90 {:.4}   p99 {:.4}   max {:.4}",
        geomean(&ratio_race),
        percentile(&sr, 0.50),
        percentile(&sr, 0.90),
        percentile(&sr, 0.99),
        percentile(&sr, 1.0),
    );
    println!();
    println!("cheap `Auto` vs `AutoRace` head to head:");
    println!(
        "  Auto ties race: {}   Auto beats race: {}   race beats Auto: {}",
        auto_ties_race, auto_beats_race, race_beats_auto
    );
    println!(
        "  auto_vs_race = fill_auto / fill_autorace geomean {:.4}",
        geomean(&auto_vs_race)
    );
    println!();
    println!("symbolic time (the cost AutoRace pays for its pick):");
    println!(
        "  total t_auto = {} us   total t_autorace = {} us   race/auto = {:.2}x",
        t_auto_total,
        t_race_total,
        t_race_total as f64 / (t_auto_total.max(1)) as f64
    );
}
