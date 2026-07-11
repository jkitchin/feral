//! Issue #110 follow-up study: the TIME/MEMORY-objective version of
//! `study_ordering_gap`. The fill study (`study_ordering_gap`) showed feral's
//! deterministic ordering selection is near-optimal *by fill* — but fill is a proxy
//! that provably diverges from factor wall-clock and peak memory (see the #110 note
//! §5.5.1 and nql180). This binary measures the real objectives directly, so we can
//! quantify the *potential* of a time/memory-aware tuner.
//!
//! For each family representative under `data/matrices/kkt`, for each of the 5
//! explicit orderings (AMD, AMF, MetisND, ScotchND, KahipND) plus `Auto` and
//! `AutoRace`, it measures:
//!   * numeric factor wall-clock (min over reps; symbolic is reused across reps, so
//!     this is the IPM-relevant per-iteration numeric cost, not the one-off analysis).
//!   * peak memory model = 16 * nnz_l (stored factor) + peak_contrib_bytes
//!     (transient multifrontal contribution-block stack) — feral's own accounting.
//!   * fill (factor_nnz_estimate), for cross-check against the fill study.
//!
//! The headline questions:
//!   1. How often does the FILL-optimal ordering (what `AutoRace` picks) differ from
//!      the TIME-optimal / MEMORY-optimal ordering? (the fill-lies rate)
//!   2. If you always pick fill-optimal, how much time / memory do you leave on the
//!      table vs a perfect per-matrix oracle? (the ceiling for a time/mem tuner)
//!   3. How much would a perfect time-oracle beat feral's actual `Auto` today?
//!
//! Timing on tiny matrices is dominated by overhead and irrelevant (nobody tunes a
//! 5 us solve), so the regret distributions are reported both over all matrices and
//! over the "meaningful" subset (min factor time >= TIME_FLOOR_US).
//!
//! Usage:
//!   cargo run --release --bin study_ordering_timemem -- data/matrices/kkt --csv /tmp/tm.csv
//!   cargo run --release --bin study_ordering_timemem -- <root> --limit 200

use feral::symbolic::{OrderingMethod, SupernodeParams};
use feral::{read_mtx, CscMatrix, Solver};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Factor-time floor (us) below which a matrix is too small for its timing to be
/// meaningful; excluded from the regret distributions (but kept in the CSV).
const TIME_FLOOR_US: u128 = 50;

/// Orderings measured, in a fixed order. Indices used throughout.
const METHODS: &[(&str, OrderingMethod)] = &[
    ("amd", OrderingMethod::Amd),
    ("amf", OrderingMethod::Amf),
    ("metis", OrderingMethod::MetisND),
    ("scotch", OrderingMethod::ScotchND),
    ("kahip", OrderingMethod::KahipND),
    ("auto", OrderingMethod::Auto),
    ("autorace", OrderingMethod::AutoRace),
];

#[derive(Clone)]
struct Meas {
    fill: u64,
    /// Dense-multifrontal work proxy: Σ over supernodes of `ncol · nrow²`. The
    /// standard "flops" estimate — a candidate cheap *time* proxy to race on instead
    /// of fill (issue-73 flagged it unreliable on nql180; this study measures how it
    /// does across the corpus).
    flop_proxy: u128,
    /// Largest frontal dimension (max `nrow`) — another cheap time/parallelism proxy.
    max_front: u64,
    nnz_l: u64,
    peak_bytes: u64,
    time_us: u128,
}

struct Rec {
    family: String,
    n: usize,
    nnz: usize,
    /// One `Meas` per entry in `METHODS` (None if that ordering failed).
    per_method: Vec<Option<Meas>>,
}

/// Factor `matrix` with a fixed `method`, reusing the cached symbolic across reps.
/// Returns (fill, nnz_l, peak_bytes, min factor time). `None` if the factor fails.
fn measure(matrix: &CscMatrix, method: OrderingMethod) -> Option<Meas> {
    let mut solver = Solver::new().with_ordering(method);
    // Warm-up factor: builds the symbolic analysis (which we do NOT time) plus the
    // first numeric factor. Its wall-clock also sets the rep budget below.
    let warm_t = Instant::now();
    if !matches!(solver.factor(matrix, None), feral::FactorStatus::Success) {
        return None;
    }
    let warm_dt = warm_t.elapsed();
    let stats = solver.last_factor_stats()?;
    let sym = solver.symbolic()?;
    let peak_bytes = 16u64.saturating_mul(stats.nnz_l as u64) + sym.peak_contrib_bytes as u64;
    let fill = sym.factor_nnz_estimate as u64;
    // Cheap symbolic time proxies (no numeric work): dense-front work Σ ncol·nrow²
    // and the largest front dimension.
    let mut flop_proxy: u128 = 0;
    let mut max_front: u64 = 0;
    for sn in &sym.supernodes {
        let ncol = sn.ncol as u128;
        let nrow = sn.nrow as u128;
        flop_proxy += ncol * nrow * nrow;
        max_front = max_front.max(sn.nrow as u64);
    }

    // Timed reps: each factor() reuses the cached symbolic (same pattern) and redoes
    // the numeric factorization — the IPM per-iteration cost. Take the min over reps
    // (least OS noise), adaptively repeating until a time budget or rep cap. The
    // minimum rep count is warmup-time-aware so multi-second factors (n~10^5) don't
    // force dozens of seconds of work: fast factors get more reps (noise), slow ones
    // get few (already stable, expensive). Note: `warm_dt` includes the one-off
    // symbolic build, so it is an upper bound on the numeric cost and never enters
    // `best`.
    let min_reps = if warm_dt > Duration::from_millis(200) {
        2
    } else if warm_dt > Duration::from_millis(20) {
        3
    } else {
        5
    };
    let mut best = Duration::from_secs(3600);
    let budget = Duration::from_millis(40);
    let started = Instant::now();
    let mut reps = 0u32;
    while reps < 200 && (reps < min_reps || started.elapsed() < budget) {
        let t = Instant::now();
        let ok = matches!(solver.factor(matrix, None), feral::FactorStatus::Success);
        let dt = t.elapsed();
        if !ok {
            return None;
        }
        if dt < best {
            best = dt;
        }
        reps += 1;
    }
    Some(Meas {
        fill,
        flop_proxy,
        max_front,
        nnz_l: stats.nnz_l as u64,
        peak_bytes,
        time_us: best.as_micros(),
    })
}

fn collect_representatives(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        eprintln!("error: cannot read {}", root.display());
        return out;
    };
    let entries: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    // Flat layout: `.mtx` files directly under `root` (e.g. tests/data/large). Use
    // each file as its own "family" so the study works on ad-hoc large-matrix dirs.
    let mut flat: Vec<PathBuf> = entries
        .iter()
        .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("mtx"))
        .cloned()
        .collect();
    if !flat.is_empty() {
        flat.sort();
        return flat;
    }
    // Per-family layout: one representative (first `.mtx`) per subdirectory.
    let mut subs: Vec<PathBuf> = entries.into_iter().filter(|p| p.is_dir()).collect();
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
    (vals.iter().map(|v| v.ln()).sum::<f64>() / vals.len() as f64).exp()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

/// Index of the min-`key` method among the first 5 (explicit) methods; `None` if all
/// failed. Ties break to the earliest (AMD < AMF < Metis < Scotch < KaHIP).
fn argmin_explicit(rec: &Rec, key: impl Fn(&Meas) -> u128) -> Option<usize> {
    (0..5)
        .filter_map(|i| rec.per_method[i].as_ref().map(|m| (i, key(m))))
        .min_by(|a, b| a.1.cmp(&b.1))
        .map(|(i, _)| i)
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut csv_path: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut positional = Vec::new();
    let mut it = raw.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--csv" => csv_path = it.next().map(PathBuf::from),
            "--limit" => limit = it.next().and_then(|s| s.parse().ok()),
            _ => positional.push(a),
        }
    }
    let root = PathBuf::from(
        positional
            .into_iter()
            .next()
            .unwrap_or_else(|| "data/matrices/kkt".to_string()),
    );

    eprintln!("scanning {} ...", root.display());
    let mut files = collect_representatives(&root);
    if let Some(l) = limit {
        files.truncate(l);
    }
    if files.is_empty() {
        eprintln!("no family representatives found");
        std::process::exit(1);
    }
    eprintln!("{} family representatives", files.len());

    let _ = SupernodeParams::default(); // (defaults used inside Solver::new)
    let mut recs: Vec<Rec> = Vec::new();
    let mut skipped = 0usize;

    for (idx, path) in files.iter().enumerate() {
        // Flat layout (file sits directly in `root`): label by file stem.
        // Family layout (file in a subdir): label by the subdir name.
        let flat = path.parent() == Some(root.as_path());
        let family = if flat {
            path.file_stem().and_then(|s| s.to_str())
        } else {
            path.parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
        }
        .unwrap_or("?")
        .to_string();
        if idx % 25 == 0 {
            eprintln!("  [{}/{}] {}", idx, files.len(), family);
        }
        let Ok(mtx) = read_mtx(path) else {
            skipped += 1;
            continue;
        };
        let Ok(csc) = mtx.to_csc() else {
            skipped += 1;
            continue;
        };
        if csc.n == 0 {
            skipped += 1;
            continue;
        }
        let per_method: Vec<Option<Meas>> = METHODS
            .iter()
            .map(|(_, m)| measure(&csc, m.clone()))
            .collect();
        // Need the 5 explicit methods to all succeed for a clean oracle comparison.
        if per_method[..5].iter().any(|m| m.is_none()) {
            skipped += 1;
            continue;
        }
        recs.push(Rec {
            family,
            n: csc.n,
            nnz: csc.row_idx.len(),
            per_method,
        });
    }

    // --- CSV ---
    if let Some(p) = &csv_path {
        if let Ok(mut f) = std::fs::File::create(p) {
            let mut hdr = String::from("family,n,nnz");
            for (name, _) in METHODS {
                hdr.push_str(&format!(",{name}_fill,{name}_nnzl,{name}_peakb,{name}_us"));
            }
            let _ = writeln!(f, "{hdr}");
            for r in &recs {
                let mut line = format!("{},{},{}", r.family, r.n, r.nnz);
                for m in &r.per_method {
                    match m {
                        Some(m) => line.push_str(&format!(
                            ",{},{},{},{}",
                            m.fill, m.nnz_l, m.peak_bytes, m.time_us
                        )),
                        None => line.push_str(",,,,"),
                    }
                }
                let _ = writeln!(f, "{line}");
            }
            eprintln!("wrote per-matrix CSV to {}", p.display());
        }
    }

    // --- aggregate ---
    // For each matrix: the fill-optimal, time-optimal, memory-optimal explicit
    // ordering. Then the regret of the fill-optimal pick on time and memory.
    let mut time_regret_all: Vec<f64> = Vec::new();
    let mut time_regret_big: Vec<f64> = Vec::new();
    let mut mem_regret_all: Vec<f64> = Vec::new();
    let (mut fill_eq_time, mut fill_eq_time_big, mut big_n) = (0usize, 0usize, 0usize);
    let mut fill_eq_mem = 0usize;
    // Totals for the "oracle vs Auto today" comparison (meaningful subset only).
    let (mut tot_auto_us, mut tot_autorace_us, mut tot_timeoracle_us) = (0u128, 0u128, 0u128);
    // Deterministic proxy-race prototype: totals + per-matrix time-regret of picking
    // the ordering that minimizes each cheap symbolic proxy (vs the time-oracle).
    // A proxy that lands near the oracle means a deterministic race on it (no ML)
    // captures the §5.6 headroom.
    let (mut tot_fill_us, mut tot_flop_us, mut tot_maxfront_us) = (0u128, 0u128, 0u128);
    let mut flop_regret: Vec<f64> = Vec::new();
    let mut maxfront_regret: Vec<f64> = Vec::new();

    for r in &recs {
        let fill_best = argmin_explicit(r, |m| m.fill as u128).unwrap();
        let time_best = argmin_explicit(r, |m| m.time_us).unwrap();
        let mem_best = argmin_explicit(r, |m| m.peak_bytes as u128).unwrap();
        let flop_best = argmin_explicit(r, |m| m.flop_proxy).unwrap();
        let maxfront_best = argmin_explicit(r, |m| m.max_front as u128).unwrap();

        let t_fillpick = r.per_method[fill_best].as_ref().unwrap().time_us.max(1);
        let t_timebest = r.per_method[time_best].as_ref().unwrap().time_us.max(1);
        let mem_fillpick = r.per_method[fill_best].as_ref().unwrap().peak_bytes.max(1);
        let mem_membest = r.per_method[mem_best].as_ref().unwrap().peak_bytes.max(1);

        let treg = t_fillpick as f64 / t_timebest as f64;
        let mreg = mem_fillpick as f64 / mem_membest as f64;
        time_regret_all.push(treg);
        mem_regret_all.push(mreg);
        if fill_best == time_best {
            fill_eq_time += 1;
        }
        if fill_best == mem_best {
            fill_eq_mem += 1;
        }

        // Meaningful subset: the time-best ordering is above the noise floor.
        if t_timebest >= TIME_FLOOR_US {
            big_n += 1;
            time_regret_big.push(treg);
            if fill_best == time_best {
                fill_eq_time_big += 1;
            }
            // Oracle vs Auto today (only where timing is meaningful).
            tot_timeoracle_us += t_timebest;
            if let Some(a) = &r.per_method[5] {
                tot_auto_us += a.time_us.max(1);
            }
            if let Some(ar) = &r.per_method[6] {
                tot_autorace_us += ar.time_us.max(1);
            }
            // Deterministic proxy-race prototype: the factor time you'd get by racing
            // on each cheap proxy and picking its argmin ordering.
            let t_flop = r.per_method[flop_best].as_ref().unwrap().time_us.max(1);
            let t_maxfront = r.per_method[maxfront_best].as_ref().unwrap().time_us.max(1);
            tot_fill_us += t_fillpick;
            tot_flop_us += t_flop;
            tot_maxfront_us += t_maxfront;
            flop_regret.push(t_flop as f64 / t_timebest as f64);
            maxfront_regret.push(t_maxfront as f64 / t_timebest as f64);
        }
    }

    let sort = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v
    };
    let tra = sort(time_regret_all.clone());
    let trb = sort(time_regret_big.clone());
    let mra = sort(mem_regret_all.clone());
    let m = recs.len().max(1);

    println!();
    println!(
        "=== issue #110 TIME/MEMORY ordering study ({} matrices, {} skipped) ===",
        recs.len(),
        skipped
    );
    println!("root: {}", root.display());
    println!(
        "timing: min-of-reps numeric factor, symbolic reused; meaningful subset = \
         time-best >= {} us ({} of {} matrices)",
        TIME_FLOOR_US, big_n, m
    );
    println!();
    println!("does the FILL-optimal ordering match the TIME-optimal ordering?");
    println!(
        "  all: {}/{} ({:.1}%) agree   |   meaningful subset: {}/{} ({:.1}%) agree",
        fill_eq_time,
        m,
        100.0 * fill_eq_time as f64 / m as f64,
        fill_eq_time_big,
        big_n.max(1),
        100.0 * fill_eq_time_big as f64 / big_n.max(1) as f64,
    );
    println!("  => the fill-lies rate (fill-best != time-best) is the complement.");
    println!();
    println!("TIME regret of always picking the fill-optimal ordering (t_fill / t_time_best):");
    println!(
        "  all matrices     : geomean {:.3}  p50 {:.3}  p90 {:.3}  p99 {:.3}  max {:.3}",
        geomean(&time_regret_all),
        percentile(&tra, 0.50),
        percentile(&tra, 0.90),
        percentile(&tra, 0.99),
        percentile(&tra, 1.0),
    );
    println!(
        "  meaningful subset: geomean {:.3}  p50 {:.3}  p90 {:.3}  p99 {:.3}  max {:.3}",
        geomean(&time_regret_big),
        percentile(&trb, 0.50),
        percentile(&trb, 0.90),
        percentile(&trb, 0.99),
        percentile(&trb, 1.0),
    );
    println!("  (1.00 = fill pick is also time-optimal; >1 = time left on the table)");
    println!();
    println!("does the FILL-optimal ordering match the MEMORY-optimal ordering?");
    println!(
        "  all: {}/{} ({:.1}%) agree",
        fill_eq_mem,
        m,
        100.0 * fill_eq_mem as f64 / m as f64
    );
    println!(
        "  MEMORY regret (peak_fill / peak_mem_best): geomean {:.3}  p90 {:.3}  p99 {:.3}  max {:.3}",
        geomean(&mem_regret_all),
        percentile(&mra, 0.90),
        percentile(&mra, 0.99),
        percentile(&mra, 1.0),
    );
    println!();
    println!("upper bound on a TIME-oracle tuner vs feral today (meaningful subset totals):");
    println!(
        "  total factor us: Auto = {}   AutoRace = {}   time-oracle = {}",
        tot_auto_us, tot_autorace_us, tot_timeoracle_us
    );
    println!(
        "  a perfect per-matrix time-oracle would cut total factor time to {:.1}% of Auto \
         ({:.2}x speedup) and {:.1}% of AutoRace",
        100.0 * tot_timeoracle_us as f64 / tot_auto_us.max(1) as f64,
        tot_auto_us as f64 / tot_timeoracle_us.max(1) as f64,
        100.0 * tot_timeoracle_us as f64 / tot_autorace_us.max(1) as f64,
    );

    // --- deterministic proxy-race prototype ---
    // How close does a *cheap symbolic proxy* race (no numeric factoring, no ML) get
    // to the time-oracle? `fill` is what AutoRace uses today; `flop`=Σncol·nrow² and
    // `max_front` are the candidate time-aware proxies. A proxy whose total ≈ oracle
    // and whose regret ≈ 1.0 means "just race on this proxy" closes the §5.6 gap with
    // no model; a stubborn residual over the best proxy is the actual learned-model case.
    let frs = sort(flop_regret.clone());
    let mfs = sort(maxfront_regret.clone());
    println!();
    println!(
        "deterministic proxy-race prototype (meaningful subset; pick argmin(proxy) per matrix):"
    );
    println!(
        "  total factor us:  fill-race = {}   flop-race = {}   maxfront-race = {}   oracle = {}",
        tot_fill_us, tot_flop_us, tot_maxfront_us, tot_timeoracle_us
    );
    println!(
        "  vs oracle:        fill-race {:.3}x   flop-race {:.3}x   maxfront-race {:.3}x   (1.00 = matches oracle)",
        tot_fill_us as f64 / tot_timeoracle_us.max(1) as f64,
        tot_flop_us as f64 / tot_timeoracle_us.max(1) as f64,
        tot_maxfront_us as f64 / tot_timeoracle_us.max(1) as f64,
    );
    println!(
        "  per-matrix time-regret vs oracle:  fill-race geo {:.3} p90 {:.3} max {:.3}",
        geomean(&time_regret_big),
        percentile(&trb, 0.90),
        percentile(&trb, 1.0),
    );
    println!(
        "                                     flop-race geo {:.3} p90 {:.3} max {:.3}",
        geomean(&flop_regret),
        percentile(&frs, 0.90),
        percentile(&frs, 1.0),
    );
    println!(
        "                                 maxfront-race geo {:.3} p90 {:.3} max {:.3}",
        geomean(&maxfront_regret),
        percentile(&mfs, 0.90),
        percentile(&mfs, 1.0),
    );
}
