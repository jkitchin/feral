//! Corpus-wide Solver-reuse benchmark.
//!
//! Walks `data/matrices/kkt/` and groups all `.mtx` files by family
//! (`<FAM>_NNNN.mtx` → `FAM`). For each family, runs two scenarios and
//! reports per-iteration cost + speedup of Solver vs free-function:
//!   - Scenario A (Solver): one persistent `Solver` factors every iterate
//!     in the family. `symbolic_factorize` runs once, the rest reuse
//!     the cached `SymbolicFactorization` and pooled `FactorWorkspace`.
//!   - Scenario B (free-fn): `symbolic_factorize` + `factorize_multifrontal`
//!     each iteration. This is what the per-matrix `bench` measures.
//!
//! Motivation: the per-matrix `bench` walks 154k matrices through the
//! free-function API and reports symbolic = 64% of wall. In production
//! IPM use, the same KKT pattern re-factorizes hundreds of times within
//! one solve; symbolic is paid once. This bench measures the realistic
//! workload so optimization effort lands on the actually-hot path.
//!
//! Usage: `cargo run --release --bin bench_solver_corpus`
//!
//! Env knobs:
//!   FERAL_BENCH_FAMILY_CAP     max iterates loaded per family (default 64)
//!   FERAL_BENCH_MIN_ITERS      skip families with fewer iterates (default 4)
//!   FERAL_BENCH_KKT_ROOT       data root (default data/matrices/kkt)
//!   FERAL_BENCH_FAMILY_FILTER  comma-sep substring filter on family names
//!   FERAL_BENCH_MAX_FAMILIES   cap families processed (default unlimited)

use feral::numeric::factorize::factorize_multifrontal;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, NumericParams, Solver, ZeroPivotAction};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn numeric_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    })
}

fn family_of(stem: &str) -> Option<&str> {
    let idx = stem.rfind('_')?;
    let suffix = &stem[idx + 1..];
    if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
        Some(&stem[..idx])
    } else {
        None
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Discover families in the corpus root. Returns a map from family name
/// to a sorted list of `.mtx` paths (sorted by filename so iterate
/// indices come in order).
fn discover_families(root: &Path) -> BTreeMap<String, Vec<PathBuf>> {
    let mut by_family: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ERROR: cannot read {}: {}", root.display(), e);
            return by_family;
        }
    };
    for ent in entries.flatten() {
        let p = ent.path();
        if !p.is_dir() {
            continue;
        }
        let dir_entries = match std::fs::read_dir(&p) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let mut paths: Vec<PathBuf> = dir_entries
            .flatten()
            .filter_map(|e| {
                let f = e.path();
                if f.extension().and_then(|s| s.to_str()) == Some("mtx") {
                    Some(f)
                } else {
                    None
                }
            })
            .collect();
        paths.sort();
        if paths.is_empty() {
            continue;
        }
        // Use the directory name as the family key. This matches the
        // CUTEst convention where `data/matrices/kkt/<FAM>/<FAM>_NNNN.mtx`.
        let fam = match p.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        // Sanity: the file stems should match `<fam>_NNNN`. Skip the
        // few directories that don't follow the convention rather than
        // mis-attributing iterate indices.
        let first_stem = paths[0].file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if let Some(detected) = family_of(first_stem) {
            if detected != fam {
                // Iterate naming and dir name disagree — keep going,
                // dir-name is authoritative for grouping.
            }
        }
        by_family.insert(fam, paths);
    }
    by_family
}

fn run_scenario_solver(iterates: &[feral::CscMatrix]) -> (Duration, usize) {
    let params = numeric_params();
    let snode = SupernodeParams::default();
    let mut solver = Solver::with_params(params, snode);
    let t0 = Instant::now();
    for csc in iterates {
        let _ = solver.factor(csc, None);
    }
    (t0.elapsed(), solver.symbolic_call_count())
}

fn run_scenario_freefn(iterates: &[feral::CscMatrix]) -> (Duration, usize) {
    let params = numeric_params();
    let snode = SupernodeParams::default();
    let mut symbolic_calls = 0usize;
    let t0 = Instant::now();
    for csc in iterates {
        let sym = match symbolic_factorize(csc, &snode) {
            Ok(s) => s,
            Err(_) => continue,
        };
        symbolic_calls += 1;
        let _ = factorize_multifrontal(csc, &sym, &params);
    }
    (t0.elapsed(), symbolic_calls)
}

#[derive(Clone)]
struct FamilyResult {
    family: String,
    n: usize,
    nnz: usize,
    iters: usize,
    solver_total_us: u128,
    solver_sym_calls: usize,
    freefn_total_us: u128,
}

fn run_family(family: &str, paths: &[PathBuf], cap: usize) -> Option<FamilyResult> {
    let take = paths.len().min(cap);
    let mut iterates: Vec<feral::CscMatrix> = Vec::with_capacity(take);
    for path in paths.iter().take(take) {
        let mtx = match read_mtx(path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(_) => continue,
        };
        iterates.push(csc);
    }
    if iterates.is_empty() {
        return None;
    }
    let n = iterates[0].n;
    let nnz = iterates[0].row_idx.len();

    // Warmup pass to neutralize cold-cache effects (allocator, page
    // faults, branch predictor). Throwaway result.
    let _ = run_scenario_solver(&iterates);
    let _ = run_scenario_freefn(&iterates);

    let (solver_dur, solver_sym_calls) = run_scenario_solver(&iterates);
    let (freefn_dur, _) = run_scenario_freefn(&iterates);

    Some(FamilyResult {
        family: family.to_string(),
        n,
        nnz,
        iters: iterates.len(),
        solver_total_us: solver_dur.as_micros(),
        solver_sym_calls,
        freefn_total_us: freefn_dur.as_micros(),
    })
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn geomean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let s: f64 = xs.iter().filter(|x| **x > 0.0).map(|x| x.ln()).sum();
    let n = xs.iter().filter(|x| **x > 0.0).count();
    if n == 0 {
        f64::NAN
    } else {
        (s / n as f64).exp()
    }
}

fn print_family_table(results: &[FamilyResult]) {
    println!(
        "\n{:<14} {:>5} {:>7} {:>5}   {:>11} {:>5}   {:>11}   {:>10} {:>10}   {:>7}",
        "family",
        "n",
        "nnz",
        "iter",
        "solver(us)",
        "syms",
        "freefn(us)",
        "solver/it",
        "freefn/it",
        "speedup",
    );
    println!("{}", "-".repeat(115));
    for r in results.iter().take(40) {
        let solver_per = r.solver_total_us as f64 / r.iters as f64;
        let freefn_per = r.freefn_total_us as f64 / r.iters as f64;
        let speedup = if r.solver_total_us > 0 {
            r.freefn_total_us as f64 / r.solver_total_us as f64
        } else {
            0.0
        };
        println!(
            "{:<14} {:>5} {:>7} {:>5}   {:>11} {:>5}   {:>11}   {:>10.1} {:>10.1}   {:>6.2}x",
            r.family,
            r.n,
            r.nnz,
            r.iters,
            r.solver_total_us,
            r.solver_sym_calls,
            r.freefn_total_us,
            solver_per,
            freefn_per,
            speedup,
        );
    }
    if results.len() > 40 {
        println!("  ... and {} more families", results.len() - 40);
    }
}

fn print_aggregate(results: &[FamilyResult]) {
    if results.is_empty() {
        println!("\nNo families processed.");
        return;
    }
    let total_solver: u128 = results.iter().map(|r| r.solver_total_us).sum();
    let total_freefn: u128 = results.iter().map(|r| r.freefn_total_us).sum();
    let total_iters: usize = results.iter().map(|r| r.iters).sum();
    let total_sym_calls: usize = results.iter().map(|r| r.solver_sym_calls).sum();
    let agg_speedup = if total_solver > 0 {
        total_freefn as f64 / total_solver as f64
    } else {
        0.0
    };

    let mut speedups: Vec<f64> = results
        .iter()
        .filter_map(|r| {
            if r.solver_total_us > 0 {
                Some(r.freefn_total_us as f64 / r.solver_total_us as f64)
            } else {
                None
            }
        })
        .collect();
    speedups.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n=== Corpus aggregate ===");
    println!("families processed:         {}", results.len());
    println!("total iterates:             {}", total_iters);
    println!(
        "solver symbolic calls:      {} (vs {} free-fn = {:.1}× fewer)",
        total_sym_calls,
        total_iters,
        if total_sym_calls > 0 {
            total_iters as f64 / total_sym_calls as f64
        } else {
            f64::NAN
        }
    );
    println!("solver total (us):          {}", total_solver);
    println!("freefn total (us):          {}", total_freefn);
    println!("aggregate speedup:          {:.2}x", agg_speedup);
    println!(
        "speedup distribution:  geomean={:.2}x  p10={:.2}x  p50={:.2}x  p90={:.2}x  max={:.2}x",
        geomean(&speedups),
        percentile(&speedups, 0.10),
        percentile(&speedups, 0.50),
        percentile(&speedups, 0.90),
        speedups.last().copied().unwrap_or(f64::NAN),
    );
    let symbolic_share_freefn =
        (total_freefn.saturating_sub(total_solver)) as f64 / total_freefn.max(1) as f64 * 100.0;
    println!(
        "implied symbolic share of freefn wall: {:.1}%  (sym amortized away by Solver)",
        symbolic_share_freefn
    );
}

fn main() {
    let cap = env_usize("FERAL_BENCH_FAMILY_CAP", 64);
    let min_iters = env_usize("FERAL_BENCH_MIN_ITERS", 4);
    let max_families = env_usize("FERAL_BENCH_MAX_FAMILIES", usize::MAX);
    let root =
        std::env::var("FERAL_BENCH_KKT_ROOT").unwrap_or_else(|_| "data/matrices/kkt".to_string());
    let filter = std::env::var("FERAL_BENCH_FAMILY_FILTER").ok();

    println!("Solver corpus benchmark");
    println!("  root:        {}", root);
    println!("  family cap:  {} iterates", cap);
    println!("  min iters:   {} (skip smaller families)", min_iters);
    if let Some(f) = &filter {
        println!("  filter:      {}", f);
    }
    if max_families != usize::MAX {
        println!("  max fams:    {}", max_families);
    }

    let by_family = discover_families(Path::new(&root));
    if by_family.is_empty() {
        eprintln!("ERROR: no families found under {}", root);
        std::process::exit(1);
    }
    println!("\ndiscovered {} families", by_family.len());

    // Apply filters and min-iters threshold up-front so we don't pay
    // load+factor cost for families we'll skip.
    let mut work: Vec<(String, Vec<PathBuf>)> = by_family
        .into_iter()
        .filter(|(name, paths)| {
            if paths.len() < min_iters {
                return false;
            }
            if let Some(f) = &filter {
                let pats: Vec<&str> = f.split(',').map(|s| s.trim()).collect();
                if !pats.iter().any(|pat| name.contains(pat)) {
                    return false;
                }
            }
            true
        })
        .collect();
    if work.len() > max_families {
        work.truncate(max_families);
    }
    println!(
        "running {} families (after min-iters / filter / cap)",
        work.len()
    );

    let mut results: Vec<FamilyResult> = Vec::with_capacity(work.len());
    let total = work.len();
    for (i, (fam, paths)) in work.into_iter().enumerate() {
        if i % 25 == 0 && i > 0 {
            eprintln!("  progress: {}/{} families", i, total);
        }
        if let Some(r) = run_family(&fam, &paths, cap) {
            results.push(r);
        }
    }

    // Sort families by iter count (descending) so the printed table
    // leads with the families that exercise reuse the most.
    let mut by_iters = results.clone();
    by_iters.sort_by_key(|r| std::cmp::Reverse(r.iters));
    print_family_table(&by_iters);

    print_aggregate(&results);

    // Worst speedups (where Solver helped least). Useful for spotting
    // pattern-changing families or cold-cache outliers.
    let mut by_speedup: Vec<&FamilyResult> = results.iter().collect();
    by_speedup.sort_by(|a, b| {
        let sa = if a.solver_total_us > 0 {
            a.freefn_total_us as f64 / a.solver_total_us as f64
        } else {
            f64::INFINITY
        };
        let sb = if b.solver_total_us > 0 {
            b.freefn_total_us as f64 / b.solver_total_us as f64
        } else {
            f64::INFINITY
        };
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("\n=== Bottom 10 families by Solver speedup ===");
    println!(
        "{:<14} {:>5} {:>5}   {:>11} {:>11}   {:>7}",
        "family", "n", "iter", "solver(us)", "freefn(us)", "speedup"
    );
    for r in by_speedup.iter().take(10) {
        let speedup = if r.solver_total_us > 0 {
            r.freefn_total_us as f64 / r.solver_total_us as f64
        } else {
            0.0
        };
        println!(
            "{:<14} {:>5} {:>5}   {:>11} {:>11}   {:>6.2}x",
            r.family, r.n, r.iters, r.solver_total_us, r.freefn_total_us, speedup
        );
    }
}
