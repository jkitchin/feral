//! Re-check of the `nemin` amalgamation lever **after** the 0.15.0
//! SIMD kernel work.
//!
//! Why this exists when the sweep is already in
//! `dev/tried-and-rejected.md` (2026-05-16, issue #10 lever 5): that
//! rejection turned on one quantity — "chain-link merges blow trailing
//! fill faster than the wider panel can amortize". The amortization
//! rate is precisely what 0.15.0 changed. The trailing update went from
//! an SSE2-baseline scalar tile walk to explicit AVX2/NEON (n=2955:
//! 4.24 s -> 1.52 s on x86; 2.25-3.19x on NEON), so a wide front is now
//! several times cheaper per flop while the fill penalty of a larger
//! `nemin` is unchanged. If the break-even moved, it moved here.
//!
//! Methodology follows `dev/decisions.md` 2026-08-09: **paired
//! alternating** A/B (all `nemin` values are timed once per pair, in
//! order, so drift hits every arm equally), `min_us` per arm, and a
//! sign test over the pairs. Do not compare medians collected at
//! different times — this container has produced a 1.9x spread on
//! identical code.
//!
//! `nemin` changes fill **and** numerics, so byte-parity does not
//! apply. Inertia and the true relative residual are reported per arm
//! as the local correctness signal; the shipping gate is a full corpus
//! run, which this container does not have.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin diag_nemin_post_simd \
//!       -- <matrix.mtx> [more.mtx ...]
//!
//! Env:
//!   FERAL_NEMIN_LIST=1,4,8,16,32,64   arms to sweep (default this list)
//!   FERAL_NEMIN_PAIRS=10              paired repetitions (default 10)
//!   FERAL_MERGE_BUDGET_LIST=0,1e3,..  sweep `merge_flop_budget` instead
//!                                     of `nemin`; arms are budgets and
//!                                     `nemin` is held at its default.
//!                                     The baseline arm is the shipped
//!                                     `None` (printed as `off`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams, SymbolicFactorization};
use feral::{read_mtx, CscMatrix, Solver};

/// One configuration under test. The baseline arm is whichever one
/// equals the shipped default; every ratio is taken against it.
#[derive(Clone, Copy)]
enum Arm {
    /// Sweep `nemin` with the cost-model guard off (shipped rule).
    Nemin(usize),
    /// Sweep `merge_flop_budget` with `nemin` at its default.
    Budget(Option<u128>),
}

impl Arm {
    fn params(self) -> SupernodeParams {
        match self {
            Arm::Nemin(nemin) => SupernodeParams {
                nemin,
                ..SupernodeParams::default()
            },
            Arm::Budget(b) => SupernodeParams {
                merge_flop_budget: b,
                ..SupernodeParams::default()
            },
        }
    }

    fn is_baseline(self) -> bool {
        let d = SupernodeParams::default();
        match self {
            Arm::Nemin(nemin) => nemin == d.nemin,
            Arm::Budget(b) => b == d.merge_flop_budget,
        }
    }

    fn label(self) -> String {
        match self {
            Arm::Nemin(nemin) => nemin.to_string(),
            Arm::Budget(None) => "off".into(),
            Arm::Budget(Some(b)) => b.to_string(),
        }
    }
}

struct Shape {
    n_snodes: usize,
    ncol_mean: f64,
    ncol_p50: usize,
    ncol_p90: usize,
    ncol_max: usize,
    /// Fraction of supernodes with <= 8 eliminated columns — the
    /// quantity the PR #150 review cited for clnlbeam (90%).
    frac_le8: f64,
    nrow_mean: f64,
}

fn shape_of(sym: &SymbolicFactorization) -> Shape {
    let mut ncols: Vec<usize> = sym.supernodes.iter().map(|s| s.ncol).collect();
    let nrow_sum: usize = sym.supernodes.iter().map(|s| s.nrow).sum();
    ncols.sort_unstable();
    let n = ncols.len().max(1);
    let pct = |p: f64| ncols[(((ncols.len().max(1) - 1) as f64) * p).round() as usize];
    Shape {
        n_snodes: ncols.len(),
        ncol_mean: ncols.iter().sum::<usize>() as f64 / n as f64,
        ncol_p50: pct(0.50),
        ncol_p90: pct(0.90),
        ncol_max: ncols.last().copied().unwrap_or(0),
        frac_le8: ncols.iter().filter(|&&c| c <= 8).count() as f64 / n as f64,
        nrow_mean: nrow_sum as f64 / n as f64,
    }
}

/// True relative residual ||Ax-b||_inf / ||b||_inf of a single solve at
/// this `nemin`, using the public `Solver` path (so scaling, refinement
/// and the permute cache are exercised exactly as a caller would).
fn residual_at(csc: &CscMatrix, arm: Arm) -> Option<(String, f64)> {
    let mut solver = Solver::with_params(NumericParams::default(), arm.params());
    let _ = solver.factor(csc, None);
    let inertia = solver.inertia()?.to_string();
    let b = vec![1.0f64; csc.n];
    let x = solver.solve(&b).ok()?;
    let mut num: f64 = 0.0;
    let mut ax = vec![0.0f64; csc.n];
    for j in 0..csc.n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = csc.values[k];
            ax[i] += v * x[j];
            if i != j {
                ax[j] += v * x[i];
            }
        }
    }
    for i in 0..csc.n {
        num = num.max((ax[i] - b[i]).abs());
    }
    Some((inertia, num))
}

fn geomean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    (xs.iter().map(|v| v.ln()).sum::<f64>() / xs.len() as f64).exp()
}

fn main() {
    let paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        eprintln!("usage: diag_nemin_post_simd <matrix.mtx> [more.mtx ...]");
        std::process::exit(2);
    }
    // Budget mode takes precedence when both are set: the budget sweep
    // is the newer lever and holding `nemin` fixed is what isolates it.
    let arms: Vec<Arm> = match std::env::var("FERAL_MERGE_BUDGET_LIST") {
        Ok(s) => {
            let mut v: Vec<Arm> = vec![Arm::Budget(None)];
            v.extend(
                s.split(',')
                    .filter_map(|t| t.trim().parse::<u128>().ok())
                    .map(|b| Arm::Budget(Some(b))),
            );
            v
        }
        Err(_) => match std::env::var("FERAL_NEMIN_LIST") {
            Ok(s) => s
                .split(',')
                .filter_map(|t| t.trim().parse().ok())
                .map(Arm::Nemin)
                .collect(),
            Err(_) => [1, 4, 8, 16, 32, 64].into_iter().map(Arm::Nemin).collect(),
        },
    };
    let pairs: usize = std::env::var("FERAL_NEMIN_PAIRS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let base_idx = arms.iter().position(|a| a.is_baseline());

    println!(
        "pairs={pairs} arms=[{}] base={}",
        arms.iter()
            .map(|a| a.label())
            .collect::<Vec<_>>()
            .join(", "),
        base_idx
            .map(|i| arms[i].label())
            .unwrap_or_else(|| "NONE".into())
    );
    // Per-arm geomean accumulators across matrices.
    let mut t_ratio: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut nnz_ratio: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];

    for path in &paths {
        let label = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let Ok(csc) = read_mtx(Path::new(path)).and_then(|m| m.to_csc()) else {
            println!("\n=== {label}: LOAD FAILED ===");
            continue;
        };
        println!("\n=== {label} n={} nnz={} ===", csc.n, csc.row_idx.len());

        // Symbolic once per arm; only the numeric phase is timed, so the
        // measurement isolates the lever from ordering/analysis noise.
        let mut syms: Vec<Option<SymbolicFactorization>> = Vec::new();
        for arm in &arms {
            syms.push(symbolic_factorize(&csc, &arm.params()).ok());
        }

        let np = NumericParams::default();
        // Warm every arm before timing so allocator and cache state are
        // comparable across arms.
        let mut nnz: Vec<Option<usize>> = Vec::new();
        for s in &syms {
            nnz.push(match s {
                Some(sym) => factorize_multifrontal(&csc, sym, &np)
                    .ok()
                    .map(|(f, _)| f.factor_nnz()),
                None => None,
            });
        }

        // Paired alternating timing.
        let mut times: Vec<Vec<u128>> = vec![Vec::new(); arms.len()];
        for _ in 0..pairs {
            for (ai, sym) in syms.iter().enumerate() {
                let Some(sym) = sym else { continue };
                let t = Instant::now();
                let ok = factorize_multifrontal(&csc, sym, &np).is_ok();
                let us = t.elapsed().as_micros();
                if ok {
                    times[ai].push(us);
                }
            }
        }

        println!(
            "{:>6}  {:>7} {:>9} {:>6} {:>6} {:>7} {:>7} {:>9}  {:>11} {:>7}  {:>9} {:>6} {:>5}  {:>10} {:>12}",
            "nemin",
            "snodes",
            "ncol_mean",
            "p50",
            "p90",
            "max",
            "%<=8",
            "nrow_mean",
            "factor_nnz",
            "x_nnz",
            "min_us",
            "x_t",
            "wins",
            "inertia",
            "resid_inf"
        );
        for (ai, &arm) in arms.iter().enumerate() {
            let name = arm.label();
            let Some(sym) = &syms[ai] else {
                println!("{name:>6}  symbolic FAILED");
                continue;
            };
            if times[ai].is_empty() {
                println!("{name:>6}  factor FAILED");
                continue;
            }
            let sh = shape_of(sym);
            let mn = times[ai].iter().copied().min().unwrap_or(0);
            // Sign test vs the baseline arm, pair by pair.
            let (x_t, wins) = match base_idx {
                Some(bi) if bi != ai && !times[bi].is_empty() => {
                    let base_mn = times[bi].iter().copied().min().unwrap_or(1).max(1);
                    let w = times[ai]
                        .iter()
                        .zip(times[bi].iter())
                        .filter(|(a, b)| a < b)
                        .count();
                    (
                        mn as f64 / base_mn as f64,
                        format!("{w}/{}", times[ai].len()),
                    )
                }
                _ => (1.0, "-".into()),
            };
            let x_nnz = match (nnz[ai], base_idx.and_then(|bi| nnz[bi])) {
                (Some(a), Some(b)) if b > 0 => a as f64 / b as f64,
                _ => f64::NAN,
            };
            let (inertia, resid) = residual_at(&csc, arm)
                .map(|(i, r)| (i, format!("{r:.3e}")))
                .unwrap_or_else(|| ("FAIL".into(), "-".into()));
            println!(
                "{:>6}  {:>7} {:>9.2} {:>6} {:>6} {:>7} {:>6.1}% {:>9.1}  {:>11} {:>7.3}  {:>9} {:>6.3} {:>5}  {:>10} {:>12}",
                name,
                sh.n_snodes,
                sh.ncol_mean,
                sh.ncol_p50,
                sh.ncol_p90,
                sh.ncol_max,
                100.0 * sh.frac_le8,
                sh.nrow_mean,
                nnz[ai].map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
                x_nnz,
                mn,
                x_t,
                wins,
                inertia,
                resid
            );
            if base_idx.is_some() {
                t_ratio[ai].push(x_t);
                if x_nnz.is_finite() {
                    nnz_ratio[ai].push(x_nnz);
                }
            }
        }
    }

    if paths.len() > 1 {
        println!(
            "\n=== geomean across {} matrices (vs base) ===",
            paths.len()
        );
        println!("{:>6}  {:>10}  {:>10}", "arm", "x_time", "x_nnz");
        for (ai, &arm) in arms.iter().enumerate() {
            println!(
                "{:>6}  {:>10.3}  {:>10.3}",
                arm.label(),
                geomean(&t_ratio[ai]),
                geomean(&nnz_ratio[ai])
            );
        }
    }
}
