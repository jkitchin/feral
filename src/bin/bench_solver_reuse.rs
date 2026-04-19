//! Solver cache-reuse benchmark.
//!
//! Quantifies the β + `Solver` win that the per-matrix `bench` cannot
//! show: across an IPM-style sequence of structurally identical KKTs,
//! one persistent `Solver` runs `symbolic_factorize` once and amortizes
//! that cost over every refactor, whereas the free-function loop re-
//! runs symbolic on every iteration.
//!
//! For each KKT family with N iterates `<FAM>_NNNN.mtx`:
//!   - Scenario A (Solver): one `Solver`, call `factor()` N times.
//!     Reports `symbolic_call_count()` (target = 1).
//!   - Scenario B (free-fn): `symbolic_factorize` + `factorize_multifrontal`
//!     each iteration.
//!   - Reports total wall time, mean per-iteration time, speedup ratio.
//!
//! Usage: `cargo run --release --bin bench_solver_reuse`

use feral::numeric::factorize::factorize_multifrontal;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, NumericParams, Solver, ZeroPivotAction};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn numeric_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    })
}

fn load_family(family: &str, max_iterates: usize) -> Vec<feral::CscMatrix> {
    let mut iterates: Vec<feral::CscMatrix> = Vec::new();
    for i in 0..max_iterates {
        let path = PathBuf::from(format!(
            "data/matrices/kkt/{}/{}_{:04}.mtx",
            family, family, i
        ));
        if !path.exists() {
            break;
        }
        let mtx = match read_mtx(&path) {
            Ok(m) => m,
            Err(_) => break,
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(_) => break,
        };
        iterates.push(csc);
    }
    iterates
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

struct FamilyResult {
    family: &'static str,
    n: usize,
    nnz: usize,
    iters: usize,
    solver_total_us: u128,
    solver_sym_calls: usize,
    freefn_total_us: u128,
    freefn_sym_calls: usize,
}

fn run_family(family: &'static str, max_iterates: usize) -> Option<FamilyResult> {
    let iterates = load_family(family, max_iterates);
    if iterates.is_empty() {
        eprintln!("SKIP {}: no iterates loaded", family);
        return None;
    }
    let n = iterates[0].n;
    let nnz = iterates[0].row_idx.len();

    // Warmup pass on each scenario to neutralize cold-cache effects
    // (allocator, page faults, branch predictor).
    let _ = run_scenario_solver(&iterates);
    let _ = run_scenario_freefn(&iterates);

    let (solver_dur, solver_sym_calls) = run_scenario_solver(&iterates);
    let (freefn_dur, freefn_sym_calls) = run_scenario_freefn(&iterates);

    Some(FamilyResult {
        family,
        n,
        nnz,
        iters: iterates.len(),
        solver_total_us: solver_dur.as_micros(),
        solver_sym_calls,
        freefn_total_us: freefn_dur.as_micros(),
        freefn_sym_calls,
    })
}

fn print_header() {
    println!(
        "{:<10} {:>5} {:>7} {:>5}   {:>11} {:>5}   {:>11} {:>5}   {:>10} {:>10}   {:>7}",
        "family",
        "n",
        "nnz",
        "iter",
        "solver(us)",
        "syms",
        "freefn(us)",
        "syms",
        "solver/it",
        "freefn/it",
        "speedup",
    );
    println!("{}", "-".repeat(110));
}

fn print_row(r: &FamilyResult) {
    let solver_per = r.solver_total_us as f64 / r.iters as f64;
    let freefn_per = r.freefn_total_us as f64 / r.iters as f64;
    let speedup = if r.solver_total_us > 0 {
        r.freefn_total_us as f64 / r.solver_total_us as f64
    } else {
        0.0
    };
    println!(
        "{:<10} {:>5} {:>7} {:>5}   {:>11} {:>5}   {:>11} {:>5}   {:>10.1} {:>10.1}   {:>6.2}x",
        r.family,
        r.n,
        r.nnz,
        r.iters,
        r.solver_total_us,
        r.solver_sym_calls,
        r.freefn_total_us,
        r.freefn_sym_calls,
        solver_per,
        freefn_per,
        speedup,
    );
}

fn main() {
    println!("Solver cache-reuse benchmark: persistent Solver vs per-iter symbolic_factorize.");
    println!("Lower solver/it and higher speedup = better cache reuse.");
    println!();
    print_header();

    // Cap iterates per family so the run finishes in reasonable time
    // on AIRPORT (428 iterates). 64 is enough to amortize symbolic.
    let cap = 64;
    let families: &[&'static str] = &["ACOPP14", "ACOPP30", "ACOPR30", "AIRPORT"];

    let mut total_solver_us: u128 = 0;
    let mut total_freefn_us: u128 = 0;
    let mut count = 0usize;
    for fam in families {
        if let Some(r) = run_family(fam, cap) {
            print_row(&r);
            total_solver_us += r.solver_total_us;
            total_freefn_us += r.freefn_total_us;
            count += 1;
        }
    }

    if count > 0 {
        println!("{}", "-".repeat(110));
        let speedup = if total_solver_us > 0 {
            total_freefn_us as f64 / total_solver_us as f64
        } else {
            0.0
        };
        println!(
            "TOTAL{:>5} families   solver: {:>11} us   freefn: {:>11} us   speedup: {:>6.2}x",
            count, total_solver_us, total_freefn_us, speedup
        );
    }
}
