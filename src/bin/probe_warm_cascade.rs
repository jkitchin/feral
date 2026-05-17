//! Warm-state cascade amplification probe — step 1 of the 4-step
//! investigation in `dev/research/warm-state-cascade-amplification-2026-05-17.md`.
//!
//! Demonstrates and bisects the FRESH-vs-warm disparity on the
//! Mittelmann KKT corpus. The thesis under test: a `Solver` that has
//! factored matrix N-1 cascades on matrix N, while a fresh `Solver`
//! handed matrix N alone factors cleanly. We isolate which piece of
//! `Solver` state carries the amplification.
//!
//! The probe runs four modes against a pair of consecutive KKT dumps
//! `<problem>_<prev>.mtx` and `<problem>_<curr>.mtx`:
//!
//!   COLD    fresh `Solver`; factor only the curr matrix.
//!   WARM    one `Solver`; factor prev then curr (reproduces cascade).
//!   SYMOFF  one `Solver`; factor prev, call `invalidate_symbolic_cache`,
//!           factor curr. Symbolic is forced fresh but the pooled
//!           numeric workspace, last_factors, and parallel_pool stay
//!           warm. If COLD ≈ SYMOFF ≪ WARM, the amplification lives
//!           in the cached symbolic factorisation.
//!   ALLNEW  rebuild the entire `Solver` between prev and curr (the
//!           probe_kkt_replay `FRESH=1` baseline; identical to COLD on
//!           the curr matrix but reads prev first to keep wall-clock
//!           apples-to-apples).
//!
//! Default pair is `pinene_3200` (prev=4, curr=5) — the agent
//! a84f721906859018f result showed warm DNF (>451 s) vs FRESH 1.56 s
//! on those exact indices. Override with positional args.
//!
//! Usage:
//!     cargo run --release --bin probe_warm_cascade
//!     cargo run --release --bin probe_warm_cascade -- marine_1600 8 9

use std::env;
use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus, Solver};
use feral::symbolic::supernode::SupernodeParams;
use feral::{read_mtx, read_sidecar, CscMatrix, Inertia};

fn build_solver() -> Solver {
    Solver::with_params(NumericParams::default(), SupernodeParams::default())
}

fn load_pair(
    problem: &str,
    prev: usize,
    curr: usize,
) -> (CscMatrix, Option<Inertia>, CscMatrix, Option<Inertia>) {
    let dir = format!("data/matrices/kkt-mittelmann/{problem}");
    let load = |i: usize| {
        let mtx = format!("{dir}/{problem}_{i:04}.mtx");
        let json = format!("{dir}/{problem}_{i:04}.json");
        let m = read_mtx(Path::new(&mtx)).expect("read_mtx");
        let c = m.to_csc().expect("to_csc");
        let exp = read_sidecar(Path::new(&json)).ok().map(|s| Inertia {
            positive: s.inertia.positive,
            negative: s.inertia.negative,
            zero: s.inertia.zero,
        });
        (c, exp)
    };
    let (cp, ep) = load(prev);
    let (cc, ec) = load(curr);
    (cp, ep, cc, ec)
}

fn factor_one(
    solver: &mut Solver,
    matrix: &CscMatrix,
    expected: Option<Inertia>,
    label: &str,
) -> f64 {
    let t0 = Instant::now();
    let status = solver.factor(matrix, expected);
    let secs = t0.elapsed().as_secs_f64();
    let tag = match status {
        FactorStatus::Success => "OK",
        FactorStatus::WrongInertia { .. } => "WrongInertia",
        FactorStatus::Singular => "Singular",
        FactorStatus::FatalError(_) => "Fatal",
    };
    println!("  {label:<8} {secs:>10.3} s  {tag}");
    secs
}

fn main() {
    let mut args = env::args().skip(1);
    let problem = args.next().unwrap_or_else(|| "pinene_3200".to_string());
    let prev: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let curr: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    println!("# probe_warm_cascade  problem={problem}  prev={prev}  curr={curr}");
    println!("# pair: data/matrices/kkt-mittelmann/{problem}/{problem}_{prev:04}.mtx -> {problem}_{curr:04}.mtx");
    let (cp, ep, cc, ec) = load_pair(&problem, prev, curr);
    println!(
        "# prev dim={}, nnz={}; curr dim={}, nnz={}",
        cp.n,
        cp.row_idx.len(),
        cc.n,
        cc.row_idx.len()
    );
    println!();

    println!("[COLD] fresh solver, curr only");
    let mut s = build_solver();
    let cold = factor_one(&mut s, &cc, ec.clone(), "curr");
    let sym_calls_cold = s.symbolic_call_count();
    drop(s);

    println!("\n[WARM] one solver, prev then curr (reproduce)");
    let mut s = build_solver();
    factor_one(&mut s, &cp, ep.clone(), "prev");
    let warm = factor_one(&mut s, &cc, ec.clone(), "curr");
    let sym_calls_warm = s.symbolic_call_count();
    drop(s);

    println!("\n[SYMOFF] one solver; factor prev, invalidate symbolic, factor curr");
    let mut s = build_solver();
    factor_one(&mut s, &cp, ep.clone(), "prev");
    s.invalidate_symbolic_cache();
    let symoff = factor_one(&mut s, &cc, ec.clone(), "curr");
    let sym_calls_symoff = s.symbolic_call_count();
    drop(s);

    println!("\n[ALLNEW] rebuild solver between prev and curr");
    let mut s = build_solver();
    factor_one(&mut s, &cp, ep.clone(), "prev");
    drop(s);
    let mut s = build_solver();
    let allnew = factor_one(&mut s, &cc, ec.clone(), "curr");
    let sym_calls_allnew = s.symbolic_call_count();
    drop(s);

    println!("\n# Summary (curr factor time only)");
    println!("# mode       curr_factor (s)   symbolic_calls");
    println!("# COLD      {cold:>15.3}   {sym_calls_cold}");
    println!("# WARM      {warm:>15.3}   {sym_calls_warm}");
    println!("# SYMOFF    {symoff:>15.3}   {sym_calls_symoff}");
    println!("# ALLNEW    {allnew:>15.3}   {sym_calls_allnew}");
    println!();
    if warm > 5.0 * cold && symoff < 2.0 * cold {
        println!("# VERDICT: amplification lives in cached symbolic state.");
        println!("#          invalidate_symbolic_cache() rescues curr to ~COLD baseline.");
    } else if warm > 5.0 * cold && symoff >= 2.0 * warm.min(symoff) {
        println!("# VERDICT: amplification persists past symbolic invalidation.");
        println!("#          bug is in workspace / last_factors / pool, not symbolic.");
    } else if warm <= 2.0 * cold {
        println!("# VERDICT: did NOT reproduce the warm-vs-cold disparity on this pair.");
        println!("#          try a different (prev,curr) — the cascade may only trip later.");
    } else {
        println!("# VERDICT: intermediate. Re-examine ratios.");
    }
}
