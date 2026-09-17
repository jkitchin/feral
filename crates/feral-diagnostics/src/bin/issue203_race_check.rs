//! Issue #203 — acceptance check for the measured ordering race.
//!
//! Three questions the plan (`dev/plans/ordering-race.md`) asks:
//!
//! 1. does the race pick the arm the standalone A/B says is fastest?
//! 2. what does the first `factor()` cost, against one un-raced
//!    `factor()`?
//! 3. does the saving actually arrive on the second and later
//!    factorizations of the same pattern, which is the IPM case?
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin issue203_race_check \
//!       -- MATRIX.mtx N_VARS [REPEATS]

use feral::symbolic::OrderingMethod;
use feral::{read_mtx, FactorStatus, Solver};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: issue203_race_check MATRIX.mtx N_VARS [REPEATS]");
        std::process::exit(2);
    }
    let n_vars: usize = args[1].parse().expect("N_VARS");
    let repeats: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let mtx = read_mtx(std::path::Path::new(&args[0])).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let expect = (n_vars, csc.n - n_vars, 0usize);
    println!(
        "matrix {} n={} inertia oracle ({}, {}, {})",
        args[0], csc.n, expect.0, expect.1, expect.2
    );

    let arms = vec![OrderingMethod::Amf, OrderingMethod::MetisND];

    // Baseline: the default path, no race.
    let mut base = Solver::new();
    let t = Instant::now();
    let st = base.factor(&csc, None);
    let base_first = t.elapsed().as_micros();
    assert!(matches!(st, FactorStatus::Success), "baseline: {st:?}");
    let mut base_rest = 0u128;
    let mut base_each: Vec<u128> = Vec::new();
    for _ in 0..repeats {
        let t = Instant::now();
        let st = base.factor(&csc, None);
        let us = t.elapsed().as_micros();
        base_each.push(us);
        base_rest += us;
        assert!(matches!(st, FactorStatus::Success));
    }

    // Raced.
    let mut raced = Solver::new().with_ordering_race(arms.clone());
    let t = Instant::now();
    let st = raced.factor(&csc, None);
    let race_first = t.elapsed().as_micros();
    assert!(matches!(st, FactorStatus::Success), "raced: {st:?}");
    let mut race_rest = 0u128;
    let mut race_each: Vec<u128> = Vec::new();
    for _ in 0..repeats {
        let t = Instant::now();
        let st = raced.factor(&csc, None);
        let us = t.elapsed().as_micros();
        race_each.push(us);
        race_rest += us;
        assert!(matches!(st, FactorStatus::Success));
    }
    println!("per-call baseline us: {base_each:?}");
    println!("per-call raced    us: {race_each:?}");

    for (label, s) in [("baseline", &base), ("raced", &raced)] {
        let i = s.inertia().expect("inertia");
        let ok = (i.positive, i.negative, i.zero) == expect;
        println!(
            "{label:<9} inertia ({}, {}, {}) {}",
            i.positive,
            i.negative,
            i.zero,
            if ok { "OK" } else { "*** WRONG ***" }
        );
    }

    if let Some(r) = raced.last_race() {
        println!("\nrace arms:");
        for a in &r.arms {
            println!(
                "  {:<10} first {:>10} us   steady {:>10} us{}",
                format!("{:?}", a.method),
                a.first_factor_us
                    .map(|v| v.to_string())
                    .unwrap_or("-".into()),
                a.factor_us.map(|v| v.to_string()).unwrap_or("-".into()),
                if a.winner { "   <-- adopted" } else { "" }
            );
        }
        if r.inertia_disagreement {
            println!("  *** arms disagreed on inertia; race declined ***");
        }
    }

    println!(
        "\nfirst factor():  baseline {base_first} us, raced {race_first} us  ({:.2}x)",
        race_first as f64 / base_first as f64
    );
    println!(
        "next {repeats}:       baseline {} us, raced {} us  ({:.2}x faster)",
        base_rest,
        race_rest,
        base_rest as f64 / race_rest.max(1) as f64
    );
    let base_total = base_first + base_rest;
    let race_total = race_first + race_rest;
    println!(
        "total {}:      baseline {} us, raced {} us  ({:.2}x faster)",
        repeats + 1,
        base_total,
        race_total,
        base_total as f64 / race_total.max(1) as f64
    );
    // Where the race breaks even, assuming the per-factor rates hold.
    let base_rate = base_rest as f64 / repeats as f64;
    let race_rate = race_rest as f64 / repeats as f64;
    if base_rate > race_rate {
        let breakeven = (race_first as f64 - base_first as f64) / (base_rate - race_rate);
        println!(
            "break-even after ~{:.1} factorizations of this pattern",
            breakeven.max(0.0)
        );
    } else {
        println!("no break-even: the raced arm is not faster per factorization here");
    }
}
