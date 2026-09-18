//! Issue #203 — which ordering *policy* should a host adopt?
//!
//! Three candidate policies, and the choice between them is an
//! economics question, not a fill question:
//!
//!   * `auto`  — today's default. Free, and `choose_adaptive` can never
//!     reach `MetisND`, so it leaves the collocation win on the table.
//!   * `metis` — ask for `MetisND` by name. Free, wins big on
//!     collocation, loses on some other shapes, and requires the caller
//!     to know which shape they have.
//!   * `race`  — measure both on the first factorization of each
//!     pattern and adopt the winner. Needs no expertise; costs `k`
//!     analyses and `3k` factorizations once per pattern.
//!
//! The race pays for itself only if the host factors one pattern often
//! enough. This probe measures, per matrix, the first-`factor()` cost
//! and the steady-state cost of each policy, then reports the total at
//! several iteration counts and the break-even point against `auto`.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin issue203_policy \
//!       -- MATRIX.mtx [STEADY_REPS]

use feral::symbolic::OrderingMethod;
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::time::Instant;

struct Policy {
    label: &'static str,
    first_us: u128,
    steady_us: u128,
    ok: bool,
    note: String,
}

fn build(label: &'static str) -> Solver {
    match label {
        "auto" => Solver::new(),
        "amf" => Solver::new().with_ordering(OrderingMethod::Amf),
        "amd" => Solver::new().with_ordering(OrderingMethod::Amd),
        "metis" => Solver::new().with_ordering(OrderingMethod::MetisND),
        // The SHIPPED race: 4 symbolic passes, keeps min factor_nnz_estimate.
        "autorace" => Solver::new().with_ordering(OrderingMethod::AutoRace),
        "race2" => {
            Solver::new().with_ordering_race(vec![OrderingMethod::Amf, OrderingMethod::MetisND])
        }
        "raceDM" => {
            Solver::new().with_ordering_race(vec![OrderingMethod::Amd, OrderingMethod::MetisND])
        }
        "race3" => Solver::new().with_ordering_race(vec![
            OrderingMethod::Amf,
            OrderingMethod::Amd,
            OrderingMethod::MetisND,
        ]),
        _ => Solver::new(),
    }
}

fn measure(label: &'static str, m: &CscMatrix, reps: usize) -> Policy {
    let mut s = build(label);
    let t = Instant::now();
    let st = s.factor(m, None);
    let first_us = t.elapsed().as_micros();
    if !matches!(st, FactorStatus::Success) {
        return Policy {
            label,
            first_us,
            steady_us: 0,
            ok: false,
            note: format!("{st:?}"),
        };
    }
    let mut steady = u128::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        let st = s.factor(m, None);
        if !matches!(st, FactorStatus::Success) {
            return Policy {
                label,
                first_us,
                steady_us: 0,
                ok: false,
                note: format!("{st:?}"),
            };
        }
        steady = steady.min(t.elapsed().as_micros());
    }
    let note = match s.last_race() {
        Some(r) => match r.winner {
            Some(w) => format!("adopted {:?}", r.arms[w].method),
            None => "race declined".to_string(),
        },
        None => String::new(),
    };
    Policy {
        label,
        first_us,
        steady_us: steady,
        ok: true,
        note,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: issue203_policy MATRIX.mtx [STEADY_REPS]");
        std::process::exit(2);
    };
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);

    let mtx = read_mtx(std::path::Path::new(path)).expect("read_mtx");
    let m = mtx.to_csc().expect("to_csc");
    println!(
        "matrix {} n={} nnz={}",
        std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path),
        m.n,
        m.row_idx.len()
    );

    // `autorace` (symbolic, min fill) against `raceDM`/`race3` (numeric,
    // min measured steady-state factor time) is the head-to-head this
    // probe exists for.
    let labels = ["auto", "amf", "amd", "metis", "autorace", "raceDM", "race3"];
    let pols: Vec<Policy> = labels.iter().map(|l| measure(l, &m, reps)).collect();

    println!(
        "\n{:<7} {:>12} {:>12}  note",
        "policy", "first_us", "steady_us"
    );
    for p in &pols {
        if p.ok {
            println!(
                "{:<7} {:>12} {:>12}  {}",
                p.label, p.first_us, p.steady_us, p.note
            );
        } else {
            println!("{:<7} FAILED: {}", p.label, p.note);
        }
    }

    // Total cost of N factorizations of this one pattern.
    let horizons = [5usize, 15, 30, 100, 180];
    print!("\n{:<7}", "policy");
    for h in horizons {
        print!(" {:>12}", format!("N={h}"));
    }
    println!("      break-even vs auto");
    let base = pols.iter().find(|p| p.label == "auto" && p.ok);
    for p in &pols {
        if !p.ok {
            continue;
        }
        print!("{:<7}", p.label);
        for h in horizons {
            let total = p.first_us + p.steady_us * (h.saturating_sub(1)) as u128;
            print!(" {total:>12}");
        }
        match base {
            Some(b) if p.label != "auto" => {
                // first + steady*(N-1) beats auto's when
                // N-1 > (first - base_first) / (base_steady - steady).
                let dfirst = p.first_us as f64 - b.first_us as f64;
                let dsteady = b.steady_us as f64 - p.steady_us as f64;
                if dsteady <= 0.0 {
                    print!("      never (not faster per factorization)");
                } else if dfirst <= 0.0 {
                    print!("      immediately");
                } else {
                    print!("      N > {:.1}", dfirst / dsteady + 1.0);
                }
            }
            _ => print!("      —"),
        }
        println!();
    }
}
