//! Which of the three differences between the bench's solve and the
//! shipped solve actually moves the number?
//!
//!   cargo run -p feral-diagnostics --bin probe_solve_reconcile --release [-- <name>...]
//!
//! Two earlier probes disagreed about whether feral#131's parallel solve
//! core is worth anything, and the disagreement turned out to be a
//! measurement artifact: they were not comparing the same two things.
//!
//!   * `probe_largen_solve_levers` compared `Solver::with_parallel(true)`
//!     against `Solver::with_parallel(false)` at `max_steps = 1`. Both
//!     sides run `SolveCore::Auto`, so that is a pure *schedule*
//!     comparison. It reported geomean 0.92 (parallel slower).
//!
//!   * `probe_bench_solve_noise` compared `Solver::with_parallel(true)`
//!     against the free `solve_sparse_refined` at `max_steps = 10`. The
//!     free function hardcodes `SolveCore::SharedVector`
//!     (`src/numeric/solve.rs:2077`) while `Solver` runs `Auto` — which
//!     picks `ContribBlock` on six of these seven matrices. So that ratio
//!     folds the *core* change in with the schedule change, at a
//!     different refinement depth. It reported geomean 1.21 (parallel
//!     faster).
//!
//! Neither number was wrong; they answer different questions. This probe
//! measures all three configurations interleaved in one process, at both
//! refinement depths, so core / schedule / depth can be attributed
//! separately:
//!
//!   sv  `solve_sparse_refined_opts`      SharedVector, serial  <- the bench
//!   au  `Solver` (serial)                Auto,         serial
//!   ap  `Solver::with_parallel(true)`    Auto,         parallel <- hosts
//!
//! and reports three ratios (>1 means the second config is faster):
//!
//!   core     sv/au   what `Auto` picking ContribBlock buys, serially
//!   sched    au/ap   what feral#131's parallel schedule buys
//!   shipped  sv/ap   what a host gets over what the bench measures
//!
//! Configurations are interleaved within each repetition rather than run
//! in blocks, so machine drift and thermal state hit all three equally.

use feral::numeric::solve::{
    solve_sparse_refined_opts, solve_sparse_refined_with_diagnostics_opts, RefineOptions,
};
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::Path;
use std::time::Instant;

/// Odd, so the median is a measured sample rather than an average.
const REPS: usize = 11;

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[s.len() / 2]
}

fn geomean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp()
}

struct Row {
    name: String,
    steps: usize,
    /// `[depth1, depth10]` medians in microseconds for sv / au / ap.
    sv: [f64; 2],
    au: [f64; 2],
    ap: [f64; 2],
}

fn run(path: &Path) -> Option<Row> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let mtx = match read_mtx(path) {
        Ok(m) => m,
        Err(e) => {
            println!("SKIP {name}: read_mtx: {e}");
            return None;
        }
    };
    let csc: CscMatrix = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            println!("SKIP {name}: to_csc: {e}");
            return None;
        }
    };
    let n = csc.n;

    let mut ser = Solver::new();
    match ser.factor(&csc, None) {
        FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
        other => {
            println!("SKIP {name}: factor {other:?}");
            return None;
        }
    }
    let mut par = Solver::new().with_parallel(true);
    match par.factor(&csc, None) {
        FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
        other => {
            println!("SKIP {name}: parallel factor {other:?}");
            return None;
        }
    }
    // `tests/parallel_parity.rs` contracts the two factors to be bit-equal,
    // so `ser` and `par` differ only in whether a pool is installed.
    let factors = ser.factors()?;

    // Well-scaled deterministic RHS with a known O(1) solution, so the
    // refinement loop does a representative amount of work rather than
    // converging on the first iterate.
    let mut rhs = vec![0.0f64; n];
    let mut v = vec![0.0f64; n];
    for (i, s) in v.iter_mut().enumerate() {
        *s = 1.0 + (i % 7) as f64 / 8.0;
    }
    csc.symv(&v, &mut rhs);

    // How many corrections the default depth actually runs. If this is 1
    // then "depth 1" and "depth 10" are the same call and the two probes
    // could not have differed on depth alone.
    let deep = RefineOptions::default();
    let steps = match solve_sparse_refined_with_diagnostics_opts(&csc, factors, &rhs, deep) {
        Ok((_, d)) => d.steps.len().saturating_sub(1),
        Err(e) => {
            println!("SKIP {name}: diagnostics: {e}");
            return None;
        }
    };

    let depths = [RefineOptions::with_max_steps(1), deep];
    let mut sv = [0.0f64; 2];
    let mut au = [0.0f64; 2];
    let mut ap = [0.0f64; 2];

    for (d, opts) in depths.iter().enumerate() {
        let (mut t_sv, mut t_au, mut t_ap) = (Vec::new(), Vec::new(), Vec::new());
        let mut x = vec![0.0f64; n];
        for _ in 0..REPS {
            let t = Instant::now();
            match solve_sparse_refined_opts(&csc, factors, &rhs, *opts) {
                Ok(y) => {
                    std::hint::black_box(&y);
                }
                Err(e) => {
                    println!("SKIP {name}: sv solve: {e}");
                    return None;
                }
            }
            t_sv.push(t.elapsed().as_secs_f64() * 1e6);

            let t = Instant::now();
            if let Err(e) = ser.solve_refined_into(&csc, &rhs, &mut x, *opts) {
                println!("SKIP {name}: au solve: {e}");
                return None;
            }
            std::hint::black_box(&x);
            t_au.push(t.elapsed().as_secs_f64() * 1e6);

            let t = Instant::now();
            if let Err(e) = par.solve_refined_into(&csc, &rhs, &mut x, *opts) {
                println!("SKIP {name}: ap solve: {e}");
                return None;
            }
            std::hint::black_box(&x);
            t_ap.push(t.elapsed().as_secs_f64() * 1e6);
        }
        sv[d] = median(&t_sv);
        au[d] = median(&t_au);
        ap[d] = median(&t_ap);
    }

    for (d, tag) in ["steps=1", "default"].iter().enumerate() {
        println!(
            "{name:<20} n={n:<8} {tag:<8} sv {:9.0}  au {:9.0}  ap {:9.0} us   \
             core {:.2}x  sched {:.2}x  shipped {:.2}x",
            sv[d],
            au[d],
            ap[d],
            sv[d] / au[d],
            au[d] / ap[d],
            sv[d] / ap[d],
        );
    }
    println!(
        "{:<20} corrections actually run at default depth: {steps}",
        ""
    );

    Some(Row {
        name,
        steps,
        sv,
        au,
        ap,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names: Vec<String> = if args.is_empty() {
        vec![
            "bcsstk38".into(),
            "r05_kkt".into(),
            "bratu3d".into(),
            "qap15_kkt".into(),
            "dirichlet120_kkt".into(),
            "cont-201".into(),
            "cont5_late_kkt".into(),
        ]
    } else {
        args
    };
    println!(
        "core / schedule / depth attribution ({REPS} reps, interleaved, 1 RHS)\n\
         sv = SharedVector serial (the bench)   au = Auto serial   ap = Auto parallel (hosts)\n\
         ratios > 1 mean the second config is faster\n"
    );
    let rows: Vec<Row> = names
        .iter()
        .filter_map(|nm| run(&Path::new("tests/data/large").join(format!("{nm}.mtx"))))
        .collect();

    println!(
        "\n{:<20} {:>8}  {:>6}  {:>6}  {:>7}",
        "geomean", "depth", "core", "sched", "shipped"
    );
    for (d, tag) in ["steps=1", "default"].iter().enumerate() {
        let core: Vec<f64> = rows.iter().map(|r| r.sv[d] / r.au[d]).collect();
        let sched: Vec<f64> = rows.iter().map(|r| r.au[d] / r.ap[d]).collect();
        let ship: Vec<f64> = rows.iter().map(|r| r.sv[d] / r.ap[d]).collect();
        println!(
            "{:<20} {tag:>8}  {:>5.2}x  {:>5.2}x  {:>6.2}x",
            "",
            geomean(&core),
            geomean(&sched),
            geomean(&ship)
        );
    }
    let deep: Vec<String> = rows
        .iter()
        .filter(|r| r.steps > 1)
        .map(|r| format!("{}({})", r.name, r.steps))
        .collect();
    println!(
        "\nmatrices where default depth runs more than one correction: {}",
        if deep.is_empty() {
            "none".to_string()
        } else {
            deep.join(" ")
        }
    );
}
