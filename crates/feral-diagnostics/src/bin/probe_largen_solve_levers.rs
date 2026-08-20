//! Quantify the two solve-phase levers claimed in feral#131 and #189 on
//! IPM-scale KKT matrices.
//!
//!   cargo run -p feral-diagnostics --bin probe_largen_solve_levers --release [-- <name>...]
//!
//! For each matrix it factors once, then times four ways of producing the
//! same `NRHS` solutions:
//!
//!   A  serial   `solve_sparse`     x NRHS   (what a host at feral_refine=no gets today)
//!   B  cb-ser   `solve_sparse_cb(false)` x NRHS
//!   C  cb-par   `solve_sparse_cb(true)`  x NRHS   (#131: the unreachable core)
//!   D  many     `solve_sparse_many(NRHS)` x 1     (batching: shares traversal)
//!
//! then sweeps `nrhs` through the BLAS-3 crossover (`BLAS3_NRHS_THRESHOLD
//! = 32`) reporting per-RHS cost of `many` against looped single, which is
//! the measurement #189 says was never taken above n = 1e4.
//!
//! Every path is reported with its own relative residual: the CB core and
//! the shared-vector core are different reassociations (#177), so parity
//! is a residual question, not a bit question.

use feral::numeric::solve::{
    solve_sparse, solve_sparse_cb, solve_sparse_many, solve_sparse_refined_auto_into, RefineOptions,
};
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::Path;
use std::time::Instant;

/// L-BFGS with `limited_memory_max_history = 6` presents this many
/// columns per low-rank block, which is the case pounce#698 hits.
const NRHS: usize = 13;
/// Best-of-`REPS`: these are wall-clock timings on a shared machine, and
/// the minimum is the least noisy estimator of the achievable cost.
const REPS: usize = 5;
const SWEEP: [usize; 10] = [1, 2, 4, 8, 13, 16, 24, 32, 48, 64];

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0f64; n];
    a.symv(x, &mut ax);
    let num = ax
        .iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f64, f64::max);
    let den = b.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1.0);
    num / den
}

/// Deterministic, well-scaled RHS block: column `c` is `A * v_c` with
/// `v_c[i] = 1 + ((i + c) % 7) as f64 / 8.0`, so each column has a known
/// O(1) solution and no two columns are collinear.
fn build_rhs_block(a: &CscMatrix, nrhs: usize) -> Vec<f64> {
    let n = a.n;
    let mut rhs = vec![0.0f64; n * nrhs];
    let mut v = vec![0.0f64; n];
    for c in 0..nrhs {
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = 1.0 + ((i + c) % 7) as f64 / 8.0;
        }
        a.symv(&v, &mut rhs[c * n..(c + 1) * n]);
    }
    rhs
}

/// Best-of-REPS wall time, in seconds, for one closure.
fn best_of(mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        f();
        let s = t.elapsed().as_secs_f64();
        if s < best {
            best = s;
        }
    }
    best
}

fn run(path: &Path) {
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    let mtx = match read_mtx(path) {
        Ok(m) => m,
        Err(e) => {
            println!("SKIP {name}: {e}");
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            println!("SKIP {name}: to_csc: {e}");
            return;
        }
    };
    let n = csc.n;
    let nnz = csc.row_idx.len();

    let mut solver = Solver::new();
    let t = Instant::now();
    let status = solver.factor(&csc, None);
    let factor_s = t.elapsed().as_secs_f64();
    match &status {
        FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
        other => {
            println!("SKIP {name}: factor {other:?}");
            return;
        }
    }
    let factors = match solver.factors() {
        Some(f) => f,
        None => {
            println!("SKIP {name}: no factors");
            return;
        }
    };

    println!();
    println!("=== {name}  n={n}  nnz={nnz}  factor={factor_s:.3}s ===");

    // Which core does `SolveCore::Auto` — the mode `Solver` uses — pick for
    // this factor? `cb_core_profitable` is private, but the two cores are
    // different reassociations, so the *bits* identify the choice: run the
    // refined entry point with `max_steps = 0` (one bare core apply, no
    // correction) and bit-compare against each explicit core.
    {
        let probe_rhs = {
            let mut v = vec![0.0f64; n];
            let mut u = vec![0.0f64; n];
            for (i, s) in u.iter_mut().enumerate() {
                *s = 1.0 + (i % 7) as f64 / 8.0;
            }
            csc.symv(&u, &mut v);
            v
        };
        let mut x_auto = vec![0.0f64; n];
        let verdict = match solve_sparse_refined_auto_into(
            &csc,
            factors,
            &probe_rhs,
            &mut x_auto,
            false,
            RefineOptions::with_max_steps(0),
        ) {
            Err(e) => format!("error: {e:?}"),
            Ok(()) => {
                let sv = solve_sparse(factors, &probe_rhs).expect("sv probe");
                let cb = solve_sparse_cb(factors, &probe_rhs, false).expect("cb probe");
                let m_sv = sv
                    .iter()
                    .zip(&x_auto)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                let m_cb = cb
                    .iter()
                    .zip(&x_auto)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                match (m_sv, m_cb) {
                    (true, false) => "SharedVector".to_string(),
                    (false, true) => "ContribBlock".to_string(),
                    (true, true) => "ambiguous (cores agree bitwise)".to_string(),
                    (false, false) => "neither (unexpected)".to_string(),
                }
            }
        };
        println!("SolveCore::Auto picks: {verdict}");
    }

    // End-to-end, through the API a host actually calls. `solve_refined` is
    // the ONLY `Solver` entry that installs the pool (`solver.rs:1719`), so
    // `with_parallel(true/false)` isolates the *schedule* — both sides run
    // whichever core `Auto` picked above, so the arithmetic is unchanged and
    // this is a pure parallel-vs-serial comparison of the shipped path.
    {
        let probe_rhs = {
            let mut v = vec![0.0f64; n];
            let mut u = vec![0.0f64; n];
            for (i, s) in u.iter_mut().enumerate() {
                *s = 1.0 + (i % 7) as f64 / 8.0;
            }
            csc.symv(&u, &mut v);
            v
        };
        let opts = RefineOptions::with_max_steps(1);
        let timed = |par: bool| -> Option<f64> {
            let mut sv = Solver::new().with_parallel(par);
            match sv.factor(&csc, None) {
                FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
                _ => return None,
            }
            let mut x = vec![0.0f64; n];
            Some(best_of(|| {
                let _ = sv.solve_refined_into(&csc, &probe_rhs, &mut x, opts);
                std::hint::black_box(&x);
            }))
        };
        match (timed(false), timed(true)) {
            (Some(ser), Some(par)) => println!(
                "E solve_refined(1 step) x1   serial {:8.2} ms   parallel {:8.2} ms   {:.2}x",
                ser * 1e3,
                par * 1e3,
                ser / par
            ),
            _ => println!("E solve_refined: factor failed"),
        }
    }

    let rhs = build_rhs_block(&csc, NRHS);
    let col = |c: usize| &rhs[c * n..(c + 1) * n];

    // Correctness first: a fast wrong path is not a lever.
    let res_a = (0..NRHS)
        .map(|c| rel_residual(&csc, &solve_sparse(factors, col(c)).expect("A"), col(c)))
        .fold(0.0f64, f64::max);
    let res_c = (0..NRHS)
        .map(|c| {
            rel_residual(
                &csc,
                &solve_sparse_cb(factors, col(c), true).expect("C"),
                col(c),
            )
        })
        .fold(0.0f64, f64::max);
    let many = solve_sparse_many(factors, &rhs, NRHS).expect("D");
    let res_d = (0..NRHS)
        .map(|c| rel_residual(&csc, &many[c * n..(c + 1) * n], col(c)))
        .fold(0.0f64, f64::max);

    let a = best_of(|| {
        for c in 0..NRHS {
            std::hint::black_box(solve_sparse(factors, col(c)).expect("A"));
        }
    });
    let b = best_of(|| {
        for c in 0..NRHS {
            std::hint::black_box(solve_sparse_cb(factors, col(c), false).expect("B"));
        }
    });
    let cpar = best_of(|| {
        for c in 0..NRHS {
            std::hint::black_box(solve_sparse_cb(factors, col(c), true).expect("C"));
        }
    });
    let d = best_of(|| {
        std::hint::black_box(solve_sparse_many(factors, &rhs, NRHS).expect("D"));
    });

    println!(
        "{:<28} {:>12} {:>10} {:>12}",
        "path (nrhs=13)", "wall(ms)", "vs A", "rel.resid"
    );
    let row = |label: &str, s: f64, res: Option<f64>| {
        let r = match res {
            Some(v) => format!("{v:.2e}"),
            None => "-".to_string(),
        };
        println!("{:<28} {:>12.2} {:>10.2} {:>12}", label, s * 1e3, a / s, r);
    };
    row("A serial solve_sparse x13", a, Some(res_a));
    row("B cb serial x13", b, None);
    row("C cb PARALLEL x13  (#131)", cpar, Some(res_c));
    row("D solve_sparse_many(13)", d, Some(res_d));

    // BLAS-3 crossover sweep: per-RHS cost, batched vs looped.
    println!();
    println!(
        "{:>6} {:>14} {:>14} {:>10}   (BLAS3 threshold = 32)",
        "nrhs", "looped us/rhs", "many us/rhs", "speedup"
    );
    for &k in SWEEP.iter() {
        let blk = build_rhs_block(&csc, k);
        let looped = best_of(|| {
            for c in 0..k {
                std::hint::black_box(solve_sparse(factors, &blk[c * n..(c + 1) * n]).expect("s"));
            }
        });
        let batched = best_of(|| {
            std::hint::black_box(solve_sparse_many(factors, &blk, k).expect("m"));
        });
        println!(
            "{:>6} {:>14.1} {:>14.1} {:>10.2}",
            k,
            looped * 1e6 / k as f64,
            batched * 1e6 / k as f64,
            looped / batched
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names: Vec<String> = if args.is_empty() {
        vec![
            "r05_kkt".into(),
            "qap15_kkt".into(),
            "dirichlet120_kkt".into(),
            "cont-201".into(),
        ]
    } else {
        args
    };
    println!(
        "rayon threads = {}  |  best-of-{} wall clock",
        rayon::current_num_threads(),
        REPS
    );
    for nm in names {
        let p = format!("tests/data/large/{nm}.mtx");
        run(Path::new(&p));
    }
}
