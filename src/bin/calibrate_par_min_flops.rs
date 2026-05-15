//! Calibration probe for `PAR_MIN_FLOPS` (issue #19 follow-up).
//!
//! Sweeps Poisson-KKT problem size, measures sequential vs forced-parallel
//! factor wall under a single persistent `rayon::ThreadPool` (matching
//! the `Solver` reuse path landed in session 2026-05-15-04), and reports
//! the crossover where parallel begins to beat sequential by ≥1.2×.
//!
//! The "forced" framing matters: this probe bypasses
//! `should_parallelize_assembly`'s flop gate by calling the parallel
//! driver directly, so we measure actual driver performance independent
//! of the current threshold. The output tells us where the threshold
//! *should* sit, not whether the current threshold fires.
//!
//! Usage:
//!     cargo run --release --bin calibrate_par_min_flops
//!     cargo run --release --bin calibrate_par_min_flops -- --reps 5
//!
//! Reports `(K, n_kkt, n_snodes, multichild, est_flops, seq_ms, par_ms,
//! par/seq, decision)` per row. `decision` is `parallel` when par/seq ≤
//! 1/1.2, `sequential` when ≥ 1.2, `tie` in between.

use std::sync::Arc;
use std::time::Instant;

use feral::numeric::factorize::{
    estimate_assembly_flops, factorize_multifrontal_supernodal_parallel,
    factorize_multifrontal_supernodal_with_workspace, FactorWorkspace, NumericParams,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn build_poisson_kkt(k: usize) -> CscMatrix {
    let m = k * k;
    let n_kkt = 3 * m;
    let h = 1.0 / (k as f64 + 1.0);
    let alpha = 0.01;
    let inv_h2 = 1.0 / (h * h);

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // (1,1) Hessian diagonal — u block then f block.
    for i in 0..m {
        rows.push(i);
        cols.push(i);
        vals.push(h * h);
    }
    for i in 0..m {
        let idx = m + i;
        rows.push(idx);
        cols.push(idx);
        vals.push(alpha * h * h);
    }

    // (2,1) Jacobian = 5-point Laplacian + (-I) coupling to f.
    for i in 0..k {
        for j in 0..k {
            let c = i * k + j;
            let con_row = 2 * m + c;

            rows.push(con_row);
            cols.push(c);
            vals.push(4.0 * inv_h2);

            if i > 0 {
                rows.push(con_row);
                cols.push((i - 1) * k + j);
                vals.push(-inv_h2);
            }
            if i + 1 < k {
                rows.push(con_row);
                cols.push((i + 1) * k + j);
                vals.push(-inv_h2);
            }
            if j > 0 {
                rows.push(con_row);
                cols.push(i * k + (j - 1));
                vals.push(-inv_h2);
            }
            if j + 1 < k {
                rows.push(con_row);
                cols.push(i * k + (j + 1));
                vals.push(-inv_h2);
            }

            rows.push(con_row);
            cols.push(m + c);
            vals.push(-1.0);
        }
    }

    CscMatrix::from_triplets(n_kkt, &rows, &cols, &vals).expect("triplet build")
}

fn numeric_params() -> NumericParams {
    let bk = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        zero_tol: 1e-10,
        zero_tol_2x2: 1e-20,
        pivot_threshold: 1e-8,
        ..BunchKaufmanParams::default()
    };
    NumericParams::with_bk(bk)
}

fn parse_reps() -> usize {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == "--reps" {
            if let Ok(n) = w[1].parse::<usize>() {
                return n.max(1);
            }
        }
    }
    3
}

fn main() {
    let reps = parse_reps();

    // Build a dedicated ThreadPool matching what `Solver` constructs.
    // Wrapping the parallel call in `pool.install(...)` binds the inner
    // `rayon::scope` to these workers and amortises cv-wait wakeup the
    // same way as in production.
    let n_threads = rayon::current_num_threads().max(1);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("threadpool"),
    );
    eprintln!(
        "calibrate_par_min_flops: reps={} threads={}",
        reps, n_threads
    );

    let ks = [15usize, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160];
    let snode_params = SupernodeParams::default();
    let nparams = numeric_params();

    println!(
        "{:>4} {:>9} {:>9} {:>5} {:>13} {:>10} {:>10} {:>9} {:>11}",
        "K", "n_kkt", "n_snode", "mc", "est_flops", "seq_ms", "par_ms", "par/seq", "decision"
    );

    for &k in &ks {
        let csc = build_poisson_kkt(k);
        let symbolic = match symbolic_factorize(&csc, &snode_params) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("K={}: symbolic failed: {}", k, e);
                continue;
            }
        };
        let n_snode = symbolic.supernodes.len();
        let multichild = symbolic.supernodes.iter().any(|s| s.children.len() >= 2);
        let est_flops = estimate_assembly_flops(&symbolic.supernodes);

        // Warm-up — one of each path to fault in workspace pages,
        // bring caches up, and prime worker threads in the pool.
        let mut ws = FactorWorkspace::new();
        let _ =
            factorize_multifrontal_supernodal_with_workspace(&csc, &symbolic, &nparams, &mut ws);
        let _ =
            pool.install(|| factorize_multifrontal_supernodal_parallel(&csc, &symbolic, &nparams));

        // Timed reps — best-of-N strips OS jitter (the noise floor we
        // care about is microseconds on small problems).
        let mut best_seq = u128::MAX;
        for _ in 0..reps {
            let t0 = Instant::now();
            let _ = factorize_multifrontal_supernodal_with_workspace(
                &csc, &symbolic, &nparams, &mut ws,
            );
            let us = t0.elapsed().as_micros();
            best_seq = best_seq.min(us);
        }

        let mut best_par = u128::MAX;
        for _ in 0..reps {
            let t0 = Instant::now();
            let _ = pool
                .install(|| factorize_multifrontal_supernodal_parallel(&csc, &symbolic, &nparams));
            let us = t0.elapsed().as_micros();
            best_par = best_par.min(us);
        }

        let ratio = best_par as f64 / best_seq as f64;
        // Decision uses the "parallel must win by ≥1.2×" criterion from
        // PAR_MIN_FLOPS's docstring ("one decimal above break-even" is
        // strong but 1.2× is a defensible production threshold).
        let decision = if ratio <= 1.0 / 1.2 {
            "parallel"
        } else if ratio >= 1.2 {
            "sequential"
        } else {
            "tie"
        };

        println!(
            "{:>4} {:>9} {:>9} {:>5} {:>13} {:>10.3} {:>10.3} {:>9.3} {:>11}",
            k,
            3 * k * k,
            n_snode,
            if multichild { "y" } else { "n" },
            est_flops,
            best_seq as f64 / 1000.0,
            best_par as f64 / 1000.0,
            ratio,
            decision,
        );
    }

    println!();
    println!(
        "Current PAR_MIN_FLOPS = {} ({:.0e})",
        feral::numeric::factorize::PAR_MIN_FLOPS,
        feral::numeric::factorize::PAR_MIN_FLOPS as f64
    );
    println!(
        "Calibrated threshold = smallest est_flops with decision=parallel. \
         Pick a value ~one decimal below the 'parallel' rows to keep margin."
    );
}
