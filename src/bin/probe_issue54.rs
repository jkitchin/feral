//! Probe for issue #54: wrong inertia on LP-shaped KKT
//! (nuffield2_trap iter=1).
//!
//! Loads the matrix `dev/repros/issue-54/nuffield2_trap_iter1.mtx` and
//! reports inertia, status, and basic factor diagnostics.

#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]

use std::path::PathBuf;

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::Solver;
use feral::read_mtx;
use feral::scaling::ScalingStrategy;
use feral::symbolic::SupernodeParams;

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("dev/repros/issue-54/nuffield2_trap_iter1.mtx"));

    let mtx = read_mtx(&path).expect("read mtx");
    let n = mtx.n;
    let a = mtx.to_csc().expect("csc");
    println!("matrix       = {:?}", path);
    println!("n            = {}", n);
    println!("nnz          = {}", mtx.entries.len());

    let mut zero_diag = 0usize;
    let mut pos_diag = 0usize;
    let mut neg_diag = 0usize;
    for col in 0..a.n {
        let start = a.col_ptr[col];
        let end = a.col_ptr[col + 1];
        let mut found_diag = false;
        for k in start..end {
            if a.row_idx[k] == col {
                found_diag = true;
                let v = a.values[k];
                if v == 0.0 {
                    zero_diag += 1;
                } else if v > 0.0 {
                    pos_diag += 1;
                } else {
                    neg_diag += 1;
                }
                break;
            }
        }
        if !found_diag {
            zero_diag += 1;
        }
    }
    println!(
        "diag         = {} pos / {} neg / {} zero (incl. missing)",
        pos_diag, neg_diag, zero_diag
    );

    let expected_neg: usize = 13202;

    let configs: Vec<(&str, Box<dyn Fn() -> Solver>)> = vec![
        (
            "default",
            Box::new(|| Solver::with_params(NumericParams::default(), SupernodeParams::default())),
        ),
        (
            "no-delayed-pivots",
            Box::new(|| {
                let mut np = NumericParams::default();
                np.allow_delayed_pivots = false;
                Solver::with_params(np, SupernodeParams::default())
            }),
        ),
        (
            "pivtol=1e-2",
            Box::new(|| {
                let mut np = NumericParams::default();
                np.bk.pivot_threshold = 1e-2;
                Solver::with_params(np, SupernodeParams::default())
            }),
        ),
        (
            "pivtol=0.0",
            Box::new(|| {
                let mut np = NumericParams::default();
                np.bk.pivot_threshold = 0.0;
                Solver::with_params(np, SupernodeParams::default())
            }),
        ),
        (
            "identity-scaling",
            Box::new(|| {
                let mut np = NumericParams::default();
                np.scaling = ScalingStrategy::Identity;
                Solver::with_params(np, SupernodeParams::default())
            }),
        ),
        (
            "infnorm-scaling",
            Box::new(|| {
                let mut np = NumericParams::default();
                np.scaling = ScalingStrategy::InfNorm;
                Solver::with_params(np, SupernodeParams::default())
            }),
        ),
        (
            "mc64-scaling",
            Box::new(|| {
                let mut np = NumericParams::default();
                np.scaling = ScalingStrategy::Mc64Symmetric;
                Solver::with_params(np, SupernodeParams::default())
            }),
        ),
        (
            "static-pivot-1e-12",
            Box::new(|| {
                let mut np = NumericParams::default();
                np.bk.static_pivot_floor = 1e-12;
                Solver::with_params(np, SupernodeParams::default())
            }),
        ),
        (
            "static-pivot-1e-8",
            Box::new(|| {
                let mut np = NumericParams::default();
                np.bk.static_pivot_floor = 1e-8;
                Solver::with_params(np, SupernodeParams::default())
            }),
        ),
        (
            "static-pivot-1e-6",
            Box::new(|| {
                let mut np = NumericParams::default();
                np.bk.static_pivot_floor = 1e-6;
                Solver::with_params(np, SupernodeParams::default())
            }),
        ),
    ];

    for (label, make_solver) in configs {
        let mut solver = make_solver();
        let status = solver.factor(&a, None);
        let got_neg = solver.num_negative_eigenvalues();
        let inertia = solver.inertia();
        let stats = solver.last_factor_stats();
        let diff = got_neg as isize - expected_neg as isize;
        println!("\n--- config: {} ---", label);
        println!("status       = {:?}", status);
        println!("inertia      = {:?}", inertia);
        println!("diff_neg     = {}", diff);
        if let Some(s) = stats {
            println!("stats        = {:?}", s);
        }
        // Solve A x = b with random b and check residual.
        if matches!(
            status,
            feral::numeric::solver::FactorStatus::Success
                | feral::numeric::solver::FactorStatus::Singular
        ) {
            // Build random b (use a deterministic pattern).
            let n = a.n;
            let mut b = vec![0.0f64; n];
            for i in 0..n {
                b[i] = (((i as u64).wrapping_mul(2654435761) % 9999) as f64) / 9999.0 - 0.5;
            }
            let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
            match solver.solve(&b) {
                Ok(x) => {
                    // Compute A x, exploiting symmetric lower-triangle CSC.
                    let mut ax = vec![0.0f64; n];
                    for j in 0..n {
                        let s = a.col_ptr[j];
                        let e = a.col_ptr[j + 1];
                        for k in s..e {
                            let i = a.row_idx[k];
                            let v = a.values[k];
                            ax[i] += v * x[j];
                            if i != j {
                                ax[j] += v * x[i];
                            }
                        }
                    }
                    let mut rnorm2 = 0.0;
                    for i in 0..n {
                        let r = ax[i] - b[i];
                        rnorm2 += r * r;
                    }
                    let rel = rnorm2.sqrt() / b_norm;
                    println!("rel_resid    = {:.3e}", rel);
                }
                Err(e) => println!("solve error  = {:?}", e),
            }
        }
    }
}
