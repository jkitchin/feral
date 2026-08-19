//! Issue #177: `FERAL_CB_THRESH` must not change a result bit.
//!
//! The reporter's repro drove the divergence with exactly this variable:
//!
//! ```sh
//! env FERAL_PAR_TASK_MIN_FLOPS=1000000000000 pounce henon120.nl ...
//! env FERAL_PAR_TASK_MIN_FLOPS=1000000000000 FERAL_CB_THRESH=1000000000000 pounce ...
//! ```
//!
//! with the factorization held sequential in both, so the solve was the
//! only thing varying — and the two runs diverged from IPM iteration 19.
//! `FERAL_CB_THRESH` fed `CbTaskPlan`'s coarsening, whose `worthwhile`
//! verdict then chose between the contribution-block and shared-vector
//! solve cores. It is now confined to scheduling.
//!
//! **This file must contain exactly one test.** It mutates the process
//! environment, which is shared by every test in a binary but not across
//! binaries; a second test here could observe a half-applied variable.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::solve_sparse_refined_auto;
use feral::numeric::solver::Solver;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, FactorStatus, ZeroPivotAction};

fn poisson_2d_spd(k: usize) -> CscMatrix {
    let n = k * k;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..k {
        for i in 0..k {
            let p = j * k + i;
            rows.push(p);
            cols.push(p);
            vals.push(4.0);
            if i + 1 < k {
                rows.push(p + 1);
                cols.push(p);
                vals.push(-1.0);
            }
            if j + 1 < k {
                rows.push(p + k);
                cols.push(p);
                vals.push(-1.0);
            }
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("poisson_2d")
}

#[test]
fn feral_cb_thresh_cannot_change_a_result_bit() {
    // n = 25600, bushy: routed to the CB core, so `FERAL_CB_THRESH` has
    // real coarsening decisions to change. A factor that never reaches
    // the CB core would make this vacuous.
    let m = poisson_2d_spd(160);
    let n = m.n;
    let sym = symbolic_factorize(&m, &SupernodeParams::default()).expect("symbolic");
    let params = NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.0,
        ..BunchKaufmanParams::default()
    });
    let (f, _) = factorize_multifrontal(&m, &sym, &params).expect("numeric");
    let b: Vec<f64> = (0..n).map(|i| 1.0 + 0.37 * (i % 7) as f64).collect();

    let mut solver = Solver::new().with_parallel(true);
    assert!(matches!(solver.factor(&m, None), FactorStatus::Success));

    let base_free = solve_sparse_refined_auto(&m, &f, &b, true).expect("baseline refine");
    let base_solver = solver
        .solve_refined(&m, &b)
        .expect("baseline solver refine");

    // The reporter's own value, plus the opposite extreme (every node its
    // own task root) and a mid-range cut.
    for v in ["1", "512", "1000000000000"] {
        // SAFETY: this binary contains exactly one test (enforced by the
        // module docs above), so no other thread is reading the
        // environment concurrently.
        unsafe { std::env::set_var("FERAL_CB_THRESH", v) };

        let x_free = solve_sparse_refined_auto(&m, &f, &b, true).expect("refine");
        let x_solver = solver.solve_refined(&m, &b).expect("solver refine");
        for i in 0..n {
            assert_eq!(
                base_free[i].to_bits(),
                x_free[i].to_bits(),
                "FERAL_CB_THRESH={v}: solve_sparse_refined_auto moved bit {i}"
            );
            assert_eq!(
                base_solver[i].to_bits(),
                x_solver[i].to_bits(),
                "FERAL_CB_THRESH={v}: Solver::solve_refined moved bit {i}"
            );
        }
    }
    // SAFETY: as above.
    unsafe { std::env::remove_var("FERAL_CB_THRESH") };
}
