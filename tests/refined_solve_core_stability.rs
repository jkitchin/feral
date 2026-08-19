//! Issue #177 — a refined solve's arithmetic must be a function of the
//! factor, never of the host.
//!
//! feral has two numerically distinct sparse solve cores: the
//! shared-global-vector core (`solve_sparse`, a flat postorder fold) and
//! the contribution-block core of issue #131 Gap A (a subtree sum tree).
//! They differ in the last bits by design — each is a valid
//! reassociation of the other.
//!
//! What #177 reports is that *which* core ran was decided by
//! `CbTaskPlan::worthwhile`, a predicate computed from
//! `rayon::current_num_threads()` and `FERAL_CB_THRESH`, and by whether a
//! thread pool had been built. So the same binary, same matrix, same
//! factor did different arithmetic on hosts with different core counts —
//! and #16 established that a ULP difference here is amplified by the IPM
//! host's line search into whole different iterate trajectories (141 vs
//! 888 outer iterations on qcqp1000-1nc). Reported on henon120.
//!
//! Contracts checked here, at the *driver* level that
//! `tests/cb_solve_parity.rs` does not reach:
//!
//!  1. worker count does not move a bit of a refined solve;
//!  2. serial vs tree-parallel execution are byte-identical — #131's
//!     contract, now literally true rather than true-when-gated;
//!  3. `Solver::solve_refined` is byte-identical whether or not a thread
//!     pool was built, and whether or not parallelism was requested at
//!     all — the latter closes `Solver::default_use_parallel()`, which
//!     reads `available_parallelism()` and so was itself a route from
//!     core count to arithmetic;
//!  4. the factor-derived core choice really does route each factor to
//!     the intended core.
//!
//! Every invariance claim is exercised on matrices sitting on *both*
//! sides of the profitability predicate, since that predicate was the
//! thing doing the switching.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::{
    solve_sparse_refined, solve_sparse_refined_auto, solve_sparse_refined_cb,
};
use feral::numeric::solver::Solver;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, FactorStatus, ZeroPivotAction};

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.0,
        ..BunchKaufmanParams::default()
    })
}

/// 2-D Poisson on a `k x k` grid: a bushy nested-dissection tree. Large
/// `k` clears the profitability predicate; small `k` does not.
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

/// A pentadiagonal chain: path-like, so the Amdahl arm of the predicate
/// rejects it however many workers the host has.
fn chain(n: usize) -> CscMatrix {
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(4.0);
        if i + 1 < n {
            rows.push(i + 1);
            cols.push(i);
            vals.push(-1.0);
        }
        if i + 2 < n {
            rows.push(i + 2);
            cols.push(i);
            vals.push(-0.5);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("chain")
}

fn factor_of(m: &CscMatrix) -> feral::numeric::factorize::SparseFactors {
    let sym = symbolic_factorize(m, &SupernodeParams::default()).expect("symbolic");
    let (f, _) = factorize_multifrontal(m, &sym, &ldlt_params()).expect("numeric");
    f
}

fn rhs_for(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 + 0.37 * (i % 7) as f64).collect()
}

fn bit_diff(a: &[f64], b: &[f64]) -> usize {
    (0..a.len())
        .filter(|&i| a[i].to_bits() != b[i].to_bits())
        .count()
}

fn assert_bit_eq(a: &[f64], b: &[f64], what: &str) {
    let d = bit_diff(a, b);
    assert_eq!(d, 0, "{what}: {d} of {} entries differ", a.len());
}

/// Contracts 1 and 2 over one matrix: the refined solve is invariant to
/// worker count and to the serial/parallel execution choice.
fn check_host_invariance(m: &CscMatrix, label: &str) {
    let f = factor_of(m);
    let n = m.n;
    let b = rhs_for(n);

    // Reference: no pool, no parallelism requested — the state a host
    // that cannot spawn threads ends up in.
    let reference = solve_sparse_refined_auto(m, &f, &b, false).expect("serial auto refine");

    for nt in [1usize, 2, 3, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .expect("pool");
        let x = pool
            .install(|| solve_sparse_refined_auto(m, &f, &b, true))
            .expect("parallel auto refine");
        assert_bit_eq(
            &reference,
            &x,
            &format!("[{label}] refined solve at {nt} workers vs no pool"),
        );
    }

    // The explicit CB entry point honours the same contract: its
    // `parallel` argument is a scheduling request, not an arithmetic one.
    let cb_serial = solve_sparse_refined_cb(m, &f, &b, false).expect("cb serial");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool");
    let cb_par = pool
        .install(|| solve_sparse_refined_cb(m, &f, &b, true))
        .expect("cb parallel");
    assert_bit_eq(
        &cb_serial,
        &cb_par,
        &format!("[{label}] CB core: serial vs tree-parallel execution"),
    );

    // And the result is a genuine solve, not merely a stable one.
    let mut ax = vec![0.0; n];
    m.symv(&reference, &mut ax);
    let (mut r2, mut b2) = (0.0f64, 0.0f64);
    for i in 0..n {
        r2 += (ax[i] - b[i]).powi(2);
        b2 += b[i] * b[i];
    }
    let rel = (r2 / b2.max(1e-300)).sqrt();
    assert!(rel < 1e-8, "[{label}] refined residual {rel:e} too large");
}

#[test]
fn refined_solve_is_host_invariant_on_a_cb_profitable_tree() {
    // n = 25600: total solve cost ~1.1e6 and a bushy tree, so the
    // predicate routes this to the CB core and the tree-parallel path
    // really runs.
    check_host_invariance(&poisson_2d_spd(160), "poisson_160");
}

#[test]
fn refined_solve_is_host_invariant_on_a_cb_rejected_tree() {
    // Path-like and below the cost floor: the predicate keeps this on the
    // shared-vector core, which is also the faster core here.
    check_host_invariance(&chain(400), "chain_400");
}

#[test]
fn refined_solve_is_host_invariant_on_a_small_bushy_tree() {
    check_host_invariance(&poisson_2d_spd(40), "poisson_40");
}

/// Contract 3, and the sharpest form of the #177 regression.
///
/// Two routes ran from the host's core count into the arithmetic:
/// `Solver::default_use_parallel()` reads `available_parallelism()`, and
/// `parallel_pool` is `None` whenever `ThreadPoolBuilder::build()` failed
/// (`wasm32-wasip1`, or a host out of threads under `RLIMIT_NPROC` /
/// cgroup `pids.max`). Both used to swap the solve core. All three
/// configurations must now agree bit for bit.
#[test]
fn solver_refine_is_identical_across_parallelism_and_pool_availability() {
    for (label, m) in [
        ("poisson_160", poisson_2d_spd(160)),
        ("chain_400", chain(400)),
    ] {
        let b = rhs_for(m.n);

        let mut pooled = Solver::new().with_parallel(true);
        assert!(matches!(pooled.factor(&m, None), FactorStatus::Success));
        let x_pooled = pooled.solve_refined(&m, &b).expect("pooled refine");

        // Parallelism never requested: what a single-core host gets from
        // `Solver::new()`, since `default_use_parallel()` is false there.
        let mut seq = Solver::new().with_parallel(false);
        assert!(matches!(seq.factor(&m, None), FactorStatus::Success));
        let x_seq = seq.solve_refined(&m, &b).expect("sequential refine");

        // Parallelism requested but no pool — the post-build-failure
        // state. `with_parallel` only flips the flag, so the factor and
        // the absent pool survive the move.
        let mut poolless = Solver::new().with_parallel(false);
        assert!(matches!(poolless.factor(&m, None), FactorStatus::Success));
        let poolless = poolless.with_parallel(true);
        assert!(poolless.parallel(), "fixture must request parallelism");
        let x_poolless = poolless.solve_refined(&m, &b).expect("pool-less refine");

        assert_bit_eq(
            &x_pooled,
            &x_seq,
            &format!("[{label}] pooled vs sequential"),
        );
        assert_bit_eq(
            &x_pooled,
            &x_poolless,
            &format!("[{label}] pooled vs pool-less"),
        );

        // The multi-RHS refine dispatches per column through the same
        // three arms.
        let many_pooled = pooled.solve_many_refined(&m, &b, 1).expect("pooled many");
        let many_seq = seq.solve_many_refined(&m, &b, 1).expect("sequential many");
        let many_poolless = poolless
            .solve_many_refined(&m, &b, 1)
            .expect("pool-less many");
        assert_bit_eq(
            &many_pooled,
            &many_seq,
            &format!("[{label}] many: pooled vs sequential"),
        );
        assert_bit_eq(
            &many_pooled,
            &many_poolless,
            &format!("[{label}] many: pooled vs pool-less"),
        );
    }
}

/// Contract 4: the factor-derived choice routes each factor to the core
/// it is supposed to reach — and the two cores really are
/// distinguishable, so the invariance assertions above are not passing
/// vacuously because every path happens to agree anyway.
#[test]
fn the_factor_selects_the_intended_core() {
    // Bushy and large: routed to the CB core, which on this factor is
    // *not* the shared-vector answer.
    let m = poisson_2d_spd(160);
    let f = factor_of(&m);
    let b = rhs_for(m.n);
    let auto = solve_sparse_refined_auto(&m, &f, &b, false).expect("auto");
    let cb = solve_sparse_refined_cb(&m, &f, &b, false).expect("cb");
    let shared = solve_sparse_refined(&m, &f, &b).expect("shared-vector");
    assert_bit_eq(&auto, &cb, "poisson_160 auto vs explicit CB");
    assert!(
        bit_diff(&auto, &shared) > 0,
        "poisson_160 cannot distinguish the cores; the invariance tests would be vacuous"
    );
    // Both are valid solves: they agree far above the bit level.
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..m.n {
        num = num.max((shared[i] - cb[i]).abs());
        den = den.max(shared[i].abs());
    }
    assert!(
        num / den.max(1.0) < 1e-9,
        "cores disagree by more than a reassociation: {:e}",
        num / den.max(1.0)
    );

    // Path-like: routed to the shared-vector core, where the CB core
    // measured 1.27x slower with nothing to show for it.
    let m = chain(400);
    let f = factor_of(&m);
    let b = rhs_for(m.n);
    let auto = solve_sparse_refined_auto(&m, &f, &b, false).expect("auto");
    let shared = solve_sparse_refined(&m, &f, &b).expect("shared-vector");
    assert_bit_eq(&auto, &shared, "chain_400 auto vs shared-vector");
}
