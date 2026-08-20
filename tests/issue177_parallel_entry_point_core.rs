//! `solve_sparse_refined_parallel` must keep the shared-vector fallback
//! it had through v0.16.0.
//!
//! # What went wrong
//!
//! Issue #177 replaced the runtime core gate — `CbSolveWorkspace::
//! worthwhile()`, derived from `rayon::current_num_threads()` and
//! `FERAL_CB_THRESH` — with an explicit [`SolveCore`] argument, because
//! the old gate made a factor's arithmetic a function of the host. In the
//! rewrite, `solve_sparse_refined_parallel` was mapped to
//! `SolveCore::ContribBlock { parallel: true }`, which runs the
//! contribution-block core *unconditionally*. Through v0.16.0 it ran the
//! CB core only when the gate accepted the factor and the shared-vector
//! core otherwise:
//!
//! ```text
//! // v0.16.0, src/numeric/solve.rs
//! let mut cb_ws: Option<CbSolveWorkspace> = if parallel_cb && n > 0 {
//!     let cb = CbSolveWorkspace::for_factors(factors);
//!     cb.worthwhile().then_some(cb)
//! } else { None };
//! ```
//!
//! So the function silently changed its answer on every factor the gate
//! used to reject — while the `[Unreleased]` CHANGELOG entry that
//! introduced `SolveCore` asserted it was unchanged. It was also the
//! slower of the two: routing rejected factors to the CB core cost 1.86x
//! on poisson_40 (659 µs → 1228 µs), which is the alternative #177's own
//! measurements rejected.
//!
//! # The oracle
//!
//! `solve_sparse_refined` — the shared-vector entry point, unchanged
//! since v0.16.0 and pinned by `tests/golden_bits.rs`. On a factor the
//! profitability predicate rejects, the parallel entry point must agree
//! with it bit-for-bit, because that is what v0.16.0 returned. The
//! oracle is therefore the released binary's behaviour, not a value
//! computed here.
//!
//! # What is pinned
//!
//!  1. on a CB-rejected factor, `solve_sparse_refined_parallel` is
//!     bit-identical to `solve_sparse_refined`;
//!  2. that check is not vacuous — the CB core genuinely returns
//!     different bits on the same factor, so a regression to
//!     `ContribBlock` would fail (1);
//!  3. on a CB-profitable factor the parallel entry point does reach the
//!     CB core, so the fallback was restored rather than the CB core
//!     disabled;
//!  4. the `_opts` and `_into` forms make the same core choice as the
//!     allocating one.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::{
    solve_sparse_refined, solve_sparse_refined_auto, solve_sparse_refined_cb,
    solve_sparse_refined_parallel, solve_sparse_refined_parallel_into,
    solve_sparse_refined_parallel_opts, RefineOptions,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.0,
        ..BunchKaufmanParams::default()
    })
}

/// Path-like and below the cost floor: `cb_core_profitable` rejects it,
/// so v0.16.0's gate rejected it too, at every worker count.
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

/// 2-D Poisson on a `k x k` grid: bushy nested dissection. At `k = 160`
/// the tree clears the predicate and the CB core is chosen.
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

/// Contracts 1 and 2. The non-vacuity half runs first: if the two cores
/// happened to agree on this matrix, contract 1 would hold no matter
/// which core the parallel entry point ran, and the test would pin
/// nothing.
#[test]
fn parallel_entry_point_keeps_the_shared_vector_core_on_a_rejected_factor() {
    let m = chain(400);
    let f = factor_of(&m);
    let b = rhs_for(m.n);

    let shared = solve_sparse_refined(&m, &f, &b).expect("shared-vector refine");
    let cb = solve_sparse_refined_cb(&m, &f, &b, false).expect("cb refine");
    assert!(
        bit_diff(&shared, &cb) > 0,
        "non-vacuity: the two cores agree bit-for-bit on chain_400, so this \
         test cannot distinguish them — pick a fixture where they differ"
    );

    let parallel = solve_sparse_refined_parallel(&m, &f, &b).expect("parallel refine");
    assert_bit_eq(
        &shared,
        &parallel,
        "chain_400: solve_sparse_refined_parallel vs solve_sparse_refined (v0.16.0 behaviour)",
    );
}

/// Contract 3: the fallback was restored, not bolted on by disabling the
/// CB core outright.
#[test]
fn parallel_entry_point_still_reaches_the_cb_core_on_a_profitable_factor() {
    let m = poisson_2d_spd(160);
    let f = factor_of(&m);
    let b = rhs_for(m.n);

    let cb = solve_sparse_refined_cb(&m, &f, &b, true).expect("cb refine");
    let parallel = solve_sparse_refined_parallel(&m, &f, &b).expect("parallel refine");
    assert_bit_eq(
        &cb,
        &parallel,
        "poisson_160: solve_sparse_refined_parallel vs the CB core",
    );
}

/// The parallel entry point is `solve_sparse_refined_auto(.., true)` —
/// which is what its documentation now says it is — on both sides of the
/// predicate.
#[test]
fn parallel_entry_point_equals_auto_with_parallelism_on() {
    for (m, label) in [
        (chain(400), "chain_400"),
        (poisson_2d_spd(40), "poisson_40"),
        (poisson_2d_spd(160), "poisson_160"),
    ] {
        let f = factor_of(&m);
        let b = rhs_for(m.n);
        let auto = solve_sparse_refined_auto(&m, &f, &b, true).expect("auto refine");
        let parallel = solve_sparse_refined_parallel(&m, &f, &b).expect("parallel refine");
        assert_bit_eq(
            &auto,
            &parallel,
            &format!("{label}: parallel vs auto(parallel = true)"),
        );
    }
}

/// Contract 4: the `_opts` and `_into` forms route to the same core. A
/// fix applied to only the allocating entry point would leave the other
/// two on the CB core.
#[test]
fn opts_and_into_forms_make_the_same_core_choice() {
    for (m, label) in [
        (chain(400), "chain_400"),
        (poisson_2d_spd(160), "poisson_160"),
    ] {
        let f = factor_of(&m);
        let b = rhs_for(m.n);
        let base = solve_sparse_refined_parallel(&m, &f, &b).expect("parallel refine");

        let via_opts = solve_sparse_refined_parallel_opts(&m, &f, &b, RefineOptions::default())
            .expect("parallel refine opts");
        assert_bit_eq(&base, &via_opts, &format!("{label}: _opts vs allocating"));

        let mut x = vec![0.0; m.n];
        solve_sparse_refined_parallel_into(&m, &f, &b, &mut x, RefineOptions::default())
            .expect("parallel refine into");
        assert_bit_eq(&base, &x, &format!("{label}: _into vs allocating"));
    }
}
