//! Issue #221 review: the delay-chain backstop (`DELAY_CHAIN_GROWTH`,
//! `DELAY_CHAIN_FLOOR` in `src/numeric/factorize.rs`) had no in-tree
//! test.
//!
//! A front engages cascade-break when the extra frontal area its
//! in-flight delay chain has cost, `chain_excess`, exceeds both
//! `DELAY_CHAIN_FLOOR` (2^20 entries) and `DELAY_CHAIN_GROWTH` (10) times
//! its subtree's symbolic area. The per-front budget (`delayed_capacity`)
//! never fires here: no front receives more than `d` delays, far under
//! its `max(3·nrow, 4096 - nrow)` growth bound.
//!
//! Matrix (a KKT in saddle-point form). Primal variables are a chain
//! `x_0 .. x_L` (tridiagonal, diagonal 4, off-diagonal 1) and a hub `h`
//! (diagonal 10, uncoupled in `H`), so `H = blockdiag(T, 10)` is SPD.
//! Constraint `z_i` (`i < d`) has a zero diagonal, couples to the hub
//! with weight 1 and to `x_i` with weight `ε = 1e-6`:
//! `B_i = ε·e_{x_i} + e_h`. The rows of `B` are independent (their
//! pairwise differences are `ε(e_{x_i} - e_{x_j})`), so `B` has full row
//! rank `d`. By the saddle-point inertia theorem the inertia of
//! `[[H, Bᵀ], [B, 0]]` is `(L + 2, d, 0)`.
//!
//! With the order `z, x_0 .. x_L, h`, `z_i` is a leaf whose parent is
//! `x_i`. Its pivot is 0 and its only fully-summed partner coupling is
//! `ε`, so every `z_i` delays (the 2×2 with `x_i` fails the growth bound
//! against the hub coupling), and the delays ride the chain to the hub.
//! Each chain front then carries about `d` delayed columns, so its excess
//! area is about `d²` while its symbolic area is 3.

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus, Solver};
use feral::sparse::csc::CscMatrix;
use feral::symbolic::{OrderingMethod, SupernodeParams};

const EPS_COUPLING: f64 = 1e-6;

/// Returns the matrix (natural order is the elimination order) and its
/// saddle-point inertia `(L + 2, d, 0)`.
fn delay_chain_kkt(d: usize, chain_len: usize) -> (CscMatrix, (usize, usize, usize)) {
    assert!(d <= chain_len);
    let nx = chain_len + 1;
    let n = d + nx + 1;
    let x = |j: usize| d + j;
    let h = n - 1;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let mut push = |i: usize, j: usize, val: f64| {
        let (i, j) = if i >= j { (i, j) } else { (j, i) };
        r.push(i);
        c.push(j);
        v.push(val);
    };
    for i in 0..d {
        // Zero diagonal stored explicitly so the pattern is a KKT's.
        push(i, i, 0.0);
        push(x(i), i, EPS_COUPLING);
        push(h, i, 1.0);
    }
    for j in 0..nx {
        push(x(j), x(j), 4.0);
        if j + 1 < nx {
            push(x(j + 1), x(j), 1.0);
        }
    }
    push(h, h, 10.0);
    let a = CscMatrix::from_triplets(n, &r, &c, &v).expect("csc");
    (a, (nx + 1, d, 0))
}

/// Default numeric parameters (cascade-break armed, `u = 1e-8`), natural
/// order, no amalgamation, so each chain variable is its own front.
fn solver(n: usize) -> Solver {
    let sn = SupernodeParams {
        nemin: 1,
        ..SupernodeParams::default()
    };
    Solver::with_params(NumericParams::default(), sn)
        .with_ordering(OrderingMethod::External((0..n).collect()))
}

struct Outcome {
    inertia: (usize, usize, usize),
    n_cb: usize,
    max_delayed_in: usize,
    first_cb_chain_index: Option<usize>,
}

fn factor(d: usize, chain_len: usize) -> (Outcome, (usize, usize, usize)) {
    let (a, want) = delay_chain_kkt(d, chain_len);
    let mut s = solver(a.n);
    let st = s.factor(&a, None);
    assert!(matches!(st, FactorStatus::Success), "factor: {st:?}");
    let i = s.inertia().expect("inertia").clone();
    let f = s.factors().expect("factors");
    let max_delayed_in = f
        .node_factors
        .iter()
        .map(|nf| nf.n_delayed_in)
        .max()
        .unwrap_or(0);
    let first_cb_chain_index = f
        .node_factors
        .iter()
        .find(|nf| nf.cascade_break)
        .map(|nf| nf.first_col - d);
    let stats = s.last_factor_stats().expect("stats");
    assert_eq!(stats.n_cascade_break_nodes, f.n_cascade_break_nodes());
    (
        Outcome {
            inertia: (i.positive, i.negative, i.zero),
            n_cb: stats.n_cascade_break_nodes,
            max_delayed_in,
            first_cb_chain_index,
        },
        want,
    )
}

/// `d = 100` delays carried along a 220-front chain. The excess reaches
/// 2^20 near chain index 170 (about `Σ_{j<100} j² + 70·101·103`), while
/// the subtree's symbolic area stays under 10^3, so the backstop fires
/// there. The inertia must still match the saddle-point theorem.
#[test]
fn delay_chain_backstop_fires_on_a_long_chain() {
    let (out, want) = factor(100, 220);
    assert_eq!(out.inertia, want, "saddle-point inertia (L + 2, d, 0)");
    assert!(
        out.max_delayed_in >= 100,
        "the chain carries every delayed column (max in {})",
        out.max_delayed_in
    );
    assert!(out.n_cb >= 1, "the chain backstop must fire");
    let at = out.first_cb_chain_index.expect("a cascade-break front");
    assert!(
        (140..=200).contains(&at),
        "fires where the excess first passes 2^20 (chain index {at})"
    );
}

/// Same delays, a chain too short for the excess to reach
/// `DELAY_CHAIN_FLOOR` (about 333K + 30·10.4K < 2^20): the delays ride to
/// the hub and no front engages cascade-break.
#[test]
fn delay_chain_backstop_does_not_fire_below_the_floor() {
    let (out, want) = factor(100, 130);
    assert_eq!(out.inertia, want, "saddle-point inertia (L + 2, d, 0)");
    assert!(out.max_delayed_in >= 100, "delays are carried");
    assert_eq!(out.n_cb, 0, "below DELAY_CHAIN_FLOOR: no backstop");
}
