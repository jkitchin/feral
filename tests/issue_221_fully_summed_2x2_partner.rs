//! Issue #221: delayed-pivot cascade when a zero-diagonal constraint
//! column's largest entry couples to a hub row that is not yet fully
//! summed.
//!
//! ## The bug
//!
//! `scalar_pivot_step` chose the 2×2 partner of a column `k` that fails
//! the 1×1 test from two candidates only: BK's magnitude argmax `r` (if
//! fully summed) or the adjacent column `k+1` (if coupled, issue #46).
//! On brainpc3 (pounce, interior-point iterations at μ ≈ 2.5e-9) each
//! constraint row couples with weight ~600 to two parameter "hub" rows
//! that are eliminated at the root, and with weight ~1 to its matched
//! state variable, whose Hessian diagonal has fallen to ~1e-13. Once a
//! matched pair is split across supernodes the state variable delays
//! into the parent, where it lands at the end of the fully-summed range,
//! not next to its constraint. The constraint's argmax is the hub (not
//! fully summed), `k+1` is uncoupled, so neither candidate applied, rook
//! rescue walked the hub rows and failed, and the column delayed again
//! at every front up the chain: about 2000 delays carried through 428
//! fronts, 9.4 s against 0.04 s on feral 0.18.0.
//!
//! ## The fix
//!
//! When neither candidate applies, the kernel also tries the largest
//! entry among the fully-summed rows (`r_fs`), as SSIDS's TPP kernel does
//! (`ldlt_tpp.cxx`, `find_row_abs_max` over uneliminated fully-summed
//! columns only). It is accepted only if it passes the Duff-Reid growth
//! bound and the SSIDS det floor.
//!
//! ## Oracles
//!
//! - Frontal test: hand calculation (2×2 block `[[0,1],[1,δ]]`, its
//!   inverse and the hub's Schur complement).
//! - Sparse test: the saddle-point inertia theorem (Benzi, Golub &
//!   Liesen, Acta Numerica 2005, §3.4). `H` is SPD and `B` has full row
//!   rank, so the inertia is exactly `(nv, nc, 0)`. The fill and
//!   delayed-column bounds are measurement oracles. Before the fix,
//!   nnz(L) was 255,962 and grew quadratically with the chain length.

use feral::dense::factor::factor_frontal;
use feral::symbolic::SupernodeParams;
use feral::{BunchKaufmanParams, CscMatrix, FactorStatus, NumericParams, Solver, SymmetricMatrix};

/// pounce's threshold (`pounce-feral` default).
const U_POUNCE: f64 = 1e-8;

/// 4×4 front, 3 fully-summed columns, row 3 a hub (not fully summed):
///
/// ```text
///         c      v      y      | hub
///   c  [  0                    |       ]
///   v  [  0      4             |       ]
///   y  [  1      0      1e-13  |       ]
///  hub [  600    0      0      |  1e4  ]
/// ```
///
/// At `k = 0` BK's argmax is the hub row (600), which cannot pivot, and
/// `k+1 = v` is uncoupled. The matched partner `y` sits at `k+2`.
fn hub_dominated_front() -> SymmetricMatrix {
    let mut m = SymmetricMatrix::zeros(4);
    m.set(0, 0, 0.0);
    m.set(1, 1, 4.0);
    m.set(2, 0, 1.0);
    m.set(2, 2, 1e-13);
    m.set(3, 0, 600.0);
    m.set(3, 3, 1e4);
    m
}

#[test]
fn constraint_pairs_with_fully_summed_partner_past_k_plus_1() {
    let params = BunchKaufmanParams {
        pivot_threshold: U_POUNCE,
        ..BunchKaufmanParams::default()
    };
    let ff = factor_frontal(&hub_dominated_front(), 3, true, &params).expect("factor_frontal");

    // Before #221: c and y were both delayed (nelim = 1).
    assert_eq!(ff.nelim, 3, "all fully-summed columns eliminate");
    assert_eq!(ff.n_delayed, 0);

    // Hand oracle: eig([[0,1],[1,1e-13]]) = {+~1, -~1}, eig(4) = +4.
    assert_eq!(
        (ff.inertia.positive, ff.inertia.negative, ff.inertia.zero),
        (2, 1, 0)
    );

    // Hand oracle for the hub's Schur complement. With P = [[0,1],[1,δ]],
    // P⁻¹ = [[-δ,1],[1,0]], so 1e4 - [600,0]·P⁻¹·[600,0]ᵀ
    // = 1e4 + 600²·δ = 1e4 + 3.6e-8.
    assert_eq!(ff.contrib_dim, 1);
    let s = ff.contrib[0];
    let expect = 1e4 + 3.6e-8;
    assert!(
        (s - expect).abs() <= 1e-15 * expect,
        "hub Schur complement {s:e}, expected {expect:e}"
    );
}

/// Lower triangle of a brainpc-shaped KKT. States `y_0..y_T` (Hessian
/// diagonal `delta`), two parameter hubs (diagonal 1), and `T` dynamics
/// constraints `y_{t+1} - 0.897·y_t + 600·p_1 - 675·p_2 = 0` with a
/// structurally explicit zero diagonal. Returns `(matrix, nv, nc)`.
fn brainpc_like_kkt(t_steps: usize, delta: f64) -> (CscMatrix, usize, usize) {
    let ny = t_steps + 1;
    let (p1, p2) = (ny, ny + 1);
    let nv = ny + 2;
    let nc = t_steps;
    let n = nv + nc;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let mut push = |i: usize, j: usize, x: f64| {
        r.push(i.max(j));
        c.push(i.min(j));
        v.push(x);
    };
    for t in 0..ny {
        push(t, t, delta);
    }
    push(p1, p1, 1.0);
    push(p2, p2, 1.0);
    for t in 0..t_steps {
        let row = nv + t;
        push(row, row, 0.0);
        push(row, t + 1, 1.0);
        push(row, t, -0.897);
        push(row, p1, 600.0);
        push(row, p2, -675.0);
    }
    let a = CscMatrix::from_triplets(n, &r, &c, &v).expect("valid lower triangle");
    (a, nv, nc)
}

#[test]
fn brainpc_like_kkt_factors_without_delay_cascade() {
    let (a, nv, nc) = brainpc_like_kkt(500, 1e-13);
    let n = a.n;
    let mut np = NumericParams::default();
    np.bk.pivot_threshold = U_POUNCE;
    let mut solver = Solver::with_params(np, SupernodeParams::default());
    let status = solver.factor(&a, None);
    assert!(matches!(status, FactorStatus::Success), "{status:?}");

    let inertia = solver.inertia().expect("inertia");
    assert_eq!(
        (inertia.positive, inertia.negative, inertia.zero),
        (nv, nc, 0),
        "saddle-point inertia theorem"
    );

    let stats = solver.last_factor_stats().expect("stats");
    assert_eq!(stats.n_tiny, 0, "no pivot floored");
    // Measured: 19,359 after the fix, 255,962 before (≈ 255·n, growing
    // quadratically with the chain length).
    assert!(
        stats.nnz_l <= 40 * n,
        "nnz(L) = {} > 40·n = {}: delayed columns are being carried up the chain",
        stats.nnz_l,
        40 * n
    );
    let factors = solver.factors().expect("factors");
    let delayed_in: usize = factors.node_factors.iter().map(|f| f.n_delayed_in).sum();
    assert!(
        delayed_in <= n,
        "{delayed_in} delayed-column arrivals: each column should delay at most about once"
    );

    // Accuracy: refined solve against b = A·1.
    let ones = vec![1.0; n];
    let mut b = vec![0.0; n];
    a.symv(&ones, &mut b);
    let x = solver.solve_refined(&a, &b).expect("solve");
    let err = x.iter().map(|xi| (xi - 1.0).abs()).fold(0.0, f64::max);
    let mut ax = vec![0.0; n];
    a.symv(&x, &mut ax);
    let res = ax
        .iter()
        .zip(&b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    let bn = b.iter().map(|q| q.abs()).fold(0.0, f64::max);
    assert!(
        res <= 1e-12 * bn,
        "residual {res:e} (|b| = {bn:e}), forward err {err:e}"
    );
}
