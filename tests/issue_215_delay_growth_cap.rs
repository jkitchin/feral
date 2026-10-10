//! Issue #215: a supernode over its delay budget engages cascade-break,
//! which floors threshold-failing pivots to `sign(d)·eps`. When `|d|` is
//! round-off, `sign(d)` is noise and the inertia count is wrong. The
//! budget used to be `4·own_ncol`, which thin supernodes in IPM KKTs
//! exceed with modest delay catchment. It now bounds front growth
//! (`symbolic::supernode::assign_delayed_capacities`), and on these
//! matrices CB must not fire at all.
//!
//! Fixtures (gitignored, so the tests skip when absent):
//! - `tests/data/large/issue215_column25_i011.mtx`: pounce#999
//!   `column25_integral` iteration 11, solve 2 (issue #215
//!   gh999-decisive-pair). Oracle: 75,596 negative eigenvalues (the
//!   reporter's scipy `eigsh`, nearest eigenvalue +3.47e-8; the congruent
//!   derivative-form matrix factors to the same count). Sylvester's law
//!   carries it to every `D·A·D` with `D > 0` diagonal.
//! - `data/matrices/kkt-mittelmann/qcqp1500-1nc/qcqp1500-1nc_0000.mtx`.
//!   Oracle: numpy `eigvalsh` on the dense matrix, 10,508 negative and
//!   1,500 positive (cluster of eigenvalues at -5.62e-9). MUMPS 5.8.2
//!   fails on this matrix.

use feral::numeric::factorize::NumericParams;
use feral::symbolic::{OrderingMethod, SupernodeParams};
use feral::{read_mtx, CscMatrix, FactorStatus, Inertia, Solver};
use std::path::Path;

/// pounce's configuration (the one the issue was reported under).
fn pounce_solver() -> Solver {
    let mut np = NumericParams::default();
    np.bk.pivot_threshold = 1e-8;
    Solver::with_params(np, SupernodeParams::default()).with_ordering(OrderingMethod::Auto)
}

fn load(path: &str) -> Option<CscMatrix> {
    let p = Path::new(path);
    if !p.is_file() {
        eprintln!("SKIP: {} not present.", p.display());
        return None;
    }
    Some(read_mtx(p).and_then(|m| m.to_csc()).expect("read fixture"))
}

/// `A ← D·A·D` with `d_i ∈ [0.7, 1.4]` from a fixed xorshift stream.
fn congruence(m: &mut CscMatrix, seed: u64) {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
    let d: Vec<f64> = (0..m.n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            0.7 + 0.7 * ((x >> 11) as f64 / (1u64 << 53) as f64)
        })
        .collect();
    for j in 0..m.n {
        for p in m.col_ptr[j]..m.col_ptr[j + 1] {
            m.values[p] *= d[j] * d[m.row_idx[p]];
        }
    }
}

fn factor_exact(m: &CscMatrix, want: Inertia, tag: &str) {
    let mut s = pounce_solver();
    assert!(
        matches!(s.factor(m, None), FactorStatus::Success),
        "{tag}: factor failed"
    );
    let st = s.last_factor_stats().expect("stats");
    assert_eq!(
        s.inertia(),
        Some(&want),
        "{tag}: wrong inertia, stats {st:?}"
    );
    assert_eq!(st.n_cascade_break_nodes, 0, "{tag}: cascade-break engaged");
    assert_eq!(st.inertia_uncertain(), 0, "{tag}: floored pivots");
}

/// Seeds 1, 4 and 8 counted +3, +1 and +1 under the `4·own_ncol` budget
/// (cb 12, 6 and 3 nodes) on an M4 Pro. ~10 s per factor, so ignored by
/// default: `cargo test --release --test issue_215_delay_growth_cap -- --ignored`.
#[test]
#[ignore]
fn column25_congruences_factor_without_cascade_break() {
    let Some(base) = load("tests/data/large/issue215_column25_i011.mtx") else {
        return;
    };
    let want = Inertia {
        positive: 75_796,
        negative: 75_596,
        zero: 0,
    };
    for seed in [1u64, 4, 8] {
        let mut m = base.clone();
        congruence(&mut m, seed);
        factor_exact(&m, want.clone(), &format!("column25 seed {seed}"));
    }
}

/// Under the old budget: correct, but with cb 16 and 2263 floored pivots.
#[test]
fn qcqp1500_1nc_factors_without_cascade_break() {
    let Some(m) = load("data/matrices/kkt-mittelmann/qcqp1500-1nc/qcqp1500-1nc_0000.mtx") else {
        return;
    };
    let want = Inertia {
        positive: 1_500,
        negative: 10_508,
        zero: 0,
    };
    factor_exact(&m, want, "qcqp1500-1nc_0000");
}
