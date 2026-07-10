//! Issue #131 Gap A — contribution-block (tree-parallel) sparse solve.
//!
//! Contracts checked:
//!  1. **serial-CB == parallel-CB, byte-identical** (`to_bits`) — the
//!     #131 bit-exactness contract: the child-reduction order is fixed
//!     (ascending child index), so thread scheduling cannot change a bit.
//!  2. **determinism** — repeated parallel runs are byte-identical.
//!  3. **valid solve** — the CB solution's relative residual is tiny, and
//!     it matches the default `solve_sparse` to ~κ·eps (the CB forward
//!     groups contributions by subtree, a valid reassociation of the
//!     default's flat postorder fold).
//!  4. Coverage: SPD chains/grids, a KKT saddle that delays pivots (the
//!     data-flow risk area), and a disjoint forest (multiple roots).

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::{solve_sparse, solve_sparse_cb};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.0,
        ..BunchKaufmanParams::default()
    })
}

fn tridiag_spd(n: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(4.0);
        if i + 1 < n {
            rows.push(i + 1);
            cols.push(i);
            vals.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("tridiag")
}

fn poisson_2d_spd(k: usize) -> CscMatrix {
    let n = k * k;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
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

fn small_kkt_saddle(m: usize, k: usize) -> CscMatrix {
    assert!(k <= m);
    let n = m + k;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..m {
        rows.push(i);
        cols.push(i);
        vals.push(4.0);
        if i + 1 < m {
            rows.push(i + 1);
            cols.push(i);
            vals.push(-1.0);
        }
    }
    for j in 0..k {
        rows.push(m + j);
        cols.push(j);
        vals.push(1.0);
        if j + 1 < m {
            rows.push(m + j);
            cols.push(j + 1);
            vals.push(-1.0);
        }
    }
    for j in 0..k {
        rows.push(m + j);
        cols.push(m + j);
        vals.push(-1e-8);
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("kkt")
}

fn disjoint_tridiag(n_block: usize) -> CscMatrix {
    let n = 2 * n_block;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for block in 0..2 {
        let off = block * n_block;
        for i in 0..n_block {
            rows.push(off + i);
            cols.push(off + i);
            vals.push(4.0);
            if i + 1 < n_block {
                rows.push(off + i + 1);
                cols.push(off + i);
                vals.push(-1.0);
            }
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("disjoint_tridiag")
}

fn rel_residual(m: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = m.n;
    let mut ax = vec![0.0; n];
    m.symv(x, &mut ax);
    let mut r2 = 0.0;
    let mut b2 = 0.0;
    for i in 0..n {
        r2 += (ax[i] - b[i]).powi(2);
        b2 += b[i] * b[i];
    }
    (r2 / b2.max(1e-300)).sqrt()
}

fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..a.len() {
        num = num.max((a[i] - b[i]).abs());
        den = den.max(a[i].abs());
    }
    num / den.max(1.0)
}

fn check(matrix: &CscMatrix, label: &str) {
    let sym = symbolic_factorize(matrix, &SupernodeParams::default()).unwrap();
    let (factors, _) = factorize_multifrontal(matrix, &sym, &ldlt_params()).unwrap();
    let n = matrix.n;
    let b: Vec<f64> = (0..n).map(|i| 1.0 + 0.37 * (i % 7) as f64).collect();

    let x_default = solve_sparse(&factors, &b).unwrap();
    let x_cb_serial = solve_sparse_cb(&factors, &b, false).unwrap();
    let x_cb_par = solve_sparse_cb(&factors, &b, true).unwrap();
    let x_cb_par2 = solve_sparse_cb(&factors, &b, true).unwrap();

    // 1. serial-CB == parallel-CB, byte-identical.
    for i in 0..n {
        assert_eq!(
            x_cb_serial[i].to_bits(),
            x_cb_par[i].to_bits(),
            "[{label}] serial-CB vs parallel-CB differ at {i}"
        );
    }
    // 2. determinism across parallel runs.
    for i in 0..n {
        assert_eq!(
            x_cb_par[i].to_bits(),
            x_cb_par2[i].to_bits(),
            "[{label}] parallel-CB not deterministic at {i}"
        );
    }
    // 3a. CB is a valid solve — no worse than the proven default path
    // (both are unrefined solves of the same factor; an indefinite KKT
    // saddle has a raw residual set by its conditioning, not the solve).
    let res_cb = rel_residual(matrix, &x_cb_serial, &b);
    let res_def = rel_residual(matrix, &x_default, &b);
    assert!(
        res_cb <= 10.0 * res_def + 1e-12,
        "[{label}] CB residual {res_cb:e} >> default {res_def:e}"
    );
    // 3b. CB matches the default solve to a reassociation (~κ·eps). The
    // KKT saddle's small (2,2) block amplifies the reassociation, so scale
    // the bound by the default solve's own residual (its conditioning).
    let d = max_rel_diff(&x_default, &x_cb_serial);
    assert!(
        d < (1e-9_f64).max(100.0 * res_def),
        "[{label}] CB vs default solve diff too large: {d:e} (res_def {res_def:e})"
    );
}

#[test]
fn cb_parity_tridiag_120() {
    check(&tridiag_spd(120), "tridiag_120");
}

#[test]
fn cb_parity_poisson_2d_10x10() {
    check(&poisson_2d_spd(10), "poisson_2d_10x10");
}

#[test]
fn cb_parity_kkt_saddle_with_delays() {
    check(&small_kkt_saddle(60, 15), "kkt_saddle_75");
}

#[test]
fn cb_parity_disjoint_tridiag() {
    check(&disjoint_tridiag(40), "disjoint_tridiag_80");
}

#[test]
fn cb_parity_poisson_2d_96x96_concurrent() {
    // n = 9216: large + bushy enough to clear the `worthwhile` gate, so
    // this exercises the actual tree-parallel task path (not the serial
    // fallback) — the byte-identity check that matters most.
    check(&poisson_2d_spd(96), "poisson_2d_96x96");
}

#[test]
fn cb_parity_dense_fast_path() {
    // n <= N_TINY routes the single-supernode dense fast path
    // (node_parents = [None]); the CB solve must handle a lone root.
    check(&tridiag_spd(8), "tridiag_8_dense");
}
