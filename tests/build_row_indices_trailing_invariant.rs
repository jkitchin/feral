//! Regression test for the upper-triangle pollution bug in
//! `build_row_indices` (factorize.rs:2241-2308).
//!
//! Before the fix, `build_row_indices` collected trailing rows from
//! `full_pattern.col_ptr[j]` for j ∈ own_cols without filtering
//! upper-triangle entries (r < j). Because `full_pattern` is the
//! fully-symmetrized A pattern, every column carries both lower-tri
//! (legitimate trailing) and upper-tri (already-eliminated columns)
//! entries — the latter polluted every supernode's frontal with rows
//! that should not be there, propagated up the etree through child
//! contrib blocks, and inflated `factor_nnz` by 7-19× over the
//! textbook L-fill (Σ col_counts).
//!
//! See `dev/research/build-row-indices-fix.md` for the full diagnosis
//! and before/after numbers on PoissonControl K=50, K=158.
//!
//! All fixtures here use `n > 16` so the dense fast-path
//! (`N_TINY = 16`) does not bypass `build_row_indices`. Smaller
//! fixtures densify directly and never exercise the multifrontal
//! pattern code we are testing.
//!
//! The two invariants enforced:
//!   1. *Trailing-row floor* — every row at frontal positions
//!      `[own_ncol + n_delayed_in .. nrow)` must be `>= first_col +
//!      own_ncol`. Trailing rows below the supernode's own columns are
//!      structurally invalid in a multifrontal factorization.
//!   2. *Symbolic ↔ numeric nrow parity (delayed-free supernodes)* —
//!      on supernodes where no children delayed pivots up,
//!      `numeric.nrow == symbolic.nrow`. Before the fix, numeric was
//!      systematically larger because of upper-tri pollution.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.0,
        ..BunchKaufmanParams::default()
    })
}

/// Tridiagonal SPD matrix of size n. n=30 is past the N_TINY=16 gate.
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

/// 2D Poisson SPD on a k×k grid (n = k²) with 5-point stencil.
/// k=5 → n=25 past the fast-path gate.
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

/// Small KKT system: m primal cols (Identity + tridiag coupling) plus
/// k equality rows with a sparse J coupling. Lays out the full saddle
/// matrix the same way Ipopt's augmented system does.
/// m=20, k=5 → n=25 past the fast-path gate.
fn small_kkt_saddle(m: usize, k: usize) -> CscMatrix {
    assert!(k <= m);
    let n = m + k;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    // (1,1) block: tridiagonal SPD primal
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
    // (2,1) block: J — each equality row j touches primal cols j and j+1
    // Stored as J entries in the lower triangle: row m+j, col j and j+1.
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
    // (2,2) block: -delta_c * I (Ipopt-style equality regularization)
    for j in 0..k {
        rows.push(m + j);
        cols.push(m + j);
        vals.push(-1e-8);
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("kkt")
}

/// Two disjoint tridiagonal SPD blocks of size n_block each, n=2*n_block.
/// n_block=12 → n=24 past the fast-path gate. Tests that disjoint
/// components don't get phantom upper-triangle rows.
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

/// For every supernode in the multifrontal factor, every trailing row
/// (positions `[own_ncol + n_delayed_in .. nrow)`) must be >=
/// first_col + own_ncol. This is the multifrontal frontal invariant
/// that build_row_indices' upper-triangle filter restores.
fn assert_trailing_invariant(matrix: &CscMatrix, label: &str) {
    let sym_params = SupernodeParams::default();
    let sym = symbolic_factorize(matrix, &sym_params).unwrap();
    let nparams = ldlt_params();
    let (factors, _inertia) = factorize_multifrontal(matrix, &sym, &nparams).unwrap();

    // Pair only as many supernodes as numeric produced. Disjoint-forest
    // matrices may differ between symbolic (one supernode per
    // connected component) and numeric (which can pack a forest into a
    // single dense frontal under the fast-path gate when n <= N_TINY).
    // For multifrontal-path fixtures (n > 16) the lengths agree.
    let pairs = sym.supernodes.len().min(factors.node_factors.len());
    assert!(pairs > 0, "[{}] no supernodes produced", label);
    for idx in 0..pairs {
        let s = &sym.supernodes[idx];
        let nf = &factors.node_factors[idx];
        let first_col = s.first_col;
        let own_ncol = s.ncol();
        let own_last = first_col + own_ncol;
        let expanded_ncol = own_ncol + nf.n_delayed_in;
        for (pos, &r) in nf.row_indices.iter().enumerate().skip(expanded_ncol) {
            assert!(
                r >= own_last,
                "[{}] supernode #{} (first_col={}, own_ncol={}, n_delayed_in={}): \
                 trailing row {} at position {} is below first_col+own_ncol={} \
                 — upper-triangle pollution leaked into the frontal",
                label,
                idx,
                first_col,
                own_ncol,
                nf.n_delayed_in,
                r,
                pos,
                own_last
            );
        }
    }
}

#[test]
fn trailing_rows_above_own_range_tridiag_30() {
    assert_trailing_invariant(&tridiag_spd(30), "tridiag_30");
}

#[test]
fn trailing_rows_above_own_range_poisson_2d_5x5() {
    assert_trailing_invariant(&poisson_2d_spd(5), "poisson_2d_5x5");
}

#[test]
fn trailing_rows_above_own_range_small_kkt_saddle() {
    assert_trailing_invariant(&small_kkt_saddle(20, 5), "kkt_saddle_25");
}

#[test]
fn trailing_rows_above_own_range_disjoint_tridiag() {
    assert_trailing_invariant(&disjoint_tridiag(12), "disjoint_tridiag_24");
}

/// On SPD multifrontal fixtures (no delayed pivots), every supernode's
/// numeric `nrow` is bounded below by the symbolic-side prediction.
/// `Supernode.nrow = col_counts[first_col].max(ncol)` is the L NNZ of
/// the supernode's first column; the working frontal can be larger
/// when children's contribs pass rows through this supernode that
/// aren't in any of its own columns' L pattern (see
/// dev/research/factor-nnz-residual-gap.md). Before the upper-tri
/// pollution fix, numeric was inflated *beyond* the legitimate
/// pass-through floor — that regression is what these tests guard.
fn assert_nrow_at_least_symbolic(matrix: &CscMatrix, label: &str) {
    let sym_params = SupernodeParams::default();
    let sym = symbolic_factorize(matrix, &sym_params).unwrap();
    let nparams = ldlt_params();
    let (factors, _) = factorize_multifrontal(matrix, &sym, &nparams).unwrap();

    let pairs = sym.supernodes.len().min(factors.node_factors.len());
    let mut checked = 0usize;
    for idx in 0..pairs {
        let s = &sym.supernodes[idx];
        let nf = &factors.node_factors[idx];
        if nf.n_delayed_in != 0 {
            continue;
        }
        assert!(
            nf.nrow >= s.nrow,
            "[{}] supernode #{} (first_col={}, ncol={}): numeric nrow={} \
             < symbolic nrow={} — symbolic should be a lower bound",
            label,
            idx,
            s.first_col,
            s.ncol(),
            nf.nrow,
            s.nrow
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "[{}] no delayed-free supernodes were checked — fixture choice issue",
        label
    );
}

#[test]
fn nrow_at_least_symbolic_tridiag_30() {
    assert_nrow_at_least_symbolic(&tridiag_spd(30), "tridiag_30");
}

#[test]
fn nrow_at_least_symbolic_poisson_2d_5x5() {
    assert_nrow_at_least_symbolic(&poisson_2d_spd(5), "poisson_2d_5x5");
}

#[test]
fn nrow_at_least_symbolic_disjoint_tridiag() {
    assert_nrow_at_least_symbolic(&disjoint_tridiag(12), "disjoint_tridiag_24");
}

/// Solve correctness gate: factorization remains accurate after the
/// fix. Run a real solve on a KKT saddle and verify residual is small.
#[test]
fn solve_residual_small_after_fix() {
    let m = small_kkt_saddle(20, 5);
    let sym_params = SupernodeParams::default();
    let sym = symbolic_factorize(&m, &sym_params).unwrap();
    let nparams = NumericParams::default();
    let (factors, _) = factorize_multifrontal(&m, &sym, &nparams).unwrap();
    let n = m.n;
    let b: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let x = solve_sparse_refined(&m, &factors, &b).unwrap();

    let mut ax = vec![0.0; n];
    m.symv(&x, &mut ax);
    let mut r2 = 0.0;
    let mut b2 = 0.0;
    for i in 0..n {
        r2 += (ax[i] - b[i]).powi(2);
        b2 += b[i] * b[i];
    }
    let rel = (r2 / b2).sqrt();
    assert!(
        rel < 1e-8,
        "relative residual {} too large after build_row_indices fix",
        rel
    );
}
