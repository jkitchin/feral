//! Issue #125: precomputed static assembly maps
//! (`SymbolicFactorization::static_rows`).
//!
//! Two guarantees:
//!
//!  1. **A/B equivalence** — factoring with the static fast path enabled
//!     (`use_static_row_indices = true`, the default) produces a
//!     **byte-identical** factor to factoring with it disabled (which
//!     forces the `build_row_indices` path). This is the real
//!     correctness contract: the static layout must reproduce
//!     `build_row_indices`' output bit-for-bit on the no-delay fast
//!     path, and fall back cleanly on delayed-pivot fronts. Covers SPD,
//!     KKT-indefinite (delayed pivots), and disjoint-forest patterns.
//!
//!  2. **Independent structural oracle** — the static layout equals a
//!     from-scratch `BTreeSet`-based recompute over the *public*
//!     symbolic data (own-column pattern reach ∪ children's separators,
//!     filtered to `>= own_last`, sorted). This is a different
//!     implementation of the same spec than `compute_static_row_indices`
//!     (seen-array + flat CSR), so it catches an error in either.

use std::collections::BTreeSet;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams, SymbolicFactorization};
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.0,
        ..BunchKaufmanParams::default()
    })
}

// ----- fixtures (mirror build_row_indices_trailing_invariant.rs) -----

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

/// KKT saddle with equality regularization that exercises delayed
/// pivoting (the (2,2) block is tiny-negative, so BK will delay some
/// pivots up the tree — the path where the static map must fall back to
/// `build_row_indices`).
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

// ----- 1. A/B byte-identical equivalence -----

/// Factor twice — static fast path on vs off — and assert the two
/// factors are byte-identical (structure and every stored f64) and the
/// inertia matches.
fn assert_static_ab_identical(matrix: &CscMatrix, label: &str) {
    let sym = symbolic_factorize(matrix, &SupernodeParams::default()).unwrap();

    let mut p_on = ldlt_params();
    p_on.use_static_row_indices = true;
    let mut p_off = ldlt_params();
    p_off.use_static_row_indices = false;

    let (f_on, i_on) = factorize_multifrontal(matrix, &sym, &p_on).unwrap();
    let (f_off, i_off) = factorize_multifrontal(matrix, &sym, &p_off).unwrap();

    assert_eq!(
        i_on, i_off,
        "[{label}] inertia differs static-on vs static-off"
    );
    assert_eq!(
        f_on.node_factors.len(),
        f_off.node_factors.len(),
        "[{label}] node count differs"
    );
    for (idx, (a, b)) in f_on
        .node_factors
        .iter()
        .zip(f_off.node_factors.iter())
        .enumerate()
    {
        assert_eq!(
            a.row_indices, b.row_indices,
            "[{label}] node {idx} row_indices differ"
        );
        assert_eq!(a.nrow, b.nrow, "[{label}] node {idx} nrow differs");
        assert_eq!(a.nelim, b.nelim, "[{label}] node {idx} nelim differs");
        assert_eq!(
            a.n_delayed_in, b.n_delayed_in,
            "[{label}] node {idx} n_delayed_in differs"
        );
        let (fa, fb) = (&a.frontal_factors, &b.frontal_factors);
        // Bit-identical numeric arrays: the static layout must not
        // perturb a single stored value.
        assert!(
            fa.l.iter()
                .zip(fb.l.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "[{label}] node {idx} L differs bitwise"
        );
        assert!(
            fa.d_diag
                .iter()
                .zip(fb.d_diag.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "[{label}] node {idx} d_diag differs bitwise"
        );
        assert!(
            fa.d_subdiag
                .iter()
                .zip(fb.d_subdiag.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "[{label}] node {idx} d_subdiag differs bitwise"
        );
        assert_eq!(fa.perm, fb.perm, "[{label}] node {idx} perm differs");
    }
}

#[test]
fn static_ab_identical_tridiag_60() {
    assert_static_ab_identical(&tridiag_spd(60), "tridiag_60");
}

#[test]
fn static_ab_identical_poisson_2d_8x8() {
    assert_static_ab_identical(&poisson_2d_spd(8), "poisson_2d_8x8");
}

#[test]
fn static_ab_identical_kkt_saddle_with_delays() {
    // KKT with a negative (2,2) block: delayed pivots occur, so this
    // exercises both the static fast path (no-delay fronts) and the
    // build_row_indices fallback (delayed fronts) in one factor.
    assert_static_ab_identical(&small_kkt_saddle(40, 10), "kkt_saddle_50");
}

#[test]
fn static_ab_identical_disjoint_tridiag() {
    assert_static_ab_identical(&disjoint_tridiag(24), "disjoint_tridiag_48");
}

// ----- 2. Independent structural oracle -----

/// Recompute each supernode's static row layout from scratch over the
/// public symbolic data with a BTreeSet — a different implementation of
/// the same spec than `compute_static_row_indices`.
fn reference_static_rows(sym: &SymbolicFactorization) -> Vec<Vec<usize>> {
    let pat = &sym.permuted_pattern;
    let n_snodes = sym.supernodes.len();
    // Per-node separator (trailing slice) once computed.
    let mut sep: Vec<Vec<usize>> = vec![Vec::new(); n_snodes];
    let mut out: Vec<Vec<usize>> = Vec::with_capacity(n_snodes);
    for (idx, s) in sym.supernodes.iter().enumerate() {
        let first_col = s.first_col;
        let own_ncol = s.ncol();
        let own_last = first_col + own_ncol;
        let mut trailing: BTreeSet<usize> = BTreeSet::new();
        // Own columns' reach in the symmetric pattern.
        for j in first_col..own_last {
            for k in pat.col_ptr[j]..pat.col_ptr[j + 1] {
                let r = pat.row_idx[k];
                if r >= own_last {
                    trailing.insert(r);
                }
            }
        }
        // Children's separators.
        for &c in &s.children {
            if c >= n_snodes {
                continue;
            }
            for &r in &sep[c] {
                if r >= own_last {
                    trailing.insert(r);
                }
            }
        }
        let trailing_vec: Vec<usize> = trailing.into_iter().collect();
        sep[idx] = trailing_vec.clone();
        let mut rows: Vec<usize> = (first_col..own_last).collect();
        rows.extend_from_slice(&trailing_vec);
        out.push(rows);
    }
    out
}

fn assert_static_matches_reference(matrix: &CscMatrix, label: &str) {
    let sym = symbolic_factorize(matrix, &SupernodeParams::default()).unwrap();
    let reference = reference_static_rows(&sym);
    assert_eq!(
        reference.len(),
        sym.supernodes.len(),
        "[{label}] reference length mismatch"
    );
    for (idx, ref_rows) in reference.iter().enumerate() {
        let got = sym.static_rows(idx);
        assert_eq!(
            got,
            ref_rows.as_slice(),
            "[{label}] supernode {idx} static_rows mismatch\n  got: {got:?}\n  ref: {ref_rows:?}"
        );
        // Structural invariants: own cols first and in order; trailing
        // sorted and strictly above own range.
        let s = &sym.supernodes[idx];
        let own_last = s.first_col + s.ncol();
        for (k, &r) in got.iter().take(s.ncol()).enumerate() {
            assert_eq!(
                r,
                s.first_col + k,
                "[{label}] node {idx} own col {k} out of place"
            );
        }
        let trailing = &got[s.ncol()..];
        assert!(
            trailing.windows(2).all(|w| w[0] < w[1]),
            "[{label}] node {idx} trailing not strictly sorted: {trailing:?}"
        );
        assert!(
            trailing.iter().all(|&r| r >= own_last),
            "[{label}] node {idx} trailing has row below own_last={own_last}"
        );
    }
}

#[test]
fn static_matches_reference_tridiag_60() {
    assert_static_matches_reference(&tridiag_spd(60), "tridiag_60");
}

#[test]
fn static_matches_reference_poisson_2d_8x8() {
    assert_static_matches_reference(&poisson_2d_spd(8), "poisson_2d_8x8");
}

#[test]
fn static_matches_reference_kkt_saddle() {
    assert_static_matches_reference(&small_kkt_saddle(40, 10), "kkt_saddle_50");
}

#[test]
fn static_matches_reference_disjoint_tridiag() {
    assert_static_matches_reference(&disjoint_tridiag(24), "disjoint_tridiag_48");
}
