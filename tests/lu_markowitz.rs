//! Threshold-Markowitz LU (`SparseLu::factor_markowitz`, issue #167).
//!
//! The shipped path picks a *static* fill-reducing column order (AMD on the AᵀA
//! pattern) and then pivots for stability inside that order. Markowitz instead
//! picks each pivot dynamically to minimise `(r_i-1)(c_j-1)` subject to
//! `|a_ij| >= u·max_k|a_kj|`. It is a different factorization path, not a
//! different ordering, so the contract it has to meet is the *same output
//! object*: `P B Q = L U` in the same `L`/`U`/`perm`/`qcol` coordinates every
//! other consumer already reads.
//!
//! The fill claim itself (geomean 1.11x against the shipped 3.00x on 16 real LP
//! bases) is measured out-of-tree against the Python oracle in
//! `dev/probes/markowitz-fill/`; those bases are not committed. What is checked
//! here is the part a corpus cannot check: that the factor is *right*, that the
//! threshold bound on `max|L|` holds, and that on structures where the dynamic
//! choice is provably better it actually finds it.

use feral::{FeralError, LuParams, LuPivoting, SparseColMatrix, SparseLu, SparseLuSymbolic};

/// Deterministic LCG — no rand dependency, and reproducible failures.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n
    }
}

fn build(m: usize, trip: &mut Vec<(usize, usize, f64)>) -> SparseColMatrix {
    trip.sort_by_key(|&(i, j, _)| (j, i));
    trip.dedup_by_key(|&mut (i, j, _)| (i, j));
    let mut col_ptr = vec![0usize; m + 1];
    for &(_, j, _) in trip.iter() {
        col_ptr[j + 1] += 1;
    }
    for j in 0..m {
        col_ptr[j + 1] += col_ptr[j];
    }
    SparseColMatrix {
        m,
        col_ptr,
        row_idx: trip.iter().map(|&(i, _, _)| i).collect(),
        values: trip.iter().map(|&(_, _, v)| v).collect(),
    }
}

fn dense_of(a: &SparseColMatrix) -> Vec<Vec<f64>> {
    let mut d = vec![vec![0.0; a.m]; a.m];
    for (j, w) in a.col_ptr.windows(2).enumerate() {
        for idx in w[0]..w[1] {
            d[a.row_idx[idx]][j] = a.values[idx];
        }
    }
    d
}

/// `max |(P B Q)_ij - (L U)_ij| / max|B|` — the correctness gate. A fill number
/// from a wrong factorization is worthless, so every fill assertion below is
/// paired with this.
fn residual(a: &SparseColMatrix, lu: &SparseLu) -> f64 {
    let m = a.m;
    let d = dense_of(a);
    let (perm, qcol) = (lu.perm(), lu.qcol());
    let mut worst = 0.0_f64;
    let mut amax = 0.0_f64;
    for row in d.iter() {
        for &v in row.iter() {
            amax = amax.max(v.abs());
        }
    }
    for i in 0..m {
        for j in 0..m {
            // (L U)[i][j] = sum_k L[i][k] U[k][j], both triangular.
            let mut s = 0.0;
            for k in 0..=i.min(j) {
                let l = lu.l_dense(i, k);
                if l != 0.0 {
                    s += l * lu.u_dense(k, j);
                }
            }
            worst = worst.max((d[perm[i]][qcol[j]] - s).abs());
        }
    }
    worst / amax.max(f64::MIN_POSITIVE)
}

fn max_abs_l(lu: &SparseLu) -> f64 {
    let m = lu.dim();
    let mut mx = 0.0_f64;
    for j in 0..m {
        for i in (j + 1)..m {
            mx = mx.max(lu.l_dense(i, j).abs());
        }
    }
    mx
}

/// A 4x4 worked by hand. Column 2 is a singleton (row 3), so Markowitz must take
/// it first at cost 0 — before anything that would fill.
#[test]
fn hand_matrix_factors_exactly() {
    let mut t = vec![
        (0, 0, 2.0),
        (1, 0, 1.0),
        (2, 0, 4.0),
        (0, 1, 3.0),
        (2, 1, 1.0),
        (3, 2, 5.0),
        (1, 3, 7.0),
        (3, 3, 2.0),
    ];
    let a = build(4, &mut t);
    let lu = SparseLu::factor_markowitz(&a, LuParams::default()).expect("factor");
    assert!(residual(&a, &lu) < 1e-14, "residual {}", residual(&a, &lu));
    // Column 2 has a single entry, so it is the cost-0 pivot and goes first.
    assert_eq!(lu.qcol()[0], 2, "qcol {:?}", lu.qcol());
    assert_eq!(lu.perm()[0], 3, "perm {:?}", lu.perm());
}

/// Arrow with the dense row and column at index 0. Eliminating in natural order
/// fills the entire trailing block; eliminating the trailing diagonal first
/// fills nothing. Markowitz must find the second order, so `nnz(L+U)` is exactly
/// `nnz(A)`.
#[test]
fn arrow_takes_zero_fill_order() {
    let m = 200;
    let mut t = vec![(0, 0, 10.0)];
    for j in 1..m {
        t.push((0, j, 1.0 + j as f64 * 0.01)); // dense row 0
        t.push((j, 0, 1.0 - j as f64 * 0.003)); // dense column 0
        t.push((j, j, 3.0 + j as f64 * 0.05)); // diagonal
    }
    let a = build(m, &mut t);
    let nnz = a.nnz();
    let lu = SparseLu::factor_markowitz(&a, LuParams::default()).expect("factor");
    assert!(residual(&a, &lu) < 1e-13, "residual {}", residual(&a, &lu));
    assert_eq!(
        lu.factor_nnz(),
        nnz,
        "arrow must factor with zero fill, got {} for nnz(A) = {}",
        lu.factor_nnz(),
        nnz
    );
}

/// The same arrow through the shipped static path, as the contrast the issue is
/// about. This is not a claim about AMD in general — it is the demonstration
/// that a static column order *can* be beaten arbitrarily badly, on a structure
/// where the dynamic choice is obvious.
#[test]
fn markowitz_fill_never_worse_than_static_on_arrow() {
    let m = 200;
    let mut t = vec![(0, 0, 10.0)];
    for j in 1..m {
        t.push((0, j, 1.0 + j as f64 * 0.01));
        t.push((j, 0, 1.0 - j as f64 * 0.003));
        t.push((j, j, 3.0 + j as f64 * 0.05));
    }
    let a = build(m, &mut t);
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    let stat = SparseLu::factor(&a, &sym, LuParams::default()).expect("static factor");
    let mark = SparseLu::factor_markowitz(&a, LuParams::default()).expect("markowitz factor");
    assert!(
        mark.factor_nnz() <= stat.factor_nnz(),
        "markowitz {} must not exceed static {}",
        mark.factor_nnz(),
        stat.factor_nnz()
    );
}

/// Random sparse matrices: `P B Q = L U`, and the threshold bound `max|L| <= 1/u`
/// that is the entire stability argument for accepting a non-maximal pivot.
#[test]
fn random_sparse_residual_and_l_bound() {
    for seed in 0..12u64 {
        let m = 60 + (seed as usize) * 7;
        let mut rng = Rng(seed * 7919 + 13);
        let mut t: Vec<(usize, usize, f64)> = Vec::new();
        for j in 0..m {
            t.push((j, j, 2.0 + rng.next_f64().abs() * 3.0)); // keep it nonsingular
            for _ in 0..4 {
                let i = rng.below(m);
                let v = rng.next_f64();
                if v != 0.0 {
                    t.push((i, j, v));
                }
            }
        }
        let a = build(m, &mut t);
        let p = LuParams {
            markowitz_threshold: 0.1,
            ..Default::default()
        };
        let lu = SparseLu::factor_markowitz(&a, p).expect("factor");
        let r = residual(&a, &lu);
        assert!(r < 1e-12, "seed {} residual {}", seed, r);
        let ml = max_abs_l(&lu);
        assert!(
            ml <= 1.0 / 0.1 + 1e-9,
            "seed {} max|L| {} exceeds 1/u = 10",
            seed,
            ml
        );
    }
}

/// A tighter threshold must bound `max|L|` more tightly. If it does not, the
/// threshold test is not actually being applied.
#[test]
fn threshold_controls_l_growth() {
    let m = 120;
    let mut rng = Rng(4242);
    let mut t: Vec<(usize, usize, f64)> = Vec::new();
    for j in 0..m {
        t.push((j, j, 0.5 + rng.next_f64().abs()));
        for _ in 0..5 {
            let i = rng.below(m);
            t.push((i, j, rng.next_f64() * 4.0));
        }
    }
    let a = build(m, &mut t);
    for &u in &[0.01_f64, 0.1, 0.5, 1.0] {
        let p = LuParams {
            markowitz_threshold: u,
            ..Default::default()
        };
        let lu = SparseLu::factor_markowitz(&a, p).expect("factor");
        assert!(residual(&a, &lu) < 1e-12, "u {} residual", u);
        assert!(
            max_abs_l(&lu) <= 1.0 / u + 1e-9,
            "u {} gave max|L| {}",
            u,
            max_abs_l(&lu)
        );
    }
}

/// The factor has to be usable, not just correct on paper: solve through it and
/// check `A x = b` in the original coordinates.
#[test]
fn solves_through_the_markowitz_factor() {
    let m = 90;
    let mut rng = Rng(99);
    let mut t: Vec<(usize, usize, f64)> = Vec::new();
    for j in 0..m {
        t.push((j, j, 3.0 + rng.next_f64().abs()));
        for _ in 0..3 {
            t.push((rng.below(m), j, rng.next_f64()));
        }
    }
    let a = build(m, &mut t);
    let x_true: Vec<f64> = (0..m).map(|_| rng.next_f64()).collect();
    let d = dense_of(&a);
    let b: Vec<f64> = (0..m)
        .map(|i| (0..m).map(|j| d[i][j] * x_true[j]).sum())
        .collect();

    let mut lu = SparseLu::factor_markowitz(&a, LuParams::default()).expect("factor");
    let mut x = b.clone();
    lu.ftran(&mut x).expect("ftran");
    let err = x
        .iter()
        .zip(x_true.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(err < 1e-9, "ftran error {}", err);

    // Btran against the transpose, same matrix.
    let y_true: Vec<f64> = (0..m).map(|_| rng.next_f64()).collect();
    let c: Vec<f64> = (0..m)
        .map(|j| (0..m).map(|i| d[i][j] * y_true[i]).sum())
        .collect();
    let mut y = c.clone();
    lu.btran(&mut y).expect("btran");
    let err = y
        .iter()
        .zip(y_true.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(err < 1e-9, "btran error {}", err);
}

/// A structurally singular basis (an empty column) must be reported as
/// `SingularBasis` with the *original* column index, matching what
/// `SparseLu::factor` promises so a simplex driver can repair the basis.
#[test]
fn structurally_singular_reports_the_original_column() {
    let m = 20;
    let mut t: Vec<(usize, usize, f64)> = Vec::new();
    for j in 0..m {
        if j == 7 {
            continue; // column 7 is empty
        }
        t.push((j, j, 1.0 + j as f64));
    }
    let a = build(m, &mut t);
    match SparseLu::factor_markowitz(&a, LuParams::default()) {
        Err(FeralError::SingularBasis { column }) => assert_eq!(column, 7),
        other => panic!(
            "expected SingularBasis {{ column: 7 }}, got {:?}",
            other.is_ok()
        ),
    }
}

/// Exact cancellation during the update must *drop* the entry, not leave a
/// stored zero — otherwise `rcnt`/`ccnt` drift and every subsequent Markowitz
/// cost is computed against a structure that is not the active submatrix.
///
/// Two assertions, and the first is what keeps the second from being vacuous:
/// replaying the elimination *symbolically* on the pivot order the factor chose
/// gives the fill a no-cancellation implementation would produce, so
/// `factor_nnz() < structural` proves entries really did cancel here. Then
/// `factor_nnz() == nonzeros` proves they were dropped rather than stored as
/// zeros.
#[test]
fn exact_cancellation_is_dropped_not_stored() {
    // Power-of-two off-diagonals: every multiplier is exact in binary, so an
    // update that lands on zero lands on it exactly rather than at 1e-17. Seed
    // 11 of this family cancels 23 entries, which is what the first assertion
    // below pins.
    let m = 24;
    let mut rng = Rng(11 * 104729 + 7);
    let mut t: Vec<(usize, usize, f64)> = Vec::new();
    for j in 0..m {
        t.push((j, j, 4.0));
        for _ in 0..3 {
            let i = rng.below(m);
            let v = [1.0_f64, 2.0, 4.0, -1.0, -2.0, -4.0][rng.below(6)];
            t.push((i, j, v));
        }
    }
    let a = build(m, &mut t);
    let lu = SparseLu::factor_markowitz(&a, LuParams::default()).expect("factor");
    assert!(residual(&a, &lu) < 1e-11, "residual {}", residual(&a, &lu));

    // Symbolic replay of the same pivot sequence: no cancellation possible.
    let d = dense_of(&a);
    let mut pat = vec![vec![false; m]; m];
    for (i, row) in pat.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = d[i][j] != 0.0;
        }
    }
    let mut alive_r = vec![true; m];
    let mut alive_c = vec![true; m];
    let mut structural = 0usize;
    for k in 0..m {
        let (pi, pj) = (lu.perm()[k], lu.qcol()[k]);
        let rows: Vec<usize> = (0..m)
            .filter(|&i| alive_r[i] && i != pi && pat[i][pj])
            .collect();
        let colsr: Vec<usize> = (0..m)
            .filter(|&j| alive_c[j] && j != pj && pat[pi][j])
            .collect();
        structural += rows.len() + colsr.len() + 1; // L column, U row, diagonal
        for &i in rows.iter() {
            for &j in colsr.iter() {
                pat[i][j] = true;
            }
        }
        alive_r[pi] = false;
        alive_c[pj] = false;
    }

    let mut nonzero = 0usize;
    for i in 0..m {
        for j in 0..m {
            if (i > j && lu.l_dense(i, j) != 0.0) || (i <= j && lu.u_dense(i, j) != 0.0) {
                nonzero += 1;
            }
        }
    }
    assert!(
        lu.factor_nnz() < structural,
        "no cancellation happened on this matrix ({} == structural {}), so the \
         stored-zero check below would be vacuous",
        lu.factor_nnz(),
        structural
    );
    assert_eq!(
        lu.factor_nnz(),
        nonzero,
        "factor stores {} entries but only {} are nonzero — a cancelled entry \
         was kept",
        lu.factor_nnz(),
        nonzero
    );
}

/// `LuParams::pivoting` defaults to `Markowitz` (issue #171), and the choice is
/// observable rather than inferred.
///
/// Both routes are silent — `factor` returns the same type either way — so a
/// test that only checked the answer would pass whichever rule ran. That is the
/// failure mode #168 recorded, hence the `used_markowitz()` assertions and the
/// arrow-matrix fill witness: the two rules must disagree on this matrix, or the
/// routing assertions would be vacuous.
#[test]
fn default_params_route_through_markowitz_and_the_choice_is_observable() {
    // Arrow with the dense row/column first: a static order that eliminates the
    // tip first fills in completely, Markowitz takes it last and fills nothing.
    let m = 40;
    let mut t: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..m {
        t.push((i, 0, if i == 0 { 4.0 } else { 1.0 }));
    }
    for j in 1..m {
        t.push((0, j, 1.0));
        t.push((j, j, 3.0));
    }
    let a = build(m, &mut t);
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");

    let dflt = SparseLu::factor(&a, &sym, LuParams::default()).expect("factor default");
    assert!(
        dflt.used_markowitz(),
        "LuParams::default() must route through Markowitz (#171)"
    );

    let gp = SparseLu::factor(
        &a,
        &sym,
        LuParams {
            pivoting: LuPivoting::GilbertPeierls,
            ..Default::default()
        },
    )
    .expect("factor gilbert-peierls");
    assert!(
        !gp.used_markowitz(),
        "an explicit GilbertPeierls request must not silently take the Markowitz route"
    );

    // The witness: without a fill gap the two assertions above could hold on a
    // factorization that ignored the rule entirely.
    assert!(
        dflt.factor_nnz() < gp.factor_nnz(),
        "the two rules must produce different fill on this arrow matrix, else the \
         routing assertions are vacuous (markowitz {} vs gilbert-peierls {})",
        dflt.factor_nnz(),
        gp.factor_nnz()
    );

    // Routing is not allowed to cost correctness: both must still solve.
    let d = dense_of(&a);
    let x_true: Vec<f64> = (0..m).map(|i| 1.0 + i as f64).collect();
    let b: Vec<f64> = (0..m)
        .map(|i| (0..m).map(|j| d[i][j] * x_true[j]).sum())
        .collect();
    for lu in [dflt, gp].iter_mut() {
        let mut x = b.clone();
        lu.ftran(&mut x).expect("ftran");
        let err = x
            .iter()
            .zip(x_true.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(err < 1e-9, "solve error {err}");
    }
}
