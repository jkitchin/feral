//! Differential test for the post-triangularization dense-bump route
//! (`LuParams::dense_bump_max_dim`).
//!
//! The dense route factors the peeled bump with the blocked dense kernel and
//! splices `L`/`U` back into the sparse structures. It must be
//! *indistinguishable* from the sparse route in what it computes: same solves,
//! same residuals, same singularity verdicts. These tests build matrices with a
//! genuine triangular border around a dense bump — the LP-basis shape the route
//! exists for — and check `A x = b` and `Aᵀ y = c` both ways.

use feral::{FeralError, LuParams, LuSingularAction, SparseColMatrix, SparseLu, SparseLuSymbolic};

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

/// `nfront` column singletons, a dense `bump x bump` block, `nback` row
/// singletons — a caricature of a simplex basis, permuted so nothing is in
/// natural order.
fn lp_like_basis(nfront: usize, bump: usize, nback: usize, seed: u64) -> SparseColMatrix {
    let m = nfront + bump + nback;
    let mut rng = Rng(seed);
    // Scramble the identity of rows and columns.
    let mut rows: Vec<usize> = (0..m).collect();
    let mut cols: Vec<usize> = (0..m).collect();
    for i in (1..m).rev() {
        rows.swap(i, rng.below(i + 1));
        cols.swap(i, rng.below(i + 1));
    }
    let mut trip: Vec<(usize, usize, f64)> = Vec::new();
    // Front: column `cols[k]` has a single entry at row `rows[k]`.
    for k in 0..nfront {
        trip.push((rows[k], cols[k], 1.0 + rng.next_f64().abs()));
    }
    // Bump: dense block on rows/cols [nfront, nfront+bump).
    for jj in 0..bump {
        for ii in 0..bump {
            let v = rng.next_f64();
            let v = if ii == jj { v + 3.0 } else { v };
            trip.push((rows[nfront + ii], cols[nfront + jj], v));
        }
    }
    // Back: row `rows[k]` has a single entry, in column `cols[k]`.
    for k in (nfront + bump)..m {
        trip.push((rows[k], cols[k], 1.0 + rng.next_f64().abs()));
    }
    // Fill above the diagonal blocks: front rows may hold entries in bump and
    // back columns (the `U_FB`/`U_FK` blocks), bump rows in back columns.
    for &r in rows.iter().take(nfront) {
        for _ in 0..3 {
            let j = nfront + rng.below(bump + nback);
            trip.push((r, cols[j], rng.next_f64()));
        }
    }
    for ii in 0..bump {
        for _ in 0..2 {
            if nback == 0 {
                break;
            }
            let j = nfront + bump + rng.below(nback);
            trip.push((rows[nfront + ii], cols[j], rng.next_f64()));
        }
    }
    build(m, &mut trip)
}

fn matvec(a: &SparseColMatrix, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.m];
    for (j, &xj) in x.iter().enumerate() {
        for idx in a.col_ptr[j]..a.col_ptr[j + 1] {
            y[a.row_idx[idx]] += a.values[idx] * xj;
        }
    }
    y
}

fn matvec_t(a: &SparseColMatrix, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.m];
    for (j, yj) in y.iter_mut().enumerate() {
        let mut s = 0.0;
        for idx in a.col_ptr[j]..a.col_ptr[j + 1] {
            s += a.values[idx] * x[a.row_idx[idx]];
        }
        *yj = s;
    }
    y
}

fn rel_resid(r: &[f64], b: &[f64]) -> f64 {
    let nr = r.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let nb = b.iter().fold(0.0_f64, |m, &x| m.max(x.abs())).max(1e-300);
    nr / nb
}

/// Solve with both routes and return `(rel_ftran_resid, rel_btran_resid, bump)`.
fn both_routes(
    a: &SparseColMatrix,
    max_dim: usize,
    seed: u64,
) -> (f64, f64, usize, Vec<f64>, bool) {
    let m = a.m;
    let sym = SparseLuSymbolic::analyze(a).expect("analyze");
    let params = LuParams {
        dense_bump_max_dim: max_dim,
        ..LuParams::default()
    };
    let mut lu = SparseLu::factor(a, &sym, params).expect("factor");

    let mut rng = Rng(seed);
    let b: Vec<f64> = (0..m).map(|_| rng.next_f64()).collect();

    let mut x = b.clone();
    lu.ftran(&mut x).expect("ftran");
    let ax = matvec(a, &x);
    let rf: Vec<f64> = ax.iter().zip(b.iter()).map(|(p, q)| p - q).collect();

    let mut y = b.clone();
    lu.btran(&mut y).expect("btran");
    let aty = matvec_t(a, &y);
    let rb: Vec<f64> = aty.iter().zip(b.iter()).map(|(p, q)| p - q).collect();

    (
        rel_resid(&rf, &b),
        rel_resid(&rb, &b),
        sym.bump_hi - sym.bump_lo,
        x,
        lu.used_dense_bump(),
    )
}

#[test]
fn dense_bump_route_solves_as_accurately_as_the_sparse_route() {
    let mut checked = 0usize;
    for (nfront, bump, nback, seed) in [
        (40usize, 12usize, 8usize, 1u64),
        (100, 25, 30, 2),
        (5, 40, 5, 3),
        (200, 8, 0, 4),
        (0, 16, 0, 5),
        (60, 2, 60, 6),
    ] {
        let a = lp_like_basis(nfront, bump, nback, seed);
        let (sf, sb, nb_s, xs, used_s) = both_routes(&a, 0, 99);
        let (df, db, nb_d, xd, used_d) = both_routes(&a, 4096, 99);
        // Rule 6: prove the route under test actually ran. Without this the
        // whole comparison degrades to sparse-vs-sparse and passes vacuously.
        assert!(!used_s, "the sparse arm must not take the dense route");
        assert!(
            used_d,
            "the dense arm fell back to sparse (seed {seed}) - test is vacuous"
        );
        assert_eq!(nb_s, nb_d, "the peel must not depend on the numeric route");
        assert!(
            nb_d >= bump,
            "expected a bump of at least {bump}, peel left {nb_d} (seed {seed})"
        );
        // Both routes must be backward stable. The dense route reorders the
        // arithmetic, so the solutions differ in the last bits, not in value.
        assert!(
            sf < 1e-9 && sb < 1e-9,
            "sparse route unstable: {sf:e} {sb:e}"
        );
        assert!(
            df < 1e-9 && db < 1e-9,
            "dense route unstable: {df:e} {db:e}"
        );
        let dx = xs
            .iter()
            .zip(xd.iter())
            .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
        let nx = xs.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1e-300);
        assert!(
            dx / nx < 1e-8,
            "routes disagree on the solution: {:e} (seed {seed})",
            dx / nx
        );
        checked += 1;
    }
    // discopt CLAUDE.md rule 6: an executed-assertion count, not a silent pass.
    assert_eq!(checked, 6, "probe did not run every case");
}

#[test]
fn dense_bump_route_is_off_by_default() {
    let a = lp_like_basis(30, 10, 10, 7);
    assert_eq!(
        LuParams::default().dense_bump_max_dim,
        0,
        "the route must be opt-in"
    );
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    let lu = SparseLu::factor(&a, &sym, LuParams::default()).expect("factor");
    assert!(lu.factor_nnz() > 0);
}

#[test]
fn bump_above_the_cap_stays_on_the_sparse_route() {
    let a = lp_like_basis(20, 30, 10, 8);
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    let bump = sym.bump_hi - sym.bump_lo;
    assert!(bump > 4, "test needs a real bump, got {bump}");
    // A cap below the bump must not take the route; the factorization must
    // still be correct and identical to the plain sparse one.
    let small = LuParams {
        dense_bump_max_dim: bump - 1,
        ..LuParams::default()
    };
    let a_lu = SparseLu::factor(&a, &sym, small).expect("factor");
    let b_lu = SparseLu::factor(&a, &sym, LuParams::default()).expect("factor");
    assert!(
        !a_lu.used_dense_bump(),
        "cap below the bump must not take the route"
    );
    assert!(!b_lu.used_dense_bump());
    assert_eq!(
        a_lu.factor_nnz(),
        b_lu.factor_nnz(),
        "a below-cap bump must be byte-identical to the sparse route"
    );
}

#[test]
fn singular_bump_is_reported_on_both_routes() {
    // A bump with two identical columns is singular whichever kernel sees it.
    let m = 12;
    let mut trip: Vec<(usize, usize, f64)> = Vec::new();
    for k in 0..4 {
        trip.push((k, k, 2.0));
    }
    for jj in 0..8 {
        for ii in 0..8 {
            // Columns 4+6 and 4+7 are made identical below.
            let v = if ii == jj {
                3.0
            } else {
                0.5 + 0.01 * (ii + jj) as f64
            };
            trip.push((4 + ii, 4 + jj, v));
        }
    }
    // Force column 11 to equal column 10.
    for ii in 0..8 {
        let v = trip
            .iter()
            .find(|&&(i, j, _)| i == 4 + ii && j == 10)
            .map(|&(_, _, v)| v)
            .expect("col 10 entry");
        if let Some(e) = trip.iter_mut().find(|(i, j, _)| *i == 4 + ii && *j == 11) {
            e.2 = v;
        }
    }
    let a = build(m, &mut trip);
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    let base = LuParams {
        on_singular: LuSingularAction::Fail,
        ..LuParams::default()
    };
    let sparse = SparseLu::factor(&a, &sym, base.clone());
    let dense = SparseLu::factor(
        &a,
        &sym,
        LuParams {
            dense_bump_max_dim: 4096,
            ..base
        },
    );
    // Matching on `SingularBasis { .. }` alone passes vacuously against a route
    // that names the wrong column, so compare the columns themselves. Both must
    // report the *original basis column*, the contract
    // `tests/lu_sparse.rs::singular_basis_reports_original_column_not_factorization_position`
    // pins for the sparse path: `factorize_packed` names a block-local index,
    // which is only the same number by accident.
    let sparse_col = match sparse {
        Err(FeralError::SingularBasis { column }) => column,
        other => panic!("sparse route missed a singular bump: {other:?}"),
    };
    let dense_col = match dense {
        Err(FeralError::SingularBasis { column }) => column,
        other => panic!("dense route missed a singular bump: {other:?}"),
    };
    assert_eq!(
        sparse_col, dense_col,
        "routes disagree on which basis column is singular \
         (sparse={sparse_col}, dense={dense_col}); the dense route must remap \
         the block-local column through `qcol[bump_lo + k]`"
    );
    // Columns 10 and 11 are the identical pair, so the singularity must be
    // pinned to one of them and not to some unrelated position.
    assert!(
        sparse_col == 10 || sparse_col == 11,
        "expected the duplicated column (10 or 11), got {sparse_col}"
    );
}
