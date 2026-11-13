//! `SparseLu::refactor_markowitz` (issue #232): an in-place refactorization
//! that reuses the previous factor's memory must be indistinguishable from a
//! fresh `SparseLu::factor_markowitz` of the same matrix.
//!
//! The oracle is the fresh factor itself, compared bitwise: pivots, every `L`
//! and `U` entry, every counter the public API exposes, and then the results
//! of the *same* script of solves and Forrest–Tomlin updates run on both.
//! Before the refactor, the reused factor is driven through updates and
//! dense and sparse solves, so its etas, pooled rows, reach and hyper-sparse
//! workspaces and lazily built `L` row index all hold state that a correct
//! refill has to discard.

use feral::{FeralError, LuParams, SparseColMatrix, SparseLu};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn f64(&mut self) -> f64 {
        (self.next() as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Near-triangular basis with a filling kernel, rows and columns scrambled —
/// the shape the in-place path exists for.
fn basis(m: usize, seed: u64) -> SparseColMatrix {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let k = m / 4;
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for (j, col) in cols.iter_mut().enumerate() {
        col.push((j, 3.0 + rng.f64().abs()));
        let (n, hi) = if j < k { (3, k) } else { (2, j.max(1)) };
        if j < k || rng.below(2) == 0 {
            for _ in 0..n {
                let i = rng.below(hi);
                let v = rng.f64();
                if v != 0.0 && !col.iter().any(|&(r, _)| r == i) {
                    col.push((i, v));
                }
            }
        }
    }
    let mut pr: Vec<usize> = (0..m).collect();
    let mut pc: Vec<usize> = (0..m).collect();
    for i in (1..m).rev() {
        pr.swap(i, rng.below(i + 1));
        pc.swap(i, rng.below(i + 1));
    }
    let mut out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for (j, col) in cols.iter().enumerate() {
        let mut c: Vec<(usize, f64)> = col.iter().map(|&(i, v)| (pr[i], v)).collect();
        c.sort_by_key(|&(i, _)| i);
        out[pc[j]] = c;
    }
    SparseColMatrix::from_sparse_columns(m, &out).expect("basis")
}

/// Everything observable about a factorization, as exact bits.
fn fingerprint(lu: &SparseLu) -> Vec<u64> {
    let m = lu.dim();
    let mut f: Vec<u64> = vec![m as u64];
    f.extend(lu.perm().iter().map(|&x| x as u64));
    f.extend(lu.qcol().iter().map(|&x| x as u64));
    f.push(lu.factor_nnz() as u64);
    f.push(lu.growth().to_bits());
    f.push(lu.u_max0().to_bits());
    f.push(lu.eta_ops() as u64);
    f.push(lu.last_eta_ops() as u64);
    f.push(lu.update_work() as u64);
    f.push(lu.last_update_work() as u64);
    f.push(lu.updates_since_refactor() as u64);
    f.push(lu.pivot_search_swaps() as u64);
    f.push(lu.hyper_sparse_sweeps() as u64);
    f.push(lu.hyper_sparse_nodes() as u64);
    f.push(lu.sparse_rhs_fallbacks() as u64);
    f.push(lu.last_sparse_solve_work() as u64);
    f.push(lu.bump_dim() as u64);
    f.push(lu.used_markowitz() as u64);
    f.push(lu.used_dense_bump() as u64);
    f.push(lu.reach_visits() as u64);
    f.push(lu.should_refactor() as u64);
    for i in 0..m {
        for j in 0..m {
            f.push(lu.l_dense(i, j).to_bits());
            f.push(lu.u_dense(i, j).to_bits());
        }
    }
    f
}

/// One step of a deterministic script: a dense and a sparse solve each way,
/// then a column replacement. Returns the bits of every result, the update's
/// outcome included (an `Err` must match too).
fn step(lu: &mut SparseLu, rng: &mut Rng) -> Vec<u64> {
    let m = lu.dim();
    let mut out = Vec::new();
    let mut x: Vec<f64> = (0..m).map(|_| rng.f64()).collect();
    out.push(format!("{:?}", lu.ftran(&mut x)).len() as u64);
    out.extend(x.iter().map(|v| v.to_bits()));
    let mut y: Vec<f64> = (0..m).map(|_| rng.f64()).collect();
    out.push(format!("{:?}", lu.btran(&mut y)).len() as u64);
    out.extend(y.iter().map(|v| v.to_bits()));

    let r = rng.below(m);
    let mut s = Vec::new();
    out.push(format!("{:?}", lu.ftran_sparse(&[(r, 1.0)], &mut s)).len() as u64);
    out.extend(s.iter().flat_map(|&(i, v)| [i as u64, v.to_bits()]));
    out.push(format!("{:?}", lu.btran_sparse(&[(r, -2.0)], &mut s)).len() as u64);
    out.extend(s.iter().flat_map(|&(i, v)| [i as u64, v.to_bits()]));

    // A sparse entering column with a dominant entry, so most updates commit;
    // whichever way each one goes, both factors must go the same way.
    let slot = rng.below(m);
    let mut col: Vec<(usize, f64)> = vec![(rng.below(m), 4.0 + rng.f64())];
    for _ in 0..3 {
        let i = rng.below(m);
        if !col.iter().any(|&(r, _)| r == i) {
            col.push((i, rng.f64()));
        }
    }
    col.sort_by_key(|&(i, _)| i);
    let res = lu.update_sparse(slot, &col);
    out.push(match res {
        Ok(()) => 0,
        Err(FeralError::NeedsRefactor) => 1,
        Err(_) => 2,
    });
    out
}

/// Dirty `lu` with `n` script steps (results discarded).
fn churn(lu: &mut SparseLu, n: usize, seed: u64) {
    let mut rng = Rng(seed);
    for _ in 0..n {
        step(lu, &mut rng);
    }
}

/// `reused` (already refactored in place onto `a`) against a fresh factor of
/// `a`: equal fingerprints now, and equal results under the same script.
fn assert_matches_fresh(reused: &mut SparseLu, a: &SparseColMatrix, params: &LuParams, what: &str) {
    let mut fresh = SparseLu::factor_markowitz(a, params.clone()).expect("fresh factor");
    assert!(
        fingerprint(reused) == fingerprint(&fresh),
        "{what}: state after refactor differs from a fresh factor"
    );
    let (mut r1, mut r2) = (Rng(77), Rng(77));
    for k in 0..12 {
        let a1 = step(reused, &mut r1);
        let a2 = step(&mut fresh, &mut r2);
        assert!(a1 == a2, "{what}: script step {k} results differ");
        assert!(
            fingerprint(reused) == fingerprint(&fresh),
            "{what}: state after script step {k} differs"
        );
    }
}

#[test]
fn refactor_after_updates_matches_fresh_factor() {
    let params = LuParams::default();
    for seed in 0..4u64 {
        let m = 70 + 9 * seed as usize;
        let mut lu = SparseLu::factor_markowitz(&basis(m, seed), params.clone()).expect("factor");
        churn(&mut lu, 15, seed + 100);
        let b = basis(m, seed + 1000);
        lu.refactor_markowitz(&b, params.clone()).expect("refactor");
        assert_matches_fresh(&mut lu, &b, &params, &format!("seed {seed}"));
    }
}

#[test]
fn refactor_across_dimension_changes_matches_fresh_factor() {
    let params = LuParams::default();
    let mut lu = SparseLu::factor_markowitz(&basis(80, 1), params.clone()).expect("factor");
    for (step_no, &m) in [50usize, 120, 120, 31, 90].iter().enumerate() {
        churn(&mut lu, 8, step_no as u64);
        let b = basis(m, 500 + step_no as u64);
        lu.refactor_markowitz(&b, params.clone()).expect("refactor");
        assert_matches_fresh(&mut lu, &b, &params, &format!("m -> {m}"));
    }
}

/// The `L` row index is built at factor time when the reach route is on and
/// lazily on first sparse solve when it is off. A refactor that turns the
/// route off must not leave the previous factor's index behind for the lazy
/// path to mistake for this one's.
#[test]
fn refactor_switching_the_reach_route_matches_fresh_factor() {
    let on = LuParams::default();
    let off = LuParams {
        hyper_sparse_max_density: 0.0,
        ..LuParams::default()
    };
    let mut lu = SparseLu::factor_markowitz(&basis(64, 3), on.clone()).expect("factor");
    churn(&mut lu, 6, 1);
    let b = basis(64, 4);
    lu.refactor_markowitz(&b, off.clone()).expect("refactor");
    assert_matches_fresh(&mut lu, &b, &off, "on -> off");
    churn(&mut lu, 6, 2);
    let c = basis(64, 5);
    lu.refactor_markowitz(&c, on.clone()).expect("refactor");
    assert_matches_fresh(&mut lu, &c, &on, "off -> on");
}

/// A failed refactor leaves the previous factorization usable and unchanged.
#[test]
fn failed_refactor_leaves_the_previous_factor_intact() {
    let params = LuParams::default();
    let mut lu = SparseLu::factor_markowitz(&basis(60, 9), params.clone()).expect("factor");
    churn(&mut lu, 7, 3);
    let mut before = lu.clone();

    // Structurally singular: column 5 empty.
    let a = basis(60, 10);
    let mut cols: Vec<Vec<(usize, f64)>> = (0..60)
        .map(|j| {
            (a.col_ptr[j]..a.col_ptr[j + 1])
                .map(|k| (a.row_idx[k], a.values[k]))
                .collect()
        })
        .collect();
    cols[5].clear();
    let singular = SparseColMatrix::from_sparse_columns(60, &cols).expect("matrix");
    assert!(matches!(
        lu.refactor_markowitz(&singular, params.clone()),
        Err(FeralError::SingularBasis { .. })
    ));

    assert!(
        fingerprint(&lu) == fingerprint(&before),
        "state changed on Err"
    );
    let (mut r1, mut r2) = (Rng(5), Rng(5));
    for k in 0..6 {
        assert!(
            step(&mut lu, &mut r1) == step(&mut before, &mut r2),
            "step {k} after a failed refactor differs"
        );
    }
}

/// `SparseLu::refactor(a, symbolic)` — the existing entry point, and the one
/// the Python `LuFactor.refactor` calls — takes the in-place route under the
/// default Markowitz pivoting and must still equal a fresh `factor`.
#[test]
fn symbolic_refactor_entry_point_matches_fresh_factor() {
    use feral::SparseLuSymbolic;
    let params = LuParams::default();
    let a = basis(72, 21);
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    let mut lu = SparseLu::factor(&a, &sym, params.clone()).expect("factor");
    churn(&mut lu, 10, 4);
    let b = basis(72, 22);
    let sym_b = SparseLuSymbolic::analyze(&b).expect("analyze");
    lu.refactor(&b, &sym_b).expect("refactor");
    assert!(lu.used_markowitz());
    assert_matches_fresh(&mut lu, &b, &params, "refactor(a, symbolic)");

    // A symbolic of the wrong dimension is still rejected, and leaves `lu` as
    // it was.
    let before = fingerprint(&lu);
    let small = basis(40, 23);
    let sym_small = SparseLuSymbolic::analyze(&small).expect("analyze");
    assert!(matches!(
        lu.refactor(&b, &sym_small),
        Err(FeralError::DimensionMismatch { .. })
    ));
    assert!(fingerprint(&lu) == before);
}
