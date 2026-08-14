//! Regression tests against **real** discopt simplex bases (issue #161B).
//!
//! These exist because the synthetic generator in `tests/lu_hyper_sparse.rs`
//! shipped a wrong default. `hyper_sparse_max_density` was set to `0.25` from a
//! sweep on that generator; on the real QPLIB_1157 basis `0.25` costs **1.6x on
//! `btran`**, and the generator structurally cannot show it — its solutions are
//! bimodal (tiny, or ~9% of `m`) with nothing in the 10–25% band where
//! QPLIB_1157's mean solution density of 14.7% lives.
//!
//! A parameter can only be validated against a basis whose solution density
//! straddles it. Hence real fixtures, and hence a guard on the *mechanism*
//! rather than on wall-clock:
//!
//! **The reach DFS costs about what the sweep it replaces costs.** Measured on
//! QPLIB_1157 at cap 0.25, the `btran` reach walks 68,925 graph edges per sweep
//! against `nnz(L) = 75,084` — 92% of what the dense `Lᵀ` sweep traverses — and
//! then the numeric sweep walks the reached columns *again*. Paying ~1.9x to
//! avoid 1x is the regression. At the shipped default the same measurement is
//! 5%. That ratio is deterministic; the timing that follows from it is not, so
//! these tests assert the former.
//!
//! Fixtures: `tests/data/lu_bases/` (see its README for provenance). Tests skip
//! with a `SKIP:` line if absent, matching the convention in
//! `tests/amf_corpus_oracle.rs`.

use std::path::PathBuf;

use feral::{LuParams, SparseColMatrix, SparseLu, SparseLuSymbolic};

fn fixture(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/lu_bases")
        .join(name);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("SKIP: {} not found", p.display());
        None
    }
}

/// Matrix Market coordinate reader, same shape as the one in
/// `examples/basis_refactor.rs`.
fn read_mtx(path: &PathBuf) -> SparseColMatrix {
    let text = std::fs::read_to_string(path).expect("read mtx");
    let mut lines = text.lines().filter(|l| !l.starts_with('%'));
    let hdr: Vec<usize> = lines
        .next()
        .expect("header")
        .split_whitespace()
        .map(|t| t.parse().expect("header int"))
        .collect();
    let (m, nnz) = (hdr[0], hdr[2]);
    let mut trip: Vec<(usize, usize, f64)> = Vec::with_capacity(nnz);
    for l in lines {
        let mut it = l.split_whitespace();
        let i: usize = it.next().expect("row").parse().expect("row int");
        let j: usize = it.next().expect("col").parse().expect("col int");
        let v: f64 = it.next().expect("val").parse().expect("val f64");
        trip.push((i - 1, j - 1, v));
    }
    assert_eq!(trip.len(), nnz, "declared nnz != entries read");
    trip.sort_by_key(|&(i, j, _)| (j, i));
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for &(i, j, v) in &trip {
        cols[j].push((i, v));
    }
    SparseColMatrix::from_sparse_columns(m, &cols).expect("basis")
}

fn factor(a: &SparseColMatrix, cap: f64) -> SparseLu {
    let sym = SparseLuSymbolic::analyze(a).expect("analyze");
    SparseLu::factor(
        a,
        &sym,
        LuParams {
            hyper_sparse_max_density: cap,
            ..LuParams::default()
        },
    )
    .expect("factor")
}

/// The guard that the synthetic fixture could not provide.
///
/// At the shipped default the `btran` route fires only on genuinely small
/// reaches. At `0.25` it admits reaches averaging a large fraction of `m` —
/// the population where the reach DFS costs more than the sweep it replaces.
/// Both quantities are deterministic (reach sizes, not times).
///
/// This fails if someone raises the default back into that band, which is
/// exactly the mistake this PR shipped once.
#[test]
fn default_cap_excludes_the_band_where_btran_reaches_are_near_dense() {
    let Some(path) = fixture("QPLIB_1157_basis.mtx") else {
        return;
    };
    let a = read_mtx(&path);
    let m = a.m;
    let probes: Vec<usize> = (0..256).map(|t| (t * 97 + 11) % m).collect();

    // Only `btran` is run on each factor, so every counted sweep is a `btran`
    // sweep and the mean below is unambiguous.
    let mean_reach = |cap: f64| -> (f64, usize) {
        let mut lu = factor(&a, cap);
        for &k in probes.iter() {
            let mut y = vec![0.0; m];
            y[k] = 1.0;
            lu.btran(&mut y).expect("btran");
        }
        let fired = lu.hyper_sparse_sweeps();
        let mean = lu.hyper_sparse_nodes() as f64 / fired.max(1) as f64;
        (mean, fired)
    };

    // The two reference caps are *fixed*, not derived from the default, so that
    // raising the default fails the assertion about the default rather than the
    // one about the fixture.
    let (mean_narrow, fired_narrow) = mean_reach(0.02);
    let (mean_wide, fired_wide) = mean_reach(0.25);
    let (mean_default, _) = mean_reach(LuParams::default().hyper_sparse_max_density);

    // Fixture sanity: this basis really does have the pathological population at
    // 0.25 and not at 0.02. If that stops holding, the fixture has changed and
    // the assertion below means nothing.
    assert!(
        fired_wide > fired_narrow * 4 && mean_wide > 0.10 * m as f64,
        "fixture changed: expected cap 0.25 to admit many near-dense btran \
         reaches and 0.02 not to; got wide {fired_wide} sweeps averaging \
         {mean_wide:.0}, narrow {fired_narrow} averaging {mean_narrow:.0} \
         (m = {m})"
    );

    // And the shipped default must behave like the narrow cap, not the wide one.
    assert!(
        mean_default < 0.02 * m as f64,
        "the default cap admits btran reaches averaging {mean_default:.0} of \
         m = {m}. On this basis the reach DFS walks ~92% of nnz(L) at that \
         density and the route costs ~1.6x more than the dense sweep it \
         replaces. Raising hyper_sparse_max_density needs a basis where that \
         band *wins*, not a synthetic sweep — see the parameter's doc comment."
    );
}

/// `ftran_sparse` / `btran_sparse` against real data.
///
/// This was the one part of the change neither the author nor the maintainer
/// had run on a real basis — correct by construction and by differential tests
/// on generated matrices, but new public API validated only against a fixture
/// tuned to it. The dense entry point on the same factor is the oracle.
#[test]
fn sparse_entry_points_agree_with_dense_on_real_bases() {
    for name in ["QPLIB_3852_basis.mtx", "QPLIB_1157_basis.mtx"] {
        let Some(path) = fixture(name) else {
            continue;
        };
        let a = read_mtx(&path);
        let m = a.m;
        let mut lu = factor(&a, LuParams::default().hyper_sparse_max_density);
        let mut out = Vec::new();
        let mut worst_f = 0.0_f64;
        let mut worst_b = 0.0_f64;

        // Every column, not a sample: these bases have a wide spread of
        // solution densities and the interesting cases are the tail.
        for k in 0..m {
            let mut expect = vec![0.0; m];
            expect[k] = 1.0;
            lu.ftran(&mut expect).expect("ftran");
            lu.ftran_sparse(&[(k, 1.0)], &mut out)
                .expect("ftran_sparse");
            let mut got = vec![0.0; m];
            for &(i, v) in out.iter() {
                got[i] = v;
            }
            worst_f = worst_f.max(
                got.iter()
                    .zip(expect.iter())
                    .fold(0.0_f64, |acc, (p, q)| acc.max((p - q).abs())),
            );

            let mut expect = vec![0.0; m];
            expect[k] = 1.0;
            lu.btran(&mut expect).expect("btran");
            lu.btran_sparse(&[(k, 1.0)], &mut out)
                .expect("btran_sparse");
            let mut got = vec![0.0; m];
            for &(i, v) in out.iter() {
                got[i] = v;
            }
            worst_b = worst_b.max(
                got.iter()
                    .zip(expect.iter())
                    .fold(0.0_f64, |acc, (p, q)| acc.max((p - q).abs())),
            );
        }
        assert!(
            worst_f < 1e-9 && worst_b < 1e-9,
            "{name}: sparse vs dense entry point disagree — ftran {worst_f:e}, \
             btran {worst_b:e}"
        );
    }
}

/// The reach-limited route must be numerically identical to the dense sweep on
/// real bases, not merely close. The omitted terms are exactly `v * 0.0`, so
/// anything other than a bit-for-bit match means the reach dropped a term the
/// solution depended on.
#[test]
fn reach_route_is_bit_identical_to_the_dense_sweep_on_real_bases() {
    for name in ["QPLIB_3852_basis.mtx", "QPLIB_1157_basis.mtx"] {
        let Some(path) = fixture(name) else {
            continue;
        };
        let a = read_mtx(&path);
        let m = a.m;
        let mut off = factor(&a, 0.0);
        let mut on = factor(&a, LuParams::default().hyper_sparse_max_density);
        assert_eq!(
            off.factor_nnz(),
            on.factor_nnz(),
            "{name}: the route must not change the factor, only how it is walked"
        );

        for k in 0..m {
            let mut e = vec![0.0; m];
            e[k] = 1.0;
            let (mut f_off, mut f_on) = (e.clone(), e.clone());
            off.ftran(&mut f_off).expect("ftran off");
            on.ftran(&mut f_on).expect("ftran on");
            assert_eq!(f_off, f_on, "{name}: ftran differs at column {k}");
            let (mut b_off, mut b_on) = (e.clone(), e.clone());
            off.btran(&mut b_off).expect("btran off");
            on.btran(&mut b_on).expect("btran on");
            assert_eq!(b_off, b_on, "{name}: btran differs at column {k}");
        }
        assert_eq!(off.hyper_sparse_sweeps(), 0);
        assert!(
            on.hyper_sparse_sweeps() > 0,
            "{name}: the reach route never fired — this test would be vacuous"
        );
    }
}

/// **Issue #164 on real data.** The whole-basis fallback and the pure reach
/// route must return the same solution on the bases the cap was measured on.
///
/// The synthetic fixture in `tests/lu_sparse_rhs.rs` pins the contract; this
/// pins it where it matters. These bases are the ones whose solution-density
/// profile straddles the cap — that is why they are in the tree at all — so
/// this is the only place the fallback runs against real fill.
///
/// Right-hand sides span the cap deliberately: unit vectors (the simplex's own,
/// almost always under it) and blocks of 64 adjacent columns (reliably over).
/// The fallback counter is asserted in both directions so neither arm can pass
/// having tested nothing.
#[test]
fn the_density_fallback_agrees_with_the_reach_route_on_real_bases() {
    for name in ["QPLIB_3852_basis.mtx", "QPLIB_1157_basis.mtx"] {
        let Some(path) = fixture(name) else {
            continue;
        };
        let a = read_mtx(&path);
        let m = a.m;
        let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
        let mk = |cap: f64| {
            SparseLu::factor(
                &a,
                &sym,
                LuParams {
                    sparse_rhs_max_density: cap,
                    ..LuParams::default()
                },
            )
            .expect("factor")
        };
        let mut never = mk(1.0);
        let mut capped = mk(LuParams::default().sparse_rhs_max_density);

        let mut rhss: Vec<Vec<(usize, f64)>> = (0..m).step_by(7).map(|k| vec![(k, 1.0)]).collect();
        for lo in (0..m).step_by(m / 8 + 1) {
            let hi = (lo + 64).min(m);
            rhss.push((lo..hi).map(|i| (i, 1.0 + (i % 5) as f64)).collect());
        }

        let mut a_out = Vec::new();
        let mut b_out = Vec::new();
        let mut worst = 0.0_f64;
        for rhs in rhss.iter() {
            for forward in [true, false] {
                if forward {
                    never.ftran_sparse(rhs, &mut a_out).expect("ftran_sparse");
                    capped.ftran_sparse(rhs, &mut b_out).expect("ftran_sparse");
                } else {
                    never.btran_sparse(rhs, &mut a_out).expect("btran_sparse");
                    capped.btran_sparse(rhs, &mut b_out).expect("btran_sparse");
                }
                let (mut x, mut y) = (vec![0.0; m], vec![0.0; m]);
                for &(i, v) in a_out.iter() {
                    x[i] = v;
                }
                for &(i, v) in b_out.iter() {
                    y[i] = v;
                }
                worst = worst.max(
                    x.iter()
                        .zip(y.iter())
                        .fold(0.0_f64, |acc, (p, q)| acc.max((p - q).abs())),
                );
            }
        }

        let fired = capped.sparse_rhs_fallbacks();
        let solves = rhss.len() * 2;
        assert_eq!(
            never.sparse_rhs_fallbacks(),
            0,
            "{name}: a cap of 1.0 must never fall back"
        );
        assert!(
            fired > 0 && fired < solves,
            "{name}: the fallback fired on {fired}/{solves} solves; this fixture \
             is supposed to straddle the cap, so 0 and {solves} are both vacuous"
        );
        assert!(
            worst < 1e-9,
            "{name}: the fallback and the reach route disagree by {worst:e}"
        );
    }
}
