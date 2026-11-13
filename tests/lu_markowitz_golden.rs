//! Golden pivot order for threshold-Markowitz LU (issue #232, PR #234).
//!
//! pounce's active-set QP uses `SparseLu` as a rank-reveal probe, so *which*
//! pivots the kernel search picks is part of the contract, not just fill. An
//! earlier revision of #232 changed the candidate-search budget, moved fill by
//! +0.3 % and made pounce's QGFRDXPN active-set solve cycle to its iteration
//! cap. These digests pin the pivot order and the factor's solves.
//!
//! **Oracle:** the expected values were computed with the feral 0.22.0
//! `markowitz_factor` (before any #232 change), using this file's digest and
//! readers, which are copied verbatim from `examples/lu_markowitz_triangular.rs`.
//! A change that alters them changes pivots; that needs a decision, not a new
//! golden value.

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{LuParams, SparseLu};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn piperout_shape(seed: u64) -> SparseColMatrix {
    let m = 17_470;
    let k = 1_459;
    let mut rng = Rng(seed);
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    // Kernel: diagonal plus ~2.6 entries per column (≈ 5 200 nnz).
    for (j, col) in cols.iter_mut().enumerate().take(k) {
        col.push((j, 1.0 + rng.f64()));
        for _ in 0..3 {
            if rng.f64() < 0.85 {
                // Mostly below the diagonal (no fill), a few anywhere, so the
                // kernel is a genuine bump with little fill like the real one.
                let i = if j + 1 < k && rng.f64() < 0.97 {
                    j + 1 + rng.below(k - j - 1)
                } else {
                    rng.below(k)
                };
                if !col.iter().any(|&(r, _)| r == i) {
                    col.push((i, rng.f64() - 0.5));
                }
            }
        }
    }
    // Triangular part: ~80 % one-entry columns, the rest carry a few entries
    // in rows above the diagonal (kernel rows included).
    for (j, col) in cols.iter_mut().enumerate().skip(k) {
        col.push((j, 2.0 + rng.f64()));
        if rng.f64() < 0.21 {
            let extra = 2 + rng.below(9);
            for _ in 0..extra {
                let i = rng.below(j);
                if !col.iter().any(|&(r, _)| r == i) {
                    col.push((i, rng.f64() - 0.5));
                }
            }
        }
    }
    let mut a = SparseColMatrix::from_sparse_columns(m, &cols).expect("basis");
    // A permutation so the triangular structure is not in index order.
    let mut p: Vec<usize> = (0..m).collect();
    for i in (1..m).rev() {
        p.swap(i, rng.below(i + 1));
    }
    let mut pc: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for j in 0..m {
        for idx in a.col_ptr[j]..a.col_ptr[j + 1] {
            pc[p[j]].push((p[a.row_idx[idx]], a.values[idx]));
        }
    }
    a = SparseColMatrix::from_sparse_columns(m, &pc).expect("basis");
    a
}

fn read_mtx(path: &str) -> SparseColMatrix {
    let text = std::fs::read_to_string(path).expect("read");
    let mut lines = text.lines().filter(|l| !l.starts_with('%'));
    let hdr: Vec<usize> = lines
        .next()
        .expect("header")
        .split_whitespace()
        .map(|t| t.parse().expect("int"))
        .collect();
    let m = hdr[0];
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for l in lines {
        let t: Vec<&str> = l.split_whitespace().collect();
        if t.len() < 3 {
            continue;
        }
        let i: usize = t[0].parse().expect("i");
        let j: usize = t[1].parse().expect("j");
        let v: f64 = t[2].parse().expect("v");
        cols[j - 1].push((i - 1, v));
    }
    SparseColMatrix::from_sparse_columns(m, &cols).expect("basis")
}

fn digest(lu: &mut SparseLu) -> u64 {
    let m = lu.dim();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    for &p in lu.perm() {
        mix(p as u64);
    }
    for &q in lu.qcol() {
        mix(q as u64);
    }
    mix(lu.factor_nnz() as u64);
    let mut x: Vec<f64> = (0..m).map(|i| 1.0 + (i % 7) as f64).collect();
    lu.ftran(&mut x).expect("ftran");
    for v in &x {
        mix(v.to_bits());
    }
    let mut y: Vec<f64> = (0..m).map(|i| 1.0 + (i % 5) as f64).collect();
    lu.btran(&mut y).expect("btran");
    for v in &y {
        mix(v.to_bits());
    }
    h
}

fn fixture(name: &str) -> Option<String> {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/lu_bases")
        .join(name);
    if p.exists() {
        Some(p.to_string_lossy().into_owned())
    } else {
        eprintln!("SKIP: {} not found", p.display());
        None
    }
}

/// Fresh factor and in-place refactor (after factoring the same pattern with
/// other values) must both reproduce the 0.22 digest.
fn check(name: &str, a: &SparseColMatrix, expected: u64) {
    let params = LuParams::default();
    let mut lu = SparseLu::factor_markowitz(a, params.clone()).expect("factor");
    assert_eq!(
        digest(&mut lu),
        expected,
        "{name}: fresh factor differs from feral 0.22"
    );

    let mut other = a.clone();
    for (k, v) in other.values.iter_mut().enumerate() {
        *v *= 1.0 + 0.25 * ((k % 3) as f64);
    }
    let mut lu2 = SparseLu::factor_markowitz(&other, params.clone()).expect("factor");
    lu2.refactor_markowitz(a, params).expect("refactor");
    assert_eq!(
        digest(&mut lu2),
        expected,
        "{name}: in-place refactor differs from feral 0.22"
    );
}

#[test]
fn markowitz_pivots_match_feral_0_22_on_real_bases() {
    for (name, expected) in [
        ("QPLIB_1157_basis.mtx", 0x9478_2c15_ef88_618e_u64),
        ("QPLIB_3852_basis.mtx", 0x3097_a9b9_db09_31e9),
        ("bchoco06_illcond_basis.mtx", 0x0c42_6e80_19bf_b167),
    ] {
        if let Some(p) = fixture(name) {
            check(name, &read_mtx(&p), expected);
        }
    }
}

/// The issue's shape: 92 % peeled by singletons, then a kernel whose heap
/// carries duplicate live entries, which is where the budget matters.
#[test]
fn markowitz_pivots_match_feral_0_22_on_the_issue_232_shape() {
    for (seed, expected) in [
        (1u64, 0x200f_c855_bc09_dc10_u64),
        (2, 0x0c45_08fb_7a46_a918),
        (3, 0x36d5_3d40_60eb_6c35),
    ] {
        check(
            &format!("piperout-shape seed {seed}"),
            &piperout_shape(seed),
            expected,
        );
    }
}
