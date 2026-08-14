//! Issue #163: score `SparseLuSymbolic::analyze` (whole-basis AMD) against
//! `analyze_triangularized` (Suhl-Suhl peel + AMD on the bump) on real bases.
//!
//! ```text
//! cargo run --release --example probe_illcond_ordering -- tests/data/lu_bases
//! ```
//!
//! Reports, per basis, each ordering's **backward** error (relative residual of
//! `A x = b` and `A^T y = c`, worst of the two) and **forward** error (relative
//! distance from a known `x_ref`, worst of the two), plus the peel/AMD ratio of
//! each and the two factor sizes.
//!
//! This is the harness behind the claim in `SparseLuSymbolic::analyze`'s rustdoc
//! that the peel is *not* the less accurate ordering. Run over the 30 bases of
//! the failing discopt trajectory it never scored the peel worse: forward-error
//! ratios 0.0x-1.0x, backward error ~1e-16 throughout. Only the worst of those
//! bases is carried in-tree (`bchoco06_illcond_basis.mtx`, forward error 2.6e-11
//! under both orderings); regenerating the full set needs the dump snippet in
//! `dev/research/lu-ordering-and-kernel-2026-08-13.md` plus a discopt checkout.

use feral::{LuParams, SparseColMatrix, SparseLu, SparseLuSymbolic};
use std::fs;

fn read_mtx(path: &str) -> SparseColMatrix {
    let text = fs::read_to_string(path).expect("read");
    let mut lines = text.lines().filter(|l| !l.starts_with('%'));
    let hdr: Vec<usize> = lines
        .next()
        .expect("header")
        .split_whitespace()
        .map(|t| t.parse().expect("dim"))
        .collect();
    let (m, n) = (hdr[0], hdr[1]);
    let mut trip: Vec<(usize, usize, f64)> = Vec::new();
    for l in lines {
        let mut it = l.split_whitespace();
        let i: usize = it.next().expect("i").parse().expect("i");
        let j: usize = it.next().expect("j").parse().expect("j");
        let v: f64 = it.next().expect("v").parse().expect("v");
        trip.push((i - 1, j - 1, v));
    }
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (i, j, v) in trip {
        cols[j].push((i, v));
    }
    SparseColMatrix::from_sparse_columns(m, &cols).expect("matrix")
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

fn rel(r: &[f64], b: &[f64]) -> f64 {
    let nr = r.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let nb = b.iter().fold(0.0_f64, |m, &x| m.max(x.abs())).max(1e-300);
    nr / nb
}

/// `(max backward resid, max forward error, factor nnz)`, or `None` if the
/// factorization was rejected outright.
fn score(a: &SparseColMatrix, sym: &SparseLuSymbolic) -> Option<(f64, f64, usize)> {
    let m = a.m;
    let mut lu = SparseLu::factor(a, sym, LuParams::default()).ok()?;
    let x_ref: Vec<f64> = (0..m).map(|i| 1.0 + ((i * 7) % 13) as f64 * 0.25).collect();

    let b = matvec(a, &x_ref);
    let mut x = b.clone();
    lu.ftran(&mut x).ok()?;
    let ax = matvec(a, &x);
    let rf: Vec<f64> = ax.iter().zip(b.iter()).map(|(p, q)| p - q).collect();

    let c = matvec_t(a, &x_ref);
    let mut y = c.clone();
    lu.btran(&mut y).ok()?;
    let aty = matvec_t(a, &y);
    let rb: Vec<f64> = aty.iter().zip(c.iter()).map(|(p, q)| p - q).collect();

    // Forward error too: two backward-stable factorizations of an
    // ill-conditioned basis can still land at different points.
    let ef: Vec<f64> = x.iter().zip(x_ref.iter()).map(|(p, q)| p - q).collect();
    let eb: Vec<f64> = y.iter().zip(x_ref.iter()).map(|(p, q)| p - q).collect();
    Some((
        rel(&rf, &b).max(rel(&rb, &c)),
        rel(&ef, &x_ref).max(rel(&eb, &x_ref)),
        lu.factor_nnz(),
    ))
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: <dir>");
    let mut names: Vec<String> = fs::read_dir(&dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".mtx"))
        .collect();
    names.sort();

    println!(
        "  basis                 amd_resid   amd_fwderr  peel_resid  peel_fwderr  \
         resid_x  fwd_x   nnz amd/peel"
    );
    let mut worst = (0.0_f64, String::new());
    for name in names {
        let path = format!("{dir}/{name}");
        let a = read_mtx(&path);
        let amd = SparseLuSymbolic::analyze(&a).expect("analyze");
        let peel = SparseLuSymbolic::analyze_triangularized(&a).expect("triangularized");
        let (af, ab, an) = match score(&a, &amd) {
            Some(v) => v,
            None => {
                println!("  {name:<20}  amd factorization rejected");
                continue;
            }
        };
        let (pf, pb, pn) = match score(&a, &peel) {
            Some(v) => v,
            None => {
                println!("  {name:<20}  PEEL FACTORIZATION REJECTED (amd ok: {af:.2e})");
                continue;
            }
        };
        let ratio = pf / af.max(1e-300);
        let fratio = pb / ab.max(1e-300);
        println!(
            "  {name:<20} {af:>11.2e} {ab:>11.2e} {pf:>11.2e} {pb:>11.2e} \
             {ratio:>7.1}x {fratio:>6.1}x {an:>6}/{pn}"
        );
        if fratio > worst.0 {
            worst = (fratio, name);
        }
    }
    println!(
        "\n  worst peel/amd forward-error ratio: {:.1}x on {}",
        worst.0, worst.1
    );
}
