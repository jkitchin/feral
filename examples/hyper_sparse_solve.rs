//! Interleaved A/B for the reach-limited ("hyper-sparse") triangular solves
//! (issue #161B).
//!
//! Issue #161 part B measured feral's `ftran` costing 0.74x as much for a
//! solution with **one** nonzero as for a solution with `m` — "2918x more work
//! than it performs". This harness reproduces that shape in-tree and measures
//! what the reach-limited route does to it.
//!
//! Two factors of the *same* basis are built, one with
//! `hyper_sparse_max_density = 0` (the pre-#161 dense sweep) and one with the
//! route on, and their solves are interleaved rep by rep so a thermal or
//! frequency drift hits both arms equally. The arms are checked to agree to
//! round-off, and the run **exits non-zero** if the reach route never actually
//! fired — a silent fallback would otherwise let this report a flattering 1.00x
//! against itself.
//!
//! Usage: cargo run --release --example hyper_sparse_solve -- [m] [bump] [band] [density]

use std::time::Instant;

use feral::{LuParams, SparseColMatrix, SparseLu, SparseLuSymbolic};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 32) as u32 as f64 / u32::MAX as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n.max(1)
    }
}

/// An LP-simplex-shaped basis: near-triangular skeleton (~2.3 nonzeros/column,
/// the density issue #161 reports for QPLIB_3852) plus a non-triangular bump,
/// row- and column-permuted. This is the structure that makes `B⁻¹eᵢ` sparse and
/// therefore the structure the route is for; a uniformly random sparse matrix
/// has a dense inverse and would measure nothing.
fn lp_basis(m: usize, bump: usize, band: usize, seed: u64) -> Vec<Vec<(usize, f64)>> {
    let mut rng = Rng(seed);
    let bump_lo = m / 3;
    let bump_hi = (bump_lo + bump).min(m);
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for (j, col) in cols.iter_mut().enumerate() {
        col.push((j, 4.0 + rng.unit()));
        if rng.unit() < 0.7 {
            let i = j + 1 + rng.below(band);
            if i < m {
                col.push((i, rng.unit() * 2.0 - 1.0));
            }
        }
        if j >= bump_lo && j < bump_hi {
            for _ in 0..2 {
                let i = bump_lo + rng.below(bump_hi - bump_lo);
                if i != j {
                    col.push((i, rng.unit() * 2.0 - 1.0));
                }
            }
        }
    }
    let mut rperm: Vec<usize> = (0..m).collect();
    let mut cperm: Vec<usize> = (0..m).collect();
    for k in (1..m).rev() {
        rperm.swap(k, rng.below(k + 1));
        cperm.swap(k, rng.below(k + 1));
    }
    let mut out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for (j, col) in cols.into_iter().enumerate() {
        let dst = &mut out[cperm[j]];
        for (i, v) in col {
            dst.push((rperm[i], v));
        }
        dst.sort_by_key(|&(i, _)| i);
        dst.dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 += a.1;
                true
            } else {
                false
            }
        });
    }
    out
}

struct Arm {
    lu: SparseLu,
    ftran_ns: u128,
    btran_ns: u128,
    ftran_each: Vec<u128>,
    btran_each: Vec<u128>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let m: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4000);
    let bump: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(m / 10);
    let band: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(6);
    // The cap under test. Sweeping it is how the shipped default was chosen —
    // see dev/research/hyper-sparse-solves-2026-08-13.md.
    let density: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.25);

    let cols = lp_basis(m, bump, band, 0xC0FFEE);
    let a = SparseColMatrix::from_sparse_columns(m, &cols).expect("basis");
    let nnz = a.nnz();
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    let mk = |d: f64| LuParams {
        hyper_sparse_max_density: d,
        ..LuParams::default()
    };

    let mut off = Arm {
        lu: SparseLu::factor(&a, &sym, mk(0.0)).expect("factor off"),
        ftran_ns: 0,
        btran_ns: 0,
        ftran_each: Vec::new(),
        btran_each: Vec::new(),
    };
    let mut on = Arm {
        lu: SparseLu::factor(&a, &sym, mk(density)).expect("factor on"),
        ftran_ns: 0,
        btran_ns: 0,
        ftran_each: Vec::new(),
        btran_each: Vec::new(),
    };
    let fnnz = off.lu.factor_nnz();
    assert_eq!(
        fnnz,
        on.lu.factor_nnz(),
        "the route must not change the factor"
    );

    println!(
        "m={m} nnz(A)={nnz} ({:.2}/col)  nnz(LU)={fnnz}  fill={:.2}x  bump={bump} band={band} density={density}",
        nnz as f64 / m as f64,
        fnnz as f64 / nnz as f64,
    );

    // Solution density of a unit-vector rhs — the quantity the route is keyed on.
    let mut nnz_hist: Vec<usize> = Vec::with_capacity(m);
    for k in 0..m {
        let mut e = vec![0.0; m];
        e[k] = 1.0;
        on.lu.ftran(&mut e).expect("ftran");
        nnz_hist.push(e.iter().filter(|&&v| v != 0.0).count());
    }
    nnz_hist.sort_unstable();
    println!(
        "  ftran(e_i) solution nnz: p50={} p90={} max={} (of m={m})",
        nnz_hist[m / 2],
        nnz_hist[m * 9 / 10],
        nnz_hist[m - 1],
    );

    // Interleaved A/B. Both arms see identical right-hand sides in identical
    // order; the arm order alternates so neither is systematically the cold one.
    let reps = 5usize;
    let probes: Vec<usize> = (0..256).map(|t| (t * 97 + 11) % m).collect();
    let mut worst = 0.0_f64;
    // Buffers are allocated once and reused. A fresh `vec![0.0; m]` per call
    // puts the first touch of every page *inside* the timed region, so the
    // measurement picks up ~m/512 page faults per solve — which on a
    // hyper-sparse solve is larger than the solve.
    let mut x1 = vec![0.0; m];
    let mut x2 = vec![0.0; m];
    let mut y1 = vec![0.0; m];
    let mut y2 = vec![0.0; m];
    for rep in 0..reps {
        for &k in probes.iter() {
            for b in [&mut x1, &mut x2, &mut y1, &mut y2] {
                b.fill(0.0);
                b[k] = 1.0;
            }
            let (first, second) = if rep % 2 == 0 {
                (&mut off, &mut on)
            } else {
                (&mut on, &mut off)
            };
            let t = Instant::now();
            first.lu.ftran(&mut x1[..]).expect("ftran");
            let ns = t.elapsed().as_nanos();
            first.ftran_ns += ns;
            first.ftran_each.push(ns);
            let t = Instant::now();
            second.lu.ftran(&mut x2[..]).expect("ftran");
            let ns = t.elapsed().as_nanos();
            second.ftran_ns += ns;
            second.ftran_each.push(ns);
            worst = worst.max(
                x1.iter()
                    .zip(x2.iter())
                    .fold(0.0_f64, |a, (p, q)| a.max((p - q).abs())),
            );

            let t = Instant::now();
            first.lu.btran(&mut y1[..]).expect("btran");
            let ns = t.elapsed().as_nanos();
            first.btran_ns += ns;
            first.btran_each.push(ns);
            let t = Instant::now();
            second.lu.btran(&mut y2[..]).expect("btran");
            let ns = t.elapsed().as_nanos();
            second.btran_ns += ns;
            second.btran_each.push(ns);
            worst = worst.max(
                y1.iter()
                    .zip(y2.iter())
                    .fold(0.0_f64, |a, (p, q)| a.max((p - q).abs())),
            );
        }
    }

    let calls = (reps * probes.len()) as f64;
    let us = |ns: u128| ns as f64 / 1000.0 / calls;
    // The mean is reported alongside the median because this basis has a
    // bimodal solution density (see the p50/p90 line above): most unit-vector
    // solves touch a handful of positions, a minority reach through the whole
    // bump. The mean is dominated by the minority, so it *understates* the win
    // on the hyper-sparse case the route exists for — and the median alone
    // would overstate it. Both are printed rather than picking the flattering
    // one.
    let pct = |v: &mut Vec<u128>, q: usize| {
        v.sort_unstable();
        v[v.len() * q / 100] as f64 / 1000.0
    };
    println!(
        "  sparse rhs (unit vector), {} calls per arm:",
        calls as usize
    );
    println!(
        "    dense sweep   ftran mean={:8.3} p50={:8.3} p90={:8.3} us | \
btran mean={:8.3} p50={:8.3} p90={:8.3} us",
        us(off.ftran_ns),
        pct(&mut off.ftran_each, 50),
        pct(&mut off.ftran_each, 90),
        us(off.btran_ns),
        pct(&mut off.btran_each, 50),
        pct(&mut off.btran_each, 90),
    );
    println!(
        "    reach-limited ftran mean={:8.3} p50={:8.3} p90={:8.3} us | \
btran mean={:8.3} p50={:8.3} p90={:8.3} us",
        us(on.ftran_ns),
        pct(&mut on.ftran_each, 50),
        pct(&mut on.ftran_each, 90),
        us(on.btran_ns),
        pct(&mut on.btran_each, 50),
        pct(&mut on.btran_each, 90),
    );
    println!(
        "    --> ftran mean {:.2}x p50 {:.2}x | btran mean {:.2}x p50 {:.2}x  \
(max |diff| = {:.3e})",
        off.ftran_ns as f64 / on.ftran_ns.max(1) as f64,
        pct(&mut off.ftran_each, 50) / pct(&mut on.ftran_each, 50).max(1e-9),
        off.btran_ns as f64 / on.btran_ns.max(1) as f64,
        pct(&mut off.btran_each, 50) / pct(&mut on.btran_each, 50).max(1e-9),
        worst,
    );

    // Dense rhs: the fallback path. This is where the route can only lose, and
    // the number that bounds how much.
    let (mut d_off, mut d_on) = (0u128, 0u128);
    let mut rng = Rng(1);
    let mut rhs = vec![0.0; m];
    let mut buf = vec![0.0; m];
    for rep in 0..reps {
        for _ in 0..32 {
            for v in rhs.iter_mut() {
                *v = rng.unit() + 0.5;
            }
            let first_is_off = rep % 2 == 0;
            let (first, second) = if first_is_off {
                (&mut off, &mut on)
            } else {
                (&mut on, &mut off)
            };
            buf.copy_from_slice(&rhs);
            let t = Instant::now();
            first.lu.ftran(&mut buf[..]).expect("ftran");
            let e1 = t.elapsed().as_nanos();
            buf.copy_from_slice(&rhs);
            let t = Instant::now();
            second.lu.ftran(&mut buf[..]).expect("ftran");
            let e2 = t.elapsed().as_nanos();
            if first_is_off {
                d_off += e1;
                d_on += e2;
            } else {
                d_on += e1;
                d_off += e2;
            }
        }
    }
    let dcalls = (reps * 32) as f64;
    println!(
        "  dense rhs (fallback path), {} calls per arm: dense={:.3} us  reach={:.3} us  \
         --> {:.2}x",
        dcalls as usize,
        d_off as f64 / 1000.0 / dcalls,
        d_on as f64 / 1000.0 / dcalls,
        d_off as f64 / d_on.max(1) as f64,
    );

    println!(
        "  reach-limited sweeps taken: off={}  on={} ({:.1} positions swept per sweep)",
        off.lu.hyper_sparse_sweeps(),
        on.lu.hyper_sparse_sweeps(),
        on.lu.hyper_sparse_nodes() as f64 / on.lu.hyper_sparse_sweeps().max(1) as f64,
    );

    // Guard against a vacuous pass: a silent fallback would report ~1.00x and
    // look like an honest null result.
    if on.lu.hyper_sparse_sweeps() == 0 {
        eprintln!("FAIL: the reach-limited route never fired; the A/B above is vacuous");
        std::process::exit(1);
    }
    if off.lu.hyper_sparse_sweeps() != 0 {
        eprintln!("FAIL: the density=0 arm took the reach route; the arms are not distinct");
        std::process::exit(1);
    }
    if worst > 1e-9 {
        eprintln!("FAIL: the arms disagree by {worst:e}");
        std::process::exit(1);
    }
}
