//! Transfer test for feral #161B (PR #162) on **real** discopt simplex bases.
//!
//! PR #162 states its fixture is synthetic and "the transfer is **unverified**"
//! because the QPLIB `.npz` bases live in the discopt tree. This runs the same
//! measurement on the real thing.
//!
//! Three questions, in the order they have to be answered:
//!
//!   1. Is the *solution* of `B⁻¹eᵢ` actually sparse on this basis? The route is
//!      keyed on that and nothing else. If the inverse fills in, no amount of
//!      reach machinery can help and the answer is structural, not a tuning
//!      problem.
//!   2. Does the reach-limited route *fire* through the dense `ftran`/`btran`
//!      entry points (the ones discopt calls today), and what does it buy?
//!      `hyper_sparse_sweeps()` is asserted, because the route is a silent
//!      fallback and "no gain" is indistinguishable from "never ran".
//!   3. What would `ftran_sparse` buy — i.e. is it worth changing discopt's
//!      call sites?
//!
//! Usage: cargo run --release --example real_basis_hyper -- <file.mtx> [density] [bump_cap]

use std::time::Instant;

use feral::{LuParams, SparseColMatrix, SparseLu, SparseLuSymbolic};

fn read_mtx(path: &str) -> SparseColMatrix {
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

fn pct(v: &[u128], p: usize) -> f64 {
    v[(v.len() - 1) * p / 100] as f64 / 1000.0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: real_basis_hyper <file.mtx> [density] [bump_cap]");
    let density: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.25);

    let a = read_mtx(&path);
    let m = a.m;
    let nnz = a.nnz();
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    let bump_cap: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4096);

    let mk = |d: f64| LuParams {
        hyper_sparse_max_density: d,
        dense_bump_max_dim: bump_cap,
        ..LuParams::default()
    };
    let mut off = SparseLu::factor(&a, &sym, mk(0.0)).expect("factor off");
    let mut on = SparseLu::factor(&a, &sym, mk(density)).expect("factor on");
    assert_eq!(
        off.factor_nnz(),
        on.factor_nnz(),
        "the route must not change the factor"
    );

    println!(
        "{}\n  m={m} nnz(A)={nnz} ({:.2}/col)  nnz(LU)={}  fill={:.2}x  \
dense_bump(cap={bump_cap}, fired={})  hyper_sparse_max_density={density}",
        path.rsplit('/').next().unwrap(),
        nnz as f64 / m as f64,
        on.factor_nnz(),
        on.factor_nnz() as f64 / nnz as f64,
        on.used_dense_bump(),
    );

    // ---- Q1: is B⁻¹eᵢ sparse on THIS basis? -------------------------------
    // This is the precondition for everything else. Sampled over every column.
    let mut hist: Vec<usize> = Vec::with_capacity(m);
    for k in 0..m {
        let mut e = vec![0.0; m];
        e[k] = 1.0;
        on.ftran(&mut e).expect("ftran");
        hist.push(e.iter().filter(|&&v| v != 0.0).count());
    }
    hist.sort_unstable();
    let mean = hist.iter().sum::<usize>() as f64 / m as f64;
    println!(
        "  Q1 ftran(e_i) solution nnz: p50={} p90={} max={} mean={:.1}  (of m={m}) \
-> mean density {:.1}% vs cap {:.1}%",
        hist[m / 2],
        hist[m * 9 / 10],
        hist[m - 1],
        mean,
        100.0 * mean / m as f64,
        100.0 * density,
    );
    let over = hist
        .iter()
        .filter(|&&h| h as f64 > density * m as f64)
        .count();
    println!(
        "  Q1 columns whose solution exceeds the cap (route MUST abort on these): \
{over}/{m} = {:.1}%",
        100.0 * over as f64 / m as f64
    );

    // ---- Q2: dense entry points, reach route off vs on ---------------------
    let probes: Vec<usize> = (0..256).map(|t| (t * 97 + 11) % m).collect();
    let reps = 5usize;
    let (mut foff, mut fon, mut boff, mut bon) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut worst = 0.0_f64;
    let (mut x1, mut x2) = (vec![0.0; m], vec![0.0; m]);
    let (mut y1, mut y2) = (vec![0.0; m], vec![0.0; m]);
    let sweeps_before = on.hyper_sparse_sweeps();
    for rep in 0..reps {
        for &k in probes.iter() {
            for b in [&mut x1, &mut x2, &mut y1, &mut y2] {
                b.fill(0.0);
                b[k] = 1.0;
            }
            // Alternate which arm goes first so neither is systematically cold.
            if rep % 2 == 0 {
                let t = Instant::now();
                off.ftran(&mut x1[..]).expect("ftran off");
                foff.push(t.elapsed().as_nanos());
                let t = Instant::now();
                on.ftran(&mut x2[..]).expect("ftran on");
                fon.push(t.elapsed().as_nanos());
                let t = Instant::now();
                off.btran(&mut y1[..]).expect("btran off");
                boff.push(t.elapsed().as_nanos());
                let t = Instant::now();
                on.btran(&mut y2[..]).expect("btran on");
                bon.push(t.elapsed().as_nanos());
            } else {
                let t = Instant::now();
                on.ftran(&mut x2[..]).expect("ftran on");
                fon.push(t.elapsed().as_nanos());
                let t = Instant::now();
                off.ftran(&mut x1[..]).expect("ftran off");
                foff.push(t.elapsed().as_nanos());
                let t = Instant::now();
                on.btran(&mut y2[..]).expect("btran on");
                bon.push(t.elapsed().as_nanos());
                let t = Instant::now();
                off.btran(&mut y1[..]).expect("btran off");
                boff.push(t.elapsed().as_nanos());
            }
            for i in 0..m {
                worst = worst.max((x1[i] - x2[i]).abs()).max((y1[i] - y2[i]).abs());
            }
        }
    }
    let sweeps = on.hyper_sparse_sweeps() - sweeps_before;
    for v in [&mut foff, &mut fon, &mut boff, &mut bon] {
        v.sort_unstable();
    }
    println!(
        "  Q2 dense ftran  off p50={:.2}us  on p50={:.2}us  -> {:.2}x   \
(mean {:.2}x)",
        pct(&foff, 50),
        pct(&fon, 50),
        pct(&foff, 50) / pct(&fon, 50).max(1e-9),
        (foff.iter().sum::<u128>() as f64) / (fon.iter().sum::<u128>() as f64).max(1e-9),
    );
    println!(
        "  Q2 dense btran  off p50={:.2}us  on p50={:.2}us  -> {:.2}x   \
(mean {:.2}x)",
        pct(&boff, 50),
        pct(&bon, 50),
        pct(&boff, 50) / pct(&bon, 50).max(1e-9),
        (boff.iter().sum::<u128>() as f64) / (bon.iter().sum::<u128>() as f64).max(1e-9),
    );
    println!(
        "  Q2 reach route FIRED {sweeps} times over {} solves; max |off-on| = {worst:.3e}",
        2 * reps * probes.len()
    );

    // ---- Q3: what would the sparse API buy? --------------------------------
    let mut out: Vec<(usize, f64)> = Vec::new();
    let mut fsp: Vec<u128> = Vec::new();
    let mut work: Vec<usize> = Vec::new();
    for _ in 0..reps {
        for &k in probes.iter() {
            let rhs = [(k, 1.0)];
            let t = Instant::now();
            on.ftran_sparse(&rhs, &mut out).expect("ftran_sparse");
            fsp.push(t.elapsed().as_nanos());
            work.push(on.last_sparse_solve_work());
        }
    }
    fsp.sort_unstable();
    work.sort_unstable();
    println!(
        "  Q3 ftran_sparse p50={:.2}us -> {:.1}x vs dense-off, {:.1}x vs dense-on   \
| work p50={} p90={} (of m={m})",
        pct(&fsp, 50),
        pct(&foff, 50) / pct(&fsp, 50).max(1e-9),
        pct(&fon, 50) / pct(&fsp, 50).max(1e-9),
        work[work.len() / 2],
        work[work.len() * 9 / 10],
    );

    // Vacuity guard: a silent fallback plus a wall-clock table is exactly how a
    // no-op reports success. If the route never fired, the Q2 numbers above are
    // the dense path measured against itself and mean nothing.
    assert!(
        worst < 1e-6,
        "arms disagree by {worst:.3e} -- not a valid A/B"
    );
    if sweeps == 0 {
        eprintln!(
            "ROUTE NEVER FIRED: every solve on this basis overran the {density} cap, so the \
Q2 columns compare the dense sweep against itself."
        );
        std::process::exit(1);
    }
}
