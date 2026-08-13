//! Is the sparse LU's cost on a real simplex basis a *pivot-ordering* question?
//!
//! Issue #161's reframing (after its original premise was retracted) asks:
//! numeric factorization is >=41% of the motivating LP's wall, the factor shows
//! 5.7x fill on a basis averaging 6.8 nonzeros per column — is the fill itself
//! the cost, i.e. would a better column ordering fix it?
//!
//! This answers it by holding everything constant except the ordering. Each arm
//! feeds a different permutation of the `AᵀA` (column-intersection) pattern to
//! `SparseLuSymbolic::with_order` and factors the identical basis, reporting the
//! fill it produces and the time the numeric step takes to produce it.
//!
//! The two questions are separable and the answer is not the same for both:
//!
//!   * across orderings, how much does **fill** move?
//!   * at a *given* fill, how much does **numeric time** move?
//!
//! Usage: cargo run --release --example lu_fill_orderings -- <file.mtx> [reps]

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
    trip.sort_by_key(|&(i, j, _)| (j, i));
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for &(i, j, v) in &trip {
        cols[j].push((i, v));
    }
    SparseColMatrix::from_sparse_columns(m, &cols).expect("basis")
}

/// The `AᵀA` pattern in the `i32` CSC form the ordering crates take.
fn ata_i32(a: &SparseColMatrix) -> (Vec<i32>, Vec<i32>) {
    let pat = a.ata_pattern();
    (
        pat.col_ptr.iter().map(|&x| x as i32).collect(),
        pat.row_idx.iter().map(|&x| x as i32).collect(),
    )
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: lu_fill_orderings <file.mtx> [reps]");
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(9);

    let a = read_mtx(&path);
    let m = a.m;
    let nnz = a.nnz();
    let (col_ptr, row_idx) = ata_i32(&a);
    let cpat = feral_ordering_core::CscPattern::new(m, &col_ptr, &row_idx).expect("AtA pattern");

    println!(
        "{}  m={m} nnz(A)={nnz} ({:.2}/col)  reps={reps}",
        path.rsplit('/').next().unwrap(),
        nnz as f64 / m as f64,
    );
    println!("  ordering        nnz(LU)     fill   numeric(ms)   ns/factor-nnz");

    let mut arms: Vec<(&str, Vec<usize>)> = Vec::new();
    arms.push(("natural", (0..m).collect()));
    for (name, perm) in [
        ("AMD", feral_amd::amd_order(&cpat).ok()),
        ("AMF", feral_amf::amf_order(&cpat).ok()),
        ("METIS", feral_metis::metis_order(&cpat).ok()),
    ] {
        match perm {
            Some(p) => arms.push((name, p.iter().map(|&x| x as usize).collect())),
            None => println!("  {name:<14} (ordering unavailable)"),
        }
    }

    let mut baseline_fill = 0.0;
    for (name, qcol) in arms {
        let sym = match SparseLuSymbolic::with_order(m, qcol) {
            Ok(s) => s,
            Err(e) => {
                println!("  {name:<14} rejected: {e:?}");
                continue;
            }
        };
        // `with_order` reports the whole basis as bump, so the dense-bump route
        // never fires here (it requires a symbolic that triangularized). Every
        // arm therefore runs the same sparse scatter kernel and the comparison
        // is ordering against ordering, nothing else.
        let p = LuParams::default();
        let mut times = Vec::with_capacity(reps);
        let mut fnnz = 0usize;
        for _ in 0..reps {
            let t = Instant::now();
            let lu = match SparseLu::factor(&a, &sym, p.clone()) {
                Ok(lu) => lu,
                Err(e) => {
                    println!("  {name:<14} factor failed: {e:?}");
                    break;
                }
            };
            times.push(t.elapsed().as_secs_f64() * 1e3);
            fnnz = lu.factor_nnz();
        }
        if times.is_empty() {
            continue;
        }
        let ms = median(&mut times);
        let fill = fnnz as f64 / nnz as f64;
        if baseline_fill == 0.0 {
            baseline_fill = fill;
        }
        println!(
            "  {name:<14} {fnnz:>8}   {fill:>6.2}x   {ms:>9.2}     {:>7.1}",
            ms * 1e6 / fnnz as f64,
        );
    }

    // The other half of the question: at a fixed ordering, what does the
    // *kernel* choice do? `analyze` + the dense-bump route factors the same
    // basis with the same fill class through a blocked dense kernel.
    let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
    for (label, cap) in [("peel+sparse", 0usize), ("peel+denseBump", 4096)] {
        let p = LuParams {
            dense_bump_max_dim: cap,
            ..LuParams::default()
        };
        let mut times = Vec::with_capacity(reps);
        let mut fnnz = 0usize;
        let mut fired = false;
        for _ in 0..reps {
            let t = Instant::now();
            let lu = SparseLu::factor(&a, &sym, p.clone()).expect("factor");
            times.push(t.elapsed().as_secs_f64() * 1e3);
            fnnz = lu.factor_nnz();
            fired = lu.used_dense_bump();
        }
        let ms = median(&mut times);
        println!(
            "  {label:<14} {fnnz:>8}   {:>6.2}x   {ms:>9.2}     {:>7.1}   (bump={}, dense route fired={fired})",
            fnnz as f64 / nnz as f64,
            ms * 1e6 / fnnz as f64,
            sym.bump_hi - sym.bump_lo,
        );
    }
}
