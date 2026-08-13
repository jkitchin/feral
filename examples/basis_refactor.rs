//! A/B a Matrix Market basis through both symbolic paths: AMD over the whole
//! basis (feral's pre-triangularization behavior) vs Suhl–Suhl peel + AMD over
//! the residual bump. Interleaved, in one process, on the identical matrix.
//!
//! Usage: cargo run --release --example basis_refactor -- <file.mtx> [reps]

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
    let mut col_ptr = vec![0usize; m + 1];
    for &(_, j, _) in &trip {
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

struct Stat {
    sym: Vec<f64>,
    num: Vec<f64>,
    nnz_lu: usize,
    bump: usize,
    runs: usize,
}

impl Stat {
    fn new() -> Self {
        Stat {
            sym: Vec::new(),
            num: Vec::new(),
            nnz_lu: 0,
            bump: 0,
            runs: 0,
        }
    }
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n.max(2.0 - 1.0);
    (m, var.sqrt())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: basis_refactor <file.mtx> [reps]");
    let reps: usize = args.get(2).map_or(20, |s| s.parse().expect("reps"));
    let a = read_mtx(path);
    println!("m={} nnz={} reps={reps}", a.m, a.row_idx.len());
    let params = LuParams::default();

    let mut peel = Stat::new();
    let mut full = Stat::new();
    let mut dense = Stat::new();
    let mut dense_fired = 0usize;
    let dense_cap: usize = std::env::var("FERAL_DENSE_BUMP_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);

    // Interleaved A/B (discopt CLAUDE.md rule 9): alternate arms so any drift in
    // machine state hits both equally.
    for _ in 0..reps {
        for arm in 0..3 {
            let s = match arm {
                0 => &mut peel,
                1 => &mut full,
                _ => &mut dense,
            };
            let p = if arm == 2 {
                LuParams {
                    dense_bump_max_dim: dense_cap,
                    ..params.clone()
                }
            } else {
                params.clone()
            };
            let t0 = Instant::now();
            let sym = if arm == 1 {
                SparseLuSymbolic::analyze_amd_only(&a).expect("analyze")
            } else {
                SparseLuSymbolic::analyze(&a).expect("analyze")
            };
            s.sym.push(t0.elapsed().as_secs_f64() * 1e3);
            let t1 = Instant::now();
            let lu = SparseLu::factor(&a, &sym, p).expect("factor");
            s.num.push(t1.elapsed().as_secs_f64() * 1e3);
            s.nnz_lu = lu.factor_nnz();
            s.bump = sym.bump_hi - sym.bump_lo;
            s.runs += 1;
            if arm == 2 && lu.used_dense_bump() {
                dense_fired += 1;
            }
        }
    }

    for (name, s) in [
        ("peel+AMD(bump)", &peel),
        ("AMD(full)", &full),
        ("peel+denseBump", &dense),
    ] {
        let (ms, ss) = mean_sd(&s.sym);
        let (mn, sn) = mean_sd(&s.num);
        println!(
            "{name:16} symbolic={ms:7.3} +- {ss:5.3} ms  numeric={mn:7.2} +- {sn:4.2} ms  \
             total={:7.2} ms  nnz(LU)={}  bump={}",
            ms + mn,
            s.nnz_lu,
            s.bump
        );
    }
    let (ps, _) = mean_sd(&peel.sym);
    let (pn, _) = mean_sd(&peel.num);
    let (fs, _) = mean_sd(&full.sym);
    let (fn_, _) = mean_sd(&full.num);
    println!(
        "speedup: symbolic {:.2}x  numeric {:.2}x  total {:.2}x   peeled={:.2}%",
        fs / ps,
        fn_ / pn,
        (fs + fn_) / (ps + pn),
        100.0 * (a.m - peel.bump) as f64 / a.m as f64,
    );

    // Rule 6: an executed count, nonzero exit if nothing ran.
    let (ds, _) = mean_sd(&dense.sym);
    let (dn, _) = mean_sd(&dense.num);
    println!(
        "dense-bump vs AMD(full): total {:.2}x   (route fired on {}/{} reps, cap={})",
        (fs + fn_) / (ds + dn),
        dense_fired,
        dense.runs,
        dense_cap,
    );

    // Rule 6: executed counts, nonzero exit if a probe never fired.
    println!(
        "runs={} {} {} dense_fired={}",
        peel.runs, full.runs, dense.runs, dense_fired
    );
    if peel.runs == 0 || full.runs == 0 || dense_fired == 0 {
        eprintln!("ERROR: an arm never ran (dense route may have fallen back)");
        std::process::exit(1);
    }
}
