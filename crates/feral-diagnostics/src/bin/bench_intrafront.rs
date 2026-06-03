//! Lever 1.1 A/B benchmark: intra-front parallel Schur update.
//!
//! Times the parallel multifrontal driver on wide-front matrices with
//! the intra-front parallelism OFF vs ON, and checks the results are
//! bit-identical (inertia + a residual). The toggle is the
//! `FERAL_INTRAFRONT` env var honored by
//! `factorize_multifrontal_supernodal_parallel`.
//!
//! Two matrix sources:
//!   * synthetic dense diagonally-dominant SPD fronts (n given by
//!     `SIZES`, default 1200,1600,2000) — a single wide root supernode,
//!     the case Lever 1.1 targets;
//!   * any `.mtx` paths passed as CLI args (e.g. a wide-root KKT).
//!
//! Run:
//!   cargo run --release --bin bench_intrafront
//!   cargo run --release --bin bench_intrafront -- path/to/a.mtx ...

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal_supernodal_parallel, NumericParams};
use feral::read_mtx;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::CscMatrix;

fn dense_spd(n: usize) -> CscMatrix {
    let cap = n * (n + 1) / 2;
    let mut rows = Vec::with_capacity(cap);
    let mut cols = Vec::with_capacity(cap);
    let mut vals = Vec::with_capacity(cap);
    for c in 0..n {
        rows.push(c);
        cols.push(c);
        vals.push(n as f64 + 1.0);
        for r in (c + 1)..n {
            rows.push(r);
            cols.push(c);
            vals.push(1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("dense SPD triplets")
}

fn min_time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        let s = t.elapsed().as_secs_f64();
        if s < best {
            best = s;
        }
    }
    best
}

fn bench_one(label: &str, a: &CscMatrix) {
    let sym = match symbolic_factorize(a, &SupernodeParams::default()) {
        Ok(s) => s,
        Err(e) => {
            println!("{label}: symbolic failed: {e:?}");
            return;
        }
    };
    let params = NumericParams::default();
    let reps = 3;

    // OFF
    std::env::set_var("FERAL_INTRAFRONT", "0");
    let (off_f, off_i) = factorize_multifrontal_supernodal_parallel(a, &sym, &params)
        .expect("factor (intrafront off)");
    let t_off = min_time(reps, || {
        let _ = factorize_multifrontal_supernodal_parallel(a, &sym, &params).unwrap();
    });

    // ON
    std::env::set_var("FERAL_INTRAFRONT", "1");
    let (on_f, on_i) = factorize_multifrontal_supernodal_parallel(a, &sym, &params)
        .expect("factor (intrafront on)");
    let t_on = min_time(reps, || {
        let _ = factorize_multifrontal_supernodal_parallel(a, &sym, &params).unwrap();
    });
    std::env::remove_var("FERAL_INTRAFRONT");

    // Correctness: inertia equal, per-supernode L bit-identical.
    let inertia_ok = off_i == on_i;
    let mut bits_ok = true;
    let (offc, onc) = (&off_f.node_factors, &on_f.node_factors);
    if offc.len() != onc.len() {
        bits_ok = false;
    } else {
        for (s, p) in offc.iter().zip(onc.iter()) {
            let (sl, pl) = (&s.frontal_factors.l, &p.frontal_factors.l);
            if sl.len() != pl.len()
                || sl
                    .iter()
                    .zip(pl.iter())
                    .any(|(x, y)| x.to_bits() != y.to_bits())
            {
                bits_ok = false;
                break;
            }
        }
    }

    let speedup = t_off / t_on;
    println!(
        "{label:<22} n={:<7} off={:8.2}ms on={:8.2}ms  speedup={:5.2}x  inertia_eq={} bit_exact={}",
        a.n,
        t_off * 1e3,
        t_on * 1e3,
        speedup,
        inertia_ok,
        bits_ok,
    );
}

fn main() {
    let threads = rayon::current_num_threads();
    println!("intra-front Schur A/B (rayon threads = {threads})\n");
    println!("matrix                 n              off          on         speedup correctness");

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        let sizes = std::env::var("SIZES").unwrap_or_else(|_| "1200,1600,2000".to_string());
        for tok in sizes.split(',') {
            if let Ok(n) = tok.trim().parse::<usize>() {
                let a = dense_spd(n);
                bench_one(&format!("dense_spd_{n}"), &a);
            }
        }
    } else {
        for path in &args {
            match read_mtx(Path::new(path)).and_then(|m| m.to_csc()) {
                Ok(a) => {
                    let name = Path::new(path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or(path);
                    bench_one(name, &a);
                }
                Err(e) => println!("{path}: load failed: {e:?}"),
            }
        }
    }
}
