//! Issue #153 — how much of feral's deficit is the dense kernel, and how
//! much is everything around it?
//!
//! `diag_153_dense_peak` measures `factor_frontal` in isolation on square
//! fronts. That answers "how fast is our kernel", but not "how much of
//! that speed does the solver actually deliver", because the in-solver
//! fronts are trapezoidal (`ncol < nrow`) and each one is wrapped in
//! assembly, extend-add, L-extract and contribution-block work.
//!
//! This binary closes that gap *shape-exactly*. It profiles a real
//! factorization to get the multiset of `(nrow, ncol)` fronts and the
//! measured wallclock of each supernode, then re-times `factor_frontal`
//! standalone on every distinct shape that occurred and reconstitutes
//! the total:
//!
//!   kernel_ns = sum over distinct shapes of count * isolated_kernel_ns
//!   loop_ns   = measured per-supernode wallclock, same run
//!   headroom  = loop_ns / kernel_ns
//!
//! `headroom` is the factor by which the solver is slower than its own
//! arithmetic kernel on the identical shape sequence. It is the ceiling
//! on what non-kernel work (assembly, allocation, indirection, cache
//! traffic) can possibly be worth, and it is measured on *this* corpus
//! rather than assumed. Whatever is left after that is a kernel problem
//! and has to be compared against `dgemm`/`dsytrf`, not against MA57.
//!
//! Caveat, stated rather than hidden: the standalone fronts are
//! diagonally dominant and run with `may_delay = false`, so they take
//! the ordinary 1x1-pivot path. Real fronts that delay pivots do more
//! work than the replay credits them with, which makes `headroom` an
//! *over*-estimate of non-kernel overhead on pivot-heavy matrices. The
//! `sum(ncol) - n` is printed as a delay indicator: a column delayed
//! out of a front is eliminated again in its parent, so it is counted
//! twice. Zero means the replay is exact on that count.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin diag_153_kernel_headroom \
//!     -- [--reps N] [--max-shapes K] <matrix.mtx>...
use feral::dense::factor::factor_frontal;
use feral::numeric::factorize::{
    factorize_multifrontal_supernodal_with_workspace, FactorWorkspace, Profiler,
};
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, NumericParams, SymmetricMatrix};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Deterministic xorshift so runs are comparable across invocations.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn make_front(n: usize) -> SymmetricMatrix {
    let mut rng = Rng(0x243F_6A88_85A3_08D3);
    let mut m = SymmetricMatrix::zeros(n);
    for j in 0..n {
        for i in j..n {
            let v = if i == j {
                n as f64 + 1.0
            } else {
                rng.next_f64()
            };
            m.set(i, j, v);
        }
    }
    m
}

/// `sum_{k<ncol} (nrow-k)^2` multiply-adds — the model used by
/// `diag_200_work_vs_ma57` and `diag_153_dense_peak`, so the MMac/s
/// columns of all three are the same unit.
fn macs(nrow: usize, ncol: usize) -> f64 {
    (0..ncol).map(|k| ((nrow - k) * (nrow - k)) as f64).sum()
}

fn bucket_of(nrow: usize) -> usize {
    match nrow {
        0..=8 => 0,
        9..=16 => 1,
        17..=32 => 2,
        33..=64 => 3,
        65..=128 => 4,
        _ => 5,
    }
}
const BUCKETS: [&str; 6] = ["<=8", "9-16", "17-32", "33-64", "65-128", ">128"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut reps = 5usize;
    let mut max_shapes = 4096usize;
    let mut paths: Vec<String> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--reps" => {
                if let Some(v) = it.next() {
                    reps = v.parse()?;
                }
            }
            "--max-shapes" => {
                if let Some(v) = it.next() {
                    max_shapes = v.parse()?;
                }
            }
            _ => paths.push(a.clone()),
        }
    }

    let bk = BunchKaufmanParams::default();
    for p in &paths {
        let csc = match read_mtx(Path::new(p)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {p}: {e:?}");
                continue;
            }
        };
        let label = Path::new(p)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let sp = SupernodeParams::default();
        let symbolic = match symbolic_factorize_with_method(&csc, &sp, OrderingMethod::Auto) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{label}: symbolic failed: {e:?}");
                continue;
            }
        };
        let nparams = NumericParams::default();
        let mut ws = FactorWorkspace::new();
        if factorize_multifrontal_supernodal_with_workspace(&csc, &symbolic, &nparams, &mut ws)
            .is_err()
        {
            eprintln!("{label}: warm-up factorization failed");
            continue;
        }

        // Best-of-N by driver wall; keep the winning run's timings.
        let mut best_wall = u64::MAX;
        let mut best: Vec<(usize, usize, u64)> = Vec::new();
        for _ in 0..reps {
            let prof = Arc::new(Mutex::new(Profiler::new()));
            let mut np = nparams.clone();
            np.profiler = Some(prof.clone());
            np.pattern_reused_hint = true;
            let t0 = Instant::now();
            let res =
                factorize_multifrontal_supernodal_with_workspace(&csc, &symbolic, &np, &mut ws);
            let wall = t0.elapsed().as_nanos() as u64;
            if let Err(e) = res {
                eprintln!("{label}: timed factorization failed: {e:?}");
                break;
            }
            let guard = match prof.lock() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("{label}: profiler poisoned: {e}");
                    break;
                }
            };
            if wall < best_wall {
                best_wall = wall;
                best = guard
                    .timings()
                    .iter()
                    .map(|t| (t.nrow, t.ncol, t.ns))
                    .collect();
            }
        }
        if best.is_empty() {
            continue;
        }

        // Group by exact (nrow, ncol) shape.
        let mut shapes: HashMap<(usize, usize), (usize, u64)> = HashMap::new();
        for &(nrow, ncol, ns) in &best {
            let e = shapes.entry((nrow, ncol)).or_insert((0, 0));
            e.0 += 1;
            e.1 += ns;
        }
        if shapes.len() > max_shapes {
            eprintln!(
                "{label}: {} distinct shapes exceeds --max-shapes {max_shapes}; skipping",
                shapes.len()
            );
            continue;
        }

        // Time the isolated kernel once per distinct front height; the
        // matrix is reused across the `ncol` variants of that height.
        let mut heights: Vec<usize> = shapes.keys().map(|&(nrow, _)| nrow).collect();
        heights.sort_unstable();
        heights.dedup();
        let mut kernel_ns: HashMap<(usize, usize), f64> = HashMap::new();
        for &nrow in &heights {
            let front = make_front(nrow);
            let mut cols: Vec<usize> = shapes
                .keys()
                .filter(|&&(r, _)| r == nrow)
                .map(|&(_, c)| c)
                .collect();
            cols.sort_unstable();
            for ncol in cols {
                let w = macs(nrow, ncol).max(1.0);
                let r = (2.0e7 / w).ceil().clamp(3.0, 5000.0) as usize;
                for _ in 0..2 {
                    let _ = factor_frontal(&front, ncol, false, &bk)?;
                }
                let mut bn = f64::INFINITY;
                for _ in 0..r {
                    let t0 = Instant::now();
                    let f = factor_frontal(&front, ncol, false, &bk)?;
                    let ns = t0.elapsed().as_nanos() as f64;
                    std::hint::black_box(&f);
                    bn = bn.min(ns);
                }
                kernel_ns.insert((nrow, ncol), bn);
            }
        }

        // Reconstitute per bucket.
        let mut b_loop = [0f64; 6];
        let mut b_kern = [0f64; 6];
        let mut b_macs = [0f64; 6];
        let mut b_cnt = [0usize; 6];
        for (&(nrow, ncol), &(count, sum_ns)) in &shapes {
            let b = bucket_of(nrow);
            b_cnt[b] += count;
            b_loop[b] += sum_ns as f64;
            b_kern[b] += count as f64 * kernel_ns.get(&(nrow, ncol)).copied().unwrap_or(0.0);
            b_macs[b] += count as f64 * macs(nrow, ncol);
        }
        let t_loop: f64 = b_loop.iter().sum();
        let t_kern: f64 = b_kern.iter().sum();
        let t_macs: f64 = b_macs.iter().sum();

        println!(
            "\n{label}  n={} snodes={} shapes={} driver_us={} extra_elims={}",
            csc.n,
            best.len(),
            shapes.len(),
            best_wall / 1000,
            best.iter().map(|t| t.1).sum::<usize>() as i64 - csc.n as i64
        );
        println!(
            "  {:<8}{:>9}{:>13}{:>13}{:>10}{:>13}{:>13}",
            "nrow", "snodes", "loop_us", "kernel_us", "headroom", "solver_Mac/s", "kernel_Mac/s"
        );
        for b in 0..6 {
            if b_cnt[b] == 0 {
                continue;
            }
            println!(
                "  {:<8}{:>9}{:>13.0}{:>13.0}{:>9.2}x{:>13.0}{:>13.0}",
                BUCKETS[b],
                b_cnt[b],
                b_loop[b] / 1000.0,
                b_kern[b] / 1000.0,
                b_loop[b] / b_kern[b].max(1.0),
                b_macs[b] / (b_loop[b] / 1000.0).max(1e-9),
                b_macs[b] / (b_kern[b] / 1000.0).max(1e-9),
            );
        }
        // The handful of shapes that actually decide the matrix.
        let mut top: Vec<((usize, usize), (usize, u64))> =
            shapes.iter().map(|(&k, &v)| (k, v)).collect();
        top.sort_by_key(|&(_, (_, ns))| std::cmp::Reverse(ns));
        println!("  top shapes by loop time:");
        for &((nrow, ncol), (count, sum_ns)) in top.iter().take(6) {
            let k = kernel_ns.get(&(nrow, ncol)).copied().unwrap_or(0.0);
            println!(
                "    nrow={nrow:<5} ncol={ncol:<5} n={count:<6} loop_us={:<9.0} kernel_us={:<9.0} headroom={:.2}x",
                sum_ns as f64 / 1000.0,
                count as f64 * k / 1000.0,
                sum_ns as f64 / (count as f64 * k).max(1.0),
            );
        }
        println!(
            "  {:<8}{:>9}{:>13.0}{:>13.0}{:>9.2}x{:>13.0}{:>13.0}",
            "TOTAL",
            best.len(),
            t_loop / 1000.0,
            t_kern / 1000.0,
            t_loop / t_kern.max(1.0),
            t_macs / (t_loop / 1000.0).max(1e-9),
            t_macs / (t_kern / 1000.0).max(1e-9),
        );
    }
    Ok(())
}
