//! Issue #132 measure-first: how much Forrest–Tomlin eta arithmetic does the
//! current `update_sparse` pay on a real simplex update trace? #132
//! (Schork–Gondzio permute-only re-triangularization) can remove the row-eta
//! only on updates whose bump digraph is acyclic. This probe bounds the
//! *ceiling* — the total eta work that exists to be removed — by replaying the
//! in-tree casctanks trace through the real `SparseLu` and recording, per
//! update, `last_eta_ops` (solve-replay op count of the eta) and
//! `last_update_work` (build-time scatter count).
//!
//! If most updates already carry a tiny eta, #132's ceiling is low regardless
//! of the acyclic fraction; if a meaningful share of the work is eta ops, the
//! next step (the acyclic-bump symbolic test) is worth building.
//!
//! Run: cargo run --release --bin probe_ft_eta -- [trace.txt]

use std::path::PathBuf;

use feral::{LuParams, LuPivoting, SparseColMatrix, SparseLu, SparseLuSymbolic};

type SparseCol = Vec<(usize, f64)>;

struct Segment {
    basis: Vec<SparseCol>,
    updates: Vec<(usize, SparseCol)>,
}

fn parse_col(fields: &[&str]) -> SparseCol {
    let mut col: SparseCol = fields
        .iter()
        .filter_map(|tok| {
            let (r, v) = tok.split_once(':')?;
            Some((r.parse::<usize>().ok()?, v.parse::<f64>().ok()?))
        })
        .collect();
    col.sort_by_key(|&(r, _)| r);
    col
}

fn to_dense(col: &SparseCol, m: usize) -> Vec<f64> {
    let mut d = vec![0.0; m];
    for &(r, v) in col {
        d[r] = v;
    }
    d
}

/// Factor pivot threshold from `FERAL_PIVTOL` (default 1.0 = strict partial
/// pivoting, feral's current default). Lower values (0.1, 0.01) enable the
/// within-threshold diagonal/sparsity-preserving pivot — issue #133 Stage 1.
fn pivtol_from_env() -> f64 {
    std::env::var("FERAL_PIVTOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0)
}

/// ‖B·x − rhs‖∞ / ‖rhs‖∞ for basis `b` (sparse columns) and solution `x`.
fn residual(b: &[SparseCol], x: &[f64], rhs: &[f64]) -> f64 {
    let mut r = vec![0.0; rhs.len()];
    for (j, col) in b.iter().enumerate() {
        let xj = x[j];
        for &(i, v) in col {
            r[i] += v * xj;
        }
    }
    let num = r
        .iter()
        .zip(rhs)
        .map(|(&ri, &bi)| (ri - bi).abs())
        .fold(0.0, f64::max);
    let den = rhs.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300);
    num / den
}

/// Synthetic set-covering basis measurement (`--cover m per_col nupd`): build
/// an m×m 0/1 covering-structured basis, factor it, and replay `nupd` random
/// single-column replacements with sparse structural columns (~per_col
/// nonzeros) — the covering-LP pivot shape where discopt reports update() as
/// 83% of the root LP and "B⁻¹ dense on covering bases". Reports the same eta
/// vs build-work split. This is where #132's ceiling would be large *if* the
/// bumps are acyclic.
fn run_cover(m: usize, per_col: usize, nupd: usize) {
    // Deterministic LCG.
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut below = |n: usize| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as usize) % n
    };
    let rand_col = |bel: &mut dyn FnMut(usize) -> usize| -> SparseCol {
        let mut rows = Vec::with_capacity(per_col);
        while rows.len() < per_col.min(m) {
            let r = bel(m);
            if !rows.iter().any(|&(rr, _)| rr == r) {
                rows.push((r, 1.0));
            }
        }
        rows.push((rows[0].0, 0.0)); // no-op to keep signature; removed below
        rows.pop();
        rows.sort_by_key(|&(r, _)| r);
        rows
    };
    // Basis: identity backbone (guarantees nonsingular) + covering entries.
    let mut basis: Vec<SparseCol> = (0..m)
        .map(|j| {
            let mut c = rand_col(&mut below);
            if !c.iter().any(|&(r, _)| r == j) {
                c.push((j, 1.0));
            }
            // strong diagonal so it factors
            for e in c.iter_mut() {
                if e.0 == j {
                    e.1 = m as f64;
                }
            }
            c.sort_by_key(|&(r, _)| r);
            c
        })
        .collect();

    let pivtol = pivtol_from_env();
    let params = LuParams {
        max_updates: 1_000_000,
        max_growth: 1e30,
        pivot_threshold: pivtol,
        pivoting: LuPivoting::GilbertPeierls,
        ..LuParams::default()
    };
    let factor = |basis: &[SparseCol]| -> Option<SparseLu> {
        let a = SparseColMatrix::from_sparse_columns(m, basis).ok()?;
        let sym = SparseLuSymbolic::analyze(&a).ok()?;
        SparseLu::factor(&a, &sym, params.clone()).ok()
    };
    let mut lu = factor(&basis).expect("cover factor");
    let fnnz0 = lu.factor_nnz();
    // Fixed RHS for a solve-accuracy (stability) check across the update chain.
    let rhs: Vec<f64> = (0..m).map(|i| 1.0 + ((i % 7) as f64) * 0.5).collect();
    let mut worst_resid = 0.0f64;
    let (mut sum_eta, mut sum_work, mut max_eta, mut n, mut refac) =
        (0u64, 0u64, 0usize, 0usize, 0usize);
    for _ in 0..nupd {
        let slot = below(m);
        let mut col = rand_col(&mut below);
        if !col.iter().any(|&(r, _)| r == slot) {
            col.push((slot, 1.0));
            col.sort_by_key(|&(r, _)| r);
        }
        let dcol = to_dense(&col, m);
        basis[slot] = col;
        n += 1;
        if lu.update(slot, &dcol).is_err() {
            refac += 1;
            lu = match factor(&basis) {
                Some(l) => l,
                None => break,
            };
            continue;
        }
        sum_eta += lu.last_eta_ops() as u64;
        sum_work += lu.last_update_work() as u64;
        max_eta = max_eta.max(lu.last_eta_ops());
        // Solve accuracy: ‖B x − rhs‖∞ / ‖rhs‖∞.
        let mut x = rhs.clone();
        if lu.ftran(&mut x).is_ok() {
            worst_resid = worst_resid.max(residual(&basis, &x, &rhs));
        }
    }
    println!(
        "synthetic set-cover m={m} per_col={per_col} nupd={n} pivtol={pivtol} \
         refactor_signals={refac}"
    );
    println!("factor_nnz(initial)={fnnz0}  worst_residual={worst_resid:.2e}");
    println!(
        "eta_ops: total={sum_eta} max={max_eta} mean={:.1}",
        if n > 0 {
            sum_eta as f64 / n as f64
        } else {
            0.0
        }
    );
    println!(
        "update_work: total={sum_work}  eta_ops/(eta_ops+update_work) = {:.1}%",
        if sum_eta + sum_work > 0 {
            100.0 * sum_eta as f64 / (sum_eta + sum_work) as f64
        } else {
            0.0
        }
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let first = args.next();
    if first.as_deref() == Some("--cover") {
        let m: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(800);
        let per_col: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(6);
        let nupd: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(500);
        run_cover(m, per_col, nupd);
        return;
    }
    let path = first
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tests/data/lu_trace/casctanks.txt"));
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {}: {e}", path.display());
            std::process::exit(1);
        }
    };

    let mut m = 0usize;
    let mut segments: Vec<Segment> = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("M") => m = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            Some("REFACTOR") => segments.push(Segment {
                basis: Vec::new(),
                updates: Vec::new(),
            }),
            Some("BCOL") => {
                let rest: Vec<&str> = it.collect();
                if let Some(seg) = segments.last_mut() {
                    seg.basis.push(parse_col(&rest[2.min(rest.len())..]));
                }
            }
            Some("UPDATE") => {
                let rest: Vec<&str> = it.collect();
                if let (Some(seg), Ok(slot)) = (segments.last_mut(), rest[0].parse::<usize>()) {
                    seg.updates
                        .push((slot, parse_col(&rest[2.min(rest.len())..])));
                }
            }
            _ => {}
        }
    }

    let pivtol = pivtol_from_env();
    let params = LuParams {
        max_updates: 1_000_000,
        max_growth: 1e30,
        pivot_threshold: pivtol,
        pivoting: LuPivoting::GilbertPeierls,
        ..LuParams::default()
    };

    let mut n_updates = 0usize;
    let mut n_refactor_signals = 0usize;
    let mut sum_fnnz0 = 0u64; // sum of initial factor_nnz across segments
    let mut worst_resid = 0.0f64;
    let rhs: Vec<f64> = (0..m).map(|i| 1.0 + ((i % 7) as f64) * 0.5).collect();
    let mut eta_zero = 0usize; // updates whose eta is trivial (ops == 0)
    let mut sum_eta = 0u64;
    let mut sum_work = 0u64;
    let mut max_eta = 0usize;
    // eta_ops histogram buckets.
    let buckets = [1usize, 10, 100, 1_000, 10_000, 100_000, usize::MAX];
    let mut hist = [0usize; 7];

    for seg in &segments {
        if seg.basis.len() != m {
            continue;
        }
        let factor = |basis: &[SparseCol]| -> Option<SparseLu> {
            let a = SparseColMatrix::from_sparse_columns(m, basis).ok()?;
            let sym = SparseLuSymbolic::analyze(&a).ok()?;
            SparseLu::factor(&a, &sym, params.clone()).ok()
        };
        let mut basis = seg.basis.clone();
        let mut lu = match factor(&basis) {
            Some(l) => l,
            None => continue,
        };
        sum_fnnz0 += lu.factor_nnz() as u64;
        for (slot, col) in &seg.updates {
            let dcol = to_dense(col, m);
            basis[*slot] = col.clone();
            n_updates += 1;
            if lu.update(*slot, &dcol).is_err() {
                n_refactor_signals += 1;
                lu = match factor(&basis) {
                    Some(l) => l,
                    None => break,
                };
                continue;
            }
            let mut x = rhs.clone();
            if lu.ftran(&mut x).is_ok() {
                worst_resid = worst_resid.max(residual(&basis, &x, &rhs));
            }
            let e = lu.last_eta_ops();
            let w = lu.last_update_work();
            sum_eta += e as u64;
            sum_work += w as u64;
            max_eta = max_eta.max(e);
            if e == 0 {
                eta_zero += 1;
            }
            for (bi, &hi) in buckets.iter().enumerate() {
                if e < hi {
                    hist[bi] += 1;
                    break;
                }
            }
        }
    }

    println!(
        "trace={} m={m} segments={} pivtol={pivtol}",
        path.display(),
        segments.len()
    );
    println!(
        "updates_committed={n_updates}  refactor_signals={n_refactor_signals}  \
         sum_factor_nnz(initial)={sum_fnnz0}  worst_residual={worst_resid:.2e}"
    );
    println!(
        "eta_ops: total={sum_eta}  max={max_eta}  \
         mean={:.1}  committed-with-zero-eta={eta_zero} ({:.1}%)",
        if n_updates > 0 {
            sum_eta as f64 / n_updates as f64
        } else {
            0.0
        },
        if n_updates > 0 {
            100.0 * eta_zero as f64 / n_updates as f64
        } else {
            0.0
        },
    );
    println!(
        "update_work(build scatters): total={sum_work}  \
         eta_ops / (eta_ops + update_work) = {:.1}%",
        if sum_eta + sum_work > 0 {
            100.0 * sum_eta as f64 / (sum_eta + sum_work) as f64
        } else {
            0.0
        },
    );
    let labels = ["0", "1-9", "10-99", "100-999", "1k-9k", "10k-99k", ">=100k"];
    // Shift: bucket 0 counts eta<1 i.e. ==0; report explicitly.
    println!("eta_ops histogram (per committed update):");
    println!("  {:>8}: {}", labels[0], eta_zero);
    for bi in 1..buckets.len() {
        println!("  {:>8}: {}", labels[bi], hist[bi]);
    }
}
