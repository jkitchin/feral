//! diag_221_rook_cost — issue #221: factor time on KKTs whose fronts take
//! many delayed columns (brainpc3 iter 37/40 from pounce's `--dump kkt`).
//!
//! Loads a pounce `kkt_solve_*.jsonl` dump (1-based triplets), factors it
//! with pounce-feral's defaults (`bk.pivot_threshold = 1e-8`), and prints
//! cold/warm time, inertia, and the delayed-front profile.
//!
//! Usage: diag_221_rook_cost <kkt.jsonl|a.mtx> [...]
//! (FERAL_221_COMPACT=1 prints one line per matrix, for corpus diffs.)

use feral::symbolic::SupernodeParams;
use feral::{CscMatrix, NumericParams, Solver};
use std::time::Instant;

fn load(path: &str) -> Result<(CscMatrix, Vec<f64>), Box<dyn std::error::Error>> {
    if path.ends_with(".mtx") {
        let a = feral::read_mtx(std::path::Path::new(path))?.to_csc()?;
        let n = a.n;
        return Ok((a, vec![1.0; n]));
    }
    let text = std::fs::read_to_string(path)?;
    let line = text.lines().next().ok_or("empty file")?;
    let o: serde_json::Value = serde_json::from_str(line)?;
    let n = o["n"].as_u64().ok_or("no n")? as usize;
    let arr = |k: &str| o[k].as_array().cloned().ok_or(format!("no {k}"));
    let (irn, jcn, vals) = (arr("irn")?, arr("jcn")?, arr("vals")?);
    let (mut r, mut c, mut v) = (vec![], vec![], vec![]);
    for k in 0..irn.len() {
        let i = irn[k].as_u64().ok_or("irn")? as usize - 1;
        let j = jcn[k].as_u64().ok_or("jcn")? as usize - 1;
        let (i, j) = if i >= j { (i, j) } else { (j, i) };
        r.push(i);
        c.push(j);
        v.push(vals[k].as_f64().ok_or("vals")?);
    }
    let rhs: Vec<f64> = match o["rhs"].as_array() {
        Some(b) => b.iter().filter_map(|x| x.as_f64()).collect(),
        None => vec![1.0; n],
    };
    Ok((CscMatrix::from_triplets(n, &r, &c, &v)?, rhs))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: diag_221_rook_cost <kkt.jsonl> [...]");
        std::process::exit(2);
    }
    for path in &args {
        let (a, b) = load(path)?;
        let mut np = NumericParams::default();
        np.bk.pivot_threshold = 1e-8;
        let mut s = Solver::with_params(np, SupernodeParams::default());
        if std::env::var_os("FERAL_221_COMPACT").is_some() {
            let t = Instant::now();
            s.factor(&a, None);
            let dt = t.elapsed().as_secs_f64();
            let st = s.last_factor_stats();
            let x = s.solve(&b).unwrap_or_default();
            let mut ax = vec![0.0; a.n];
            a.symv(&x, &mut ax);
            let r = ax
                .iter()
                .zip(&b)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f64::max);
            println!(
                "{path} {dt:.4} {:?} nnz_l={:?} tiny={:?} res={r:.2e}",
                s.inertia().map(|i| (i.positive, i.negative, i.zero)),
                st.as_ref().map(|x| x.nnz_l),
                st.as_ref().map(|x| x.n_tiny),
            );
            continue;
        }
        println!("== {path} (n={})", a.n);
        for pass in ["cold", "warm"] {
            let t = Instant::now();
            let st = s.factor(&a, None);
            let dt = t.elapsed().as_secs_f64();
            let stats = s.last_factor_stats();
            println!(
                "  {pass}: {dt:.3}s status={st:?} inertia={:?} nnz_l={:?} tiny={:?} cb={:?}",
                s.inertia(),
                stats.as_ref().map(|x| x.nnz_l),
                stats.as_ref().map(|x| x.n_tiny),
                stats.as_ref().map(|x| x.n_cascade_break_nodes),
            );
            if let Some(x) = stats.as_ref() {
                println!(
                    "    ordering={:?} scaling={:?}",
                    x.ordering_info, x.scaling_info
                );
            }
        }
        // Normwise backward error of the plain and refined solves.
        let berr = |x: &[f64]| {
            let mut ax = vec![0.0; a.n];
            a.symv(x, &mut ax);
            let r = ax
                .iter()
                .zip(&b)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f64::max);
            let xn = x.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let bn = b.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let an = a.values.iter().map(|v| v.abs()).fold(0.0, f64::max);
            r / (an * xn + bn)
        };
        if let Ok(x) = s.solve(&b) {
            print!("  berr plain={:.2e}", berr(&x));
        }
        if let Ok(x) = s.solve_refined(&a, &b) {
            print!(" refined={:.2e}", berr(&x));
        }
        println!();
        if let Some(f) = s.factors() {
            let mut fronts: Vec<_> = f
                .node_factors
                .iter()
                .filter(|nf| nf.n_delayed_in > 0)
                .map(|nf| (nf.n_delayed_in, nf.nrow, nf.ncol, nf.nelim))
                .collect();
            let total_in: usize = fronts.iter().map(|x| x.0).sum();
            fronts.sort_by_key(|x| std::cmp::Reverse(x.0));
            println!(
                "  delayed fronts: {} of {}, sum n_delayed_in = {total_in}",
                fronts.len(),
                f.node_factors.len()
            );
            for (din, nrow, ncol, nelim) in fronts.iter().take(12) {
                println!(
                    "    n_delayed_in={din:>6} nrow={nrow:>6} ncol={ncol:>6} nelim={nelim:>6}"
                );
            }
            // Fronts that originate delays (none delayed in, some out).
            let mut shown = 0;
            for nf in &f.node_factors {
                if nf.n_delayed_in == 0 && nf.nelim < nf.ncol && shown < 4 {
                    shown += 1;
                    let orig: Vec<usize> = nf.row_indices.iter().map(|&r| f.perm[r]).collect();
                    println!(
                        "    origin front ncol={} nelim={} nrow={} cols={:?} rows={:?}",
                        nf.ncol,
                        nf.nelim,
                        nf.nrow,
                        &orig[..nf.ncol],
                        &orig[nf.ncol..]
                    );
                }
            }
            // Optional: write, per original column, how many fronts it
            // arrived at as a delayed column (FERAL_221_DUMP=<path>).
            if let Ok(out) = std::env::var("FERAL_221_DUMP") {
                let mut hops = vec![0usize; f.n];
                for nf in &f.node_factors {
                    let own = nf.ncol - nf.n_delayed_in;
                    for &r in &nf.row_indices[own..nf.ncol] {
                        if r < f.n {
                            hops[f.perm[r]] += 1;
                        }
                    }
                }
                let txt: Vec<String> = hops.iter().map(|h| h.to_string()).collect();
                std::fs::write(out, txt.join("\n"))?;
            }
        }
    }
    Ok(())
}
