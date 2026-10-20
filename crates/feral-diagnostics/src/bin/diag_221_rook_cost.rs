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

/// `A <- D·A·D` with `d_i` in [0.7, 1.4] from a fixed xorshift stream
/// (same generator as tests/issue_215_delay_growth_cap.rs). Sylvester's
/// law keeps the inertia, so every seed must count the same.
fn congruence(m: &mut CscMatrix, seed: u64) {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
    let d: Vec<f64> = (0..m.n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            0.7 + 0.7 * ((x >> 11) as f64 / (1u64 << 53) as f64)
        })
        .collect();
    for j in 0..m.n {
        for p in m.col_ptr[j]..m.col_ptr[j + 1] {
            m.values[p] *= d[j] * d[m.row_idx[p]];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: diag_221_rook_cost <kkt.jsonl> [...]");
        std::process::exit(2);
    }
    let diag = std::env::var_os("FERAL_221_PANEL_DIAG").is_some();
    if diag {
        feral::dense::factor::PANEL_DIAG_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let phases = std::env::var_os("FERAL_221_PHASES").is_some();
    if phases {
        feral::dense::factor::PHASE_TIMING_ENABLED
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
    for path in &args {
        let (mut a, b) = load(path)?;
        if let Some(seed) = feral::env::u64_var("FERAL_221_SEED") {
            congruence(&mut a, seed);
        }
        let mut np = NumericParams::default();
        np.bk.pivot_threshold = feral::env::f64_var("FERAL_221_U").unwrap_or(1e-8);
        let mut s = Solver::with_params(np, SupernodeParams::default());
        if std::env::var_os("FERAL_221_NOESC").is_some() {
            s = s.with_ordering_escalation(None);
        }
        if let Some(e) = feral::env::f64_var("FERAL_221_CB_EPS") {
            s = s.with_cascade_break_eps(e);
        }
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
            let sym = s.work_estimate().map(|w| w.factor_nnz).unwrap_or(0);
            let (mut carry, mut sym_area) = (0usize, 0usize);
            if let Some(f) = s.factors() {
                for nf in &f.node_factors {
                    carry += nf.n_delayed_in * nf.nrow;
                    sym_area += (nf.ncol - nf.n_delayed_in) * (nf.nrow - nf.n_delayed_in);
                }
            }
            // Chain-excess calibration (issue #221 backstop): per node,
            // excess = actual front area - symbolic area, summed over
            // children that still delay; sym = subtree symbolic area.
            let (mut worst_ratio, mut worst_excess) = (0.0f64, 0usize);
            if let (Some(f), Some(sy)) = (s.factors(), s.symbolic()) {
                let ns = sy.supernodes.len();
                let mut ex = vec![0usize; ns];
                let mut sa = vec![0usize; ns];
                for (i, sn) in sy.supernodes.iter().enumerate() {
                    let nf = &f.node_factors[i];
                    let own_sym = sn.ncol * sn.nrow;
                    let own_act = nf.ncol * nf.nrow;
                    let mut e = own_act.saturating_sub(own_sym);
                    let mut a = own_sym;
                    for &c in &sn.children {
                        a += sa[c];
                        if f.node_factors[c].ncol > f.node_factors[c].nelim {
                            e += ex[c];
                        }
                    }
                    sa[i] = a;
                    ex[i] = e;
                    let ratio = e as f64 / a.max(1) as f64;
                    if ratio > worst_ratio && e > 0 {
                        worst_ratio = ratio;
                        worst_excess = e;
                    }
                }
            }
            println!("    chain: worst excess/subtree = {worst_ratio:.2} at excess {worst_excess}");
            println!(
                "{path} {dt:.4} {:?} nnz_l={:?} tiny={:?} res={r:.2e} grow={:.2} carry/area={:.2} tfa={:?} cb={:?}",
                s.inertia().map(|i| (i.positive, i.negative, i.zero)),
                st.as_ref().map(|x| x.nnz_l),
                st.as_ref().map(|x| x.n_tiny),
                st.as_ref()
                    .map(|x| x.nnz_l as f64 / sym.max(1) as f64)
                    .unwrap_or(0.0),
                carry as f64 / sym_area.max(1) as f64,
                st.as_ref().map(|x| x.n_threshold_fail_accepts),
                st.as_ref().map(|x| x.n_cascade_break_nodes),
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
        if diag {
            use feral::dense::factor::panel_diag as pd;
            let g =
                |c: &std::sync::atomic::AtomicU64| c.swap(0, std::sync::atomic::Ordering::Relaxed);
            println!(
                "  delays (cold+warm): 1x1={} 1x1_tiny={} 2x2_growth={} 2x2_det={} 2x2_both={} 2x2_negdet={}",
                g(&pd::SCALAR_1X1_DELAY),
                g(&pd::SCALAR_1X1_DELAY_TINY),
                g(&pd::SCALAR_2X2_DELAY_GROWTH),
                g(&pd::SCALAR_2X2_DELAY_DET),
                g(&pd::SCALAR_2X2_DELAY_BOTH),
                g(&pd::SCALAR_2X2_DELAY_NEGDET),
            );
            println!(
                "  panel: full={} partial={} delayed={} inline_pivots={} scalar_pivots={} scalar_tail={} fb_bound={} fb_swap1x1={} fb_lapack={} fb_growth={}",
                g(&pd::PANEL_FULL),
                g(&pd::PANEL_PARTIAL),
                g(&pd::PANEL_DELAYED),
                g(&pd::PIVOTS_INLINE),
                g(&pd::PIVOTS_SCALAR),
                g(&pd::SCALAR_TAIL_STEPS),
                g(&pd::FALLBACK_2X2_NEED_SWAP_OR_BOUND),
                g(&pd::FALLBACK_2X2_SWAP_1X1_WINS),
                g(&pd::FALLBACK_2X2_LAPACK_1X1_WINS),
                g(&pd::FALLBACK_2X2_GROWTH_OR_DET),
            );
        }
        if phases {
            use feral::dense::factor::phase_timing as pt;
            let g = |c: &std::sync::atomic::AtomicU64| {
                c.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9
            };
            let dense = g(&pt::DENSEFACTOR_NS);
            let panel = g(&pt::PANELFACTOR_NS);
            let schur = g(&pt::SCHUR_NS);
            let tail = g(&pt::SCALARTAIL_NS);
            println!(
                "  phases (s, summed over threads): dense={dense:.2} panel={panel:.2} schur={schur:.2} scalar_tail={tail:.2} other_dense={:.2} assembly={:.2} extend_add={:.2}",
                dense - panel - schur - tail,
                g(&pt::ASSEMBLY_NS),
                g(&pt::EXTENDADD_NS),
            );
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
