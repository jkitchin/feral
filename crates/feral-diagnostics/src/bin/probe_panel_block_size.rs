//! probe_panel_block_size — does the panel block size sit in the wrong
//! place for wide-panel fronts?
//!
//! Issue #153 item 2. `probe_front_bucket_phases` localized dtoc1nd's
//! cost to 148 fronts of mean shape `ncol = 62`, `nrow = 88`, and inside
//! those the **panel factor** is 56% of the time while the trailing
//! Schur update is 16%. Every other fixture in the #153 set is the other
//! way round (panel 7-15%, Schur 26-31%), and the reason is shape:
//! `lblt_panel_frontal` is left-looking, so a panel of `bs` columns
//! costs ~`bs^2 * nrow / 2` in rank-1/rank-2 axpy traffic, while the
//! blocked Schur update between panels is BLAS-3. `bs =
//! params.block_size.min(ncol)` (`src/dense/factor.rs:2171`) and
//! `block_size` defaults to 64, so a 62-column front runs as **one**
//! panel and does all of its elimination in the BLAS-2 kernel.
//!
//! The hypothesis this probe tests: lowering `block_size` splits those
//! fronts into several narrower panels and moves work from the BLAS-2
//! panel into the BLAS-3 trailing update, which should be a net win once
//! `ncol` is large enough for the quadratic term to matter.
//!
//! **Answered: the split moves, the clock does not.** `FERAL_BUCKET_BS`
//! on `probe_front_bucket_phases` shows panel% falling 54 → 15 and
//! schur% rising 18 → 57 as `bs` goes 64 → 8, exactly as predicted; this
//! probe's 60-pair run then puts `bs = 48` ahead of the default by only
//! **1.3%** (43/60 wins — real, but tiny), and finds nothing at all on
//! the two other corpus matrices with fronts wider than 48. Kept as a
//! re-runnable instrument, not as a shipped tuning.
//! `dev/research/issue-153-dtoc1nd-dense-front-2026-08-19.md`,
//! `dev/tried-and-rejected.md` 2026-08-19.
//!
//! Methodology follows `dev/decisions.md` 2026-08-09: **paired
//! alternating** arms (every block size is timed once per pair, in
//! order, so drift hits every arm equally), `min_us` per arm, and a sign
//! test over the pairs. Do not compare medians collected at different
//! times.
//!
//! `block_size` changes the pivot *blocking*, not the pivot sequence's
//! definition — the panel path is documented bit-exact with the scalar
//! path (`tests/blocked_ldlt.rs`). Whether that makes every arm
//! byte-identical is exactly what this probe checks rather than assumes:
//! each arm reports its inertia and its true relative residual, and the
//! `L`/`D` digest so a silent numeric change cannot hide behind an
//! unchanged inertia.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin probe_panel_block_size \
//!       -- <a.mtx> [b.mtx ...]
//!
//! Env:
//!   FERAL_BS_LIST=8,16,24,32,48,64   arms to sweep (default this list)
//!   FERAL_BS_PAIRS=8                 paired repetitions (default 8)

use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{
    factorize_multifrontal_supernodal_with_workspace, FactorWorkspace, NumericParams,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams, SymbolicFactorization};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix};

fn params_at(bs: usize) -> NumericParams {
    NumericParams {
        bk: BunchKaufmanParams {
            block_size: bs,
            ..BunchKaufmanParams::default()
        },
        ..NumericParams::default()
    }
}

/// Digest of the factor's numeric content: the raw bits of every `D`
/// entry and every `L` value, in storage order, plus each node's
/// `nelim`. Two arms that agree here produced byte-identical factors;
/// two that differ produced a different pivot sequence, a different
/// delayed-pivot count, or different rounding — all three of which
/// matter and none of which an unchanged inertia would reveal.
fn digest(csc: &CscMatrix, sym: &SymbolicFactorization, bs: usize) -> Option<(String, u64)> {
    let mut ws = FactorWorkspace::default();
    let params = params_at(bs);
    let (factors, inertia) =
        factorize_multifrontal_supernodal_with_workspace(csc, sym, &params, &mut ws).ok()?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let mut delayed = 0usize;
    for node in factors.node_factors.iter() {
        node.nelim.hash(&mut h);
        delayed += node.ncol - node.nelim;
        let f = &node.frontal_factors;
        for v in f.d_diag.iter().chain(f.d_subdiag.iter()).chain(f.l.iter()) {
            v.to_bits().hash(&mut h);
        }
    }
    Some((
        format!(
            "{}/{}/{}+d{delayed}",
            inertia.positive, inertia.negative, inertia.zero
        ),
        h.finish(),
    ))
}

/// True relative residual `||Ax-b||_inf / ||b||_inf` for one arm, from a
/// single unrefined solve, so a numeric change shows up rather than
/// being absorbed by iterative refinement.
fn residual(csc: &CscMatrix, sym: &SymbolicFactorization, bs: usize) -> Option<f64> {
    let mut ws = FactorWorkspace::default();
    let params = params_at(bs);
    let (factors, _) =
        factorize_multifrontal_supernodal_with_workspace(csc, sym, &params, &mut ws).ok()?;
    let b = vec![1.0f64; csc.n];
    let x = feral::numeric::solve::solve_sparse(&factors, &b).ok()?;
    let mut ax = vec![0.0f64; csc.n];
    for j in 0..csc.n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = csc.values[k];
            ax[i] += v * x[j];
            if i != j {
                ax[j] += v * x[i];
            }
        }
    }
    let num = (0..csc.n).fold(0.0f64, |m, i| m.max((ax[i] - b[i]).abs()));
    let den = b
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    Some(num / den)
}

fn time_one(
    csc: &CscMatrix,
    sym: &SymbolicFactorization,
    bs: usize,
    ws: &mut FactorWorkspace,
) -> Option<u128> {
    let params = params_at(bs);
    let t = Instant::now();
    factorize_multifrontal_supernodal_with_workspace(csc, sym, &params, ws).ok()?;
    Some(t.elapsed().as_micros())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: probe_panel_block_size <a.mtx> [b.mtx ...]");
        std::process::exit(2);
    }
    let arms: Vec<usize> =
        feral::env::usize_list_var("FERAL_BS_LIST").unwrap_or_else(|| vec![8, 16, 24, 32, 48, 64]);
    let pairs: usize = feral::env::usize_var("FERAL_BS_PAIRS").unwrap_or(8).max(1);
    let baseline = BunchKaufmanParams::default().block_size;

    for path in &args {
        let Ok(csc) = read_mtx(Path::new(path)).and_then(|m| m.to_csc()) else {
            eprintln!("{path}: load failed");
            continue;
        };
        let Ok(sym) = symbolic_factorize(&csc, &SupernodeParams::default()) else {
            eprintln!("{path}: symbolic failed");
            continue;
        };
        // Front-shape context: the lever is quadratic in the panel
        // width, so the arm ranking is only interpretable next to how
        // wide this matrix's paying fronts actually are.
        let mut ncols: Vec<usize> = sym.supernodes.iter().map(|s| s.ncol).collect();
        ncols.sort_unstable();
        let ncol_max = ncols.last().copied().unwrap_or(0);
        let ncol_p90 = ncols[(ncols.len().saturating_sub(1)) * 9 / 10];

        let label = Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        println!(
            "=== {label} (n={}, snodes={}, ncol p90={}, ncol max={}) ===",
            csc.n,
            sym.supernodes.len(),
            ncol_p90,
            ncol_max
        );

        // Paired alternating timing: every arm runs once per pair, in
        // list order, so machine drift is shared across arms.
        let mut mins: Vec<u128> = vec![u128::MAX; arms.len()];
        let mut wins: Vec<usize> = vec![0; arms.len()];
        let mut ws = FactorWorkspace::default();
        for (i, &bs) in arms.iter().enumerate() {
            // Warm each arm's workspace once before the paired loop.
            if time_one(&csc, &sym, bs, &mut ws).is_none() {
                eprintln!("{label}: factor failed at block_size={bs}");
                mins[i] = 0;
            }
        }
        for _ in 0..pairs {
            let mut pair: Vec<u128> = Vec::with_capacity(arms.len());
            for &bs in &arms {
                pair.push(time_one(&csc, &sym, bs, &mut ws).unwrap_or(u128::MAX));
            }
            if let Some(best) = pair
                .iter()
                .enumerate()
                .min_by_key(|&(_, v)| *v)
                .map(|(i, _)| i)
            {
                wins[best] += 1;
            }
            for (i, v) in pair.into_iter().enumerate() {
                mins[i] = mins[i].min(v);
            }
        }

        let base_idx = arms.iter().position(|&b| b == baseline);
        let base_min = base_idx.map(|i| mins[i]).unwrap_or(0);

        println!(
            "{:>6}{:>10}{:>9}{:>8}{:>20}{:>20}{:>10}",
            "bs", "min_us", "vs_base", "wins", "inertia/delayed", "digest", "resid"
        );
        for (i, &bs) in arms.iter().enumerate() {
            let (inertia, dig) = digest(&csc, &sym, bs)
                .map(|(a, b)| (a, format!("{b:016x}")))
                .unwrap_or_else(|| ("failed".into(), "-".into()));
            let res = residual(&csc, &sym, bs).unwrap_or(f64::NAN);
            let ratio = if base_min > 0 && mins[i] > 0 {
                mins[i] as f64 / base_min as f64
            } else {
                f64::NAN
            };
            println!(
                "{:>6}{:>10}{:>9.3}{:>8}{:>20}{:>20}{:>10.2e}{}",
                bs,
                mins[i],
                ratio,
                wins[i],
                inertia,
                dig,
                res,
                if bs == baseline { "  <- default" } else { "" }
            );
        }
        println!();
    }
}
