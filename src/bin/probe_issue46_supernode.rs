//! Issue #46 — does the MC64 pairing survive into the supernode tree?
//!
//! `probe_issue46_preprocess` established that `LdltCompress` activation
//! and the 21660-pair compression ALREADY work on the CHO KKT — the
//! cascade (28M factor-nnz / ~17 s) happens anyway. This probe localises
//! *where* the pairing is lost:
//!
//!   gap A — symbolic ordering. If `factor_nnz_estimate` (the no-delay
//!           symbolic fill prediction) is already ~28M, the compressed
//!           ordering itself is bad — the cascade is a fill problem, not
//!           a delayed-pivot problem.
//!   gap B — supernode split. If the symbolic estimate is small but the
//!           MC64 pairs land in *different* supernodes, the numeric
//!           kernel cannot form the saddle 2×2 in-front and delays.
//!   gap C — numeric gate. If pairs are co-located in supernodes but the
//!           numeric factor still blows up, the `scalar_pivot_step` 2×2
//!           gate is rejecting the in-front saddle pivot.
//!
//! For each of `LdltCompress` and `None` it reports the symbolic fill
//! estimate, supernode-size histogram, MC64-pair co-location counts, and
//! the real numeric factor (time / nnz / inertia).
//!
//! Usage: cargo run --release --bin probe_issue46_supernode [-- <kkt.mtx>]

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::factorize_multifrontal;
use feral::scaling::mc64_matching;
use feral::symbolic::{
    build_supermap, symbolic_factorize_with_method, OrderingMethod, OrderingPreprocess,
    SupernodeParams, SymbolicFactorization,
};
use feral::{read_mtx, CscMatrix, NumericParams};

const DEFAULT_MTX: &str =
    "/Users/jkitchin/projects/pounce/benchmarks/cho/feral_repro/cho_iter0_kkt.mtx";

/// Build `col_to_snode[c] = supernode index owning postordered column c`.
fn col_to_snode(sym: &SymbolicFactorization) -> Vec<usize> {
    let mut map = vec![usize::MAX; sym.n];
    for (s, sn) in sym.supernodes.iter().enumerate() {
        for slot in map.iter_mut().skip(sn.first_col).take(sn.ncol()) {
            *slot = s;
        }
    }
    map
}

fn run(
    label: &str,
    m: &CscMatrix,
    preprocess: OrderingPreprocess,
    allow_delayed: bool,
    pairs: &[(usize, usize)],
) {
    let snode = SupernodeParams {
        preprocess,
        ..SupernodeParams::default()
    };
    let sym = match symbolic_factorize_with_method(m, &snode, OrderingMethod::Auto) {
        Ok(s) => s,
        Err(e) => {
            println!("{label}: symbolic failed: {e:?}");
            return;
        }
    };

    // Supernode-size histogram + largest front.
    let mut hist = [0usize; 6]; // ncol: 1, 2-15, 16-63, 64-255, 256-1023, >=1024
    let mut max_ncol = 0usize;
    for sn in &sym.supernodes {
        let c = sn.ncol();
        max_ncol = max_ncol.max(c);
        let bucket = match c {
            1 => 0,
            2..=15 => 1,
            16..=63 => 2,
            64..=255 => 3,
            256..=1023 => 4,
            _ => 5,
        };
        hist[bucket] += 1;
    }

    // MC64-pair co-location: where does each matched pair end up?
    let c2s = col_to_snode(&sym);
    let mut same_snode = 0usize;
    let mut adjacent = 0usize; // same supernode AND consecutive columns
    let mut split = 0usize;
    for &(a, b) in pairs {
        let na = sym.perm_inv[a];
        let nb = sym.perm_inv[b];
        if c2s[na] == c2s[nb] {
            same_snode += 1;
            if na.abs_diff(nb) == 1 {
                adjacent += 1;
            }
        } else {
            split += 1;
        }
    }

    println!("--- {label} ---");
    println!(
        "  resolved_preprocess={:?}  resolved_method={:?}",
        sym.resolved_preprocess, sym.resolved_method
    );
    println!(
        "  supernodes={}  max ncol={max_ncol}  factor_nnz_estimate={} (symbolic, no-delay)",
        sym.supernodes.len(),
        sym.factor_nnz_estimate
    );
    println!(
        "  ncol hist: [=1]={} [2-15]={} [16-63]={} [64-255]={} [256-1023]={} [>=1024]={}",
        hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]
    );
    println!(
        "  MC64 pairs ({} total): same-supernode={same_snode} (of those, adjacent cols={adjacent})  split-across-supernodes={split}",
        pairs.len()
    );

    // Real numeric factor — exposes delayed-pivot blowup vs symbolic estimate.
    let np = NumericParams {
        allow_delayed_pivots: allow_delayed,
        ..NumericParams::default()
    };
    feral::dense::factor::panel_diag::reset();
    feral::dense::factor::PANEL_DIAG_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    let t = Instant::now();
    let result = factorize_multifrontal(m, &sym, &np);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    feral::dense::factor::PANEL_DIAG_ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
    match result {
        Ok((factors, inertia)) => {
            let fnnz = factors.factor_nnz();
            let blowup = fnnz as f64 / sym.factor_nnz_estimate.max(1) as f64;
            println!(
                "  numeric (allow_delayed={allow_delayed}): {ms:.0}ms  factor_nnz={fnnz}  blowup={blowup:.2}x  inertia=({},{},{})",
                inertia.positive, inertia.negative, inertia.zero
            );
            println!("  {}", factors.summary());
        }
        Err(e) => println!("  numeric failed: {e:?}"),
    }
    let snap = feral::dense::factor::panel_diag::snapshot();
    let parts: Vec<String> = snap
        .iter()
        .filter(|(_, v)| *v != 0)
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    println!("  panel_diag: {}", parts.join(" "));
    println!();
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MTX.to_string());
    if !Path::new(&path).exists() {
        eprintln!("SKIP: {path} not present");
        std::process::exit(2);
    }
    let csc = match read_mtx(Path::new(&path)).and_then(|m| m.to_csc()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("load failed: {e:?}");
            std::process::exit(1);
        }
    };

    // The MC64 matched pairs, in ORIGINAL numbering — the same object
    // `LdltCompress` co-locates. `perm_inv` maps each original index to
    // its final postordered column.
    let pairs = match mc64_matching(&csc) {
        Ok((perm, _)) => build_supermap(&perm).pairs,
        Err(e) => {
            eprintln!("mc64_matching failed: {e:?}");
            std::process::exit(1);
        }
    };
    println!(
        "CHO KKT  n={}  nnz(lower)={}  MC64 pairs={}\n",
        csc.n,
        csc.nnz(),
        pairs.len()
    );

    run(
        "LdltCompress, delayed",
        &csc,
        OrderingPreprocess::LdltCompress,
        true,
        &pairs,
    );
    run(
        "LdltCompress, static (no delay)",
        &csc,
        OrderingPreprocess::LdltCompress,
        false,
        &pairs,
    );
    run(
        "None, delayed",
        &csc,
        OrderingPreprocess::None,
        true,
        &pairs,
    );
}
