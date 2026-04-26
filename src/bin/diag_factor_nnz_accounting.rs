//! Audit feral's `factor_nnz()` accounting against MUMPS `INFOG(9)`
//! and SSIDS `inform%num_factor` to confirm whether the persistent
//! 1.75× nnzL/SSIDS gap is a counting artifact or real fill.
//!
//! Per supernode the L block is `nrow × nelim` column-major with
//! unit-lower-triangular structure in the leading `nelim × nelim`
//! eliminated block. feral currently counts the full dense block
//! (`nrow * nelim`), which includes the strict-upper triangle and
//! the unit diagonal of the eliminated block — both structurally
//! known to be zero or 1 respectively.
//!
//! Three candidate counts per supernode:
//!
//!   A. Strict-lower of eliminated + trailing rect:
//!      `nelim*(nelim-1)/2 + (nrow-nelim)*nelim`
//!      (excludes diagonal and strict-upper)
//!
//!   B. Lower-tri inc diagonal of eliminated + trailing rect:
//!      `nelim*(nelim+1)/2 + (nrow-nelim)*nelim`
//!      (includes diagonal, excludes strict-upper)
//!
//!   C. feral current `factor_nnz()` = `nrow*nelim`
//!      (includes diagonal + strict-upper + strict-lower + trailing)
//!
//! Difference C - A = `nelim*(nelim+1)/2` (upper triangle including
//! diagonal of the eliminated block — structurally always zero/one).
//! Difference C - B = `nelim*(nelim-1)/2` (strict upper triangle).
//!
//! Usage: `cargo run --release --bin diag_factor_nnz_accounting`.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, ZeroPivotAction};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Clone)]
struct Counts {
    strict_lower: usize,       // A
    lower_with_diag: usize,    // B
    current_nrow_nelim: usize, // C
    n_supernodes: usize,
    nelim_total: usize,
    nrow_total: usize,
}

fn run_one(path: &Path) -> Option<(Counts, Option<u64>, Option<u64>)> {
    let mtx = read_mtx(path).ok()?;
    let csc = mtx.to_csc().ok()?;

    let bk = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    let snode_params = SupernodeParams::default();
    let factor_params = NumericParams::with_bk(bk);

    let sym = symbolic_factorize(&csc, &snode_params).ok()?;
    let (factors, _inertia) = factorize_multifrontal(&csc, &sym, &factor_params).ok()?;

    let mut c = Counts {
        strict_lower: 0,
        lower_with_diag: 0,
        current_nrow_nelim: 0,
        n_supernodes: factors.node_factors.len(),
        nelim_total: 0,
        nrow_total: 0,
    };
    for nf in &factors.node_factors {
        let ff = &nf.frontal_factors;
        let nrow = ff.nrow;
        let nelim = ff.nelim;
        let trailing_rect = nrow.saturating_sub(nelim) * nelim;
        let strict_upper_inc_diag = nelim * (nelim + 1) / 2;
        let strict_upper_excl_diag = nelim * nelim.saturating_sub(1) / 2;
        let strict_lower = nelim * nelim.saturating_sub(1) / 2 + trailing_rect;
        let lower_with_diag = strict_lower + nelim;
        c.strict_lower += strict_lower;
        c.lower_with_diag += lower_with_diag;
        c.current_nrow_nelim += nrow * nelim;
        c.nelim_total += nelim;
        c.nrow_total += nrow;
        // sanity: A + diag + strict_upper = C
        debug_assert_eq!(strict_lower + nelim + strict_upper_excl_diag, nrow * nelim);
        let _ = strict_upper_inc_diag;
    }

    let mumps = read_oracle_factor_nnz(&path.with_extension("mumps.json"));
    let ssids = read_oracle_factor_nnz(&path.with_extension("ssids.json"));
    Some((c, mumps, ssids))
}

fn read_oracle_factor_nnz(path: &Path) -> Option<u64> {
    let s = std::fs::read_to_string(path).ok()?;
    let v: Value = serde_json::from_str(&s).ok()?;
    v.get("factor_nnz").and_then(|x| x.as_u64())
}

fn ratio(a: usize, b: u64) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

fn main() {
    let mut paths: Vec<PathBuf> = Vec::new();

    // A representative sample across families known to have ratio ≈ 1.75
    // vs SSIDS in the 2026-04-26-02 bench. Hand-picked from bench output:
    // ALLINITA, BIGGSC4, CONCON, HS118, MCONCON, SSI, SSINE, BATCH, CORE1.
    let families = [
        "ALLINITA",
        "BIGGSC4",
        "CONCON",
        "HS118",
        "MCONCON",
        "SSI",
        "SSINE",
        "BATCH",
        "CORE1",
        "CERI651ALS",
        "PALMER5A",
        "PFIT4",
        "DJTL",
        "AVION2",
        "ACOPR30",
    ];
    for fam in families {
        for idx in [0, 1, 100, 500, 1000usize] {
            let p = PathBuf::from(format!("data/matrices/kkt/{fam}/{fam}_{idx:04}.mtx"));
            if p.exists() {
                paths.push(p);
            }
        }
    }

    println!("matrix                 n  ferl_C  ferl_B  ferl_A    MUMPS    SSIDS  C/SSIDS  B/SSIDS  A/SSIDS  C/MUMPS  B/MUMPS  A/MUMPS  snodes  fronts(avg)");
    println!("{}", "-".repeat(160));

    let mut all_c_ssids: Vec<f64> = Vec::new();
    let mut all_a_ssids: Vec<f64> = Vec::new();
    let mut all_b_ssids: Vec<f64> = Vec::new();

    for p in &paths {
        let (c, mumps, ssids) = match run_one(p) {
            Some(x) => x,
            None => continue,
        };
        let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let ssids_v = ssids.unwrap_or(0);
        let mumps_v = mumps.unwrap_or(0);
        let n = match read_mtx(p) {
            Ok(m) => m.n,
            _ => 0,
        };
        let r_c_ssids = ratio(c.current_nrow_nelim, ssids_v);
        let r_b_ssids = ratio(c.lower_with_diag, ssids_v);
        let r_a_ssids = ratio(c.strict_lower, ssids_v);
        let r_c_mumps = ratio(c.current_nrow_nelim, mumps_v);
        let r_b_mumps = ratio(c.lower_with_diag, mumps_v);
        let r_a_mumps = ratio(c.strict_lower, mumps_v);
        let avg_front = if c.n_supernodes > 0 {
            c.nrow_total as f64 / c.n_supernodes as f64
        } else {
            0.0
        };
        println!(
            "{:<20}  {:>4}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6.2}  {:>6.2}  {:>6.2}  {:>6.2}  {:>6.2}  {:>6.2}  {:>6}  {:>6.1}",
            name,
            n,
            c.current_nrow_nelim,
            c.lower_with_diag,
            c.strict_lower,
            mumps_v,
            ssids_v,
            r_c_ssids,
            r_b_ssids,
            r_a_ssids,
            r_c_mumps,
            r_b_mumps,
            r_a_mumps,
            c.n_supernodes,
            avg_front,
        );
        if r_c_ssids.is_finite() {
            all_c_ssids.push(r_c_ssids);
            all_b_ssids.push(r_b_ssids);
            all_a_ssids.push(r_a_ssids);
        }
    }

    println!();
    println!(
        "=== Summary across {} matrices with SSIDS sidecars ===",
        all_c_ssids.len()
    );
    let geomean = |xs: &[f64]| -> f64 {
        if xs.is_empty() {
            return f64::NAN;
        }
        let s: f64 = xs.iter().map(|x| x.ln()).sum::<f64>() / xs.len() as f64;
        s.exp()
    };
    let median = |xs: &[f64]| -> f64 {
        let mut v: Vec<f64> = xs.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() {
            f64::NAN
        } else {
            v[v.len() / 2]
        }
    };
    println!(
        "  C / SSIDS (current feral count):     geomean={:.3}  median={:.3}",
        geomean(&all_c_ssids),
        median(&all_c_ssids)
    );
    println!(
        "  B / SSIDS (lower-tri inc diag):      geomean={:.3}  median={:.3}",
        geomean(&all_b_ssids),
        median(&all_b_ssids)
    );
    println!(
        "  A / SSIDS (strict-lower only):       geomean={:.3}  median={:.3}",
        geomean(&all_a_ssids),
        median(&all_a_ssids)
    );
    println!();
    println!("If A/SSIDS  ≈ 1.00 → SSIDS counts strict-lower; current C overcounts.");
    println!("If B/SSIDS  ≈ 1.00 → SSIDS counts lower-tri including diag; ditto.");
    println!("If C/SSIDS  ≈ 1.00 → no artifact; the gap is real fill.");
}
