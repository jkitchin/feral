//! Compare the dense BK path against the sparse multifrontal path
//! on matrices that fail the parity panel, using each path's
//! production BunchKaufmanParams config.
//!
//! Post-Phase-2.3 the two paths no longer use identical params:
//!   - Dense:  pivot_threshold = 0.0 (no delays in dense kernel)
//!   - Sparse: pivot_threshold = 0.01 (delays enabled at non-roots)
//!
//! Historical hypothesis (still relevant on matrices that remain
//! failing after Phase 2.3): the dense path applies Knight-Ruiz
//! ∞-norm equilibration before BK (src/dense/equilibrate.rs)
//! while the sparse path does not. On matrices where MC64 is a
//! no-op this means dense factors an equilibrated matrix and
//! sparse factors a raw matrix.
//!
//! Run with:  cargo run --release --example dense_vs_sparse

use std::path::Path;

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    factor, read_mtx, read_sidecar, solve_refined, BunchKaufmanParams, CscMatrix, Inertia,
    ZeroPivotAction,
};

fn dense_params() -> BunchKaufmanParams {
    // Dense production path: pivot_threshold = 0.0 because the
    // dense kernel does not implement delayed pivoting.
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    }
}

fn sparse_params() -> BunchKaufmanParams {
    // Sparse production path (Phase 2.3): pivot_threshold = 0.01
    // because delayed pivoting gives rejected pivots a landing zone
    // at the parent supernode. Matches bench::params_kkt.
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    }
}

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        rs += (ax[i] - b[i]).powi(2);
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

fn read_mumps_oracle(path: &Path) -> Option<(Inertia, f64)> {
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let inertia = data.get("inertia")?;
    let pos = inertia.get("positive")?.as_u64()? as usize;
    let neg = inertia.get("negative")?.as_u64()? as usize;
    let zero = inertia.get("zero")?.as_u64()? as usize;
    let residual = data.get("residual_2norm_relative")?.as_f64()?;
    Some((Inertia::new(pos, neg, zero), residual))
}

fn compare(stem: &str) {
    let base = format!("data/matrices/kkt/{}", stem);
    let mtx_path = format!("{}.mtx", base);
    let sidecar_path = format!("{}.json", base);
    let mumps_path = format!("{}.mumps.json", base);

    let mtx = read_mtx(Path::new(&mtx_path)).expect("mtx");
    let csc = mtx.to_csc().expect("csc");
    let dense = mtx.to_dense();
    let sc = read_sidecar(Path::new(&sidecar_path)).expect("sidecar");
    let rhs = sc.finite_rhs().expect("rhs");
    let mumps = read_mumps_oracle(Path::new(&mumps_path));

    println!("\n=== {} ===", stem);
    println!("  n={}  nnz={}", csc.n, csc.values.len());
    if let Some((i, r)) = &mumps {
        println!("  MUMPS oracle: inertia={}  residual={:.3e}", i, r);
    }

    // Dense path (with Knight-Ruiz equilibration)
    let dense_p = dense_params();
    match factor(&dense, &dense_p) {
        Ok((factors, dense_inertia)) => {
            let x = solve_refined(&dense, &factors, &rhs).expect("dense solve");
            let res = rel_residual(&csc, &x, &rhs);
            let d_eq_range = {
                let mut lo = f64::INFINITY;
                let mut hi = 0.0f64;
                for &v in &factors.d_eq {
                    if v > 0.0 {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
                (lo, hi)
            };
            let matches = mumps
                .as_ref()
                .map(|(i, _)| &dense_inertia == i)
                .unwrap_or(false);
            println!(
                "  DENSE  (Knight-Ruiz + BK):  inertia={}  residual={:.3e}  d_eq=[{:.2e}, {:.2e}]  match={}",
                dense_inertia,
                res,
                d_eq_range.0,
                d_eq_range.1,
                if matches { "YES" } else { "no" },
            );
        }
        Err(e) => println!("  DENSE: factor failed: {}", e),
    }

    // Sparse path with whatever scaling is the current default.
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("sym");
    let scale_range = {
        let mut lo = f64::INFINITY;
        let mut hi = 0.0f64;
        for &v in &sym.scaling {
            if v > 0.0 {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        (lo, hi)
    };
    let sparse_p = sparse_params();
    let (sparse_fac, sparse_inertia) =
        factorize_multifrontal(&csc, &sym, &sparse_p).expect("factorize");
    let sparse_x = solve_sparse_refined(&csc, &sparse_fac, &rhs).expect("solve");
    let sparse_res = rel_residual(&csc, &sparse_x, &rhs);
    let matches = mumps
        .as_ref()
        .map(|(i, _)| &sparse_inertia == i)
        .unwrap_or(false);
    println!(
        "  SPARSE (default scaling):  inertia={}  residual={:.3e}  scale=[{:.2e}, {:.2e}]  match={}",
        sparse_inertia,
        sparse_res,
        scale_range.0,
        scale_range.1,
        if matches { "YES" } else { "no" },
    );
}

fn main() {
    for stem in [
        "HYDCAR20/HYDCAR20_0000",
        "METHANL8/METHANL8_0000",
        "SWOPF/SWOPF_0000",
        "HATFLDG/HATFLDG_0005",
        "HATFLDBNE/HATFLDBNE_1586",
        "ACOPP30/ACOPP30_0000",
        // Controls that should already pass on both paths:
        "CHWIRUT1/CHWIRUT1_0000",
        "CRESC100/CRESC100_0000",
    ] {
        compare(stem);
    }
}
