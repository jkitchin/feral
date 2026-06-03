//! Minimal profile target for issue #8 root-cause investigation.
//!
//! Factors `pinene_3200_0009` exactly once with `NumericParams::default()`
//! (FMA off, the production cross-arch path). Phase 3 verification
//! showed this iterate takes ~87s with FMA off and ~87s with FMA on,
//! while iterate 0008 (identical n and nnz) takes 44ms. We profile
//! 0009 alone to localize where the 87s is actually spent.
//!
//! Usage:
//!     cargo build --release --bin diag_pinene_0009_profile
//!     samply record target/release/diag_pinene_0009_profile

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{
    factorize_multifrontal_parallel_with_workspace, FactorWorkspace, NumericParams,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, Inertia};

fn main() -> std::io::Result<()> {
    let base = "data/matrices/kkt-mittelmann/pinene_3200/pinene_3200_0009";
    let mtx_path = format!("{base}.mtx");
    let json_path = format!("{base}.json");
    assert!(Path::new(&mtx_path).is_file(), "missing {mtx_path}");

    let mtx = read_mtx(Path::new(&mtx_path)).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let sidecar = read_sidecar(Path::new(&json_path)).expect("sidecar");
    let oracle = Inertia::new(
        sidecar.inertia.positive,
        sidecar.inertia.negative,
        sidecar.inertia.zero,
    );

    eprintln!("pinene_3200_0009: n={} nnz={}", csc.n, csc.row_idx.len());
    eprintln!("oracle inertia: {oracle}");

    let snode = SupernodeParams::default();
    let t0 = Instant::now();
    let sym = symbolic_factorize(&csc, &snode).expect("symbolic");
    eprintln!("symbolic: {:.3}s", t0.elapsed().as_secs_f64());

    let params = NumericParams::default(); // FMA off — production path
    let mut ws = FactorWorkspace::new();

    eprintln!("starting numeric factor (FMA off)...");
    let t0 = Instant::now();
    let (_factors, inertia) =
        factorize_multifrontal_parallel_with_workspace(&csc, &sym, &params, &mut ws)
            .expect("factor");
    let dt = t0.elapsed().as_secs_f64();
    eprintln!("numeric factor: {dt:.3}s  inertia={inertia}");
    assert_eq!(inertia, oracle, "inertia mismatch");

    Ok(())
}
