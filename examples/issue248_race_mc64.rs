//! Issue #248 probe: time `AutoRace` symbolic analysis on a Matrix Market
//! KKT and print the winner's per-stage profile. Pair with
//! `FERAL_MC64_TRACE=1` to see each MC64 matching (count and wall time).
//!
//! Usage: `cargo run --release --example issue248_race_mc64 -- <file.mtx> [method]`
//! where `method` is `autorace` (default), `auto`, `amd`, `amf`, `metis`
//! or `scotch`.

use feral::symbolic::supernode::SupernodeParams;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SymbolicProfiler};
use std::sync::{Arc, Mutex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: issue248_race_mc64 <file.mtx> [method]")?;
    let method = match args.next().as_deref().unwrap_or("autorace") {
        "autorace" => OrderingMethod::AutoRace,
        "auto" => OrderingMethod::Auto,
        "amd" => OrderingMethod::Amd,
        "amf" => OrderingMethod::Amf,
        "metis" => OrderingMethod::MetisND,
        "scotch" => OrderingMethod::ScotchND,
        other => return Err(format!("unknown method {other}").into()),
    };
    let t = std::time::Instant::now();
    let m = feral::read_mtx(std::path::Path::new(&path))?.to_csc()?;
    eprintln!(
        "read n={} nnz={} in {:.1} s",
        m.n,
        m.row_idx.len(),
        t.elapsed().as_secs_f64()
    );

    let prof = Arc::new(Mutex::new(SymbolicProfiler::new()));
    let params = SupernodeParams {
        symbolic_profiler: Some(prof.clone()),
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let sym = symbolic_factorize_with_method(&m, &params, method.clone())?;
    let secs = t.elapsed().as_secs_f64();
    println!(
        "{method:?}: {secs:.1} s, factor_nnz_estimate {:.4e}, supernodes {}",
        sym.factor_nnz_estimate as f64,
        sym.supernodes.len()
    );
    let p = prof.lock().map_err(|_| "profiler poisoned")?;
    for s in p.stages() {
        println!("  {:<24} {:>10.2} s", s.name, s.us as f64 / 1e6);
    }
    Ok(())
}
