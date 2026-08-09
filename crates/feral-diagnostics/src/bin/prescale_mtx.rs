//! Pre-scale a matrix with MC64 symmetric scaling and write the result.
//!
//! Exists to make the feral-vs-MA57 factorization comparison
//! apples-to-apples. Neither natural arm is clean: MA57 with
//! `ICNTL(15)=1` computes MC64 inside the timed `MA57BD`, while feral
//! computes its MC64 in untimed analysis; MA57 with `ICNTL(15)=0` does
//! no scaling at all while feral still applies its own. Either way the
//! two solvers factorize different numbers.
//!
//! Writing `A' = D A D` to disk once removes the asymmetry: both
//! solvers then read the same pre-scaled matrix and are configured to
//! do no scaling of their own, so `factor_us` measures factorization
//! and nothing else.
//!
//! Usage: prescale_mtx <in.mtx> <out.mtx>

use std::path::Path;

use feral::read_mtx;
use feral::scaling::{compute_scaling, ScalingStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: prescale_mtx <in.mtx> <out.mtx>");
        std::process::exit(2);
    }
    let csc = read_mtx(Path::new(&args[0]))?.to_csc()?;
    let n = csc.n;

    let (d, info) = compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric)?;
    if d.len() != n {
        return Err(format!("scaling vector length {} != n {}", d.len(), n).into());
    }

    // Report what actually happened. `Mc64Symmetric` can fall back to
    // infinity-norm equilibration on a degenerate matching, and a run
    // that silently fell back is not the experiment we think it is.
    eprintln!("{}: n={} scaling_info={:?}", args[0], n, info);

    let mut out = String::new();
    out.push_str("%%MatrixMarket matrix coordinate real symmetric\n");
    out.push_str(&format!("% pre-scaled A' = D A D, D from {:?}\n", info));
    out.push_str(&format!("{} {} {}\n", n, n, csc.values.len()));
    for j in 0..n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = csc.values[k] * d[i] * d[j];
            out.push_str(&format!("{} {} {:.17e}\n", i + 1, j + 1, v));
        }
    }
    std::fs::write(&args[1], out)?;
    Ok(())
}
