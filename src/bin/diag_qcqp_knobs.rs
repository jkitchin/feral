//! One-shot knob sweep on a single Mittelmann matrix.
//!
//! Usage:
//!   cargo run --release --bin diag_qcqp_knobs -- <path-to.mtx>

use std::path::PathBuf;
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::scaling::ScalingStrategy;
use feral::symbolic::{
    symbolic_factorize_with_method, AmalgamationStrategy, OrderingMethod, OrderingPreprocess,
    SupernodeParams,
};
use feral::{read_mtx, BunchKaufmanParams};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = if args.is_empty() {
        PathBuf::from("data/matrices/kkt-mittelmann/qcqp1500-1c/qcqp1500-1c_0000.mtx")
    } else {
        PathBuf::from(&args[0])
    };
    let mtx = read_mtx(&path).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    println!(
        "matrix: {}\n  n={} stored_nnz={}",
        path.display(),
        csc.n,
        csc.row_idx.len()
    );

    let cases: Vec<(
        &str,
        OrderingMethod,
        usize,
        AmalgamationStrategy,
        OrderingPreprocess,
    )> = vec![
        (
            "default(MetisND,nemin=32,Renumber,Compress)",
            OrderingMethod::MetisND,
            32,
            AmalgamationStrategy::Renumber,
            OrderingPreprocess::LdltCompress,
        ),
        (
            "MetisND,nemin=5",
            OrderingMethod::MetisND,
            5,
            AmalgamationStrategy::Renumber,
            OrderingPreprocess::LdltCompress,
        ),
        (
            "MetisND,nemin=8,no-compress",
            OrderingMethod::MetisND,
            8,
            AmalgamationStrategy::Renumber,
            OrderingPreprocess::None,
        ),
        (
            "Amd,nemin=32",
            OrderingMethod::Amd,
            32,
            AmalgamationStrategy::Auto,
            OrderingPreprocess::Auto,
        ),
        (
            "Amd,nemin=5",
            OrderingMethod::Amd,
            5,
            AmalgamationStrategy::Auto,
            OrderingPreprocess::Auto,
        ),
        (
            "Amf,nemin=32",
            OrderingMethod::Amf,
            32,
            AmalgamationStrategy::Auto,
            OrderingPreprocess::Auto,
        ),
        (
            "Amf,nemin=5",
            OrderingMethod::Amf,
            5,
            AmalgamationStrategy::Auto,
            OrderingPreprocess::Auto,
        ),
        (
            "ScotchND,nemin=8",
            OrderingMethod::ScotchND,
            8,
            AmalgamationStrategy::Auto,
            OrderingPreprocess::Auto,
        ),
    ];

    println!(
        "{:<46} {:>10} {:>10} {:>10} {:>9} {:>9}",
        "case", "sym_us", "num_us", "total_us", "nnz_L", "snodes"
    );
    for (name, method, nemin, amalg, preproc) in cases {
        let snode = SupernodeParams {
            nemin,
            preprocess: preproc,
            amalgamation_strategy: amalg,
            ..SupernodeParams::default()
        };
        let t0 = Instant::now();
        let sym = match symbolic_factorize_with_method(&csc, &snode, method) {
            Ok(s) => s,
            Err(e) => {
                println!("{:<46} SYM_ERR {}", name, e);
                continue;
            }
        };
        let sym_us = t0.elapsed().as_micros();
        let np = NumericParams {
            bk: BunchKaufmanParams::default(),
            scaling: ScalingStrategy::Auto,
            ..NumericParams::default()
        };
        let t1 = Instant::now();
        let (factors, _inertia) = match factorize_multifrontal(&csc, &sym, &np) {
            Ok(p) => p,
            Err(e) => {
                println!("{:<46} sym_us={} NUM_ERR {}", name, sym_us, e);
                continue;
            }
        };
        let num_us = t1.elapsed().as_micros();
        println!(
            "{:<46} {:>10} {:>10} {:>10} {:>9} {:>9}",
            name,
            sym_us,
            num_us,
            sym_us + num_us,
            factors.factor_nnz(),
            factors.node_factors.len()
        );
    }
}
