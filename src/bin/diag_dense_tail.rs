//! Phase 2.4.4 (proposed) diagnostic: decide whether the dense tail
//! on HAHN1 / CRESC100 / GAUSS2 is driven by (a) ordering packing
//! these families into giant supernodes, or (b) the dense kernel
//! itself being slow on n≈700 frontals.
//!
//! For each target matrix, this binary runs `symbolic_factorize`
//! under five parameter configs and reports supernode-size stats
//! plus factor time vs the MUMPS oracle:
//!
//!   1. default (nemin=32, preprocess=None) — production path.
//!   2. nemin=5 (MUMPS default), preprocess=None.
//!   3. nemin=1 (no amalgamation), preprocess=None.
//!   4. default + LdltCompress.
//!   5. nemin=5 + LdltCompress.
//!
//! Interpretation:
//!
//! - If the largest supernode under config 1 is ~n (the whole
//!   matrix), ordering is packing everything into one dense frontal
//!   and we have an ordering problem. Configs 2/3 will split it.
//!
//! - If configs 2/3/4/5 all produce largest_ncol ≈ 700 too, it is
//!   NOT amalgamation — AMD itself is giving a tree that bottoms
//!   out in a dense root and the symbolic phase has no lever.
//!
//! - If configs 2/3/4/5 produce small supernodes but factor time
//!   stays in the 10ms range, the dense kernel is the bottleneck
//!   regardless of ordering.
//!
//! Target matrices (top of `cargo run --bin bench` top-10 worst):
//! HAHN1_0153/0154/0454, CRESC100_0000/0027, GAUSS2_0025/0029/0035.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, OrderingPreprocess, SupernodeParams};
use feral::{read_mtx, CscMatrix};
use std::path::Path;
use std::time::Instant;

fn read_mumps_factor_us(path: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let data: serde_json::Value = serde_json::from_str(&text).ok()?;
    if data["factorization_status"].as_str() != Some("ok") {
        return None;
    }
    Some(data["factor_us"].as_u64().unwrap_or(0))
}

struct ConfigSpec {
    label: &'static str,
    params: SupernodeParams,
}

fn configs() -> Vec<ConfigSpec> {
    vec![
        ConfigSpec {
            label: "default(nemin=32)",
            params: SupernodeParams {
                nemin: 32,
                preprocess: OrderingPreprocess::None,
                small_leaf: Default::default(),
            },
        },
        ConfigSpec {
            label: "nemin=5",
            params: SupernodeParams {
                nemin: 5,
                preprocess: OrderingPreprocess::None,
                small_leaf: Default::default(),
            },
        },
        ConfigSpec {
            label: "nemin=1",
            params: SupernodeParams {
                nemin: 1,
                preprocess: OrderingPreprocess::None,
                small_leaf: Default::default(),
            },
        },
        ConfigSpec {
            label: "nemin=32+compress",
            params: SupernodeParams {
                nemin: 32,
                preprocess: OrderingPreprocess::LdltCompress,
                small_leaf: Default::default(),
            },
        },
        ConfigSpec {
            label: "nemin=5+compress",
            params: SupernodeParams {
                nemin: 5,
                preprocess: OrderingPreprocess::LdltCompress,
                small_leaf: Default::default(),
            },
        },
    ]
}

struct Stats {
    n_nodes: usize,
    max_ncol: usize,
    max_nrow: usize,
    sum_ncol_sq_nrow: u128,
    sym_us: u128,
    fac_us: u128,
}

fn run_config(a: &CscMatrix, cfg: &ConfigSpec) -> Option<Stats> {
    let t_sym = Instant::now();
    let sym = match symbolic_factorize(a, &cfg.params) {
        Ok(s) => s,
        Err(_) => return None,
    };
    let sym_us = t_sym.elapsed().as_micros();

    let mut max_ncol = 0usize;
    let mut max_nrow = 0usize;
    let mut sum_work: u128 = 0;
    for s in &sym.supernodes {
        if s.ncol > max_ncol {
            max_ncol = s.ncol;
        }
        if s.nrow > max_nrow {
            max_nrow = s.nrow;
        }
        let nc = s.ncol as u128;
        let nr = s.nrow as u128;
        sum_work += nc * nc * nr;
    }

    let t_fac = Instant::now();
    let res = factorize_multifrontal(a, &sym, &NumericParams::default());
    let fac_us = t_fac.elapsed().as_micros();
    let _ = res.ok()?;

    Some(Stats {
        n_nodes: sym.supernodes.len(),
        max_ncol,
        max_nrow,
        sum_ncol_sq_nrow: sum_work,
        sym_us,
        fac_us,
    })
}

fn report_matrix(family: &str, stem: &str) {
    let base = format!("data/matrices/kkt/{}/{}", family, stem);
    let mtx = match read_mtx(Path::new(&format!("{}.mtx", base))) {
        Ok(m) => m,
        Err(e) => {
            println!("  SKIP {} — read_mtx: {:?}", stem, e);
            return;
        }
    };
    let a = match mtx.to_csc() {
        Ok(a) => a,
        Err(e) => {
            println!("  SKIP {} — to_csc: {:?}", stem, e);
            return;
        }
    };

    let mumps_us = read_mumps_factor_us(Path::new(&format!("{}.mumps.json", base))).unwrap_or(0);
    println!(
        "\n{} n={} nnz={} mumps_factor_us={}",
        stem,
        a.n,
        a.row_idx.len(),
        mumps_us
    );
    println!(
        "{:22} {:>7} {:>9} {:>9} {:>14} {:>9} {:>9} {:>8}",
        "config", "nodes", "max_ncol", "max_nrow", "sum_nc2*nr", "sym_us", "fac_us", "vs_mumps"
    );

    for cfg in configs() {
        let Some(s) = run_config(&a, &cfg) else {
            println!("{:22} FAILED", cfg.label);
            continue;
        };
        let ratio = if mumps_us > 0 {
            format!("{:.2}x", s.fac_us as f64 / mumps_us as f64)
        } else {
            "?".into()
        };
        println!(
            "{:22} {:>7} {:>9} {:>9} {:>14} {:>9} {:>9} {:>8}",
            cfg.label,
            s.n_nodes,
            s.max_ncol,
            s.max_nrow,
            s.sum_ncol_sq_nrow,
            s.sym_us,
            s.fac_us,
            ratio
        );
    }
}

fn main() {
    println!("=== Phase 2.4.4 dense-tail diagnostic ===");
    println!("sum_nc2*nr = sum_supernodes (ncol^2 * nrow)  — dense-factor");
    println!("  work proxy; lower is better.");

    let targets: &[(&str, &[&str])] = &[
        ("HAHN1", &["HAHN1_0153", "HAHN1_0154", "HAHN1_0454"]),
        ("CRESC100", &["CRESC100_0000", "CRESC100_0027"]),
        ("GAUSS2", &["GAUSS2_0025", "GAUSS2_0029", "GAUSS2_0035"]),
    ];

    for (family, stems) in targets {
        println!("\n--- {} ---", family);
        for stem in *stems {
            report_matrix(family, stem);
        }
    }
}
