//! ACOPR14 BK pivot divergence diagnostic (carried over from F3.3).
//!
//! Investigates the family of ACOPR14 KKT matrices where feral's
//! Schur block disagrees with MUMPS at 1e-6 to 1e-9 relative.
//! ACOPP30 (same OPF/KKT family) showed the same disagreement
//! between MUMPS and a dense Gaussian-elimination oracle, so the
//! divergence is likely conditioning + pivot-ordering, not a feral
//! bug. This binary characterises the disagreement and sweeps BK
//! pivot params.
//!
//! Usage:
//!     cargo run --release --bin diag_acopr14
//!     cargo run --release --bin diag_acopr14 -- ACOPR14_0001
//!     cargo run --release --bin diag_acopr14 -- ACOPR14_0002 ACOPR14_0001 ACOPR14_0000

use feral::numeric::factorize::{factorize_multifrontal_with_schur, NumericParams};
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize_with_schur, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, ZeroPivotAction};

fn dense_oracle_schur(a_full: &[Vec<f64>], schur_indices: &[usize]) -> Vec<Vec<f64>> {
    let n = a_full.len();
    let n_schur = schur_indices.len();
    let in_schur: std::collections::HashSet<usize> = schur_indices.iter().copied().collect();
    let elim: Vec<usize> = (0..n).filter(|i| !in_schur.contains(i)).collect();
    let n_elim = elim.len();

    let mut a_ff = vec![vec![0.0_f64; n_elim]; n_elim];
    for (i, &gi) in elim.iter().enumerate() {
        for (j, &gj) in elim.iter().enumerate() {
            a_ff[i][j] = a_full[gi][gj];
        }
    }
    let mut a_fs = vec![vec![0.0_f64; n_schur]; n_elim];
    for (i, &gi) in elim.iter().enumerate() {
        for (j, &gj) in schur_indices.iter().enumerate() {
            a_fs[i][j] = a_full[gi][gj];
        }
    }
    let mut a_ss = vec![vec![0.0_f64; n_schur]; n_schur];
    for (i, &gi) in schur_indices.iter().enumerate() {
        for (j, &gj) in schur_indices.iter().enumerate() {
            a_ss[i][j] = a_full[gi][gj];
        }
    }

    // [A_FF | A_FS] partial-pivot Gaussian elim.
    let mut aug = vec![vec![0.0_f64; n_elim + n_schur]; n_elim];
    for (i, row) in aug.iter_mut().enumerate() {
        row[..n_elim].copy_from_slice(&a_ff[i][..n_elim]);
        row[n_elim..n_elim + n_schur].copy_from_slice(&a_fs[i][..n_schur]);
    }
    for k in 0..n_elim {
        let mut piv = k;
        let mut piv_val = aug[k][k].abs();
        for (i, row) in aug.iter().enumerate().skip(k + 1) {
            if row[k].abs() > piv_val {
                piv = i;
                piv_val = row[k].abs();
            }
        }
        if piv != k {
            aug.swap(piv, k);
        }
        if aug[k][k].abs() < 1e-300 {
            return a_ss; // singular — caller will see big disagreement
        }
        for i in (k + 1)..n_elim {
            let f = aug[i][k] / aug[k][k];
            let pivot_row = aug[k].clone();
            for (j, slot) in aug[i].iter_mut().enumerate().skip(k) {
                *slot -= f * pivot_row[j];
            }
        }
    }
    let mut x = vec![vec![0.0_f64; n_schur]; n_elim];
    for c in 0..n_schur {
        for i in (0..n_elim).rev() {
            let mut s = aug[i][n_elim + c];
            for j in (i + 1)..n_elim {
                s -= aug[i][j] * x[j][c];
            }
            x[i][c] = s / aug[i][i];
        }
    }
    let mut s = a_ss.clone();
    for i in 0..n_schur {
        for j in 0..n_schur {
            let mut acc = 0.0_f64;
            for k in 0..n_elim {
                acc += a_fs[k][i] * x[k][j];
            }
            s[i][j] -= acc;
        }
    }
    s
}

fn max_rel_block(a: &[f64], b: &[f64], dim: usize) -> (f64, (usize, usize), f64, f64) {
    let mut max_rel = 0.0;
    let mut loc = (0, 0);
    let mut at_max_a = 0.0;
    let mut at_max_b = 0.0;
    for j in 0..dim {
        for i in 0..dim {
            let av = a[j * dim + i];
            let bv = b[j * dim + i];
            let denom = av.abs().max(bv.abs()).max(1.0);
            let rel = (av - bv).abs() / denom;
            if rel > max_rel {
                max_rel = rel;
                loc = (i, j);
                at_max_a = av;
                at_max_b = bv;
            }
        }
    }
    (max_rel, loc, at_max_a, at_max_b)
}

fn flatten_oracle(or: &[Vec<f64>]) -> Vec<f64> {
    let dim = or.len();
    let mut out = vec![0.0; dim * dim];
    for j in 0..dim {
        for i in 0..dim {
            out[j * dim + i] = or[i][j];
        }
    }
    out
}

fn run_one(matrix_name: &str) {
    println!("\n=== {} ===", matrix_name);
    let path = format!(
        "/Users/jkitchin/Dropbox/projects/feral/data/matrices/kkt/ACOPR14/{}.mtx",
        matrix_name
    );
    let mtx = match read_mtx(std::path::Path::new(&path)) {
        Ok(m) => m,
        Err(e) => {
            println!("  cannot read {}: {:?}", path, e);
            return;
        }
    };
    let n = mtx.n;
    let mut a_full = vec![vec![0.0_f64; n]; n];
    for &(r, c, v) in &mtx.entries {
        a_full[r][c] = v;
        if r != c {
            a_full[c][r] = v;
        }
    }
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            println!("  to_csc fail: {:?}", e);
            return;
        }
    };

    // Read MUMPS Schur sidecar.
    let bin_path = std::path::Path::new(&path).with_extension("mumps_schur.bin");
    let json_path = std::path::Path::new(&path).with_extension("mumps_schur.json");
    let json_text = match std::fs::read_to_string(&json_path) {
        Ok(s) => s,
        Err(_) => {
            println!("  no MUMPS sidecar");
            return;
        }
    };
    let json: serde_json::Value = match serde_json::from_str(&json_text) {
        Ok(v) => v,
        Err(_) => return,
    };
    let schur: Vec<usize> = json["schur_indices_0indexed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let n_schur = schur.len();
    let bytes = std::fs::read(&bin_path).unwrap();
    let mumps: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| {
            let mut b = [0u8; 8];
            b.copy_from_slice(c);
            f64::from_le_bytes(b)
        })
        .collect();

    println!("  n={} n_schur={}", n, n_schur);

    // Dense Gaussian elimination oracle (partial-pivot, full
    // double precision but no compensated summation — used as a
    // third reference, not as ground truth).
    let oracle = dense_oracle_schur(&a_full, &schur);
    let oracle_flat = flatten_oracle(&oracle);

    // Three-way: feral, MUMPS, dense oracle.
    let (mr_mvo, _, _, _) = max_rel_block(&mumps, &oracle_flat, n_schur);
    println!("  MUMPS  vs dense oracle: max_rel = {:.3e}", mr_mvo);

    // BK parameter sweep.
    let snode = SupernodeParams::default();
    let configs: Vec<(&str, BunchKaufmanParams)> = vec![
        (
            "default(thresh=0.01)",
            BunchKaufmanParams {
                on_zero_pivot: ZeroPivotAction::ForceAccept,
                pivot_threshold: 0.01,
                ..BunchKaufmanParams::default()
            },
        ),
        (
            "thresh=0.0",
            BunchKaufmanParams {
                on_zero_pivot: ZeroPivotAction::ForceAccept,
                pivot_threshold: 0.0,
                ..BunchKaufmanParams::default()
            },
        ),
        (
            "thresh=0.001",
            BunchKaufmanParams {
                on_zero_pivot: ZeroPivotAction::ForceAccept,
                pivot_threshold: 0.001,
                ..BunchKaufmanParams::default()
            },
        ),
        (
            "thresh=0.1",
            BunchKaufmanParams {
                on_zero_pivot: ZeroPivotAction::ForceAccept,
                pivot_threshold: 0.1,
                ..BunchKaufmanParams::default()
            },
        ),
        (
            "thresh=0.5",
            BunchKaufmanParams {
                on_zero_pivot: ZeroPivotAction::ForceAccept,
                pivot_threshold: 0.5,
                ..BunchKaufmanParams::default()
            },
        ),
    ];

    println!(
        "  {:<25} {:>14} {:>14} {:>14}",
        "config", "vs_MUMPS", "vs_oracle", "MUMPS-feral_vs_MUMPS-oracle"
    );
    for (label, bk) in configs.iter() {
        let np = NumericParams {
            scaling: ScalingStrategy::Identity,
            ..NumericParams::with_bk(bk.clone())
        };
        let sym = match symbolic_factorize_with_schur(&csc, &snode, &schur) {
            Ok(s) => s,
            Err(e) => {
                println!("  {:<25} symbolic fail: {:?}", label, e);
                continue;
            }
        };
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            factorize_multifrontal_with_schur(&csc, &sym, &np)
        }));
        let sb = match res {
            Ok(Ok((_, _, sb))) => sb,
            _ => {
                println!("  {:<25} factor fail", label);
                continue;
            }
        };
        let mut feral_flat = vec![0.0_f64; n_schur * n_schur];
        for j in 0..n_schur {
            for i in 0..n_schur {
                feral_flat[j * n_schur + i] = sb.get(i, j);
            }
        }
        let (mr_fvm, loc_fvm, av, bv) = max_rel_block(&feral_flat, &mumps, n_schur);
        let (mr_fvo, _, _, _) = max_rel_block(&feral_flat, &oracle_flat, n_schur);
        println!(
            "  {:<25} {:>14.3e} {:>14.3e}    f={:+.3e} m={:+.3e} @ ({},{})",
            label, mr_fvm, mr_fvo, av, bv, loc_fvm.0, loc_fvm.1
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names: Vec<String> = if args.is_empty() {
        vec![
            "ACOPR14_0002".into(),
            "ACOPR14_0001".into(),
            "ACOPR14_0000".into(),
            "ACOPR14_0003".into(),
        ]
    } else {
        args
    };
    for n in names {
        run_one(&n);
    }
}
