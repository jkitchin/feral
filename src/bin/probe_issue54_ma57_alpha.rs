//! Issue #54 — MA57 head-to-head on the α-shift sweep.
//!
//! Generates the same α-shifted matrices the feral probe uses
//! (uniform `α·I` and x-block `α·diag(pos-diag rows)` sweeps),
//! writes them to disk as MatrixMarket files, builds the
//! manifest for `external_benchmarks/ma57_oracle/ma57_bench`,
//! invokes the MA57 oracle, and parses the per-α sidecars.
//!
//! Verdict logic mirrors `probe_issue54_alpha_shift`: MA57's
//! `(INFO(24), N - INFO(25))` gives `(neg, zero)`; positive =
//! n - neg - zero. Weyl monotonicity is checked on `neg` and
//! on `neg + zero`. The probe is "fix-pointable" — if MA57's
//! sweep is monotone where feral's is not, feral's BK kernel
//! has a fixable bug; if both jitter, the matrix is genuinely
//! beyond dense direct solver capability in the bad α regime.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use feral::read_mtx;
use feral::CscMatrix;

fn shifted_mask(a: &CscMatrix, alpha: f64, mask: &[bool]) -> CscMatrix {
    let n = a.n;
    let col_ptr = a.col_ptr.clone();
    let row_idx = a.row_idx.clone();
    let mut values = a.values.clone();
    let mut diag_present = vec![false; n];
    for col in 0..n {
        let s = col_ptr[col];
        let e = col_ptr[col + 1];
        for k in s..e {
            if row_idx[k] == col {
                if mask[col] {
                    values[k] += alpha;
                }
                diag_present[col] = true;
                break;
            }
        }
    }
    let mut rows = Vec::with_capacity(row_idx.len() + n);
    let mut cols = Vec::with_capacity(row_idx.len() + n);
    let mut vals = Vec::with_capacity(row_idx.len() + n);
    for col in 0..n {
        let s = col_ptr[col];
        let e = col_ptr[col + 1];
        let mut inserted = diag_present[col];
        for k in s..e {
            let r = row_idx[k];
            let v = values[k];
            if !inserted && r > col {
                if mask[col] && alpha != 0.0 {
                    rows.push(col);
                    cols.push(col);
                    vals.push(alpha);
                }
                inserted = true;
            }
            rows.push(r);
            cols.push(col);
            vals.push(v);
        }
        if !inserted && mask[col] && alpha != 0.0 {
            rows.push(col);
            cols.push(col);
            vals.push(alpha);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("rebuild csc")
}

fn classify_pos_diag(a: &CscMatrix) -> Vec<bool> {
    let n = a.n;
    let mut mask = vec![false; n];
    for col in 0..n {
        let s = a.col_ptr[col];
        let e = a.col_ptr[col + 1];
        for k in s..e {
            if a.row_idx[k] == col {
                if a.values[k] > 0.0 {
                    mask[col] = true;
                }
                break;
            }
        }
    }
    mask
}

fn write_mtx_symmetric(a: &CscMatrix, path: &Path) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "%%MatrixMarket matrix coordinate real symmetric")?;
    let mut nnz = 0usize;
    for col in 0..a.n {
        let s = a.col_ptr[col];
        let e = a.col_ptr[col + 1];
        for k in s..e {
            if a.row_idx[k] >= col {
                nnz += 1;
            }
        }
    }
    writeln!(w, "{} {} {}", a.n, a.n, nnz)?;
    for col in 0..a.n {
        let s = a.col_ptr[col];
        let e = a.col_ptr[col + 1];
        for k in s..e {
            let r = a.row_idx[k];
            if r >= col {
                writeln!(w, "{} {} {:.17e}", r + 1, col + 1, a.values[k])?;
            }
        }
    }
    Ok(())
}

fn write_rhs(n: usize, path: &Path) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    for i in 0..n {
        let v = (((i as u64).wrapping_mul(2654435761) % 9999) as f64) / 9999.0 - 0.5;
        writeln!(w, "{:.17e}", v)?;
    }
    Ok(())
}

#[derive(Debug, Default)]
struct Sidecar {
    n: i64,
    inertia_pos: i64,
    inertia_neg: i64,
    inertia_zero: i64,
    rel_res: f64,
    info1: i64,
    status: String,
}

fn parse_sidecar(path: &Path) -> std::io::Result<Sidecar> {
    let s = std::fs::read_to_string(path)?;
    let mut out = Sidecar::default();
    for line in s.lines() {
        let mut it = line.split_whitespace();
        let key = it.next().unwrap_or("");
        let val = it.next().unwrap_or("");
        match key {
            "n" => out.n = val.parse().unwrap_or(-1),
            "inertia_pos" => out.inertia_pos = val.parse().unwrap_or(-1),
            "inertia_neg" => out.inertia_neg = val.parse().unwrap_or(-1),
            "inertia_zero" => out.inertia_zero = val.parse().unwrap_or(-1),
            "rel_res" => out.rel_res = val.parse().unwrap_or(f64::NAN),
            "info1" => out.info1 = val.parse().unwrap_or(0),
            "status" => out.status = val.to_string(),
            _ => {}
        }
    }
    Ok(out)
}

fn run_ma57_sweep(
    label: &str,
    a0: &CscMatrix,
    shift_kind: &str,
    mask: Option<&[bool]>,
    alphas: &[f64],
    workdir: &Path,
) {
    println!("\n=== MA57 | {} | shift={} ===", label, shift_kind);

    let ma57_bin = std::env::current_dir()
        .unwrap()
        .join("external_benchmarks/ma57_oracle/ma57_bench");
    if !ma57_bin.exists() {
        eprintln!("ma57_bench binary not found at {:?}", ma57_bin);
        return;
    }

    let rhs_path = workdir.join("shared.rhs.txt");
    write_rhs(a0.n, &rhs_path).expect("write rhs");

    let mut mtx_paths: Vec<PathBuf> = Vec::new();
    let mut out_paths: Vec<PathBuf> = Vec::new();
    for (i, &alpha) in alphas.iter().enumerate() {
        let a_shift = match mask {
            Some(m) => shifted_mask(a0, alpha, m),
            None => shifted_mask(a0, alpha, &vec![true; a0.n]),
        };
        let mtx_path = workdir.join(format!("{}_alpha{}.mtx", label, i));
        write_mtx_symmetric(&a_shift, &mtx_path).expect("write mtx");
        let out_path = workdir.join(format!("{}_alpha{}.out", label, i));
        mtx_paths.push(mtx_path);
        out_paths.push(out_path);
    }

    let manifest = workdir.join(format!("{}.manifest", label));
    let mut m = BufWriter::new(File::create(&manifest).expect("manifest"));
    for (mp, op) in mtx_paths.iter().zip(out_paths.iter()) {
        writeln!(
            m,
            "{} {} {}",
            mp.display(),
            rhs_path.display(),
            op.display()
        )
        .expect("write line");
    }
    drop(m);

    let status = Command::new(&ma57_bin)
        .arg(&manifest)
        .status()
        .expect("run ma57_bench");
    if !status.success() {
        eprintln!("ma57_bench returned {:?}", status);
    }

    println!(
        " {:<10} {:<8} {:<8} {:<8} {:<10} {:<10} status        Δneg",
        "alpha", "neg", "zero", "pos", "neg+zero", "rel_resid",
    );
    let mut last_neg: Option<isize> = None;
    let mut last_nz: Option<isize> = None;
    let mut weyl_violations = 0;
    let mut nz_violations = 0;
    for (i, &alpha) in alphas.iter().enumerate() {
        let s = match parse_sidecar(&out_paths[i]) {
            Ok(s) => s,
            Err(e) => {
                println!(" {:<10.1e} parse_err={:?}", alpha, e);
                continue;
            }
        };
        let neg = s.inertia_neg as isize;
        let nz = (s.inertia_neg + s.inertia_zero) as isize;
        let neg_str = match last_neg {
            None => "          ".to_string(),
            Some(prev) => {
                let d = neg - prev;
                if d > 0 {
                    weyl_violations += 1;
                    format!(" Δ={:+5} W!", d)
                } else {
                    format!(" Δ={:+5}   ", d)
                }
            }
        };
        let nz_str = match last_nz {
            None => "".to_string(),
            Some(prev) => {
                let d = nz - prev;
                if d > 0 {
                    nz_violations += 1;
                    format!(" Δnz={:+5} W!", d)
                } else {
                    format!(" Δnz={:+5}   ", d)
                }
            }
        };
        println!(
            " {:<10.1e} {:<8} {:<8} {:<8} {:<10} {:<10.3e} {}{}{}",
            alpha,
            s.inertia_neg,
            s.inertia_zero,
            s.inertia_pos,
            s.inertia_neg + s.inertia_zero,
            s.rel_res,
            s.status,
            neg_str,
            nz_str,
        );
        last_neg = Some(neg);
        last_nz = Some(nz);
    }
    println!(
        " summary: Weyl(neg-monotone) violations = {}, (neg+zero) violations = {}",
        weyl_violations, nz_violations,
    );
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("dev/repros/issue-54/nuffield2_trap_iter1.mtx"));

    let mtx = read_mtx(&path).expect("read mtx");
    let a0 = mtx.to_csc().expect("to_csc");
    let n = a0.n;
    let pos_mask = classify_pos_diag(&a0);
    let n_pos: usize = pos_mask.iter().filter(|p| **p).count();
    println!("matrix       = {:?}", path);
    println!("n            = {}", n);
    println!("|pos-diag|   = {}", n_pos);
    println!("|zero-diag|  = {}", n - n_pos);

    let workdir = std::env::temp_dir().join("feral_issue54_ma57");
    std::fs::create_dir_all(&workdir).expect("workdir");
    println!("workdir      = {:?}", workdir);

    let alphas = [
        0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e8, 1e12, 1e16, 1e20,
    ];

    run_ma57_sweep("uniform", &a0, "uniform α·I", None, &alphas, &workdir);
    run_ma57_sweep(
        "xblock",
        &a0,
        "x-block α·diag(pos-diag rows)",
        Some(&pos_mask),
        &alphas,
        &workdir,
    );

    println!(
        "\nVerdict legend:\n\
         - 'W!' on Δneg / Δnz = Weyl monotonicity violation.\n\
         - For uniform α·I shifts, neg must be non-increasing.\n\
         - For x-block α·diag(1_x) shifts (PSD perturbation), neg\n\
           must also be non-increasing.\n\
         - Compare against feral output in alpha_shift_v4.out.\n\
         - If MA57 stays monotone where feral does not, feral's BK\n\
           kernel has a fixable bug. If MA57 also jitters, the\n\
           matrix is beyond dense direct solver capability in the\n\
           bad α regime.\n"
    );
}
