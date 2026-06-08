//! Real-world unsymmetric LU validation harness (issue #81, epic Phase 7).
//!
//! Reads square general (unsymmetric) Matrix Market files — e.g. the
//! SuiteSparse square-unsymmetric matrices fetched by
//! `scripts/fetch_lu_corpus.py` — and validates the sparse LU on each against a
//! KNOWN solution oracle: pick `x_true`, form `a = B·x_true`, solve `B x = a`,
//! and report `‖x − x_true‖/‖x_true‖` and `‖Bx − a‖/‖a‖`. Also exercises a
//! self-replace rank-1 update (must leave the basis unchanged). No external
//! solver is required (the known-x oracle is self-contained); a companion
//! `scripts/lu_reference_scipy.py` cross-checks against SciPy's SuperLU.
//!
//! Run: `cargo run -p feral-diagnostics --release --bin lu_reference -- [DIR]`
//! (DIR defaults to `data/matrices/lu-corpus`).

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{LuParams, LuScaling, LuSingularAction, SparseLu, SparseLuSymbolic};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Parsed general Matrix Market matrix (square).
struct Mtx {
    n: usize,
    nnz: usize,
    triplets: Vec<(usize, usize, f64)>,
}

/// Parse a real `coordinate` Matrix Market file (`general` or `symmetric`).
/// Symmetric files are expanded to both triangles. Requires a square matrix.
fn parse_general_mtx(text: &str) -> Result<Mtx, String> {
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty file")?.to_ascii_lowercase();
    if !header.starts_with("%%matrixmarket matrix coordinate real") {
        return Err(format!("unsupported header: {header}"));
    }
    let symmetric = header.contains("symmetric");
    if header.contains("pattern") || header.contains("complex") {
        return Err("pattern/complex not supported".into());
    }
    // Skip comments and find the size line.
    let mut size_line = None;
    for line in lines.by_ref() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('%') {
            continue;
        }
        size_line = Some(t.to_string());
        break;
    }
    let size_line = size_line.ok_or("missing size line")?;
    let mut it = size_line.split_whitespace();
    let rows: usize = it.next().and_then(|s| s.parse().ok()).ok_or("bad rows")?;
    let cols: usize = it.next().and_then(|s| s.parse().ok()).ok_or("bad cols")?;
    let nnz: usize = it.next().and_then(|s| s.parse().ok()).ok_or("bad nnz")?;
    if rows != cols {
        return Err(format!("not square: {rows}x{cols}"));
    }
    let mut triplets = Vec::with_capacity(if symmetric { 2 * nnz } else { nnz });
    for line in lines {
        let t = line.trim();
        if t.is_empty() || t.starts_with('%') {
            continue;
        }
        let mut p = t.split_whitespace();
        let r: usize = p.next().and_then(|s| s.parse().ok()).ok_or("bad row idx")?;
        let c: usize = p.next().and_then(|s| s.parse().ok()).ok_or("bad col idx")?;
        let v: f64 = p.next().and_then(|s| s.parse().ok()).ok_or("bad value")?;
        let (r0, c0) = (r - 1, c - 1); // 1-based -> 0-based
        triplets.push((r0, c0, v));
        if symmetric && r0 != c0 {
            triplets.push((c0, r0, v));
        }
    }
    Ok(Mtx {
        n: rows,
        nnz: triplets.len(),
        triplets,
    })
}

fn to_sparse_cols(m: &Mtx) -> Result<SparseColMatrix, String> {
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m.n];
    for &(r, c, v) in m.triplets.iter() {
        cols[c].push((r, v));
    }
    SparseColMatrix::from_sparse_columns(m.n, &cols).map_err(|e| format!("{e}"))
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

fn rel(num: f64, den: f64) -> f64 {
    num / den.max(1e-300)
}

/// Validate one matrix; returns a one-line report.
fn check(path: &Path) -> String {
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => return format!("{name:24}  READ-ERR {e}"),
    };
    let mtx = match parse_general_mtx(&text) {
        Ok(m) => m,
        Err(e) => return format!("{name:24}  SKIP {e}"),
    };
    let b = match to_sparse_cols(&mtx) {
        Ok(b) => b,
        Err(e) => return format!("{name:24}  SKIP {e}"),
    };
    let n = b.m;
    // Known solution oracle.
    let x_true: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
    let mut a = vec![0.0; n];
    b.matvec(&x_true, &mut a);

    // Factor with MC64 scaling + PerturbToEps (so near-singular reals survive).
    let params = LuParams {
        scaling: LuScaling::Mc64,
        on_singular: LuSingularAction::PerturbToEps { abs_floor: 1e-12 },
        refine_steps: 2,
        refine_tol: 1e-14,
        ..LuParams::default()
    };
    let symbolic = match SparseLuSymbolic::analyze(&b) {
        Ok(s) => s,
        Err(e) => return format!("{name:24}  SKIP analyze {e}"),
    };
    let t0 = Instant::now();
    let mut lu = match SparseLu::factor(&b, &symbolic, params) {
        Ok(lu) => lu,
        Err(e) => return format!("{name:24}  n={n:<7} FACTOR-FAIL {e}"),
    };
    let factor_us = t0.elapsed().as_secs_f64() * 1e6;
    let fill = lu.factor_nnz() as f64 / (mtx.nnz.max(1) as f64);

    // Plain solve.
    let mut x = a.clone();
    if lu.ftran(&mut x).is_err() {
        return format!("{name:24}  n={n:<7} FTRAN-ERR");
    }
    let err = rel(
        inf_norm(
            &x.iter()
                .zip(&x_true)
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>(),
        ),
        inf_norm(&x_true),
    );
    let mut bx = vec![0.0; n];
    b.matvec(&x, &mut bx);
    let resid = rel(
        inf_norm(&bx.iter().zip(&a).map(|(&p, &q)| p - q).collect::<Vec<_>>()),
        inf_norm(&a),
    );
    // Refined solve.
    let mut xr = a.clone();
    let _ = lu.ftran_refined(&b, &mut xr);
    let mut bxr = vec![0.0; n];
    b.matvec(&xr, &mut bxr);
    let resid_ref = rel(
        inf_norm(&bxr.iter().zip(&a).map(|(&p, &q)| p - q).collect::<Vec<_>>()),
        inf_norm(&a),
    );

    // Self-replace rank-1 update: replacing a column with itself must leave the
    // factorization solving the same system.
    let slot = n / 2;
    let (rows, vals) = b.column(slot);
    let same_col: Vec<(usize, f64)> = rows.iter().copied().zip(vals.iter().copied()).collect();
    let update_note = match lu.update_sparse(slot, &same_col) {
        Ok(()) => {
            let mut xu = a.clone();
            let _ = lu.ftran(&mut xu);
            let mut bxu = vec![0.0; n];
            b.matvec(&xu, &mut bxu);
            let ru = rel(
                inf_norm(&bxu.iter().zip(&a).map(|(&p, &q)| p - q).collect::<Vec<_>>()),
                inf_norm(&a),
            );
            format!("upd={ru:.1e}")
        }
        Err(e) => format!("upd={e}"),
    };

    // A large forward error with a machine-precision residual is matrix
    // conditioning, not a solver fault: the solve is backward-stable
    // (‖Bx−a‖/‖a‖ tiny) but the near-singular matrix can't recover x_true.
    // Label that ILL so it is not conflated with a genuine BAD solve.
    let verdict = if err < 1e-6 {
        "OK "
    } else if resid < 1e-8 {
        "ILL"
    } else {
        "BAD"
    };
    format!(
        "{name:24}  n={n:<7} nnz={:<9} {verdict} err={err:.2e} resid={resid:.2e} ref={resid_ref:.2e} fill={fill:.1} {update_note} ({factor_us:.0}µs)",
        mtx.nnz
    )
}

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/matrices/lu-corpus"));
    println!("LU reference harness — corpus dir: {}", dir.display());
    let mut files: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mtx"))
            .collect(),
        Err(e) => {
            eprintln!(
                "cannot read {}: {e}\nFetch matrices first: python scripts/fetch_lu_corpus.py",
                dir.display()
            );
            return;
        }
    };
    files.sort();
    if files.is_empty() {
        eprintln!(
            "no .mtx files in {} — run scripts/fetch_lu_corpus.py",
            dir.display()
        );
        return;
    }
    let mut ok = 0usize;
    let mut ill = 0usize;
    let mut total = 0usize;
    for f in files.iter() {
        let line = check(f);
        if line.contains(" OK ") {
            ok += 1;
        }
        if line.contains(" ILL ") {
            ill += 1;
        }
        if line.contains(" OK ")
            || line.contains(" ILL ")
            || line.contains(" BAD ")
            || line.contains("FAIL")
        {
            total += 1;
        }
        println!("{line}");
    }
    println!(
        "\n{ok}/{total} solved to err < 1e-6 (known-x oracle); \
         {ill} ILL (backward-stable, resid < 1e-8, but large forward error from conditioning)."
    );
}
