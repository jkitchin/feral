//! Phase 2.2.3 follow-up: curate the 30-matrix parity panel used by
//! `tests/parity.rs`.
//!
//! For each matrix in `data/matrices/kkt/`:
//!   1. read the matrix + its MUMPS and SSIDS oracles
//!   2. run feral's sparse path (default config, matching what
//!      `tests/parity.rs` will run) and compute inertia_ok and
//!      residual_ratio = feral_residual / mumps_residual
//!   3. bucket the result by (size_bucket, result_bucket, family)
//!
//! Then sample 30 matrices stratified across buckets with a handful of
//! required inclusions (CHWIRUT1, CRESC100, CRESC132, ACOPP30, and the
//! family worst-offenders SWOPF, HYDCAR20). Write the chosen matrices
//! into `tests/data/parity/<family>/` together with a `manifest.json`
//! and regenerate `tests/parity.rs`.
//!
//! Run with:  cargo run --release --example select_parity_panel
//!
//! Re-run this whenever the solver's behavior changes enough that the
//! panel should be refreshed. The test file it generates is checked
//! in.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, Inertia, KktSidecar, ZeroPivotAction,
};

/// Relative residual tolerance factor K: parity is "feral residual
/// within K*MUMPS residual." Must match `tests/parity.rs::K_RESIDUAL`.
const K_RESIDUAL: f64 = 10.0;

/// Absolute residual floor. For matrices where MUMPS's residual is
/// smaller than this, we accept feral's residual as parity as long as
/// it is also at or below this floor. Prevents penalizing feral on
/// trivial matrices where MUMPS happens to reach sub-machine-precision
/// residuals (e.g. the RHS is structured to cancel to ~1e-30).
const ABS_FLOOR: f64 = 1.0e-14;

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct MatrixResult {
    name: String,
    family: String,
    stem: String,
    mtx_path: PathBuf,
    n: usize,
    mumps_inertia: Inertia,
    mumps_residual: f64,
    ssids_inertia: Option<Inertia>,
    ssids_residual: Option<f64>,
    feral_inertia: Inertia,
    feral_residual: f64,
    inertia_ok: bool,
    residual_ratio: f64,
}

fn family_of(name: &str) -> String {
    if let Some(idx) = name.rfind('_') {
        let suffix = &name[idx + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            return name[..idx].to_string();
        }
    }
    name.to_string()
}

fn size_bucket(n: usize) -> &'static str {
    match n {
        0..=49 => "tiny",
        50..=499 => "small",
        500..=1999 => "medium",
        _ => "large",
    }
}

/// Three-way result bucket: does feral match MUMPS on this matrix?
/// Mirrors the exact test gate so panel-time classifications match
/// what `tests/parity.rs` will see.
fn result_bucket(r: &MatrixResult) -> &'static str {
    if !r.inertia_ok {
        return "fail_inertia";
    }
    // The test gate allows feral to pass if either its residual is
    // within K*mumps, OR it is at or below the absolute floor. Apply
    // the same rule here so buckets agree with test outcomes.
    let target = (K_RESIDUAL * r.mumps_residual).max(ABS_FLOOR);
    if r.feral_residual <= target {
        return "pass";
    }
    if r.residual_ratio.is_nan() || r.feral_residual > 100.0 * target {
        return "fail_residual";
    }
    "close"
}

fn read_oracle(path: &Path) -> Option<(Inertia, f64)> {
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let inertia = data.get("inertia")?;
    let pos = inertia.get("positive")?.as_u64()? as usize;
    let neg = inertia.get("negative")?.as_u64()? as usize;
    let zero = inertia.get("zero")?.as_u64()? as usize;
    let residual = data.get("residual_2norm_relative")?.as_f64()?;
    Some((Inertia::new(pos, neg, zero), residual))
}

fn params() -> BunchKaufmanParams {
    // Phase 2.3: restored pivot_threshold = 0.01 (SSIDS/MUMPS default)
    // now that delayed pivoting gives rejected pivots a landing zone
    // at the parent supernode. Keep in sync with the test-file
    // template below and with bench::params_kkt.
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

struct KktInput {
    mtx_path: PathBuf,
    csc: CscMatrix,
    sidecar: KktSidecar,
    stem: String,
}

fn load_kkt_inputs(dir: &Path) -> Vec<KktInput> {
    let mut out = Vec::new();
    let Ok(subdirs) = std::fs::read_dir(dir) else {
        return out;
    };
    let mut subs: Vec<_> = subdirs.filter_map(|e| e.ok()).collect();
    subs.sort_by_key(|e| e.file_name());
    for sub in subs {
        let sp = sub.path();
        if !sp.is_dir() {
            continue;
        }
        let Ok(files) = std::fs::read_dir(&sp) else {
            continue;
        };
        let mut mtxs: Vec<_> = files
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "mtx"))
            .collect();
        mtxs.sort_by_key(|e| e.file_name());
        for e in mtxs {
            let mtx_path = e.path();
            let stem = mtx_path.file_stem().unwrap().to_string_lossy().to_string();
            let json_path = mtx_path.with_extension("json");
            if !json_path.exists() {
                continue;
            }
            let Ok(mtx) = read_mtx(&mtx_path) else {
                continue;
            };
            let Ok(sidecar) = read_sidecar(&json_path) else {
                continue;
            };
            if sidecar.finite_rhs().is_none() {
                continue;
            }
            if mtx.entries.iter().any(|(_, _, v)| !v.is_finite()) {
                continue;
            }
            let expected_dim = sidecar.n + sidecar.m;
            if mtx.n != expected_dim {
                continue;
            }
            let Ok(csc) = mtx.to_csc() else {
                continue;
            };
            out.push(KktInput {
                mtx_path,
                csc,
                sidecar,
                stem,
            });
        }
    }
    out
}

fn main() {
    let kkt_dir = Path::new("data/matrices/kkt");
    let out_dir = Path::new("tests/data/parity");
    let test_file = Path::new("tests/parity.rs");

    println!("Loading KKT matrices from {} ...", kkt_dir.display());
    let inputs = load_kkt_inputs(kkt_dir);
    println!("  loaded {} matrices", inputs.len());
    if inputs.is_empty() {
        eprintln!("No matrices found. Aborting.");
        std::process::exit(1);
    }

    let mut results: Vec<MatrixResult> = Vec::with_capacity(inputs.len());
    let mut n_mumps_missing = 0usize;
    let mut n_factor_err = 0usize;
    let mut n_solve_err = 0usize;
    let mut processed = 0usize;

    for input in &inputs {
        processed += 1;
        if processed % 5000 == 0 {
            println!("  processed {}/{}", processed, inputs.len());
        }

        let mumps_path = input.mtx_path.with_extension("mumps.json");
        let ssids_path = input.mtx_path.with_extension("ssids.json");

        let Some((mumps_inertia, mumps_residual)) = read_oracle(&mumps_path) else {
            n_mumps_missing += 1;
            continue;
        };
        let (ssids_inertia, ssids_residual) = match read_oracle(&ssids_path) {
            Some((i, r)) => (Some(i), Some(r)),
            None => (None, None),
        };

        let Some(rhs) = input.sidecar.finite_rhs() else {
            continue;
        };

        let Ok(sym) = symbolic_factorize(&input.csc, &SupernodeParams::default()) else {
            n_factor_err += 1;
            continue;
        };
        let Ok((fac, feral_inertia)) = factorize_multifrontal(&input.csc, &sym, &params()) else {
            n_factor_err += 1;
            continue;
        };
        let Ok(x) = solve_sparse_refined(&input.csc, &fac, &rhs) else {
            n_solve_err += 1;
            continue;
        };
        let feral_residual = rel_residual(&input.csc, &x, &rhs);

        let inertia_ok = feral_inertia == mumps_inertia;
        let residual_ratio = if mumps_residual > 0.0 {
            feral_residual / mumps_residual
        } else {
            feral_residual / 1e-16
        };

        results.push(MatrixResult {
            name: input.stem.clone(),
            family: family_of(&input.stem),
            stem: input.stem.clone(),
            mtx_path: input.mtx_path.clone(),
            n: input.csc.n,
            mumps_inertia,
            mumps_residual,
            ssids_inertia,
            ssids_residual,
            feral_inertia,
            feral_residual,
            inertia_ok,
            residual_ratio,
        });
    }

    println!(
        "\nProcessed {} matrices: {} mumps-missing, {} factor-err, {} solve-err, {} scored",
        inputs.len(),
        n_mumps_missing,
        n_factor_err,
        n_solve_err,
        results.len()
    );

    // Bucket and print a size x result table.
    let mut table: BTreeMap<(&'static str, &'static str), usize> = BTreeMap::new();
    for r in &results {
        *table
            .entry((size_bucket(r.n), result_bucket(r)))
            .or_insert(0) += 1;
    }
    println!("\nBucket table (size x result):");
    let sizes = ["tiny", "small", "medium", "large"];
    let kinds = ["pass", "close", "fail_inertia", "fail_residual"];
    print!("{:<8}", "");
    for k in &kinds {
        print!("{:>16}", k);
    }
    println!();
    for s in &sizes {
        print!("{:<8}", s);
        for k in &kinds {
            let v = table.get(&(s, k)).copied().unwrap_or(0);
            print!("{:>16}", v);
        }
        println!();
    }

    // Select the panel.
    let mut selected: Vec<MatrixResult> = Vec::new();
    let mut picked_names: BTreeSet<String> = BTreeSet::new();
    let mut picked_families: HashMap<String, usize> = HashMap::new();

    // Required matrices: the four mc64_regression targets, plus two
    // worst-offender families we want explicit coverage of.
    let required_names = [
        "CHWIRUT1_0000",
        "CRESC100_0000",
        "CRESC132_0000",
        "ACOPP30_0000",
    ];
    let required_families = ["SWOPF", "HYDCAR20"];

    for name in required_names {
        if let Some(r) = results.iter().find(|r| r.name == name) {
            if picked_names.insert(r.name.clone()) {
                *picked_families.entry(r.family.clone()).or_insert(0) += 1;
                selected.push(r.clone());
            }
        }
    }
    for fam in required_families {
        if let Some(r) = results
            .iter()
            .find(|r| r.family == fam && !picked_names.contains(&r.name))
        {
            if picked_names.insert(r.name.clone()) {
                *picked_families.entry(r.family.clone()).or_insert(0) += 1;
                selected.push(r.clone());
            }
        }
    }

    // Target allocation across (size, result) buckets. 30 total minus
    // the required matrices above. Under the absolute-floor gate
    // (fail_residual is nearly empty), the frontier is almost
    // entirely inertia mismatches, so we weight those buckets more.
    let target: &[(&str, &str, usize)] = &[
        ("tiny", "pass", 2),
        ("small", "pass", 3),
        ("medium", "pass", 2),
        ("large", "pass", 1),
        ("tiny", "close", 3),
        ("small", "close", 2),
        ("medium", "close", 1),
        ("tiny", "fail_inertia", 4),
        ("small", "fail_inertia", 4),
        ("medium", "fail_inertia", 2),
        ("large", "fail_inertia", 1),
    ];

    for (sz, kind, count) in target {
        // Candidates in this bucket, preferring matrices whose family
        // isn't already represented.
        let mut candidates: Vec<&MatrixResult> = results
            .iter()
            .filter(|r| {
                size_bucket(r.n) == *sz
                    && result_bucket(r) == *kind
                    && !picked_names.contains(&r.name)
            })
            .collect();
        // Sort by: new family first, then by residual ratio (more
        // diverse failure modes), then alphabetical for determinism.
        candidates.sort_by(|a, b| {
            let fam_a = picked_families.contains_key(&a.family) as u8;
            let fam_b = picked_families.contains_key(&b.family) as u8;
            fam_a
                .cmp(&fam_b)
                .then_with(|| {
                    b.residual_ratio
                        .partial_cmp(&a.residual_ratio)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.name.cmp(&b.name))
        });
        for c in candidates.into_iter().take(*count) {
            if picked_names.insert(c.name.clone()) {
                *picked_families.entry(c.family.clone()).or_insert(0) += 1;
                selected.push(c.clone());
            }
        }
    }

    selected.sort_by(|a, b| a.name.cmp(&b.name));
    println!("\nSelected {} matrices:", selected.len());
    for r in &selected {
        println!(
            "  {:<28} n={:<6} bucket={:<14} inertia_ok={} res_ratio={:.2e} feral_res={:.2e} mumps_res={:.2e}",
            r.name,
            r.n,
            format!("{}/{}", size_bucket(r.n), result_bucket(r)),
            r.inertia_ok,
            r.residual_ratio,
            r.feral_residual,
            r.mumps_residual
        );
    }

    // Write files and manifest.
    std::fs::create_dir_all(out_dir).expect("mkdir parity");
    let mut manifest_entries = Vec::new();

    for r in &selected {
        let fam_dir = out_dir.join(r.family.to_ascii_lowercase());
        std::fs::create_dir_all(&fam_dir).expect("mkdir fam");

        let stems = [
            (r.mtx_path.clone(), format!("{}.mtx", r.stem)),
            (
                r.mtx_path.with_extension("mumps.json"),
                format!("{}.mumps.json", r.stem),
            ),
            (
                r.mtx_path.with_extension("json"),
                format!("{}.json", r.stem),
            ),
        ];
        for (src, dst_name) in stems {
            let dst = fam_dir.join(dst_name);
            if let Err(e) = std::fs::copy(&src, &dst) {
                eprintln!("copy {} -> {}: {}", src.display(), dst.display(), e);
            }
        }
        // SSIDS sidecar is optional.
        let ssids_src = r.mtx_path.with_extension("ssids.json");
        if ssids_src.exists() {
            let _ = std::fs::copy(&ssids_src, fam_dir.join(format!("{}.ssids.json", r.stem)));
        }

        manifest_entries.push(format!(
            "    {{\"name\":\"{}\",\"family\":\"{}\",\"dir\":\"{}\",\"n\":{},\"bucket\":\"{}/{}\",\"inertia_ok_at_panel\":{},\"residual_ratio_at_panel\":{:.4e},\"feral_residual_at_panel\":{:.4e},\"mumps_residual_at_panel\":{:.4e}}}",
            r.name,
            r.family,
            r.family.to_ascii_lowercase(),
            r.n,
            size_bucket(r.n),
            result_bucket(r),
            r.inertia_ok,
            r.residual_ratio,
            r.feral_residual,
            r.mumps_residual,
        ));
    }

    let manifest = format!(
        "{{\n  \"k_residual\": {},\n  \"count\": {},\n  \"generated_by\": \"examples/select_parity_panel.rs\",\n  \"note\": \"The *_at_panel fields record feral's output at panel-selection time, not the current truth. Use tests/parity.rs as the live gate.\",\n  \"entries\": [\n{}\n  ]\n}}\n",
        K_RESIDUAL,
        selected.len(),
        manifest_entries.join(",\n"),
    );
    std::fs::write(out_dir.join("manifest.json"), manifest).expect("write manifest");

    // Regenerate tests/parity.rs.
    let mut test_body = String::new();
    test_body.push_str("//! Phase 2.2.3 follow-up — parity panel.\n");
    test_body.push_str("//!\n");
    test_body.push_str("//! For each curated matrix in `tests/data/parity/`, assert feral's\n");
    test_body.push_str("//! multi-frontal solve matches the MUMPS oracle exactly on inertia\n");
    test_body.push_str("//! and within K*MUMPS on relative residual. Regenerate this file by\n");
    test_body.push_str("//! running:\n");
    test_body.push_str("//!     cargo run --release --example select_parity_panel\n");
    test_body.push_str("//!\n");
    test_body.push_str("//! Do NOT edit tests/parity.rs by hand. The file is generated.\n\n");
    test_body.push_str("use std::path::Path;\n\n");
    test_body.push_str("use feral::numeric::factorize::factorize_multifrontal;\n");
    test_body.push_str("use feral::numeric::solve::solve_sparse_refined;\n");
    test_body.push_str("use feral::symbolic::{symbolic_factorize, SupernodeParams};\n");
    test_body.push_str(
        "use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};\n\n",
    );
    test_body.push_str(&format!("const K_RESIDUAL: f64 = {:.1};\n", K_RESIDUAL));
    test_body.push_str(&format!("const ABS_FLOOR: f64 = {:e};\n\n", ABS_FLOOR));
    test_body.push_str("fn ldlt_params() -> BunchKaufmanParams {\n");
    test_body.push_str("    BunchKaufmanParams {\n");
    test_body.push_str("        on_zero_pivot: ZeroPivotAction::ForceAccept,\n");
    test_body.push_str("        pivot_threshold: 0.01,\n");
    test_body.push_str("        ..BunchKaufmanParams::default()\n");
    test_body.push_str("    }\n");
    test_body.push_str("}\n\n");
    test_body.push_str("fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {\n");
    test_body.push_str("    let n = a.n;\n");
    test_body.push_str("    let mut ax = vec![0.0; n];\n");
    test_body.push_str("    a.symv(x, &mut ax);\n");
    test_body.push_str("    let mut rs = 0.0;\n");
    test_body.push_str("    let mut bs = 0.0;\n");
    test_body.push_str("    for i in 0..n {\n");
    test_body.push_str("        rs += (ax[i] - b[i]).powi(2);\n");
    test_body.push_str("        bs += b[i] * b[i];\n");
    test_body.push_str("    }\n");
    test_body.push_str("    if bs > 0.0 { (rs / bs).sqrt() } else { rs.sqrt() }\n");
    test_body.push_str("}\n\n");
    test_body.push_str("fn read_oracle(path: &Path) -> (Inertia, f64) {\n");
    test_body.push_str("    let data: serde_json::Value = serde_json::from_str(\n");
    test_body.push_str("        &std::fs::read_to_string(path).expect(\"read oracle\"),\n");
    test_body.push_str("    )\n");
    test_body.push_str("    .expect(\"parse oracle\");\n");
    test_body
        .push_str("    let pos = data[\"inertia\"][\"positive\"].as_u64().unwrap() as usize;\n");
    test_body
        .push_str("    let neg = data[\"inertia\"][\"negative\"].as_u64().unwrap() as usize;\n");
    test_body.push_str("    let zero = data[\"inertia\"][\"zero\"].as_u64().unwrap() as usize;\n");
    test_body.push_str("    let res = data[\"residual_2norm_relative\"].as_f64().unwrap();\n");
    test_body.push_str("    (Inertia::new(pos, neg, zero), res)\n");
    test_body.push_str("}\n\n");
    test_body.push_str("fn run_parity(fam: &str, stem: &str) {\n");
    test_body.push_str("    let base = format!(\"tests/data/parity/{}/{}\", fam, stem);\n");
    test_body.push_str(
        "    let mtx = read_mtx(Path::new(&format!(\"{}.mtx\", base))).expect(\"read mtx\");\n",
    );
    test_body.push_str("    let csc = mtx.to_csc().expect(\"to_csc\");\n");
    test_body.push_str("    let sidecar = read_sidecar(Path::new(&format!(\"{}.json\", base))).expect(\"sidecar\");\n");
    test_body.push_str("    let rhs = sidecar.finite_rhs().expect(\"finite rhs\");\n");
    test_body.push_str("    let (mumps_inertia, mumps_res) = read_oracle(Path::new(&format!(\"{}.mumps.json\", base)));\n\n");
    test_body.push_str("    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect(\"symbolic\");\n");
    test_body.push_str("    let (fac, inertia) = factorize_multifrontal(&csc, &sym, &ldlt_params()).expect(\"factor\");\n");
    test_body.push_str("    let x = solve_sparse_refined(&csc, &fac, &rhs).expect(\"solve\");\n");
    test_body.push_str("    let feral_res = rel_residual(&csc, &x, &rhs);\n\n");
    test_body.push_str("    assert_eq!(\n");
    test_body.push_str("        inertia, mumps_inertia,\n");
    test_body.push_str("        \"{} inertia: feral={} mumps={}\", stem, inertia, mumps_inertia\n");
    test_body.push_str("    );\n");
    test_body.push_str("    // Gate: feral residual must be within K*MUMPS residual, OR at or\n");
    test_body.push_str("    // below the absolute floor. The floor catches matrices where MUMPS\n");
    test_body.push_str("    // produces sub-machine-precision residuals (e.g. 1e-30) that feral\n");
    test_body.push_str("    // cannot and should not be expected to match.\n");
    test_body.push_str("    let target = (K_RESIDUAL * mumps_res).max(ABS_FLOOR);\n");
    test_body.push_str("    assert!(\n");
    test_body.push_str("        feral_res <= target,\n");
    test_body.push_str(
        "        \"{} residual: feral={:.3e} > max(K*mumps={:.3e}, floor={:.3e}) = {:.3e}\",\n",
    );
    test_body.push_str("        stem, feral_res, K_RESIDUAL * mumps_res, ABS_FLOOR, target\n");
    test_body.push_str("    );\n");
    test_body.push_str("}\n\n");

    // Count passing at panel time for a progress summary.
    let pass_count = selected
        .iter()
        .filter(|r| result_bucket(r) == "pass")
        .count();
    test_body.push_str(&format!(
        "// Panel snapshot: {}/{} matrices pass MUMPS parity at panel time.\n\
         // Failing matrices are `#[ignore]`'d with the panel-time failure\n\
         // mode in the attribute comment. Passing matrices run as regular\n\
         // tests and protect against regression. As fixes land, rerun\n\
         // `cargo run --release --example select_parity_panel` to refresh\n\
         // the panel and un-ignore the now-passing matrices.\n\n",
        pass_count,
        selected.len(),
    ));

    for r in &selected {
        let fn_name = format!(
            "parity_{}",
            r.name.to_ascii_lowercase().replace(['-', '.'], "_")
        );
        let is_pass = result_bucket(r) == "pass";
        if is_pass {
            test_body.push_str(&format!(
                "#[test]\nfn {}() {{\n    run_parity(\"{}\", \"{}\");\n}}\n\n",
                fn_name,
                r.family.to_ascii_lowercase(),
                r.name,
            ));
        } else {
            let reason = match result_bucket(r) {
                "fail_inertia" => format!(
                    "inertia mismatch (feral={} mumps={})",
                    r.feral_inertia, r.mumps_inertia
                ),
                "fail_residual" => format!(
                    "residual ratio {:.2e} (feral={:.2e}, mumps={:.2e})",
                    r.residual_ratio, r.feral_residual, r.mumps_residual
                ),
                "close" => format!(
                    "residual ratio {:.2e} > K=10 (feral={:.2e}, mumps={:.2e})",
                    r.residual_ratio, r.feral_residual, r.mumps_residual
                ),
                _ => "unknown".into(),
            };
            test_body.push_str(&format!(
                "// Panel time: {}\n#[test]\n#[ignore]\nfn {}() {{\n    run_parity(\"{}\", \"{}\");\n}}\n\n",
                reason,
                fn_name,
                r.family.to_ascii_lowercase(),
                r.name,
            ));
        }
    }

    std::fs::write(test_file, test_body).expect("write tests/parity.rs");

    println!(
        "\nWrote tests/parity.rs with {} tests and tests/data/parity/manifest.json",
        selected.len()
    );
}
