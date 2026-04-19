//! Diff per-matrix bench dumps across scaling policies and
//! cross-reference each regression against the matrix's verdict
//! file.
//!
//! Reads three CSVs produced by `cargo run --bin bench --release`
//! with `FERAL_BENCH_DUMP=path.csv` set:
//!   - dev/results/lever-c/dump-baseline.csv
//!   - dev/results/lever-c/dump-mc64.csv
//!   - dev/results/lever-c/dump-adaptive.csv
//!
//! For each (policy ∈ {mc64, adaptive}), enumerates the matrices
//! that:
//!   (R) regressed from baseline residual_ok=1 to policy residual_ok=0
//!   (I) regressed from baseline inertia_ok=1 to policy inertia_ok=0
//! Then loads `data/matrices/kkt/<family>/<name>.verdict.json`
//! for each regressed matrix and reports the
//! `inertia_agreement` field, so we can tell consensus regressions
//! from oracle-disagreement noise.
//!
//! Usage: `cargo run --release --bin dump_diff`.
//!
//! No production code change. The output is the data the lever-C
//! Policy 4 decision needs.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
#[allow(dead_code)] // act/exp parsed for completeness; printed by Debug only.
struct Row {
    name: String,
    n: usize,
    inertia_ok: bool,
    residual_ok: bool,
    rel_res: f64,
    act: (usize, usize, usize),
    exp: (usize, usize, usize),
}

fn parse_csv(path: &str) -> Vec<Row> {
    let txt = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("FAIL reading {}: {}", path, e);
            return Vec::new();
        }
    };
    let mut rows = Vec::new();
    for (lineno, line) in txt.lines().enumerate() {
        if lineno == 0 || line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 13 {
            eprintln!("skipping malformed row {}: {}", lineno, line);
            continue;
        }
        let name = cols[0].to_string();
        let n = cols[1].parse().unwrap_or(0);
        let exp = (
            cols[4].parse().unwrap_or(0),
            cols[5].parse().unwrap_or(0),
            cols[6].parse().unwrap_or(0),
        );
        let act = (
            cols[7].parse().unwrap_or(0),
            cols[8].parse().unwrap_or(0),
            cols[9].parse().unwrap_or(0),
        );
        let inertia_ok = cols[10] == "1";
        let rel_res: f64 = cols[11].parse().unwrap_or(f64::NAN);
        let residual_ok = cols[12] == "1";
        rows.push(Row {
            name,
            n,
            inertia_ok,
            residual_ok,
            rel_res,
            act,
            exp,
        });
    }
    rows
}

fn family_of(name: &str) -> &str {
    if let Some(i) = name.rfind('_') {
        let suf = &name[i + 1..];
        if !suf.is_empty() && suf.chars().all(|c| c.is_ascii_digit()) {
            return &name[..i];
        }
    }
    name
}

/// Best-effort parse of `.verdict.json`'s `inertia_agreement` and
/// `verdict` fields without pulling in serde. The file is small
/// (<1 KB per matrix) so a substring scan is fine.
fn read_verdict(name: &str) -> (String, String) {
    let fam = family_of(name);
    let path = format!("data/matrices/kkt/{}/{}.verdict.json", fam, name);
    let txt = match fs::read_to_string(Path::new(&path)) {
        Ok(t) => t,
        Err(_) => return ("(no-verdict)".to_string(), "(no-verdict)".to_string()),
    };
    let agree = extract_str_field(&txt, "inertia_agreement").unwrap_or("(none)".to_string());
    let verdict = extract_str_field(&txt, "verdict").unwrap_or("(none)".to_string());
    (agree, verdict)
}

fn extract_str_field(txt: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\":", key);
    let i = txt.find(&needle)? + needle.len();
    let rest = txt[i..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn diff(label: &str, baseline: &HashMap<String, Row>, policy: &HashMap<String, Row>) {
    let mut residual_regressions: Vec<(Row, Row)> = Vec::new();
    let mut inertia_regressions: Vec<(Row, Row)> = Vec::new();
    let mut residual_recoveries = 0usize;
    let mut inertia_recoveries = 0usize;
    for (name, b) in baseline.iter() {
        let p = match policy.get(name) {
            Some(p) => p,
            None => continue,
        };
        if b.residual_ok && !p.residual_ok {
            residual_regressions.push((b.clone(), p.clone()));
        } else if !b.residual_ok && p.residual_ok {
            residual_recoveries += 1;
        }
        if b.inertia_ok && !p.inertia_ok {
            inertia_regressions.push((b.clone(), p.clone()));
        } else if !b.inertia_ok && p.inertia_ok {
            inertia_recoveries += 1;
        }
    }

    println!();
    println!("============ baseline → {} ============", label);
    println!(
        "residual: {} regressions / {} recoveries (net {})",
        residual_regressions.len(),
        residual_recoveries,
        residual_recoveries as i64 - residual_regressions.len() as i64,
    );
    println!(
        "inertia:  {} regressions / {} recoveries (net {})",
        inertia_regressions.len(),
        inertia_recoveries,
        inertia_recoveries as i64 - inertia_regressions.len() as i64,
    );

    if residual_regressions.is_empty() && inertia_regressions.is_empty() {
        println!("  (no regressions to enumerate)");
        return;
    }

    // Combined regressed set, deduplicated by name. The (R, I)
    // tuple keys each entry by which regression bucket(s) hit it.
    type RegPair = (Row, Row);
    type Buckets = (Option<RegPair>, Option<RegPair>);
    use std::collections::BTreeMap;
    let mut combined: BTreeMap<String, Buckets> = BTreeMap::new();
    for (b, p) in &residual_regressions {
        combined.entry(b.name.clone()).or_default().0 = Some((b.clone(), p.clone()));
    }
    for (b, p) in &inertia_regressions {
        combined.entry(b.name.clone()).or_default().1 = Some((b.clone(), p.clone()));
    }

    println!();
    println!(
        "{:<22} {:>5} {:>3} {:>3} {:>11} {:>11} {:<28} {:<22}",
        "name",
        "n",
        "ΔR",
        "ΔI",
        "base_res",
        "pol_res",
        "verdict.inertia_agreement",
        "verdict.verdict"
    );
    println!("{}", "-".repeat(115));

    let mut by_agreement: HashMap<String, usize> = HashMap::new();
    let mut by_verdict: HashMap<String, usize> = HashMap::new();

    for (name, (rres, rinr)) in &combined {
        let r_flag = if rres.is_some() { "R" } else { " " };
        let i_flag = if rinr.is_some() { "I" } else { " " };
        // At least one of rres / rinr is Some by construction:
        // entries are inserted only when one bucket fires.
        let row_for_meta = match rres
            .as_ref()
            .map(|(b, _)| b)
            .or_else(|| rinr.as_ref().map(|(b, _)| b))
        {
            Some(r) => r,
            None => continue,
        };
        let pol_for_res = rres
            .as_ref()
            .map(|(_, p)| p.rel_res)
            .or_else(|| rinr.as_ref().map(|(_, p)| p.rel_res))
            .unwrap_or(f64::NAN);
        let (agree, verdict) = read_verdict(name);
        *by_agreement.entry(agree.clone()).or_insert(0) += 1;
        *by_verdict.entry(verdict.clone()).or_insert(0) += 1;
        println!(
            "{:<22} {:>5} {:>3} {:>3} {:>11.2e} {:>11.2e} {:<28} {:<22}",
            name, row_for_meta.n, r_flag, i_flag, row_for_meta.rel_res, pol_for_res, agree, verdict,
        );
    }

    println!();
    println!(
        "Regression cross-reference summary ({} matrices):",
        combined.len()
    );
    println!("  by inertia_agreement:");
    let mut by_agreement: Vec<_> = by_agreement.into_iter().collect();
    by_agreement.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (k, c) in by_agreement {
        println!("    {:<28} {:>5}", k, c);
    }
    println!("  by verdict.verdict:");
    let mut by_verdict: Vec<_> = by_verdict.into_iter().collect();
    by_verdict.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (k, c) in by_verdict {
        println!("    {:<28} {:>5}", k, c);
    }
}

fn main() {
    let baseline = parse_csv("dev/results/lever-c/dump-baseline.csv");
    let mc64 = parse_csv("dev/results/lever-c/dump-mc64.csv");
    let adaptive = parse_csv("dev/results/lever-c/dump-adaptive.csv");
    println!(
        "loaded: baseline={}  mc64={}  adaptive={}",
        baseline.len(),
        mc64.len(),
        adaptive.len()
    );

    let baseline_map: HashMap<String, Row> = baseline
        .iter()
        .map(|r| (r.name.clone(), r.clone()))
        .collect();
    let mc64_map: HashMap<String, Row> = mc64.iter().map(|r| (r.name.clone(), r.clone())).collect();
    let adaptive_map: HashMap<String, Row> = adaptive
        .iter()
        .map(|r| (r.name.clone(), r.clone()))
        .collect();

    // Sanity: counts match.
    println!(
        "policy-1 residual passes: {}  inertia passes: {}",
        baseline.iter().filter(|r| r.residual_ok).count(),
        baseline.iter().filter(|r| r.inertia_ok).count(),
    );
    println!(
        "policy-2 residual passes: {}  inertia passes: {}",
        mc64.iter().filter(|r| r.residual_ok).count(),
        mc64.iter().filter(|r| r.inertia_ok).count(),
    );
    println!(
        "policy-3 residual passes: {}  inertia passes: {}",
        adaptive.iter().filter(|r| r.residual_ok).count(),
        adaptive.iter().filter(|r| r.inertia_ok).count(),
    );

    diff("Policy 2 (mc64)", &baseline_map, &mc64_map);
    diff("Policy 3 (adaptive)", &baseline_map, &adaptive_map);
}
