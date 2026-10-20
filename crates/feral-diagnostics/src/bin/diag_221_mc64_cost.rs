//! diag_221_mc64_cost — issue #221 follow-up: MC64 Hungarian matching
//! cost on pounce KKT dumps (brainpc1 restoration cycling).
//!
//! Prints wall time and the Hungarian work counters for each matrix.
//! Usage: diag_221_mc64_cost <kkt.jsonl> [...]

use feral::scaling::diagnose_mc64_matching;
use feral::CscMatrix;
use std::time::Instant;

fn load(path: &str) -> Result<CscMatrix, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let line = text.lines().next().ok_or("empty file")?;
    let o: serde_json::Value = serde_json::from_str(line)?;
    let n = o["n"].as_u64().ok_or("no n")? as usize;
    let arr = |k: &str| o[k].as_array().cloned().ok_or(format!("no {k}"));
    let (irn, jcn, vals) = (arr("irn")?, arr("jcn")?, arr("vals")?);
    let (mut r, mut c, mut v) = (vec![], vec![], vec![]);
    for k in 0..irn.len() {
        let i = irn[k].as_u64().ok_or("irn")? as usize - 1;
        let j = jcn[k].as_u64().ok_or("jcn")? as usize - 1;
        r.push(i.max(j));
        c.push(i.min(j));
        v.push(vals[k].as_f64().ok_or("vals")?);
    }
    Ok(CscMatrix::from_triplets(n, &r, &c, &v)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for path in std::env::args().skip(1) {
        let a = load(&path)?;
        let t = Instant::now();
        let s = diagnose_mc64_matching(&a)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "{path} {ms:.1}ms searches={} touched={} heap_init={} phase3={} edge_scans={}",
            s.augment_searches,
            s.touched_total,
            s.heap_init_slots,
            s.phase3_inner_iters,
            s.main_loop_edge_scans
        );
    }
    Ok(())
}
