//! Issue #227 golden test: `ReanalysisPolicy::Never` (the default) must
//! stay bit-identical to feral before the policy existed.
//!
//! The digests below were computed on `main` @ 95ba43b (before any #227
//! source change) by running this test with `FERAL_227_PRINT=1`. They
//! hash every factor's global `perm`, each front's `nelim`/`nrow`, local
//! `perm`, and the bits of `l`, `d_diag`, `d_subdiag`, plus the inertia.
//! A digest change is a change to factorization results — a correctness
//! event, not a refactor. Keep this file unchanged through the #227 work.

#[path = "common227/mod.rs"]
mod common227;

use common227::{factor_digest, synthetic_inertia, synthetic_kkt, Fnv, N_ITER};
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::{Path, PathBuf};

const SYNTHETIC_DIGESTS: [u64; N_ITER] = [
    0xc29fd80a7defa603,
    0xcc811d34dde9fc35,
    0x5dfbbe09c2c20057,
    0x67f1c1de8441fb62,
    0x8203996eccfba0b4,
    0xb98ffa5b29046d6a,
    0x6c1b1b6ec45e207f,
    0x94989d5e8d87d735,
];
const PARITY_DIGEST: u64 = 0xc9f250bc555067d9;
const PARITY_COUNT: usize = 122;

fn printing() -> bool {
    std::env::var("FERAL_227_PRINT").is_ok()
}

fn synthetic_digests(parallel: bool) -> Vec<u64> {
    let mut s = Solver::new().with_parallel(parallel);
    let mut out = Vec::new();
    for k in 0..N_ITER {
        let a = synthetic_kkt(k);
        let st = s.factor(&a, Some(synthetic_inertia()));
        assert!(
            matches!(st, FactorStatus::Success),
            "synthetic iterate {k}: {st:?} (Sylvester inertia {:?})",
            synthetic_inertia()
        );
        let (f, i) = match (s.factors(), s.inertia()) {
            (Some(f), Some(i)) => (f, i),
            _ => panic!("no factor stored at iterate {k}"),
        };
        out.push(factor_digest(f, i));
    }
    out
}

fn parity_files() -> Vec<PathBuf> {
    let root = Path::new("tests/data/parity");
    let mut dirs: Vec<PathBuf> = match std::fs::read_dir(root) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(e) => panic!("read_dir {root:?}: {e}"),
    };
    dirs.sort();
    let mut files = Vec::new();
    for d in dirs {
        let mut fs: Vec<PathBuf> = match std::fs::read_dir(&d) {
            Ok(rd) => rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
                .collect(),
            Err(_) => continue,
        };
        fs.sort();
        files.extend(fs);
    }
    files
}

fn load(p: &Path) -> CscMatrix {
    match read_mtx(p).and_then(|m| m.to_csc()) {
        Ok(a) => a,
        Err(e) => panic!("{p:?}: {e}"),
    }
}

/// Every parity matrix factored twice by one `Solver` (the second call
/// reuses the cached analysis), folded into one digest.
fn parity_digest(parallel: bool) -> (u64, usize) {
    let mut h = Fnv::new();
    let mut count = 0;
    for p in parity_files() {
        let a = load(&p);
        let mut s = Solver::new().with_parallel(parallel);
        for rep in 0..2 {
            let _ = s.factor(&a, None);
            match (s.factors(), s.inertia()) {
                (Some(f), Some(i)) => {
                    let d = factor_digest(f, i);
                    if printing() && rep == 0 {
                        eprintln!("{p:?} {d:#018x}");
                    }
                    h.word(d);
                }
                _ => h.word(u64::MAX),
            }
            count += 1;
        }
    }
    (h.0, count)
}

#[test]
fn synthetic_kkt_path_is_bit_identical_to_main() {
    let par = synthetic_digests(true);
    let seq = synthetic_digests(false);
    if printing() {
        eprintln!("SYNTHETIC_DIGESTS = {par:#018x?}");
    }
    assert_eq!(par, seq, "parallel and sequential digests differ");
    assert_eq!(par, SYNTHETIC_DIGESTS.to_vec());
}

#[test]
fn parity_corpus_is_bit_identical_to_main() {
    let (dp, np) = parity_digest(true);
    let (ds, ns) = parity_digest(false);
    if printing() {
        eprintln!("PARITY_DIGEST = {dp:#018x} PARITY_COUNT = {np}");
    }
    assert_eq!((dp, np), (ds, ns), "parallel and sequential digests differ");
    assert_eq!(np, PARITY_COUNT);
    assert_eq!(dp, PARITY_DIGEST);
}
