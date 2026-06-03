//! Run the parallel driver twice on ACOPP14_0006 with
//! `FERAL_HASH_FRONTAL=1` set so `factor_one_supernode` prints the
//! pre-factor frontal hash for every supernode. If the hashes are
//! identical across both runs but the factor outputs differ, the
//! dense kernel is non-deterministic. If the hashes differ, the
//! assembly path itself is diverging.

use feral::numeric::factorize::{
    factorize_multifrontal_supernodal_parallel, NumericParams, SparseFactors,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, Inertia, ZeroPivotAction};
use std::hash::{Hash, Hasher};
use std::path::Path;

fn factor_hash(f: &SparseFactors) -> (u64, Inertia) {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let mut t = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    for n in f.node_factors.iter() {
        t.positive += n.inertia.positive;
        t.negative += n.inertia.negative;
        t.zero += n.inertia.zero;
        for v in n.frontal_factors.l.iter() {
            v.to_bits().hash(&mut h);
        }
        for v in n.frontal_factors.d_diag.iter() {
            v.to_bits().hash(&mut h);
        }
        for v in n.frontal_factors.d_subdiag.iter() {
            v.to_bits().hash(&mut h);
        }
    }
    (h.finish(), t)
}

fn main() {
    let path = Path::new("data/matrices/kkt/ACOPR14/ACOPR14_0003.mtx");
    let mtx = read_mtx(path).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let sp = SupernodeParams::default();
    let sym = symbolic_factorize(&csc, &sp).expect("symbolic");
    let params = NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    });
    // Loop until we hit a divergent pair, printing frontal hashes
    // with FERAL_HASH_FRONTAL. The caller must set this env var.
    for attempt in 0..200 {
        eprintln!("=== attempt {} RUN A ===", attempt);
        let a = factorize_multifrontal_supernodal_parallel(&csc, &sym, &params).expect("par A");
        eprintln!("=== attempt {} RUN B ===", attempt);
        let b = factorize_multifrontal_supernodal_parallel(&csc, &sym, &params).expect("par B");
        let (ha, ia) = factor_hash(&a.0);
        let (hb, ib) = factor_hash(&b.0);
        if ha != hb || ia.positive != ib.positive || ia.negative != ib.negative {
            eprintln!(
                "*** DIVERGED attempt {}: a={:?} b={:?} (hash neq: {})",
                attempt,
                ia,
                ib,
                ha != hb
            );
            return;
        }
    }
    eprintln!("no divergence in 200 attempts");
}
