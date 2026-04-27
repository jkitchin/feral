//! Measure ORBIT2_0000 nnz_L with the feral-metis quasi-dense
//! quotient (Fix A) enabled vs disabled.
//!
//! See `dev/research/orbit2-cluster-regression.md` §6 for the
//! technique. This binary is a one-off diagnostic; it is not part
//! of CI and is safe to remove once the win is recorded.
//!
//! Note: the public `OrderingMethod::MetisND` path inside the
//! symbolic pipeline calls `feral_metis::metis_order` (no opts), so
//! it always uses the default `MetisOptions`. To bench the
//! quotient-disabled baseline under the symbolic pipeline you must
//! either flip the default in `crates/feral-metis/src/lib.rs::Default`
//! and rebuild, or extend `OrderingMethod` to accept a custom opts
//! struct (out of scope for this binary).

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams};
use feral_metis::{metis_order_full, MetisOptions};
use feral_ordering_core::CscPattern;
use std::path::Path;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/matrices/kkt-expansion/ORBIT2/ORBIT2_0000.mtx".into());
    let mtx = read_mtx(Path::new(&path)).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    println!("matrix: {}, n={}, nnz={}", path, csc.n, csc.row_idx.len());

    // Build a CscPattern (full-symmetric pattern) for direct
    // metis_order_full calls.
    let pat_owned = csc.symmetric_pattern();
    let col_ptr_i32: Vec<i32> = pat_owned.col_ptr.iter().map(|&x| x as i32).collect();
    let row_idx_i32: Vec<i32> = pat_owned.row_idx.iter().map(|&x| x as i32).collect();
    let cpat = CscPattern::new(pat_owned.n, &col_ptr_i32, &row_idx_i32).expect("valid");

    // Per-column off-diagonal degree summary.
    let n = pat_owned.n;
    let thresh = ((10.0 * (n as f64).sqrt()).ceil() as usize).max(40);
    let mut max_deg = 0usize;
    let mut n_dense = 0usize;
    for c in 0..n {
        let lo = pat_owned.col_ptr[c];
        let hi = pat_owned.col_ptr[c + 1];
        let mut d = 0usize;
        for k in lo..hi {
            if pat_owned.row_idx[k] != c {
                d += 1;
            }
        }
        if d > max_deg {
            max_deg = d;
        }
        if d > thresh {
            n_dense += 1;
        }
    }
    println!(
        "default threshold = max(40, ceil(10*sqrt({}))) = {}",
        n, thresh
    );
    println!("max off-diagonal degree = {}", max_deg);
    println!("number of columns above threshold = {}", n_dense);

    // Tail of permutation under each setting.
    for &enabled in &[false, true] {
        let opts = MetisOptions {
            dense_quotient_enabled: enabled,
            ..Default::default()
        };
        let (perm, _, _) = metis_order_full(&cpat, &opts).expect("metis ok");
        let last3: Vec<i32> = perm.iter().rev().take(3).copied().collect();
        println!(
            "dense_quotient_enabled={:?} → perm.last3 (rev) = {:?}",
            enabled, last3
        );
    }

    // Symbolic+numeric path (uses default MetisOptions, so Fix A
    // takes whatever value `MetisOptions::default()` has at compile
    // time).
    let snode_params = SupernodeParams::default();
    let bk = BunchKaufmanParams::default();
    let nparams = NumericParams::with_bk(bk);

    let sym = symbolic_factorize_with_method(&csc, &snode_params, OrderingMethod::MetisND)
        .expect("sym ok");
    let (f, _) = factorize_multifrontal(&csc, &sym, &nparams).expect("num ok");
    println!(
        "[OrderingMethod::MetisND, default MetisOptions] nnz_L = {}",
        f.factor_nnz()
    );

    // Compare against feral-amd, which already implements Davis 1996
    // §5 dense-row deferral via `AmdOptions::dense_alpha = 10.0`.
    let sym_amd = symbolic_factorize_with_method(&csc, &snode_params, OrderingMethod::Amd)
        .expect("sym_amd ok");
    let (f_amd, _) = factorize_multifrontal(&csc, &sym_amd, &nparams).expect("num_amd ok");
    println!(
        "[OrderingMethod::Amd, default AmdOptions] nnz_L = {}",
        f_amd.factor_nnz()
    );

    // Compare against feral-amf (HAMF4 quotient-graph fill metric).
    // ORBIT2 is the kkt-expansion shape that motivated the AMF
    // clean-room: AMD orders ORBIT2_0000 into a ~1.4M-nnz_L factor;
    // AMF cuts it to ~32k.
    let sym_amf = symbolic_factorize_with_method(&csc, &snode_params, OrderingMethod::Amf)
        .expect("sym_amf ok");
    let (f_amf, _) = factorize_multifrontal(&csc, &sym_amf, &nparams).expect("num_amf ok");
    println!(
        "[OrderingMethod::Amf, default AmfOptions] nnz_L = {}",
        f_amf.factor_nnz()
    );

    // Direct feral-amd probe: did dense deferral actually fire, where
    // does column 2697 (the only super-dense column) land?
    use feral_amd::{amd_order_opts, AmdOptions};
    for &alpha in &[10.0_f64, 5.0, 2.0, 1.0, 0.5] {
        let amd_opts = AmdOptions {
            aggressive: true,
            dense_alpha: alpha,
        };
        let (perm_amd, stats) = amd_order_opts(&cpat, &amd_opts).expect("amd ok");
        let pos_2697 = perm_amd
            .iter()
            .position(|&p| p == 2697)
            .unwrap_or(usize::MAX);
        let last3: Vec<i32> = perm_amd.iter().rev().take(3).copied().collect();
        println!(
            "[feral-amd alpha={:.1}] n_dense_deferred={}, col2697 at perm[{}], last3={:?}",
            alpha, stats.n_dense_deferred, pos_2697, last3
        );
    }
}
