//! Issue #246: AMF must not invert its fill-score priority at large n.
//!
//! MUMPS HAMF4 quantizes a score `RMF >= i32::MAX` as `RMF / N`, which
//! files near-dense variables as cheap. On the fhnw-binschedule1 LP KKT
//! (n = 1.9M) that gave AMF nnz(L) = 1.39e10, 32x AMD's, and exhausted
//! memory in the symbolic phase. See
//! `dev/research/issue-246-amf-quantization.md`.
//!
//! Oracle: the AMD reference ordering on the same pattern. AMF is a
//! fill-minimizing heuristic, so it must not exceed AMD's fill here
//! (measured after the fix: AMF 1.835e8, AMD 4.369e8).
//!
//! The fixture is generated and gitignored. Regenerate it with
//! `dev/scripts/mps_to_kkt.py` (see its docstring). The test SKIPs when it is
//! absent (always on CI). It is `#[ignore]`d because AMD+AMF on n = 1.9M take
//! about 3.5 minutes in release and far longer in debug:
//!
//!     cargo test --release --test issue246_amf_quantization -- --ignored

use feral::ordering::amd::permute_pattern;
use feral::ordering::elimination_tree::EliminationTree;
use feral::ordering::postorder::postorder;
use feral::symbolic::{column_counts_gnp, total_factor_nnz};
use feral::CscPattern;
use std::path::Path;

fn nnz_l(full: &CscPattern, perm: &[usize]) -> usize {
    let p0 = permute_pattern(full, perm);
    let (post, _) = postorder(&EliminationTree::from_pattern(&p0));
    let perm2: Vec<usize> = post.iter().map(|&p| perm[p]).collect();
    let pp = permute_pattern(full, &perm2);
    let et = EliminationTree::from_pattern(&pp);
    total_factor_nnz(&column_counts_gnp(&pp, &et))
}

#[test]
#[ignore]
fn fhnw_amf_fill_not_worse_than_amd() {
    let path = Path::new("tests/data/large/fhnw-binschedule1.mtx");
    if !path.exists() {
        eprintln!("SKIP: {} not found", path.display());
        return;
    }
    let a = feral::read_mtx(path)
        .and_then(|m| m.to_csc())
        .expect("fixture must parse");
    let full = a.symmetric_pattern();
    let col: Vec<i32> = full.col_ptr.iter().map(|&x| x as i32).collect();
    let row: Vec<i32> = full.row_idx.iter().map(|&x| x as i32).collect();
    let pat = feral_ordering_core::CscPattern::new(full.n, &col, &row).expect("pattern");
    let to_usize = |p: Vec<i32>| -> Vec<usize> { p.into_iter().map(|x| x as usize).collect() };
    let amd = to_usize(feral_amd::amd_order(&pat).expect("amd"));
    let amf = to_usize(feral_amf::amf_order(&pat).expect("amf"));
    let (l_amd, l_amf) = (nnz_l(&full, &amd), nnz_l(&full, &amf));
    eprintln!("fhnw nnz(L): AMD {l_amd}, AMF {l_amf}");
    assert!(
        l_amf <= l_amd,
        "AMF nnz(L) {l_amf} exceeds AMD nnz(L) {l_amd} (issue #246 regression)"
    );
}
