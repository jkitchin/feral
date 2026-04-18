pub mod column_counts;
pub mod supernode;

use crate::error::FeralError;
use crate::ordering::amd::{amd_order, permute_pattern};
use crate::ordering::elimination_tree::EliminationTree;
use crate::ordering::postorder::postorder;
use crate::sparse::csc::{CscMatrix, CscPattern};

pub use column_counts::{column_counts, total_factor_nnz};
pub use supernode::{find_supernodes, Supernode, SupernodeParams};

/// Which fill-reducing ordering to use in [`symbolic_factorize_with_method`].
///
/// Dispatches at the single call site in `symbolic_factorize_with_method`
/// that today hardwires `amd_order`. All methods produce a permutation;
/// the downstream postorder composition, etree construction, column
/// counts, supernode detection, and memory planning are identical
/// regardless of method.
///
/// `Amd` uses the in-tree implementation at `src/ordering/amd.rs`.
/// The workspace `feral-amd` crate is not yet routed here; retiring
/// the in-tree AMD is a separate decision (see
/// `dev/plans/ordering-scotch.md` §"Public API and Integration" and
/// the session 2026-04-18 checkpoint).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum OrderingMethod {
    /// In-tree AMD (`src/ordering/amd.rs`). Default.
    #[default]
    Amd,
    /// feral-metis multilevel nested dissection.
    MetisND,
    /// feral-scotch nested dissection.
    ScotchND,
}

/// The complete output of symbolic factorization.
///
/// Produced before any numeric work begins. Contains everything needed
/// to allocate memory and drive the numeric factorization.
#[derive(Debug)]
pub struct SymbolicFactorization {
    /// Matrix dimension.
    pub n: usize,

    /// Fill-reducing permutation (new-to-old mapping).
    /// Column `perm[k]` of the original matrix becomes column k.
    pub perm: Vec<usize>,

    /// Inverse permutation (old-to-new mapping).
    pub perm_inv: Vec<usize>,

    /// Supernodes in postorder (children before parents).
    pub supernodes: Vec<Supernode>,

    /// Estimated total NNZ in the L factor across all supernodes.
    pub factor_nnz_estimate: usize,

    /// Slack factor applied to factor_nnz_estimate. Default 1.2.
    pub factor_slack: f64,

    /// For each supernode: the size (in f64s) of its contribution block.
    pub contrib_sizes: Vec<usize>,

    /// Peak contribution pool depth (sum of all live contribution blocks
    /// at the deepest point of the tree traversal).
    pub peak_contrib_bytes: usize,

    /// Elimination tree of the permuted matrix.
    pub etree: EliminationTree,

    /// Full symmetric pattern of the permuted matrix.
    pub permuted_pattern: CscPattern,

    /// Column counts of L.
    pub col_counts: Vec<usize>,

    /// Global symmetric scaling vector in **user-order** indexing.
    /// Length `n`. Applied symmetrically as `D · A · D` where
    /// `D = diag(scaling)`. This is the ground-truth vector — the
    /// pivot-order cache `scaling_pivot_order` is derived from it.
    ///
    /// Solve time uses this user-order vector to pre-scale the RHS
    /// and post-scale the solution at the permutation boundary.
    pub scaling: Vec<f64>,

    /// Pivot-order cache of `scaling`: for each pivot index `k`,
    /// `scaling_pivot_order[k] == scaling[perm[k]]`. Length `n`.
    ///
    /// Used during frontal assembly, where the assembly loop walks
    /// rows and columns in pivot-order indexing and needs an O(1)
    /// per-entry lookup of the scaling factor. Without this cache
    /// the loop would have to indirect through `perm` on every
    /// scattered entry.
    pub scaling_pivot_order: Vec<f64>,

    /// Diagnostic info about how `scaling` was produced.
    pub scaling_info: crate::scaling::ScalingInfo,
}

/// Perform symbolic factorization of a sparse symmetric matrix.
///
/// Steps:
/// 1. Compute fill-reducing ordering (AMD)
/// 2. Build elimination tree of the permuted matrix
/// 3. Compute column counts (fill prediction)
/// 4. Detect and amalgamate supernodes
/// 5. Compute MemoryPlan (factor NNZ, contribution sizes, peak memory)
pub fn symbolic_factorize(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
) -> Result<SymbolicFactorization, FeralError> {
    symbolic_factorize_with_method(matrix, snode_params, OrderingMethod::Amd)
}

/// Convert an owned-`usize` `CscPattern` into the contract's borrowed-`i32`
/// shape used by `feral-metis` / `feral-scotch`. Returns buffers the
/// caller must keep alive for the lifetime of the produced `CscPattern<'_>`.
fn to_contract_pattern_bufs(pattern: &CscPattern) -> Result<(Vec<i32>, Vec<i32>), FeralError> {
    let col_ptr: Result<Vec<i32>, _> = pattern.col_ptr.iter().map(|&x| i32::try_from(x)).collect();
    let col_ptr = col_ptr.map_err(|_| {
        FeralError::InvalidInput("matrix too large for i32-indexed ordering crates".to_string())
    })?;
    let row_idx: Result<Vec<i32>, _> = pattern.row_idx.iter().map(|&x| i32::try_from(x)).collect();
    let row_idx = row_idx.map_err(|_| {
        FeralError::InvalidInput("matrix too large for i32-indexed ordering crates".to_string())
    })?;
    Ok((col_ptr, row_idx))
}

/// Run an external (contract-conforming) ordering crate on `pattern` and
/// return the permutation as `Vec<usize>` in the in-tree convention
/// (new-to-old: `perm[k]` is the original column that became column `k`).
fn run_external_ordering(
    pattern: &CscPattern,
    method: OrderingMethod,
) -> Result<Vec<usize>, FeralError> {
    let (col_buf, row_buf) = to_contract_pattern_bufs(pattern)?;
    let pat = feral_ordering_core::CscPattern::new(pattern.n, &col_buf, &row_buf)
        .ok_or_else(|| FeralError::InvalidInput("malformed CSC pattern".to_string()))?;
    let perm_i32 = match method {
        OrderingMethod::MetisND => feral_metis::metis_order(&pat),
        OrderingMethod::ScotchND => feral_scotch::scotch_order(&pat),
        OrderingMethod::Amd => {
            // Unreachable: the caller handles Amd via the in-tree
            // amd_order path. Included so this function can stay
            // total without a panic.
            return Err(FeralError::InvalidInput(
                "run_external_ordering called with Amd variant".to_string(),
            ));
        }
    };
    let perm_i32 = perm_i32
        .map_err(|e| FeralError::InvalidInput(format!("external ordering failed: {}", e)))?;
    if perm_i32.len() != pattern.n {
        return Err(FeralError::InvalidInput(format!(
            "external ordering returned {} entries for n={}",
            perm_i32.len(),
            pattern.n
        )));
    }
    let mut out: Vec<usize> = Vec::with_capacity(perm_i32.len());
    for x in perm_i32 {
        let u = usize::try_from(x).map_err(|_| {
            FeralError::InvalidInput("external ordering returned negative index".to_string())
        })?;
        if u >= pattern.n {
            return Err(FeralError::InvalidInput(
                "external ordering returned out-of-range index".to_string(),
            ));
        }
        out.push(u);
    }
    Ok(out)
}

/// Like [`symbolic_factorize`] but lets the caller pick the
/// fill-reducing ordering via [`OrderingMethod`].
///
/// `symbolic_factorize(m, p) == symbolic_factorize_with_method(m, p,
/// OrderingMethod::Amd)`.
pub fn symbolic_factorize_with_method(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
    method: OrderingMethod,
) -> Result<SymbolicFactorization, FeralError> {
    let n = matrix.n;

    // Phase 2.2.1 Step 5: compute global symmetric scaling before ordering.
    // Scaling is a congruence transform and is independent of any
    // downstream symbolic work, so it can run as early as we like. We
    // run it first and cache the result for the whole pipeline.
    //
    // Returns a vector in user-order indexing; we permute it into
    // pivot-order at the end of the function so the numeric phase can
    // consume it by direct pivot index.
    let (scaling_user, scaling_info) =
        crate::scaling::compute_scaling(matrix, &snode_params.scaling_strategy)?;
    if let crate::scaling::ScalingInfo::PartialSingular { n_unmatched } = &scaling_info {
        // No project-wide logging framework yet; mirror the Phase 1
        // convention of eprintln! for unusual diagnostics so this is
        // visible in bench output without being a hard failure.
        // Structurally singular matrices are allowed to proceed — they
        // will typically surface the issue as a zero pivot during
        // numeric factorization, which is the right layer to reject.
        eprintln!(
            "warning: MC64 matching left {} of {} variables unmatched; \
             scaling is identity on those rows/columns",
            n_unmatched, n
        );
    }

    // Step 1: Fill-reducing ordering. Dispatch on `method`. The
    // downstream pipeline (postorder composition, etree, column counts,
    // supernode amalgamation, memory plan) is identical regardless of
    // which ordering produced `initial_perm`.
    let full_pattern = matrix.symmetric_pattern();
    let amd_perm: Vec<usize> = match method {
        OrderingMethod::Amd => amd_order(&full_pattern),
        OrderingMethod::MetisND | OrderingMethod::ScotchND => {
            run_external_ordering(&full_pattern, method)?
        }
    };

    // Step 2: Build the etree on the permuted pattern. This etree is
    // intermediate — we use it to compute the postorder and then discard it.
    // The local name `amd_*` is kept from the AMD-only era to minimise the
    // diff; semantically these are now "ordering output" and "permuted
    // pattern from that ordering", regardless of method.
    let amd_pattern = permute_pattern(&full_pattern, &amd_perm);
    let amd_etree = EliminationTree::from_pattern(&amd_pattern);

    // Step 3: Postorder the etree (CHOLMOD-style composition).
    // Without this step, supernode amalgamation merges columns whose indices
    // are not consecutive in the column numbering, and downstream code that
    // assumes `first_col..first_col+ncol` is the eliminated set silently
    // factors the wrong columns. See dev/research/postorder-pipeline.md.
    let (post, post_inv) = postorder(&amd_etree);

    // Step 4: Compose AMD perm with the postorder.
    //   final_perm[k] = amd_perm[post[k]]
    // The composition maps postorder position k to the original column.
    let perm: Vec<usize> = post.iter().map(|&p| amd_perm[p]).collect();
    let mut perm_inv = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        perm_inv[old] = new;
    }

    // Step 5: Re-permute the matrix on the composed permutation.
    let permuted_pattern = permute_pattern(&full_pattern, &perm);

    // Step 5b: Build the final elimination tree by renumbering `amd_etree`
    // through the postorder. Postorder is a topological relabeling of the
    // elimination tree nodes, so `etree(P·A·Pᵀ) = post-renumbering of
    // etree(A)` when P is a postorder of etree(A) — the tree structure is
    // preserved and only the node labels change. This lets us produce the
    // final etree in O(n) instead of re-running `from_pattern` at
    // O(nnz · α(n)). A 3-run bench shows ~3% small-frontal p90 improvement
    // over the old two-from_pattern approach.
    let final_parent: Vec<Option<usize>> = (0..n)
        .map(|new| {
            let old_amd = post[new];
            amd_etree.parent[old_amd].map(|old_par| post_inv[old_par])
        })
        .collect();
    let etree = EliminationTree {
        parent: final_parent,
        n,
    };

    // Step 6: Column counts on the final pattern + etree
    let col_counts = column_counts(&permuted_pattern, &etree);
    let factor_nnz = total_factor_nnz(&col_counts);

    // Step 7: Supernode detection on the postordered etree
    let supernodes = find_supernodes(&etree, &col_counts, snode_params);

    // Step 5: Compute contribution sizes and peak memory
    let contrib_sizes: Vec<usize> = supernodes.iter().map(|s| s.contrib_size()).collect();

    let peak_contrib_bytes = compute_peak_contrib(&supernodes, &contrib_sizes);

    let factor_slack = 1.2;

    // Phase 2.2.1 Step 5: build the pivot-order cache of the scaling
    // vector. `perm` is new-to-old: perm[k] is the user column that
    // became pivot column k. So
    //     scaling_pivot_order[k] = scaling_user[perm[k]]
    // matches the assembly-time lookup pattern in factorize.rs where
    // `permute_csc_values` produces a matrix indexed by pivot positions
    // and we want `scaling[pivot_row] * scaling[pivot_col]` on each
    // scattered entry. (See dev/plans/mc64-scaling.md §"Step 5".)
    let scaling_pivot_order: Vec<f64> = perm.iter().map(|&old| scaling_user[old]).collect();

    Ok(SymbolicFactorization {
        n,
        perm,
        perm_inv,
        supernodes,
        factor_nnz_estimate: (factor_nnz as f64 * factor_slack) as usize,
        factor_slack,
        contrib_sizes,
        peak_contrib_bytes,
        etree,
        permuted_pattern,
        col_counts,
        scaling: scaling_user,
        scaling_pivot_order,
        scaling_info,
    })
}

/// Compute the peak contribution pool size needed during postorder traversal.
///
/// At any point during traversal, the live contribution blocks are those
/// of nodes that have been factored but whose contribution has not yet
/// been assembled into their parent. In serial postorder, a node's
/// contribution is consumed when its parent is factored.
fn compute_peak_contrib(supernodes: &[Supernode], contrib_sizes: &[usize]) -> usize {
    let n_snodes = supernodes.len();
    if n_snodes == 0 {
        return 0;
    }

    // Simulate the postorder traversal:
    // - When we process supernode k: allocate contrib[k], free contrib[child] for each child
    // - Track peak allocation
    let mut live = vec![false; n_snodes];
    let mut current_size = 0usize;
    let mut peak = 0usize;

    for k in 0..n_snodes {
        // Allocate this node's contribution block
        current_size += contrib_sizes[k];
        live[k] = true;

        if current_size > peak {
            peak = current_size;
        }

        // Free children's contribution blocks (they've been assembled)
        for &child in &supernodes[k].children {
            if live[child] {
                current_size -= contrib_sizes[child];
                live[child] = false;
            }
        }
    }

    peak * std::mem::size_of::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_factorize_basic() {
        // Simple tridiagonal
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();

        let params = SupernodeParams {
            nemin: 32,
            ..Default::default()
        };
        let sym = symbolic_factorize(&m, &params).unwrap();

        assert_eq!(sym.n, 4);
        assert_eq!(sym.perm.len(), 4);
        assert_eq!(sym.perm_inv.len(), 4);

        // Permutation should be valid
        let mut sorted_perm = sym.perm.clone();
        sorted_perm.sort();
        assert_eq!(sorted_perm, vec![0, 1, 2, 3]);

        // Factor NNZ estimate should be >= actual NNZ
        assert!(sym.factor_nnz_estimate > 0);

        // Total supernode columns = n
        let total_cols: usize = sym.supernodes.iter().map(|s| s.ncol()).sum();
        assert_eq!(total_cols, 4);
    }

    #[test]
    fn test_symbolic_factorize_dense() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1, 2, 2], &[0, 0, 0, 1, 1, 2], &[1.0; 6])
            .unwrap();

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let sym = symbolic_factorize(&m, &params).unwrap();

        // For a dense matrix, factor NNZ = n*(n+1)/2 = 6
        assert!(sym.factor_nnz_estimate >= 6);
    }

    #[test]
    fn test_symbolic_factorize_kkt() {
        // Small KKT matrix
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 2, 2, 2],
            &[0, 1, 0, 1, 2],
            &[2.0, 3.0, 1.0, 1.0, -1e-8],
        )
        .unwrap();

        let params = SupernodeParams::default();
        let sym = symbolic_factorize(&m, &params).unwrap();

        assert_eq!(sym.n, 3);
        let total_cols: usize = sym.supernodes.iter().map(|s| s.ncol()).sum();
        assert_eq!(total_cols, 3);
    }

    #[test]
    fn test_perm_inverse_consistency() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();

        let params = SupernodeParams::default();
        let sym = symbolic_factorize(&m, &params).unwrap();

        // perm and perm_inv are inverses
        for i in 0..5 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
            assert_eq!(sym.perm_inv[sym.perm[i]], i);
        }
    }

    #[test]
    fn test_contrib_sizes_nonnegative() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let sym = symbolic_factorize(&m, &params).unwrap();

        for &cs in &sym.contrib_sizes {
            // Contribution sizes should be non-negative (they're usize, always >= 0)
            // and for the root node it should be 0
            assert!(cs < 100000, "unreasonable contrib size: {}", cs);
        }

        // Root supernode should have 0 contribution block
        if let Some(last) = sym.supernodes.last() {
            assert_eq!(
                last.contrib_size(),
                0,
                "root should have no contribution block"
            );
        }
    }

    fn small_grid_5x5() -> CscMatrix {
        // 5x5 grid graph stored as CscMatrix (full symmetric, lower
        // triangle only). Used as a structurally non-trivial test
        // case where AMD, METIS, and SCOTCH all produce permutations
        // and the downstream pipeline must accept any of them.
        let m = 5;
        let n = 5;
        let idx = |r: usize, c: usize| r * n + c;
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for r in 0..m {
            for c in 0..n {
                let k = idx(r, c);
                rows.push(k);
                cols.push(k);
                vals.push(4.0);
                if r + 1 < m {
                    rows.push(idx(r + 1, c));
                    cols.push(k);
                    vals.push(-1.0);
                }
                if c + 1 < n {
                    rows.push(idx(r, c + 1));
                    cols.push(k);
                    vals.push(-1.0);
                }
            }
        }
        CscMatrix::from_triplets(m * n, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn symbolic_factorize_metis_produces_valid_perm() {
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::MetisND).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
    }

    #[test]
    fn symbolic_factorize_scotch_produces_valid_perm() {
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::ScotchND).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
    }

    #[test]
    fn symbolic_factorize_default_matches_amd() {
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let a = symbolic_factorize(&m, &params).unwrap();
        let b = symbolic_factorize_with_method(&m, &params, OrderingMethod::Amd).unwrap();
        assert_eq!(
            a.perm, b.perm,
            "symbolic_factorize must equal symbolic_factorize_with_method(Amd)"
        );
        assert_eq!(a.factor_nnz_estimate, b.factor_nnz_estimate);
    }
}
