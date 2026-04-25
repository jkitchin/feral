pub mod column_counts;
pub mod ldlt_compress;
pub mod profiler;
pub mod small_leaf;
pub mod supernode;

use crate::error::FeralError;
use crate::ordering::amd::permute_pattern;
use crate::ordering::elimination_tree::EliminationTree;
use crate::ordering::postorder::{biased_postorder, postorder};
use crate::sparse::csc::{CscMatrix, CscPattern};

pub use column_counts::{column_counts, column_counts_gnp, total_factor_nnz};
pub use ldlt_compress::{build_supermap, compress_pattern, expand_permutation, SuperMap};
pub use profiler::{record_stage, StagePct, StageTiming, SymbolicProfileReport, SymbolicProfiler};
pub use small_leaf::{find_small_leaf_groups, SmallLeafGroup, SmallLeafParams};
pub use supernode::{
    find_supernodes, AmalgamationStrategy, OrderingPreprocess, Supernode, SupernodeParams,
};

/// Which fill-reducing ordering to use in [`symbolic_factorize_with_method`].
///
/// All methods produce a permutation; the downstream postorder
/// composition, etree construction, column counts, supernode detection,
/// and memory planning are identical regardless of method.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum OrderingMethod {
    /// Approximate Minimum Degree (`feral-amd` crate: approximate
    /// external degree with aggressive element absorption and
    /// supervariable detection, per Amestoy/Davis/Duff 1996+2004).
    /// Default. Matches SuiteSparse/faer on the oracle fixture suite.
    ///
    /// The simplified exact-external-degree implementation at
    /// `src/ordering/amd.rs` remains on disk as a reference for the
    /// algorithm's skeleton but is no longer reachable from the
    /// symbolic pipeline. See
    /// `dev/journal/2026-04-18-03.org` for the retirement evidence
    /// (34-matrix bakeoff: geomean fill tied on parity, crate
    /// 17-23% better and 18-88× faster on large).
    #[default]
    Amd,
    /// feral-metis multilevel nested dissection.
    MetisND,
    /// feral-scotch nested dissection.
    ScotchND,
    /// feral-kahip flow-based nested dissection.
    ///
    /// Includes K1 (Ost-Schulz-Strash 2021 Rule 1, conservative
    /// termination) preprocessing inside the KaHIP pipeline. Wired
    /// at `crates/feral-kahip/src/node_nd.rs`.
    ///
    /// **Not selected by `pick_default_method`.** The session 08
    /// 41-matrix bake-off (`dev/research/ordering-kahip-driver-
    /// integration.md`) showed `KahipND` ties `MetisND` on fill
    /// geomean (1.023 vs 1.024 relative to AMD) at 4-6× the per-call
    /// symbolic-time cost (81s vs 68s vs AMD 14s, total). On the
    /// 154 588-matrix IPM bench KaHIP would only match METIS where
    /// the existing narrow `n>=5000 && nnz/n<6 → MetisND` rule
    /// already fires (e.g. CRESC132). Reachable explicitly via
    /// `symbolic_factorize_with_method` for callers who want it.
    KahipND,
    /// Adaptive dispatcher: picks a concrete method per-matrix from
    /// cheap pattern features (n and average degree nnz/n).
    ///
    /// The heuristic is derived from the 41-matrix shape bakeoff:
    ///   - large-and-sparse (n > 100_000, nnz/n < 5) → `ScotchND`
    ///     (SCOTCH dominates c-big-class arrow / KKT matrices).
    ///   - small-and-sparse (n < 10_000, nnz/n < 15)   → `KahipND`
    ///     (K1 reductions find short cycles AMD misses).
    ///   - everything else                             → `Amd`.
    ///
    /// **Opt-in only.** The 154k-matrix IPM bench (2026-04-18) showed
    /// `Auto` regresses sparse factor/MUMPS geomean from 0.44 (AMD)
    /// to 0.58 because the small-and-sparse branch routes thousands
    /// of n<500 IPM iteration dumps to KaHIP, where K1 + multilevel
    /// setup costs 2-3× per call vs AMD. The 0.988 fill geomean from
    /// the shape bakeoff is real but does not translate to numeric
    /// speedup when the corpus is dominated by tiny matrices.
    ///
    /// Use `Auto` only when the workload is known to be dominated by
    /// large or `cresc132`-class matrices where the per-call setup
    /// cost amortizes. The default `symbolic_factorize` keeps `Amd`.
    /// See `dev/tried-and-rejected.md` for the full evidence.
    ///
    /// Applying `Auto` to `Auto` loops once through the dispatcher and
    /// then runs the chosen concrete method.
    Auto,
}

/// Resolve an `Auto` ordering to a concrete method from cheap pattern
/// features. Non-`Auto` inputs pass through unchanged.
///
/// The rule set is intentionally small and has a break-even fallback
/// to `Amd`, so a pattern that matches no branch still gets a valid
/// ordering. See `OrderingMethod::Auto` for the rationale.
fn choose_adaptive(pattern: &CscPattern, method: OrderingMethod) -> OrderingMethod {
    if method != OrderingMethod::Auto {
        return method;
    }
    let n = pattern.n;
    let nnz = pattern.row_idx.len();
    if n == 0 {
        return OrderingMethod::Amd;
    }
    let avg_deg = nnz as f64 / n as f64;
    if n > 100_000 && avg_deg < 5.0 {
        OrderingMethod::ScotchND
    } else if n < 10_000 && avg_deg < 15.0 {
        OrderingMethod::KahipND
    } else {
        OrderingMethod::Amd
    }
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

    /// Phase 2.9 small-leaf-subtree groups (`dev/plans/phase-2.9-
    /// small-leaf-subtree.md`). Populated unconditionally at
    /// symbolic time; used at numeric time only when
    /// `NumericParams::small_leaf == SmallLeafBatch::On`.
    pub small_leaf_groups: Vec<SmallLeafGroup>,

    /// For each supernode index, `Some(g)` if the supernode is a
    /// member of `small_leaf_groups[g]`, else `None`. Length
    /// `supernodes.len()`.
    pub snode_group: Vec<Option<usize>>,

    /// Cached MC64 matching produced by the `LdltCompress`
    /// preprocessor. When `Some`, the numeric phase reuses it to
    /// derive the `Mc64Symmetric` scaling vector in O(n) instead of
    /// rerunning the Hungarian kernel. `None` when no MC64 matching
    /// was computed during symbolic factorization.
    pub(crate) cached_mc64: Option<crate::scaling::Mc64Cache>,
}

/// Pick a default ordering for `symbolic_factorize` from cheap matrix
/// dimensions (no pattern walk). Narrow on purpose — see comment on
/// `Auto` for why a broad dispatcher regressed the IPM bench.
///
/// Current rule:
///   - `n >= 5000 && nnz/n < 6` → `MetisND` (catches bordered KKT
///     structures like CUTEst CRESC132 where AMD orders the constraint
///     block into a near-dense root frontal that swallows ~96% of n
///     and drives a ~5000-column delay cascade. CRESC132_0000 with AMD
///     factors in 5.4 s; with METIS it factors in 480 ms — 11× win.)
///   - everything else                                 → `Amd`
///
/// `nnz` here is the matrix's *stored* nnz (lower triangle for
/// symmetric matrices), not the symmetric pattern's. The threshold is
/// calibrated to that convention; using the symmetric pattern would
/// roughly double the ratio and shift the rule.
///
/// All entries in the IPM corpus's top families have `n < 5000` (the
/// largest are HAHN1 n=715 and VESUVIO n=3083), so the bordered rule
/// only fires on a handful of large matrices and pays its small extra
/// symbolic cost on those alone.
fn pick_default_method(n: usize, stored_nnz: usize) -> OrderingMethod {
    if n == 0 {
        return OrderingMethod::Amd;
    }
    let avg_deg = stored_nnz as f64 / n as f64;
    if n >= 5000 && avg_deg < 6.0 {
        OrderingMethod::MetisND
    } else {
        OrderingMethod::Amd
    }
}

/// Resolve [`OrderingPreprocess::Auto`] to a concrete preprocessor
/// choice based on cheap O(nnz) shape predicates.
///
/// Returns [`OrderingPreprocess::LdltCompress`] when two conditions hold:
///
/// 1. `n >= MIN_N_FOR_COMPRESSION` (size floor). Below this, numeric
///    factor time is in the sub-ms range and the ~100-400μs compression
///    symbolic overhead dominates. Calibrated from the 154 588-matrix
///    bench: geomean regressed 0.36 → 0.48 with unconditional
///    compression, driven by small-matrix symbolic overhead.
///
/// 2. `low_degree_cols / n >= LOW_DEGREE_THRESHOLD` (arrow-KKT
///    signature). Columns with stored degree ≤ 2 (the diagonal plus at
///    most one off-diagonal) are the structural fingerprint of IPM KKT
///    slack blocks (`IpStdAugSystemSolver.cpp:250-305`: `Σ_s + δ_s I`
///    coupled to the d-row by a single identity off-diagonal). Many
///    such columns means the MC64 matching has abundant 2-cycle
///    structure for compression to exploit. This broadens the
///    `diag_only / n` predicate from `pick_scaling_strategy` because
///    Ipopt slack columns are degree-2, not degree-1.
///
/// Otherwise returns [`OrderingPreprocess::None`].
///
/// Parallels [`crate::scaling::pick_scaling_strategy`] in spirit.
/// Both predicates are O(nnz) and allocation-free.
///
/// No published compression-benefit predictor exists in the MUMPS /
/// SPRAL literature (see consult of 2026-04-23). These thresholds are
/// calibrated against the feral corpus and documented in
/// `dev/journal/2026-04-23-02.org`.
pub fn pick_ordering_preprocess(matrix: &CscMatrix) -> OrderingPreprocess {
    const MIN_N_FOR_COMPRESSION: usize = 128;
    const LOW_DEGREE_THRESHOLD: f64 = 0.30;

    let n = matrix.n;
    if n < MIN_N_FOR_COMPRESSION {
        return OrderingPreprocess::None;
    }

    let mut low_degree = 0usize;
    for j in 0..n {
        let nnz_col = matrix.col_ptr[j + 1] - matrix.col_ptr[j];
        if nnz_col <= 2 {
            low_degree += 1;
        }
    }

    if low_degree as f64 / n as f64 >= LOW_DEGREE_THRESHOLD {
        OrderingPreprocess::LdltCompress
    } else {
        OrderingPreprocess::None
    }
}

/// Perform symbolic factorization of a sparse symmetric matrix.
///
/// Defaults to AMD, but applies a narrow bordered-KKT fallback rule to
/// catch the AMD-bad structures (see [`pick_default_method`]). Callers
/// who want a literal AMD ordering with no dispatcher should call
/// `symbolic_factorize_with_method(matrix, params, OrderingMethod::Amd)`
/// explicitly.
///
/// Steps:
/// 1. Pick fill-reducing ordering (AMD or MetisND depending on pattern)
/// 2. Build elimination tree of the permuted matrix
/// 3. Compute column counts (fill prediction)
/// 4. Detect and amalgamate supernodes
/// 5. Compute MemoryPlan (factor NNZ, contribution sizes, peak memory)
pub fn symbolic_factorize(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
) -> Result<SymbolicFactorization, FeralError> {
    let method = pick_default_method(matrix.n, matrix.row_idx.len());
    symbolic_factorize_with_method(matrix, snode_params, method)
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
    let resolved = choose_adaptive(pattern, method);
    let perm_i32 = match resolved {
        OrderingMethod::Amd => feral_amd::amd_order(&pat),
        OrderingMethod::MetisND => feral_metis::metis_order(&pat),
        OrderingMethod::ScotchND => feral_scotch::scotch_order(&pat),
        OrderingMethod::KahipND => feral_kahip::kahip_order(&pat),
        OrderingMethod::Auto => unreachable!("choose_adaptive resolves Auto"),
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

    // Phase 2.13b per-stage profiler. Every timer is `Some` only when
    // `snode_params.symbolic_profiler.is_some()`; the `None` path
    // does no `Instant::now()` calls. See
    // `dev/research/phase-2.13b-symbolic-profiler.md`.
    let prof = snode_params.symbolic_profiler.as_ref();
    let t_total = prof.map(|_| std::time::Instant::now());

    // β refactor: scaling is no longer computed here. It moved to
    // `factorize_multifrontal` so that `SymbolicFactorization`
    // depends only on the matrix pattern (not its values) and can
    // be reused across multiple numeric factorizations of
    // structurally identical KKTs. See
    // `dev/plans/scaling-in-numeric.md`.

    // Step 1: Fill-reducing ordering. Dispatch on `method`. The
    // downstream pipeline (postorder composition, etree, column counts,
    // supernode amalgamation, memory plan) is identical regardless of
    // which ordering produced `initial_perm`.
    //
    // If `snode_params.preprocess == LdltCompress`, run MC64 symmetric
    // matching, build the super-variable map, order the compressed
    // graph, and expand the resulting super-permutation back to
    // length `n` before handing it to the rest of the pipeline. See
    // `src/symbolic/ldlt_compress.rs` and
    // `dev/plans/phase-2.6.5-ldlt-compressed-graph.md`.
    let t_sym = prof.map(|_| std::time::Instant::now());
    let full_pattern = matrix.symmetric_pattern();
    if let Some(t) = t_sym {
        record_stage(prof, "symmetric_pattern", t);
    }

    let mut cached_mc64: Option<crate::scaling::Mc64Cache> = None;
    // Resolve `Auto` to `None` or `LdltCompress` before entering the
    // dispatch. Keeps the match below exhaustive on the two concrete
    // variants and keeps the dispatcher logic in one testable place.
    let t_pick = prof.map(|_| std::time::Instant::now());
    let resolved_preprocess = match snode_params.preprocess {
        OrderingPreprocess::Auto => pick_ordering_preprocess(matrix),
        other => other,
    };
    if let Some(t) = t_pick {
        record_stage(prof, "pick_preprocess", t);
    }
    let t_ord = prof.map(|_| std::time::Instant::now());
    let amd_perm: Vec<usize> = match resolved_preprocess {
        OrderingPreprocess::None => run_external_ordering(&full_pattern, method)?,
        OrderingPreprocess::Auto => unreachable!("resolved above"),
        OrderingPreprocess::LdltCompress => {
            // Run the full MC64 pipeline once and keep the cache so the
            // numeric phase can reuse it for `Mc64Symmetric` scaling
            // (Phase 2.4.4: eliminates ~70% of compression symbolic
            // overhead on matrices where scaling also runs MC64).
            let cache = crate::scaling::compute_mc64_cache(matrix)?;
            let map = build_supermap(&cache.perm);
            let perm = if map.ncmp() == n {
                // Matching gives no compression leverage; fall through
                // to the uncompressed path rather than build and walk
                // an identical-size graph.
                run_external_ordering(&full_pattern, method)?
            } else {
                let cpat = compress_pattern(&full_pattern, &map);
                let super_perm = run_external_ordering(&cpat, method)?;
                expand_permutation(&super_perm, &map)
            };
            cached_mc64 = Some(cache);
            perm
        }
    };
    if let Some(t) = t_ord {
        record_stage(prof, "ordering", t);
    }

    // Step 2: Build the etree on the permuted pattern. This etree is
    // intermediate — we use it to compute the postorder and then discard it.
    // The local name `amd_*` is kept from the AMD-only era to minimise the
    // diff; semantically these are now "ordering output" and "permuted
    // pattern from that ordering", regardless of method.
    let t_perm1 = prof.map(|_| std::time::Instant::now());
    let amd_pattern = permute_pattern(&full_pattern, &amd_perm);
    if let Some(t) = t_perm1 {
        record_stage(prof, "permute1", t);
    }
    let t_etree0 = prof.map(|_| std::time::Instant::now());
    let amd_etree = EliminationTree::from_pattern(&amd_pattern);
    if let Some(t) = t_etree0 {
        record_stage(prof, "etree_initial", t);
    }

    // Step 3: Postorder the etree (CHOLMOD-style composition).
    // Without this step, supernode amalgamation merges columns whose indices
    // are not consecutive in the column numbering, and downstream code that
    // assumes `first_col..first_col+ncol` is the eliminated set silently
    // factors the wrong columns. See dev/research/postorder-pipeline.md.
    let t_post = prof.map(|_| std::time::Instant::now());
    let (post, post_inv) = postorder(&amd_etree);
    if let Some(t) = t_post {
        record_stage(prof, "postorder", t);
    }

    // Step 4: Compose AMD perm with the postorder.
    //   final_perm[k] = amd_perm[post[k]]
    // The composition maps postorder position k to the original column.
    let t_compose = prof.map(|_| std::time::Instant::now());
    let perm: Vec<usize> = post.iter().map(|&p| amd_perm[p]).collect();
    let mut perm_inv = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        perm_inv[old] = new;
    }
    if let Some(t) = t_compose {
        record_stage(prof, "perm_compose", t);
    }

    // Step 5: Re-permute the matrix on the composed permutation.
    let t_perm2 = prof.map(|_| std::time::Instant::now());
    let permuted_pattern = permute_pattern(&full_pattern, &perm);
    if let Some(t) = t_perm2 {
        record_stage(prof, "permute2", t);
    }

    // Step 5b: Build the final elimination tree by renumbering `amd_etree`
    // through the postorder. Postorder is a topological relabeling of the
    // elimination tree nodes, so `etree(P·A·Pᵀ) = post-renumbering of
    // etree(A)` when P is a postorder of etree(A) — the tree structure is
    // preserved and only the node labels change. This lets us produce the
    // final etree in O(n) instead of re-running `from_pattern` at
    // O(nnz · α(n)). A 3-run bench shows ~3% small-frontal p90 improvement
    // over the old two-from_pattern approach.
    let t_relabel = prof.map(|_| std::time::Instant::now());
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
    if let Some(t) = t_relabel {
        record_stage(prof, "etree_relabel", t);
    }

    // Step 6: Column counts on the final pattern + etree.
    // Phase 2.5.1 switched this from the O(n²) elimination simulation
    // (still available as `column_counts`) to Gilbert-Ng-Peyton at
    // O(nnz(A) + n·α(n)). Bit-exact equivalence verified on 169585
    // KKT matrices — see `dev/validation/phase-2.5.1-*`.
    let t_cc = prof.map(|_| std::time::Instant::now());
    let mut col_counts = column_counts_gnp(&permuted_pattern, &etree);
    if let Some(t) = t_cc {
        record_stage(prof, "col_counts", t);
    }

    // Phase 2.12: optional SSIDS-style merge-biased postorder.
    // Predict desired merges using only the etree + column counts,
    // then re-postorder the etree so desired-merge children are
    // emitted adjacent to their parents. The downstream
    // `find_supernodes` adjacency check then succeeds for those
    // merges naturally.
    //
    // Rebuild path: compose perm with the bias-driven post2,
    // re-permute the matrix, rebuild etree and col_counts. The
    // structural properties are invariant under within-subtree
    // relabeling (CHOLMOD/SSIDS observation, see
    // `dev/research/phase-2.12-column-renumbering.md` §5.1).
    //
    // Fast-path: when no bias is requested (no desired merges, OR
    // the strategy is `Adjacency`), the second pass is skipped and
    // the pipeline behaves identically to pre-Phase-2.12.
    let mut permuted_pattern = permuted_pattern;
    let mut perm = perm;
    let mut etree = etree;
    let t_renumber = prof.map(|_| std::time::Instant::now());
    if matches!(
        snode_params.amalgamation_strategy,
        supernode::AmalgamationStrategy::Renumber
    ) {
        let bias = supernode::predict_merges(&etree, &col_counts, snode_params);
        if bias.iter().any(|&b| b) {
            let (post2, _post2_inv) = biased_postorder(&etree, &bias);
            // Compose: perm₂[k] = perm[post2[k]]; the existing
            // `perm` already encodes AMD ∘ post1.
            let new_perm: Vec<usize> = post2.iter().map(|&p| perm[p]).collect();
            let mut new_perm_inv = vec![0usize; n];
            for (new, &old) in new_perm.iter().enumerate() {
                new_perm_inv[old] = new;
            }
            let new_permuted_pattern = permute_pattern(&full_pattern, &new_perm);
            // Rebuild the etree on the renumbered pattern. We could
            // relabel the existing etree through post2 in O(n) (as
            // Step 5b does for the postorder), but since the
            // permutation invariant is critical and post2 is a
            // postorder of `etree`, the relabeled tree is equivalent
            // by construction. Re-derive from scratch as a defense
            // against the etree-invariance claim being subtly wrong;
            // O(nnz · α(n)) is small for the matrices we target.
            let new_etree = EliminationTree::from_pattern(&new_permuted_pattern);
            let new_col_counts = column_counts_gnp(&new_permuted_pattern, &new_etree);

            perm = new_perm;
            perm_inv = new_perm_inv;
            permuted_pattern = new_permuted_pattern;
            etree = new_etree;
            col_counts = new_col_counts;
        }
    }
    if let Some(t) = t_renumber {
        record_stage(prof, "renumber", t);
    }
    let factor_nnz = total_factor_nnz(&col_counts);

    // Step 7: Supernode detection on the postordered etree
    let t_find = prof.map(|_| std::time::Instant::now());
    let supernodes = find_supernodes(&etree, &col_counts, snode_params);
    if let Some(t) = t_find {
        record_stage(prof, "find_supernodes", t);
    }

    // Step 7b: Phase 2.9 small-leaf grouping. Runs unconditionally;
    // the groups are consumed at numeric time only when the
    // `small_leaf` gate is `On`. O(n_snodes), no allocations beyond
    // the groups themselves.
    let t_slg = prof.map(|_| std::time::Instant::now());
    let (small_leaf_groups, snode_group) =
        find_small_leaf_groups(&supernodes, &permuted_pattern, &snode_params.small_leaf);
    if let Some(t) = t_slg {
        record_stage(prof, "small_leaf_groups", t);
    }

    // Step 5: Compute contribution sizes and peak memory
    let t_pk = prof.map(|_| std::time::Instant::now());
    let contrib_sizes: Vec<usize> = supernodes.iter().map(|s| s.contrib_size()).collect();

    let peak_contrib_bytes = compute_peak_contrib(&supernodes, &contrib_sizes);
    if let Some(t) = t_pk {
        record_stage(prof, "peak_contrib", t);
    }

    let factor_slack = 1.2;

    if let (Some(arc), Some(t)) = (prof, t_total) {
        if let Ok(mut p) = arc.lock() {
            p.set_total(t.elapsed().as_micros() as u64);
        }
    }

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
        small_leaf_groups,
        snode_group,
        cached_mc64,
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
    fn symbolic_factorize_kahip_produces_valid_perm() {
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::KahipND).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
    }

    #[test]
    fn symbolic_factorize_auto_produces_valid_perm() {
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::Auto).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
    }

    #[test]
    fn choose_adaptive_rules() {
        // Pattern helper: diagonal pattern with n cols, nnz = density*n.
        fn pat_bufs(n: usize, avg_deg: usize) -> (Vec<usize>, Vec<usize>) {
            let total = n * avg_deg.max(1);
            let mut col_ptr = Vec::with_capacity(n + 1);
            let mut row_idx = Vec::with_capacity(total);
            let per = avg_deg.max(1);
            for j in 0..n {
                col_ptr.push(row_idx.len());
                for t in 0..per {
                    row_idx.push((j + t) % n.max(1));
                }
            }
            col_ptr.push(row_idx.len());
            (col_ptr, row_idx)
        }
        // Large-and-sparse → SCOTCH.
        let (cp, ri) = pat_bufs(200_000, 3);
        let p = CscPattern {
            n: 200_000,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::Auto),
            OrderingMethod::ScotchND
        );
        // Small-and-sparse → KaHIP.
        let (cp, ri) = pat_bufs(500, 6);
        let p = CscPattern {
            n: 500,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::Auto),
            OrderingMethod::KahipND
        );
        // Everything else → AMD.
        let (cp, ri) = pat_bufs(50_000, 20);
        let p = CscPattern {
            n: 50_000,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::Auto),
            OrderingMethod::Amd
        );
        // Non-Auto passes through.
        let (cp, ri) = pat_bufs(500, 6);
        let p = CscPattern {
            n: 500,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::MetisND),
            OrderingMethod::MetisND
        );
    }

    #[test]
    fn symbolic_factorize_default_uses_amd_for_small_matrices() {
        // Below the bordered-fallback threshold (n < 5000), the default
        // entry point must dispatch to AMD.
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let a = symbolic_factorize(&m, &params).unwrap();
        let b = symbolic_factorize_with_method(&m, &params, OrderingMethod::Amd).unwrap();
        assert_eq!(
            a.perm, b.perm,
            "symbolic_factorize on n<5000 must equal symbolic_factorize_with_method(Amd)"
        );
        assert_eq!(a.factor_nnz_estimate, b.factor_nnz_estimate);
    }

    #[test]
    fn pick_default_method_rules() {
        // CRESC132-shaped: n=5314, stored_nnz=22566 → avg_deg=4.25.
        // Triggers the bordered-KKT fallback.
        assert_eq!(pick_default_method(5314, 22566), OrderingMethod::MetisND);
        // VESUVIO-shaped: n=3083 < 5000 → AMD even though avg_deg<6.
        assert_eq!(pick_default_method(3083, 9484), OrderingMethod::Amd);
        // Large but dense (avg_deg≥6): keep AMD.
        assert_eq!(pick_default_method(10_000, 100_000), OrderingMethod::Amd);
        // Boundary at n=5000: triggers (>=).
        assert_eq!(pick_default_method(5000, 20_000), OrderingMethod::MetisND);
        // Empty matrix: AMD (avoids /0 and external-crate weirdness).
        assert_eq!(pick_default_method(0, 0), OrderingMethod::Amd);
    }

    #[test]
    fn pick_default_method_never_returns_kahip() {
        // Pins the session-08 driver-integration decision: KaHIP is
        // reachable only via explicit `with_method` or `Auto`. The
        // dispatcher must never return it on its own. See
        // `dev/research/ordering-kahip-driver-integration.md` for
        // the bake-off evidence (KaHIP ties METIS on fill at 4-6×
        // the per-call cost on 41 matrices). If a future change wants
        // to route some pattern to KaHIP by default, the maintainer
        // must consciously update this test and the research note.
        let shapes: &[(usize, usize)] = &[
            (0, 0),
            (10, 30),
            (500, 1500),
            (3083, 13333), // VESUVIOU
            (5314, 22566), // CRESC132
            (10_000, 50_000),
            (100_000, 500_000),
            (345_241, 1_343_126), // c-big from the shape bake-off
        ];
        for &(n, nnz) in shapes {
            let m = pick_default_method(n, nnz);
            assert_ne!(
                m,
                OrderingMethod::KahipND,
                "pick_default_method({}, {}) returned KahipND; \
                 see dev/research/ordering-kahip-driver-integration.md",
                n,
                nnz
            );
        }
    }
}
