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
    find_supernodes, pick_amalgamation_strategy, AmalgamationStrategy, OrderingPreprocess,
    Supernode, SupernodeParams, AUTO_MULTI_CHILD_FRAC_THRESHOLD,
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
    /// Approximate Minimum Fill (`feral-amf` crate: HAMF4 variant
    /// of Amestoy 1999 — quotient-graph elimination scored by
    /// approximate fill `RMF(i) = (deg(i)·(deg(i)-1+2·degme) -
    /// WF(i)) / (nv(i)+1)` rather than approximate degree).
    /// Same downstream pipeline as `Amd`.
    ///
    /// Default for `n <= 10_000` per `pick_default_method`,
    /// matching MUMPS's `ana_set_ordering.F` rule for SYM=2 small
    /// matrices. Validated against MUMPS HAMF4 on the 183_293-
    /// sidecar corpus by `tests/amf_corpus_oracle.rs`: feral nnz_L
    /// is within 1.10× MUMPS HAMF4 nnz_L on 183_277 matrices, with
    /// CHARDIS1_0000 the lone documented metric-divergence skip.
    Amf,
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

    /// Concrete ordering method actually dispatched. Records the
    /// `OrderingMethod::Auto → AMD/MetisND/ScotchND/KahipND`
    /// resolution made by `choose_adaptive`. For non-`Auto` callers
    /// this is identical to the requested method.
    pub resolved_method: OrderingMethod,
    /// Concrete amalgamation strategy actually used.
    /// `AmalgamationStrategy::Auto` is resolved by
    /// `pick_amalgamation_strategy` before supernode detection; this
    /// field records the resolved value.
    pub resolved_amalgamation: supernode::AmalgamationStrategy,
    /// Concrete ordering preprocessor actually used.
    /// `OrderingPreprocess::Auto` is resolved by
    /// `pick_ordering_preprocess`; this field records `None` or
    /// `LdltCompress` after that dispatch.
    pub resolved_preprocess: supernode::OrderingPreprocess,

    /// F3.2: When this factorization was produced by
    /// [`symbolic_factorize_with_schur`], records the size of the Schur
    /// tail. The last `n_schur` columns of `perm` correspond to the
    /// user-supplied `schur_indices` in the supplied order. `None` for
    /// factorizations produced by [`symbolic_factorize`] or
    /// [`symbolic_factorize_with_method`]. The numeric phase reads this
    /// to enforce the per-front NPIV ≤ NASS − NVSCHUR stopping rule
    /// (F3.2b).
    pub is_schur_tail: Option<usize>,
}

/// Pick a default ordering for `symbolic_factorize` from cheap matrix
/// dimensions (no pattern walk). Narrow on purpose — see comment on
/// `Auto` for why a broad dispatcher regressed the IPM bench.
///
/// Current rule (Phase D of `dev/plans/amf-clean-room.md`, mirrors
/// MUMPS's `ana_set_ordering.F` AMF-vs-METIS heuristic):
///   - `n == 0`                                        → `Amd`
///     (avoids /0 and external-crate weirdness on the empty pattern)
///   - `n >= 5000 && nnz/n < 6` → `MetisND` (bordered-KKT catch:
///     CUTEst CRESC132 where AMD/AMF order the constraint block into a
///     near-dense root frontal that swallows ~96% of n and drives a
///     ~5000-column delay cascade. CRESC132_0000 with AMD factors in
///     5.4 s; with METIS it factors in 480 ms — 11× win.)
///   - `n >= 2000 && nnz/n < 4` → `MetisND` (chain-pattern catch:
///     CHAINWOO/HYDROELL/DIXMAANH from the kkt-expansion corpus.
///     n≈3000–4033, stored avg-deg 2–3. AMD/AMF order these chain-
///     like KKT systems into a dense root that triggers a runaway
///     delay cascade — CHAINWOO_0000 produces 2.10M nnz_L with AMD
///     vs 282k with METIS. Also catches VESUVIO. Verified via
///     `diag_chainwoo` on 2026-04-27.)
///   - `n <= 10_000`                                   → `Amf`
///     (MUMPS-style "small symmetric" rule: HAMF4 fill metric is
///     within 1.10× of MUMPS HAMF4 on 183_277 of 183_293 sidecar'd
///     matrices in `tests/amf_corpus_oracle.rs`, and the in-tree
///     audit (`diag_amf_vs_amd`) shows AMF strictly better than AMD
///     on 83/782 matrices, tied on 589, AMD better on 110, geomean
///     ratio 1.003. ORBIT2_0000 alone goes from AMD's 1.4M nnz_L
///     down to AMF's 32_105.)
///   - everything else (`n > 10_000`)                  → `MetisND`
///     (large patterns where nested dissection is the standard win.)
///
/// `nnz` here is the matrix's *stored* nnz (lower triangle for
/// symmetric matrices), not the symmetric pattern's. The threshold is
/// calibrated to that convention; using the symmetric pattern would
/// roughly double the ratio and shift the rule.
///
/// The `n >= 2000` floor on the chain-pattern catch protects the
/// IPM corpus's tiny-matrix tail (e.g. HAHN1 n=715), where AMF
/// remains the best default at small scale.
fn pick_default_method(n: usize, stored_nnz: usize) -> OrderingMethod {
    if n == 0 {
        return OrderingMethod::Amd;
    }
    let avg_deg = stored_nnz as f64 / n as f64;
    if (n >= 5000 && avg_deg < 6.0) || (n >= 2000 && avg_deg < 4.0) {
        return OrderingMethod::MetisND;
    }
    if n <= 10_000 {
        OrderingMethod::Amf
    } else {
        OrderingMethod::MetisND
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
/// (new-to-old: `perm[k]` is the original column that became column `k`),
/// along with the concrete `OrderingMethod` actually dispatched (matters
/// when `method == Auto` is resolved adaptively).
fn run_external_ordering(
    pattern: &CscPattern,
    method: OrderingMethod,
) -> Result<(Vec<usize>, OrderingMethod), FeralError> {
    let (col_buf, row_buf) = to_contract_pattern_bufs(pattern)?;
    let pat = feral_ordering_core::CscPattern::new(pattern.n, &col_buf, &row_buf)
        .ok_or_else(|| FeralError::InvalidInput("malformed CSC pattern".to_string()))?;
    let resolved = choose_adaptive(pattern, method);
    let perm_i32 = match resolved {
        OrderingMethod::Amd => feral_amd::amd_order(&pat),
        OrderingMethod::Amf => feral_amf::amf_order(&pat),
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
    Ok((out, resolved))
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
    let (amd_perm, resolved_method): (Vec<usize>, OrderingMethod) = match resolved_preprocess {
        OrderingPreprocess::None => run_external_ordering(&full_pattern, method)?,
        OrderingPreprocess::Auto => unreachable!("resolved above"),
        OrderingPreprocess::LdltCompress => {
            // Run the full MC64 pipeline once and keep the cache so the
            // numeric phase can reuse it for `Mc64Symmetric` scaling
            // (Phase 2.4.4: eliminates ~70% of compression symbolic
            // overhead on matrices where scaling also runs MC64).
            let cache = crate::scaling::compute_mc64_cache(matrix)?;
            let map = build_supermap(&cache.perm);
            let pair = if map.ncmp() == n {
                // Matching gives no compression leverage; fall through
                // to the uncompressed path rather than build and walk
                // an identical-size graph.
                run_external_ordering(&full_pattern, method)?
            } else {
                let cpat = compress_pattern(&full_pattern, &map);
                let (super_perm, resolved) = run_external_ordering(&cpat, method)?;
                (expand_permutation(&super_perm, &map), resolved)
            };
            cached_mc64 = Some(cache);
            pair
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

    // Phase 2.13a: resolve `Auto` to a concrete strategy via a cheap
    // O(n) etree shape predicate. The downstream Renumber gate and
    // `find_supernodes` reverse-iteration check need a concrete
    // variant — `Auto` is a top-level dispatch sentinel only.
    let mut effective_params = snode_params.clone();
    if matches!(
        effective_params.amalgamation_strategy,
        supernode::AmalgamationStrategy::Auto
    ) {
        effective_params.amalgamation_strategy = supernode::pick_amalgamation_strategy(&etree);
    }
    let snode_params: &SupernodeParams = &effective_params;

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
        resolved_method,
        resolved_amalgamation: snode_params.amalgamation_strategy,
        resolved_preprocess,
        is_schur_tail: None,
    })
}

/// Symbolic factorization with a user-supplied Schur tail (F3.2a).
///
/// Like [`symbolic_factorize_with_method`] except the last `n_schur`
/// columns of the produced permutation are pinned to `schur_indices` in
/// the supplied order — i.e. `perm[n - n_schur + i] == schur_indices[i]`
/// for every `i`. This is the symbolic side of the Schur-complement API
/// described in `dev/research/schur-complement.md` (F3.0).
///
/// The pipeline diverges from [`symbolic_factorize_with_method`] in
/// three places:
///
/// 1. **Ordering.** The fill-reducing ordering is fixed to AMD on the
///    non-Schur subgraph, via [`crate::ordering::schur::compute_schur_aware_perm`]
///    (F3.1). Other methods are not yet wired in for the Schur path
///    because each external ordering crate would need a "constrained
///    ordering" or subgraph hook to honour the Schur tail invariant.
///    See `dev/research/schur-complement.md` D3.
///
/// 2. **Postorder.** Standard CHOLMOD postorder is replaced by
///    [`crate::ordering::postorder::schur_constrained_postorder`], which
///    pins Schur nodes to their etree-index positions. The Schur subset
///    forms a top-forest of the etree (parent always strictly greater
///    than child, and Schur indices occupy `[n - n_schur, n)`), so the
///    constraint is satisfiable.
///
/// 3. **Preprocessor / amalgamation strategy.** The
///    [`OrderingPreprocess::LdltCompress`] preprocessor and the
///    [`AmalgamationStrategy::Renumber`] reorderer both rewrite the
///    column numbering and would break the tail invariant. The Schur
///    path forces `preprocess == None` and `amalgamation_strategy ==
///    Adjacency` regardless of what the caller passed in
///    `snode_params`.
///
/// Empty `schur_indices` ⇒ returns the same result as
/// [`symbolic_factorize_with_method`] with `OrderingMethod::Amd`.
///
/// `schur_indices.len() == n` ⇒ `InvalidInput` (the elimination set
/// would be empty; almost certainly an upstream logic bug).
///
/// Returns `is_schur_tail = Some(n_schur)` so the numeric phase (F3.2b)
/// can enforce the per-front `NPIV ≤ NASS − NVSCHUR` stopping rule.
pub fn symbolic_factorize_with_schur(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
    schur_indices: &[usize],
) -> Result<SymbolicFactorization, FeralError> {
    let n = matrix.n;
    let n_schur = schur_indices.len();

    if n_schur == 0 {
        // Empty Schur ⇒ standard symbolic factorization with AMD.
        return symbolic_factorize_with_method(matrix, snode_params, OrderingMethod::Amd);
    }

    // Force the preprocessor and amalgamation strategy to values that
    // preserve the column numbering. LdltCompress rewrites columns via
    // the MC64 supermap; Renumber re-postorders. Both would break the
    // Schur tail invariant.
    let mut effective_params = snode_params.clone();
    effective_params.preprocess = OrderingPreprocess::None;
    effective_params.amalgamation_strategy = supernode::AmalgamationStrategy::Adjacency;

    // Step 1: Schur-aware ordering. AMD on the non-Schur subgraph,
    // followed by the Schur tail in user-supplied order. Validates
    // schur_indices (duplicates / out-of-range / full-n).
    let initial_perm = crate::ordering::schur::compute_schur_aware_perm(matrix, schur_indices)?;

    // Step 2: build full symmetric pattern + permute.
    let full_pattern = matrix.symmetric_pattern();
    let initial_permuted = permute_pattern(&full_pattern, &initial_perm);

    // Step 3: etree of permuted pattern. By construction Schur columns
    // sit at indices [n - n_schur, n); etree.parent[j] > j for every j,
    // so the Schur subset is closed under `parent` (top-forest).
    let initial_etree = EliminationTree::from_pattern(&initial_permuted);

    // Step 4: Schur-constrained postorder. Non-Schur descendants of
    // Schur nodes emit first (subtree-size order); Schur nodes emit at
    // their etree-index positions, preserving the user's input order.
    // Mark the highest n_schur indices in the etree as Schur. By
    // construction (compute_schur_aware_perm appends the Schur tail at
    // the end of initial_perm), these positions correspond to the user's
    // schur_indices in user-supplied order.
    let mut is_schur = vec![false; n];
    for slot in is_schur.iter_mut().skip(n - n_schur) {
        *slot = true;
    }
    let (post, post_inv) =
        crate::ordering::postorder::schur_constrained_postorder(&initial_etree, &is_schur);

    // Postorder identity check on the Schur tail (defensive — the
    // top-forest invariant should make this hold by construction).
    for (k, &p) in post.iter().enumerate().skip(n - n_schur) {
        debug_assert_eq!(
            p, k,
            "schur_constrained_postorder violated tail identity at k={}",
            k
        );
    }

    // Step 5: compose perm₀ with the postorder.
    let perm: Vec<usize> = post.iter().map(|&p| initial_perm[p]).collect();
    let mut perm_inv = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        perm_inv[old] = new;
    }

    // Tail-invariant assertion: this is the F3.2a contract.
    debug_assert_eq!(
        &perm[n - n_schur..],
        schur_indices,
        "Schur tail invariant violated"
    );

    // Step 6: re-permute and rebuild etree on the final pattern.
    let permuted_pattern = permute_pattern(&full_pattern, &perm);
    let final_parent: Vec<Option<usize>> = (0..n)
        .map(|new| {
            let old_initial = post[new];
            initial_etree.parent[old_initial].map(|old_par| post_inv[old_par])
        })
        .collect();
    let etree = EliminationTree {
        parent: final_parent,
        n,
    };

    // Step 7: column counts on the final pattern + etree.
    let col_counts = column_counts::column_counts_gnp(&permuted_pattern, &etree);
    let factor_nnz = column_counts::total_factor_nnz(&col_counts);

    // Step 8: supernode detection. Adjacency strategy only — Renumber
    // would re-postorder and break the tail invariant.
    let supernodes = supernode::find_supernodes(&etree, &col_counts, &effective_params);

    // Step 9: small-leaf grouping (consumed at numeric time only when
    // the small_leaf gate is On). Same as the standard pipeline.
    let (small_leaf_groups, snode_group) =
        find_small_leaf_groups(&supernodes, &permuted_pattern, &effective_params.small_leaf);

    // Step 10: contribution sizes + peak memory.
    let contrib_sizes: Vec<usize> = supernodes.iter().map(|s| s.contrib_size()).collect();
    let peak_contrib_bytes = compute_peak_contrib(&supernodes, &contrib_sizes);

    let factor_slack = 1.2;

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
        cached_mc64: None,
        resolved_method: OrderingMethod::Amd,
        resolved_amalgamation: effective_params.amalgamation_strategy,
        resolved_preprocess: OrderingPreprocess::None,
        is_schur_tail: Some(n_schur),
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
    fn symbolic_factorize_amf_produces_valid_perm() {
        // Phase D wire-up smoke test: OrderingMethod::Amf must
        // produce a valid permutation through the full symbolic
        // pipeline (postorder composition, etree, column counts,
        // supernodes). This pins the dispatch wiring; bit-parity vs
        // MUMPS HAMF4 is the job of tests/amf_corpus_oracle.rs.
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::Amf).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
        assert_eq!(sym.resolved_method, OrderingMethod::Amf);
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
    fn symbolic_factorize_default_uses_amf_for_small_matrices() {
        // Per Phase D of dev/plans/amf-clean-room.md: small matrices
        // (n <= 10_000) that don't trigger the bordered-KKT or
        // chain-pattern escape hatches default to AMF, mirroring
        // MUMPS's ana_set_ordering.F rule for SYM=2 N≤10000.
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let a = symbolic_factorize(&m, &params).unwrap();
        let b = symbolic_factorize_with_method(&m, &params, OrderingMethod::Amf).unwrap();
        assert_eq!(
            a.perm, b.perm,
            "symbolic_factorize on small dense matrices must equal \
             symbolic_factorize_with_method(Amf)"
        );
        assert_eq!(a.factor_nnz_estimate, b.factor_nnz_estimate);
        assert_eq!(a.resolved_method, OrderingMethod::Amf);
    }

    #[test]
    fn pick_default_method_rules() {
        // CRESC132-shaped: n=5314, stored_nnz=22566 → avg_deg=4.25.
        // Triggers the bordered-KKT fallback.
        assert_eq!(pick_default_method(5314, 22566), OrderingMethod::MetisND);
        // VESUVIO-shaped: n=3083, avg_deg=3.07. Triggers the
        // chain-pattern branch (n>=2000 && avg_deg<4). Verified
        // 2026-04-27: METIS gives 1.35× less fill than AMD here.
        assert_eq!(pick_default_method(3083, 9484), OrderingMethod::MetisND);
        // CHAINWOO-shaped: n=4000, avg_deg=2.0 → MetisND
        // (chain-pattern catch). 7.5× less fill than AMD.
        assert_eq!(pick_default_method(4000, 7999), OrderingMethod::MetisND);
        // DIXMAANH-shaped: n=3000, avg_deg=3.0 → MetisND
        // (chain-pattern catch). 11.6× less fill than AMD.
        assert_eq!(pick_default_method(3000, 8999), OrderingMethod::MetisND);
        // HAHN1-shaped: n=715 < 2000, n <= 10_000 → AMF. Below the
        // chain-pattern floor; AMF is the small-matrix default per
        // the Phase D MUMPS-style rule.
        assert_eq!(pick_default_method(715, 2839), OrderingMethod::Amf);
        // n=10_000 dense (avg_deg≥6): exactly at the AMF/MetisND
        // boundary, n <= 10_000 wins → AMF.
        assert_eq!(pick_default_method(10_000, 100_000), OrderingMethod::Amf);
        // Boundary at n=5000 with low avg_deg: bordered-KKT
        // catch fires first → MetisND.
        assert_eq!(pick_default_method(5000, 20_000), OrderingMethod::MetisND);
        // Boundary at n=2000 with avg_deg<4: chain-pattern fires
        // first → MetisND.
        assert_eq!(pick_default_method(2000, 6000), OrderingMethod::MetisND);
        // n>=2000 but avg_deg≥4 and <5000, n<=10_000: → AMF
        // (no escape hatch fires, falls through to AMF default).
        assert_eq!(pick_default_method(3000, 13_000), OrderingMethod::Amf);
        // Empty matrix: AMD (avoids /0 and external-crate weirdness).
        assert_eq!(pick_default_method(0, 0), OrderingMethod::Amd);
        // Large dense matrix (n > 10_000, avg_deg high): MetisND
        // (the n>10k tail switches off AMF in favor of nested
        // dissection per MUMPS).
        assert_eq!(
            pick_default_method(20_000, 200_000),
            OrderingMethod::MetisND
        );
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

    /// 6×6 KKT-shaped matrix: leading 4×4 identity-like block, dense
    /// trailing 2×2 Schur, with off-diagonal coupling A_FS connecting
    /// rows {0..4} to columns {4,5}. Same structure used in the F3.1
    /// schur.rs unit tests.
    fn small_kkt_6x6() -> CscMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        // Diagonal in non-Schur block (1..=4 along positions 0..4).
        for i in 0..4 {
            rows.push(i);
            cols.push(i);
            vals.push((i + 1) as f64);
        }
        // Coupling A_FS: column 4 connects to rows 0,2; column 5 connects to rows 1,3.
        rows.push(4);
        cols.push(0);
        vals.push(0.5);
        rows.push(4);
        cols.push(2);
        vals.push(0.7);
        rows.push(5);
        cols.push(1);
        vals.push(0.3);
        rows.push(5);
        cols.push(3);
        vals.push(0.9);
        // Trailing 2×2 Schur block, dense.
        rows.push(4);
        cols.push(4);
        vals.push(1.5);
        rows.push(5);
        cols.push(4);
        vals.push(0.2);
        rows.push(5);
        cols.push(5);
        vals.push(2.5);
        CscMatrix::from_triplets(6, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn schur_symbolic_tail_invariant_user_order() {
        // schur_indices = [4, 5] in user order.
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_schur(&m, &params, &[4, 5]).unwrap();
        assert_eq!(sym.n, 6);
        assert_eq!(sym.is_schur_tail, Some(2));
        assert_eq!(&sym.perm[4..], &[4, 5]);
    }

    #[test]
    fn schur_symbolic_tail_invariant_reversed_user_order() {
        // schur_indices = [5, 4] — user-supplied order MUST be preserved
        // exactly, not sorted.
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_schur(&m, &params, &[5, 4]).unwrap();
        assert_eq!(sym.is_schur_tail, Some(2));
        assert_eq!(&sym.perm[4..], &[5, 4]);
    }

    #[test]
    fn schur_symbolic_perm_is_valid_permutation() {
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_schur(&m, &params, &[4, 5]).unwrap();
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
        // perm_inv consistency.
        for (new, &old) in sym.perm.iter().enumerate() {
            assert_eq!(sym.perm_inv[old], new);
        }
    }

    #[test]
    fn schur_symbolic_empty_falls_back_to_standard() {
        // Empty schur_indices must produce a SymbolicFactorization with
        // is_schur_tail = None (delegates to symbolic_factorize_with_method).
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_schur(&m, &params, &[]).unwrap();
        assert_eq!(sym.is_schur_tail, None);
    }

    #[test]
    fn schur_symbolic_full_n_rejected() {
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let result = symbolic_factorize_with_schur(&m, &params, &[0, 1, 2, 3, 4, 5]);
        assert!(matches!(result, Err(FeralError::InvalidInput(_))));
    }

    #[test]
    fn schur_symbolic_duplicate_rejected() {
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let result = symbolic_factorize_with_schur(&m, &params, &[4, 4]);
        assert!(matches!(result, Err(FeralError::InvalidInput(_))));
    }

    #[test]
    fn schur_symbolic_supernodes_cover_n() {
        // Sanity check: the supernode layout still covers all n columns.
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_schur(&m, &params, &[4, 5]).unwrap();
        let total: usize = sym.supernodes.iter().map(|s| s.ncol()).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn schur_symbolic_single_schur_index() {
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_schur(&m, &params, &[5]).unwrap();
        assert_eq!(sym.is_schur_tail, Some(1));
        assert_eq!(sym.perm[5], 5);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }
}
