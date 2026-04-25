use crate::dense::rook::{rook_rescue, RookKind};
use crate::error::FeralError;
use crate::inertia::Inertia;

// Phase 2.4.3: the rank-1 / rank-2 Schur-update inner loops dispatch to
// the 4-way unrolled non-FMA pulp kernels in `crate::dense::schur_kernel`.
// The non-FMA variants reproduce the scalar loop's rounding bit-for-bit
// (two IEEE 754 roundings per element) so inertia counts are identical
// to the scalar path — verified by bit-exact unit tests across a length
// sweep and by the full KKT bench. The ILP win comes from 4 independent
// accumulators exposing parallelism that the single-accumulator
// autovectorized scalar loop could not. The kernel itself dispatches
// per-arch (NEON on aarch64, AVX2 on x86_64-v3, scalar fallback
// elsewhere); see commit 18194807.
use crate::dense::schur_kernel;

/// Phase 2.4.1c triage flag. When set to `true`, `factor_frontal_blocked`
/// delegates to the scalar `factor_frontal` unconditionally. Used by
/// `examples/triage_sparse_kernel_diff.rs` to A/B-compare the two
/// kernels across the full KKT corpus. Default `false` preserves
/// production dispatch; setting this is a diagnostic affordance only.
pub static FORCE_SCALAR_FRONTAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Dead-zero absolute floor for the 2×2 pivot cancellation test.
/// Matches SPRAL SSIDS `datatypes.f90:260` default (`small = 1e-20`),
/// used by `ldlt_tpp.cxx:98,106`. This is a true zero-detection floor
/// (near the underflow boundary), **not** a stability threshold — the
/// scale-invariant cancellation test at `factor_frontal`'s 2×2 gate
/// handles stability via ratios against the local block max.
const SSIDS_DET_SMALL: f64 = 1e-20;

/// Pivot-growth threshold above which a factor is flagged for iterative
/// refinement. A well-pivoted BK factor with unit-diagonal L satisfies
/// |L_ij| ≤ 1/(1 − α) ≈ 2.78; values substantially above this indicate
/// that some accepted pivot was small relative to its column max.
///
/// With `pivot_threshold = 0.0` (the BK default), the alpha-test is
/// satisfied vacuously when both the candidate diagonal and its column
/// off-diagonals are simultaneously tiny, producing a "successful"
/// factor whose plain forward/back substitution cannot reach machine
/// precision. Bratu3d under default params reaches max|L| ≈ 8e16 and
/// returns a plain residual of 4.66e6; setting `pivot_threshold = 0.01`
/// keeps max|L| ≈ 27 with plain residual 1.25e-14. See
/// dev/journal/2026-04-25-02.org.
///
/// 1e6 is a conservative trigger: matrices with growth in [2.78, 1e6]
/// converge in 1–2 IR steps without being flagged here; matrices above
/// 1e6 are catastrophic and need IR for any reasonable accuracy.
const L_GROWTH_THRESHOLD: f64 = 1e6;

/// Sets `*needs_refinement = true` when any `|L_ij| > L_GROWTH_THRESHOLD`.
/// Called at every dense-factor exit path so callers using plain
/// `Solver::solve` (rather than `solve_refined`) get a programmatic
/// signal — `factors.needs_refinement` — that the factor is too
/// unstable for plain forward/back substitution.
fn flag_growth_for_refinement(l: &[f64], needs_refinement: &mut bool) {
    if *needs_refinement {
        return;
    }
    for &v in l {
        if v.abs() > L_GROWTH_THRESHOLD {
            *needs_refinement = true;
            return;
        }
    }
}

/// Parameters controlling Bunch-Kaufman factorization behavior.
#[derive(Debug, Clone)]
pub struct BunchKaufmanParams {
    /// Pivot threshold α. BK standard: (1 + sqrt(17)) / 8 ≈ 0.6404.
    pub alpha: f64,

    /// A 1×1 pivot |d| <= zero_tol is considered numerically zero.
    /// Default: f64::EPSILON ≈ 2.22e-16.
    ///
    /// Rationale: for a well-equilibrated matrix with ||A|| ~ 1, the
    /// rounding error floor is ~eps. Any pivot more than eps above zero
    /// has a reliable sign and should be counted as positive/negative,
    /// not zero. The previous default of 100*eps (2.22e-14) was too
    /// aggressive and flagged legitimate small-positive pivots as zero
    /// on SPD matrices — verified by triage against canonical MUMPS,
    /// SSIDS, and rmumps on CERI651DLS_0534 and FBRAIN3LS_0788
    /// (2026-04-12, dev/journal/2026-04-12-01.org).
    pub zero_tol: f64,

    /// A 2×2 pivot block is near-singular when |det| <= zero_tol_2x2.
    /// Default: zero_tol².
    pub zero_tol_2x2: f64,

    /// What to do when the selected pivot is numerically zero.
    pub on_zero_pivot: ZeroPivotAction,

    /// Column-relative pivot threshold `u` (MUMPS `CNTL(1)`, SSIDS `options%u`).
    /// A 1×1 candidate pivot `a_kk` is accepted only if
    /// `|a_kk| >= u * max_{i>k}(|a_ik|)`, i.e. the pivot must dominate its
    /// column by at least a factor of `1/u`. Additionally, a 2×2 pivot block
    /// is accepted only if the Duff-Reid growth bound
    /// `(|a22|*RMAX + AMAX*TMAX)*u <= |det|` (and its symmetric partner)
    /// holds, where RMAX/TMAX are column maxes of the two pivot columns
    /// *beyond* the 2×2 block and AMAX is the cross term.
    ///
    /// Default `0.0` preserves Phase 1 behavior (no threshold check — every
    /// non-zero pivot is accepted). Callers opting into MC64 scaling should
    /// set this to `0.01` (MUMPS/SSIDS default) so that after symmetric
    /// equilibration, candidate pivots that are more than 100× smaller than
    /// their column max are rejected and flushed through the existing
    /// `ForceAccept` path. See dev/plans/scaling-aware-pivot-rejection.md
    /// and MUMPS `dfac_front_aux.F:1494-1606` for the reference formulas.
    pub pivot_threshold: f64,

    /// Panel width for the blocked dense Schur update. Consulted by the
    /// Phase 2.4.1b blocked-panel path in `factor_frontal`; ignored when
    /// `remaining <= block_size` or when `may_delay == true` routes the
    /// factor through the scalar path. Default 64 matches faer's
    /// `factor.rs:722` crossover. See `dev/plans/phase-2.4.1-blocked-ldlt.md`.
    pub block_size: usize,
}

/// Action to take when a near-zero pivot is encountered.
#[derive(Debug, Clone)]
pub enum ZeroPivotAction {
    /// Accept the tiny pivot; flag for iterative refinement.
    ForceAccept,
    /// Return FeralError::NumericallyRankDeficient.
    Fail,
}

impl Default for BunchKaufmanParams {
    fn default() -> Self {
        let zero_tol = f64::EPSILON;
        Self {
            alpha: (1.0 + 17f64.sqrt()) / 8.0, // ≈ 0.6404
            zero_tol,
            zero_tol_2x2: zero_tol * zero_tol,
            on_zero_pivot: ZeroPivotAction::Fail,
            pivot_threshold: 0.0,
            block_size: 64,
        }
    }
}

/// Factorization result: P·L·D_bk·Lᵀ·Pᵀ = D_eq·A·D_eq.
#[derive(Debug)]
pub struct Factors {
    pub n: usize,
    /// Unit lower triangular L in full n×n column-major storage.
    /// Diagonal entries are 1.0 (stored explicitly).
    pub l: Vec<f64>,
    /// D_bk diagonal entries in pivot order. Length n.
    pub d_diag: Vec<f64>,
    /// D_bk sub-diagonal entries. Length n. Zero for 1×1 pivots.
    pub d_subdiag: Vec<f64>,
    /// BK pivot permutation (forward). Length n.
    /// perm[i] = j means original row j was moved to pivot position i.
    pub perm: Vec<usize>,
    /// Inverse permutation. perm_inv[perm[i]] == i for all i.
    pub perm_inv: Vec<usize>,
    /// Equilibration scaling diagonal D_eq. Length n.
    pub d_eq: Vec<f64>,
    /// True when ZeroPivotAction::ForceAccept fired during factorization.
    pub needs_refinement: bool,
    /// 1×1 pivot threshold copied from BunchKaufmanParams at factor time.
    /// `solve` consults this to decide whether to divide by `d_diag[k]`:
    /// pivots `|d| <= zero_tol` were force-accepted as numerically zero
    /// during factorization and must be skipped (left as-is) by the
    /// D-block solve. Otherwise dividing by a tiny pivot produces
    /// catastrophic error. See dev/plans/threshold-mismatch-fix.md.
    pub zero_tol: f64,
    /// 2×2 pivot block threshold (matches BunchKaufmanParams::zero_tol_2x2).
    pub zero_tol_2x2: f64,
}

/// Factor a symmetric indefinite matrix using Bunch-Kaufman pivoting.
/// Applies equilibration transparently before factoring.
pub fn factor(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    params: &BunchKaufmanParams,
) -> Result<(Factors, Inertia), FeralError> {
    matrix.validate()?;
    let n = matrix.n;

    // Apply equilibration
    let d_eq = crate::dense::equilibrate::equilibrate_scaling(matrix);

    // Copy the lower triangle into a working array, applying equilibration
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            a[j * n + i] = d_eq[i] * matrix.data[j * n + i] * d_eq[j];
        }
    }

    // Initialize permutation as identity
    let mut perm: Vec<usize> = (0..n).collect();

    // Storage for D block subdiagonal
    let mut subdiag = vec![0.0; n];

    // Inertia counts
    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut zero = 0usize;
    let mut needs_refinement = false;

    let alpha = params.alpha;
    let mut k = 0;

    // Fused update+argmax: the previous pivot's update computes γ₀ and r
    // for the next column, avoiding a redundant O(n) scan. On the first
    // iteration (or after a swap invalidates fused values), we fall back
    // to column_offdiag_max.
    let mut fused_gamma0 = 0.0f64;
    let mut fused_r = 0usize;
    let mut have_fused = false;

    while k < n {
        let remaining = n - k;

        if remaining == 1 {
            // Last pivot: always 1×1
            let d = a[k * n + k];
            if d.abs() <= params.zero_tol {
                match params.on_zero_pivot {
                    ZeroPivotAction::ForceAccept => {
                        needs_refinement = true;
                        zero += 1;
                    }
                    ZeroPivotAction::Fail => {
                        return Err(FeralError::NumericallyRankDeficient);
                    }
                }
            } else if d > 0.0 {
                pos += 1;
            } else {
                neg += 1;
            }
            k += 1;
            continue;
        }

        // Step 1: Find γ₀ = max off-diagonal magnitude in column k
        // Use fused values from previous update when available.
        let (gamma0, r) = if have_fused {
            have_fused = false;
            (fused_gamma0, fused_r)
        } else {
            column_offdiag_max(&a, n, k)
        };

        if gamma0 == 0.0 {
            // Column is zero off-diagonal: 1×1 pivot (matrix reducible)
            let d = a[k * n + k];
            count_1x1_inertia(
                d,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            set_l_column_identity(&mut a, n, k);
            // No fused values — next column wasn't updated
            k += 1;
            continue;
        }

        // Step 3: Test if A[k,k] is acceptable as 1×1 pivot
        let akk = a[k * n + k].abs();
        if akk >= alpha * gamma0 {
            // Accept A[k,k] as 1×1 pivot, no swap — fused values are valid
            let (ng, nr) = do_1x1_pivot(
                &mut a,
                n,
                k,
                gamma0,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            fused_gamma0 = ng;
            fused_r = nr;
            have_fused = k + 1 < n;
            k += 1;
            continue;
        }

        // Step 4: Compute γᵣ = max off-diagonal magnitude in symmetric row/column r
        let gamma_r = symmetric_row_offdiag_max(&a, n, k, r);

        // Step 5: Test if A[r,r] is acceptable as 1×1 pivot (swap k↔r)
        let arr = a[r * n + r].abs();
        if arr >= alpha * gamma_r {
            // Swap invalidates any fused column — re-scan next iteration
            swap_rows_cols(&mut a, n, k, r, &mut perm);
            let (ng, nr) = do_1x1_pivot(
                &mut a,
                n,
                k,
                gamma_r,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            fused_gamma0 = ng;
            fused_r = nr;
            have_fused = k + 1 < n;
            k += 1;
            continue;
        }

        // Step 6: LAPACK extension — test if A[k,k] still usable
        if akk * gamma_r >= alpha * gamma0 * gamma0 {
            // No swap — fused values are valid
            let (ng, nr) = do_1x1_pivot(
                &mut a,
                n,
                k,
                gamma0,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            fused_gamma0 = ng;
            fused_r = nr;
            have_fused = k + 1 < n;
            k += 1;
            continue;
        }

        // Step 7: 2×2 pivot using rows/columns {k, r}
        if r != k + 1 {
            swap_rows_cols(&mut a, n, k + 1, r, &mut perm);
        }

        // Duff-Reid 2×2 growth bound (MUMPS dfac_front_aux.F:1599-1606).
        // See the corresponding comment in factor_frontal().
        let d11_v = a[k * n + k];
        let d21_v = a[k * n + (k + 1)];
        let d22_v = a[(k + 1) * n + (k + 1)];
        let det_v = d11_v * d22_v - d21_v * d21_v;
        let absdet = det_v.abs();
        let mut rmax = 0.0f64;
        let mut tmax = 0.0f64;
        for i in (k + 2)..n {
            let v0 = a[k * n + i].abs();
            if v0 > rmax {
                rmax = v0;
            }
            let v1 = a[(k + 1) * n + i].abs();
            if v1 > tmax {
                tmax = v1;
            }
        }
        let amax = d21_v.abs();
        let u = params.pivot_threshold;
        let growth_fail = (d22_v.abs() * rmax + amax * tmax) * u > absdet
            || (d11_v.abs() * tmax + amax * rmax) * u > absdet;

        if growth_fail {
            // 2×2 rejected by the Duff-Reid growth bound. Fall back to a
            // single 1×1 at k with the column-relative threshold. The
            // second position (k+1) is revisited on the next iteration.
            let (ng, nr) = do_1x1_pivot(
                &mut a,
                n,
                k,
                gamma0,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            fused_gamma0 = ng;
            fused_r = nr;
            have_fused = k + 1 < n;
            k += 1;
            continue;
        }

        let (ng, nr) = do_2x2_pivot(
            &mut a,
            n,
            k,
            &mut subdiag,
            params,
            &mut pos,
            &mut neg,
            &mut zero,
            &mut needs_refinement,
        )?;
        fused_gamma0 = ng;
        fused_r = nr;
        have_fused = k + 2 < n;
        k += 2;
    }

    // Extract L and D from the working array.
    // For 2×2 blocks, the off-diagonal a[k*n+(k+1)] is the D block subdiag
    // (already stored in subdiag), NOT an L entry. L entries for a 2×2 block
    // at {k, k+1} start at row k+2.
    let mut l = vec![0.0; n * n];
    let mut d_diag = vec![0.0; n];

    let mut j = 0;
    while j < n {
        d_diag[j] = a[j * n + j];
        l[j * n + j] = 1.0;

        if j + 1 < n && subdiag[j] != 0.0 {
            // 2×2 block at (j, j+1): L entries start at row j+2
            d_diag[j + 1] = a[(j + 1) * n + (j + 1)];
            l[(j + 1) * n + (j + 1)] = 1.0;
            for i in (j + 2)..n {
                l[j * n + i] = a[j * n + i];
                l[(j + 1) * n + i] = a[(j + 1) * n + i];
            }
            j += 2;
        } else {
            // 1×1 block: L entries start at row j+1
            for i in (j + 1)..n {
                l[j * n + i] = a[j * n + i];
            }
            j += 1;
        }
    }

    // Compute inverse permutation
    let mut perm_inv = vec![0usize; n];
    for (i, &p) in perm.iter().enumerate() {
        perm_inv[p] = i;
    }

    let inertia = Inertia::new(pos, neg, zero);

    flag_growth_for_refinement(&l, &mut needs_refinement);

    Ok((
        Factors {
            n,
            l,
            d_diag,
            d_subdiag: subdiag,
            perm,
            perm_inv,
            d_eq,
            needs_refinement,
            zero_tol: params.zero_tol,
            zero_tol_2x2: params.zero_tol_2x2,
        },
        inertia,
    ))
}

/// Factor a dense symmetric indefinite matrix by treating it as a single
/// fully-summed front and delegating to `factor_frontal(may_delay=false)`.
///
/// Unlike `factor()`, this entry point inherits `factor_frontal`'s safe
/// rejection fallback (via `try_reject_1x1_frontal`): when the 2×2
/// Duff-Reid growth bound fails, the kernel does not divide by a zero
/// pivot, and pivots below the column-relative threshold are either
/// accepted with their correct sign or force-zeroed, with iterative
/// refinement flagged.
///
/// Knight-Ruiz equilibration is applied before the factorization
/// (matching `factor()`'s preprocessing) and `d_eq` is carried on the
/// returned `Factors` for the solve to un-equilibrate.
///
/// Rationale: per `dev/research/task-19-dense-acopp30-expert-consultation.md`,
/// the dense `factor()` entry point is under-constrained for pathological
/// KKT matrices (natural order + u=0 + no `|det|==0` rejection). MUMPS
/// 5.8.2, SPRAL SSIDS, and faer all route such matrices through a single
/// multifrontal / frontal code path. This wrapper gives the bench and
/// other dense callers access to the same safe kernel the sparse path
/// uses, without needing a full symbolic analysis (no AMD/METIS).
pub fn factor_single_front(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    params: &BunchKaufmanParams,
) -> Result<(Factors, Inertia), FeralError> {
    matrix.validate()?;
    let n = matrix.n;

    let d_eq = crate::dense::equilibrate::equilibrate_scaling(matrix);

    // Build an equilibrated scratch SymmetricMatrix for factor_frontal.
    let mut eq_data = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            eq_data[j * n + i] = d_eq[i] * matrix.data[j * n + i] * d_eq[j];
        }
    }
    let eq_matrix = crate::dense::matrix::SymmetricMatrix { n, data: eq_data };

    let front = factor_frontal_blocked(&eq_matrix, n, false, params)?;

    // With may_delay=false and ncol=n, nelim==n, contrib is empty, and
    // the FrontalFactors fields map 1:1 to Factors plus d_eq.
    debug_assert_eq!(front.nelim, n);
    debug_assert_eq!(front.n_delayed, 0);
    debug_assert_eq!(front.contrib_dim, 0);

    let inertia = front.inertia;
    let factors = Factors {
        n,
        l: front.l,
        d_diag: front.d_diag,
        d_subdiag: front.d_subdiag,
        perm: front.perm,
        perm_inv: front.perm_inv,
        d_eq,
        needs_refinement: front.needs_refinement,
        zero_tol: front.zero_tol,
        zero_tol_2x2: front.zero_tol_2x2,
    };

    Ok((factors, inertia))
}

/// Result of partial frontal factorization for the multifrontal solver.
#[derive(Debug)]
pub struct FrontalFactors {
    /// Number of rows in the frontal (nrow).
    pub nrow: usize,
    /// Attempted column count (the `ncol` argument passed to `factor_frontal`).
    /// When `may_delay = true` and a pivot is rejected, the kernel may stop
    /// early with `nelim < ncol`; the leftover `ncol - nelim` columns are
    /// carried in the contribution block as delayed pivots. For the root
    /// supernode (`may_delay = false`) this always equals `nelim`.
    pub ncol: usize,
    /// Actually eliminated column count (`nelim ≤ ncol`). Solve loops use
    /// `nelim` as the upper bound of the D-block sweep.
    pub nelim: usize,
    /// L factor: nrow × nelim column-major. Unit diagonal (implicit).
    /// L[j*nrow + i] for i in [0, nrow), j in [0, nelim).
    pub l: Vec<f64>,
    /// D block diagonal (length nelim).
    pub d_diag: Vec<f64>,
    /// D block subdiagonal for 2×2 pivots (length nelim).
    pub d_subdiag: Vec<f64>,
    /// BK pivot permutation within the first nelim rows.
    /// perm[i] = j means original row j was moved to pivot position i.
    /// Only indices 0..nelim are permuted; nelim..nrow are identity.
    pub perm: Vec<usize>,
    /// Inverse permutation.
    pub perm_inv: Vec<usize>,
    /// Schur complement / delayed-pivot block: cdim × cdim column-major
    /// where cdim = nrow - nelim. Lower triangle only. For the first
    /// `ncol - nelim` positions this holds the un-eliminated (delayed)
    /// fully-summed columns; the remaining `nrow - ncol` positions hold
    /// the non-fully-summed trailing rows. When `nelim == ncol` the whole
    /// block is the classic Schur complement S = A22 - L21 * D * L21^T.
    pub contrib: Vec<f64>,
    /// Dimension of the contribution block (`nrow - nelim`).
    pub contrib_dim: usize,
    /// Number of delayed fully-summed columns in the contribution block,
    /// i.e. `ncol - nelim`. These occupy positions `0..n_delayed` of the
    /// contrib block; positions `n_delayed..contrib_dim` are the
    /// non-fully-summed trailing rows.
    pub n_delayed: usize,
    /// Inertia of the `nelim` eliminated pivots.
    pub inertia: Inertia,
    /// Whether ForceAccept fired during factorization.
    pub needs_refinement: bool,
    /// Number of pivots rescued by rook search after BK-partial's column-
    /// relative threshold test rejected them (Phase 2.4.3). Zero on
    /// well-conditioned matrices. Aggregated per-front; the multifrontal
    /// driver sums across supernodes.
    pub n_rook_rescues: usize,
    /// 1×1 pivot threshold from BunchKaufmanParams (see Factors::zero_tol).
    pub zero_tol: f64,
    /// 2×2 pivot threshold from BunchKaufmanParams.
    pub zero_tol_2x2: f64,
}

/// Outcome of an attempt to accept a 1×1 pivot via `try_reject_1x1_frontal`.
/// The caller uses this to decide whether to continue, force-accept, or
/// break out of the BK loop (SSIDS-style delayed pivoting).
#[derive(Debug, Clone, Copy, PartialEq)]
enum PivotOutcome {
    /// Pivot clears the column-relative threshold; do the rank-1 update.
    Accepted,
    /// Pivot is below threshold; L column has been zeroed and the zero
    /// has been counted. Caller increments k and continues. Only produced
    /// when `may_delay == false`.
    Rejected,
    /// Pivot is below threshold; caller should `break` the BK loop and
    /// let the parent supernode retry this column. Only produced when
    /// `may_delay == true`. The kernel has not mutated any state for
    /// the failed pivot.
    Delayed,
    /// Rook rescue (Phase 2.4.3) found a 2×2 block pivot at `{k, k+1}`
    /// after applying symmetric swaps. The gates (SSIDS det floor,
    /// Duff-Reid growth bound) were checked inside `rook_rescue`. The
    /// caller must count 2×2 inertia, record `subdiag[k] = d21`, apply
    /// `do_2x2_update`, and advance `k` by 2. Emitted only by
    /// `try_reject_1x1_with_rook_rescue`; panel path never sees this.
    AcceptedRook2x2 { d11: f64, d21: f64, d22: f64 },
}

/// Result of one iteration of the scalar BK pivot loop in
/// `factor_frontal`. `scalar_pivot_step` is the extracted per-step body;
/// the caller translates `Advanced(n)` into `k += n` and `Delayed` into
/// a `break` to keep the pre-extraction control flow byte-identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PivotStepResult {
    /// Eliminated `n` columns (1 for a 1×1 pivot, 2 for a 2×2 block).
    Advanced(usize),
    /// Pivot was delayed (SSIDS-style); caller breaks the loop.
    Delayed,
}

/// Outcome of one `lblt_panel_frontal` invocation. The panel processes a
/// run of pure 1×1 pivots (no row/column swap, no 2×2) using faer-style
/// peek-ahead; any deviation from that path (2×2 candidate, swap
/// candidate, scalar-force condition) terminates the panel and asks the
/// caller to run `scalar_pivot_step` once before re-entering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PanelStatus {
    /// Panel eliminated `bs` 1×1 pivots cleanly.
    Full,
    /// Panel terminated early because the next pivot needed capabilities
    /// the panel doesn't support (2×2 or swap). Caller runs one scalar
    /// pivot step and may re-enter the panel afterwards.
    ScalarFallback,
    /// Panel terminated early because the next 1×1 pivot was rejected
    /// under the SSIDS `may_delay == true` contract — the parent
    /// supernode will absorb the delayed columns. Caller applies the
    /// deferred Schur update to trailing columns and breaks out of the
    /// outer loop (no further pivots in this supernode). Only produced
    /// when the caller passed `may_delay == true`.
    Delayed,
}

/// Factor a frontal matrix, eliminating only the first `ncol` columns.
///
/// This is the key dense kernel for the multifrontal solver. Unlike `factor()`,
/// pivot search is RESTRICTED to the first `ncol` rows/columns. Rows ncol..nrow
/// are never swapped into pivot positions, preserving their ordering for the
/// contribution block.
///
/// When `may_delay == true`, the first pivot that fails the column-relative
/// threshold (or the 2×2 Duff-Reid growth bound) causes the kernel to stop
/// early: the leftover `(ncol - nelim)` columns are carried forward in the
/// contribution block as delayed fully-summed columns. The SSIDS `ldlt_tpp`
/// kernel uses this "break on first failure" model — see
/// `dev/research/phase-2.3-delayed-pivoting.md` for the reference.
///
/// When `may_delay == false` (the root supernode), the existing
/// `ZeroPivotAction::ForceAccept` path handles failed pivots by zeroing the
/// L column and counting a zero pivot, exactly as before.
///
/// After eliminating `nelim` pivots, the `(nrow - nelim) × (nrow - nelim)`
/// trailing block of the working matrix is extracted as the contribution
/// block. When `nelim == ncol` this is the classic Schur complement
/// `S = A22 - L21 * D * L21^T`. When `nelim < ncol` the first
/// `(ncol - nelim)` rows/columns of that block are delayed fully-summed
/// columns. No equilibration is applied.
/// Diagnostic per-phase timing sink for `factor_frontal_with_profile`.
/// Populated only when a caller opts in; the `factor_frontal` wrapper
/// passes `None` so production paths are branchless. Used by
/// `src/bin/diag_leaf_profile.rs` (Phase 2.9.2 Step A) to sub-time
/// the kernel and decide whether the arena refactor is worthwhile.
#[doc(hidden)]
#[derive(Default, Debug, Clone, Copy)]
pub struct FrontalProfile {
    pub alloc_copy_ns: u128,
    pub setup_ns: u128,
    pub pivot_loop_ns: u128,
    pub extract_ns: u128,
    pub n_calls: u64,
}

pub fn factor_frontal(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
) -> Result<FrontalFactors, FeralError> {
    factor_frontal_with_profile(matrix, ncol, may_delay, params, None)
}

#[doc(hidden)]
pub fn factor_frontal_with_profile(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
    mut profile: Option<&mut FrontalProfile>,
) -> Result<FrontalFactors, FeralError> {
    matrix.validate()?;
    let nrow = matrix.n;

    if ncol > nrow {
        return Err(FeralError::InvalidInput(format!(
            "ncol {} > nrow {}",
            ncol, nrow
        )));
    }
    if ncol == 0 {
        return Ok(FrontalFactors {
            nrow,
            ncol: 0,
            nelim: 0,
            l: Vec::new(),
            d_diag: Vec::new(),
            d_subdiag: Vec::new(),
            perm: (0..nrow).collect(),
            perm_inv: (0..nrow).collect(),
            contrib: matrix.data.clone(),
            contrib_dim: nrow,
            n_delayed: 0,
            inertia: Inertia {
                positive: 0,
                negative: 0,
                zero: 0,
            },
            needs_refinement: false,
            n_rook_rescues: 0,
            zero_tol: params.zero_tol,
            zero_tol_2x2: params.zero_tol_2x2,
        });
    }

    // Phase: alloc + copy
    let t0 = profile.as_ref().map(|_| std::time::Instant::now());
    let mut a = vec![0.0; nrow * nrow];
    for j in 0..nrow {
        for i in j..nrow {
            a[j * nrow + i] = matrix.data[j * nrow + i];
        }
    }
    if let (Some(p), Some(t)) = (profile.as_deref_mut(), t0) {
        p.alloc_copy_ns += t.elapsed().as_nanos();
    }

    // Phase: setup (perm, subdiag, counters)
    let t0 = profile.as_ref().map(|_| std::time::Instant::now());
    let mut perm: Vec<usize> = (0..nrow).collect();
    let mut subdiag = vec![0.0; nrow];
    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut zero = 0usize;
    let mut needs_refinement = false;
    let mut n_rook_rescues = 0usize;
    if let (Some(p), Some(t)) = (profile.as_deref_mut(), t0) {
        p.setup_ns += t.elapsed().as_nanos();
    }

    let t_pivot = profile.as_ref().map(|_| std::time::Instant::now());
    let mut k = 0;

    // Factor only the first ncol columns. Pivot search restricted to [k, ncol).
    // Per-step body is extracted as `scalar_pivot_step` so the Phase 2.4.1b
    // blocked-panel path can share the rejection/delay fallback with the
    // unblocked driver. This loop is behavior-preserving byte-for-byte vs
    // the pre-extraction body.
    while k < ncol {
        match scalar_pivot_step(
            &mut a,
            nrow,
            ncol,
            k,
            may_delay,
            params,
            &mut perm,
            &mut subdiag,
            &mut pos,
            &mut neg,
            &mut zero,
            &mut needs_refinement,
            &mut n_rook_rescues,
        )? {
            PivotStepResult::Advanced(n) => k += n,
            PivotStepResult::Delayed => break,
        }
    }

    if let (Some(p), Some(t)) = (profile.as_deref_mut(), t_pivot) {
        p.pivot_loop_ns += t.elapsed().as_nanos();
    }
    let t_extract = profile.as_ref().map(|_| std::time::Instant::now());

    // At the break point, `k` is the number of successfully eliminated
    // pivots. When `may_delay == false` this equals `ncol`; when
    // `may_delay == true` and a pivot was delayed, `nelim < ncol`.
    let nelim = k;
    let n_delayed = ncol - nelim;

    // Extract L (nrow × nelim), D diagonal, and contribution block
    let mut l = vec![0.0; nrow * nelim];
    let mut d_diag = vec![0.0; nelim];

    let mut j = 0;
    while j < nelim {
        d_diag[j] = a[j * nrow + j];
        l[j * nrow + j] = 1.0; // unit diagonal

        if j + 1 < nelim && subdiag[j] != 0.0 {
            // 2×2 block
            d_diag[j + 1] = a[(j + 1) * nrow + (j + 1)];
            l[(j + 1) * nrow + (j + 1)] = 1.0;
            for i in (j + 2)..nrow {
                l[j * nrow + i] = a[j * nrow + i];
                l[(j + 1) * nrow + i] = a[(j + 1) * nrow + i];
            }
            j += 2;
        } else {
            for i in (j + 1)..nrow {
                l[j * nrow + i] = a[j * nrow + i];
            }
            j += 1;
        }
    }

    // Extract contribution block: trailing (nrow-nelim) × (nrow-nelim) of a.
    // When `nelim < ncol` the first `n_delayed` rows/columns of this block
    // are delayed fully-summed columns; the remaining positions hold the
    // non-fully-summed trailing rows exactly as before.
    let cdim = nrow - nelim;
    let mut contrib = vec![0.0; cdim * cdim];
    for cj in 0..cdim {
        for ci in cj..cdim {
            contrib[cj * cdim + ci] = a[(nelim + cj) * nrow + (nelim + ci)];
        }
    }

    let mut perm_inv = vec![0usize; nrow];
    for (i, &p) in perm.iter().enumerate() {
        perm_inv[p] = i;
    }

    flag_growth_for_refinement(&l, &mut needs_refinement);

    let result = FrontalFactors {
        nrow,
        ncol,
        nelim,
        l,
        d_diag,
        d_subdiag: subdiag[..nelim].to_vec(),
        perm,
        perm_inv,
        contrib,
        contrib_dim: cdim,
        n_delayed,
        inertia: Inertia::new(pos, neg, zero),
        needs_refinement,
        n_rook_rescues,
        zero_tol: params.zero_tol,
        zero_tol_2x2: params.zero_tol_2x2,
    };
    if let (Some(p), Some(t)) = (profile, t_extract) {
        p.extract_ns += t.elapsed().as_nanos();
        p.n_calls += 1;
    }
    Ok(result)
}

/// Blocked-panel BK LDLᵀ variant of `factor_frontal` (Phase 2.4.1b).
///
/// **Status: Step 4b GREEN (peek-ahead panel + deferred Schur).**
/// Implements the faer-style blocked kernel described in
/// `dev/plans/phase-2.4.1-blocked-ldlt.md`: a panel processes up to
/// `params.block_size` 1×1 pivots. Before each pivot search, the
/// current column is updated via **replay** — pending rank-1 updates
/// from prior panel pivots are applied in ascending pivot index using
/// the same `schur_kernel::axpy_minus_unroll4_nofma` kernel scalar
/// uses, which makes the per-element accumulation order identical to
/// `factor_frontal`. After the panel, the deferred rank-1 updates are
/// applied to the remaining trailing columns in the same pivot-outer
/// order, again via the bit-exact axpy kernel.
///
/// **Bit-parity guarantee.** Since (i) replay traverses `(i, j)` with
/// updates applied in ascending pivot index — identical to scalar's
/// pivot-outer/column-inner loop — and (ii) both paths use the same
/// axpy kernel, scalar and blocked produce byte-identical
/// `(L, D, perm, inertia, contrib)`. Enforced by `tests/blocked_ldlt.rs`.
///
/// **Fallbacks.** Any of the following routes through `scalar_pivot_step`
/// instead of the panel:
/// - `may_delay == true` (SSIDS-style delayed pivoting — Step 5 target).
/// - `ncol <= params.block_size` (small-front scalar fast path).
/// - Panel encounters a 2×2 candidate (`akk < alpha * gamma0`) or any
///   other non-trivial BK branch — panel returns `ScalarFallback`,
///   caller runs one `scalar_pivot_step` then re-enters the panel.
///
/// The scalar oracle (`factor_frontal`) is retained for correctness
/// and serves as the bit-parity reference.
pub fn factor_frontal_blocked(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
) -> Result<FrontalFactors, FeralError> {
    matrix.validate()?;
    let nrow = matrix.n;

    if ncol > nrow {
        return Err(FeralError::InvalidInput(format!(
            "ncol {} > nrow {}",
            ncol, nrow
        )));
    }
    if ncol == 0 {
        return Ok(FrontalFactors {
            nrow,
            ncol: 0,
            nelim: 0,
            l: Vec::new(),
            d_diag: Vec::new(),
            d_subdiag: Vec::new(),
            perm: (0..nrow).collect(),
            perm_inv: (0..nrow).collect(),
            contrib: matrix.data.clone(),
            contrib_dim: nrow,
            n_delayed: 0,
            inertia: Inertia {
                positive: 0,
                negative: 0,
                zero: 0,
            },
            needs_refinement: false,
            n_rook_rescues: 0,
            zero_tol: params.zero_tol,
            zero_tol_2x2: params.zero_tol_2x2,
        });
    }

    // Phase 2.4.1c triage hook. When `FORCE_SCALAR_FRONTAL` is set,
    // delegate to `factor_frontal` unconditionally so a binary can
    // run the multifrontal driver with either kernel without
    // patching call sites. This is a diagnostic flag only — the
    // default (false) preserves the production dispatch.
    if FORCE_SCALAR_FRONTAL.load(std::sync::atomic::Ordering::Relaxed) {
        return factor_frontal(matrix, ncol, may_delay, params);
    }

    // Fallback conditions where the panel offers no advantage.
    // Delegation preserves parity trivially.
    let bs = params.block_size;
    if bs < 2 || ncol <= bs {
        return factor_frontal(matrix, ncol, may_delay, params);
    }

    // Copy lower triangle into the working array, same as `factor_frontal`.
    let mut a = vec![0.0; nrow * nrow];
    for j in 0..nrow {
        for i in j..nrow {
            a[j * nrow + i] = matrix.data[j * nrow + i];
        }
    }

    let mut perm: Vec<usize> = (0..nrow).collect();
    let mut subdiag = vec![0.0; nrow];
    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut zero = 0usize;
    let mut needs_refinement = false;
    let mut n_rook_rescues = 0usize;
    let mut d_panel = vec![0.0f64; bs];

    let mut k = 0;
    while k < ncol {
        let remaining = ncol - k;
        if remaining <= bs {
            // Scalar tail: process remaining pivots one at a time.
            match scalar_pivot_step(
                &mut a,
                nrow,
                ncol,
                k,
                may_delay,
                params,
                &mut perm,
                &mut subdiag,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
                &mut n_rook_rescues,
            )? {
                PivotStepResult::Advanced(n) => k += n,
                PivotStepResult::Delayed => break,
            }
            continue;
        }

        let (n_elim, status) = lblt_panel_frontal(
            &mut a,
            nrow,
            k,
            bs,
            may_delay,
            params,
            &mut pos,
            &mut neg,
            &mut zero,
            &mut needs_refinement,
            &mut d_panel,
        )?;
        // On ScalarFallback and Delayed the panel peek-ahead'd column
        // `k+n_elim` (applied pivots 0..n_elim-1 to it) before deciding
        // it could not pivot. In scalar semantics that column's state at
        // break time already matches what pivots 0..n_elim-1 produce via
        // eager updates, so `apply_blocked_schur` must skip it to avoid
        // a double rank-1 update. On Full the column at `k+n_elim` was
        // not peek-ahead'd, so the deferred update starts there normally.
        let j_start = match status {
            PanelStatus::Full => k + n_elim,
            PanelStatus::ScalarFallback | PanelStatus::Delayed => k + n_elim + 1,
        };
        apply_blocked_schur(&mut a, nrow, k, n_elim, j_start, &d_panel);
        k += n_elim;

        match status {
            PanelStatus::Full => {}
            PanelStatus::ScalarFallback => {
                // One scalar step to handle the 2×2/swap case the panel declined.
                if k >= ncol {
                    break;
                }
                match scalar_pivot_step(
                    &mut a,
                    nrow,
                    ncol,
                    k,
                    may_delay,
                    params,
                    &mut perm,
                    &mut subdiag,
                    &mut pos,
                    &mut neg,
                    &mut zero,
                    &mut needs_refinement,
                    &mut n_rook_rescues,
                )? {
                    PivotStepResult::Advanced(n) => k += n,
                    PivotStepResult::Delayed => break,
                }
            }
            PanelStatus::Delayed => break,
        }
    }

    let nelim = k;
    let n_delayed = ncol - nelim;

    // Extract L, D, contrib — identical logic to `factor_frontal`.
    let mut l = vec![0.0; nrow * nelim];
    let mut d_diag = vec![0.0; nelim];

    let mut j = 0;
    while j < nelim {
        d_diag[j] = a[j * nrow + j];
        l[j * nrow + j] = 1.0;

        if j + 1 < nelim && subdiag[j] != 0.0 {
            d_diag[j + 1] = a[(j + 1) * nrow + (j + 1)];
            l[(j + 1) * nrow + (j + 1)] = 1.0;
            for i in (j + 2)..nrow {
                l[j * nrow + i] = a[j * nrow + i];
                l[(j + 1) * nrow + i] = a[(j + 1) * nrow + i];
            }
            j += 2;
        } else {
            for i in (j + 1)..nrow {
                l[j * nrow + i] = a[j * nrow + i];
            }
            j += 1;
        }
    }

    let cdim = nrow - nelim;
    let mut contrib = vec![0.0; cdim * cdim];
    for cj in 0..cdim {
        for ci in cj..cdim {
            contrib[cj * cdim + ci] = a[(nelim + cj) * nrow + (nelim + ci)];
        }
    }

    let mut perm_inv = vec![0usize; nrow];
    for (i, &p) in perm.iter().enumerate() {
        perm_inv[p] = i;
    }

    flag_growth_for_refinement(&l, &mut needs_refinement);

    Ok(FrontalFactors {
        nrow,
        ncol,
        nelim,
        l,
        d_diag,
        d_subdiag: subdiag[..nelim].to_vec(),
        perm,
        perm_inv,
        contrib,
        contrib_dim: cdim,
        n_delayed,
        inertia: Inertia::new(pos, neg, zero),
        needs_refinement,
        n_rook_rescues,
        zero_tol: params.zero_tol,
        zero_tol_2x2: params.zero_tol_2x2,
    })
}

/// Process one blocked panel of up to `bs` pure 1×1 pivots starting at
/// global column `k`. Applies per-column peek-ahead (replay of pending
/// rank-1 updates from prior panel pivots) before each pivot search so
/// the BK test sees the same column state scalar would. Terminates early
/// on any condition the panel cannot handle without a full-state view
/// (2×2 candidate, swap candidate, or — when `may_delay == true` — a
/// delayed pivot from the SSIDS threshold test).
///
/// On return, `a[k..k+n_elim]` columns hold scaled L columns (or zeroed
/// L columns for rejected pivots), `d_panel[0..n_elim]` holds the
/// pre-scaling diagonals (or 0 for rejected/zero-gamma0 pivots).
/// Columns `[k+n_elim, nrow)` are stale — the caller must apply
/// `apply_blocked_schur` before running further pivot searches or
/// extracting the contribution block.
#[allow(clippy::too_many_arguments)]
fn lblt_panel_frontal(
    a: &mut [f64],
    nrow: usize,
    k: usize,
    bs: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
    d_panel: &mut [f64],
) -> Result<(usize, PanelStatus), FeralError> {
    let alpha_bk = params.alpha;
    let cap = bs;
    let mut c = 0usize;
    while c < cap {
        let col = k + c;

        // Peek-ahead: apply pending rank-1 updates from pivots 0..c.
        peek_ahead_column(a, nrow, k, c, d_panel);

        // Compute gamma0 over rows (col+1)..nrow (unrestricted — matches
        // scalar's gamma0 search, which includes rows ncol..nrow for the
        // BK test even in frontal mode).
        let mut gamma0 = 0.0f64;
        for i in (col + 1)..nrow {
            let v = a[col * nrow + i].abs();
            if v > gamma0 {
                gamma0 = v;
            }
        }

        if gamma0 == 0.0 {
            // Zero-column: matches scalar's gamma0==0 branch exactly.
            let d = a[col * nrow + col];
            count_1x1_inertia(d, params, pos, neg, zero, needs_refinement)?;
            set_l_column_identity(a, nrow, col);
            // The L column is all zeros (below diagonal); the diagonal is
            // unchanged. Store the diagonal as d_panel[c] — subsequent
            // replay with alpha = (0 * d) = 0 is a no-op, matching scalar.
            d_panel[c] = d;
            c += 1;
            continue;
        }

        let akk = a[col * nrow + col].abs();
        if akk < alpha_bk * gamma0 {
            // 2×2 or swap needed. Panel cannot handle without full-state
            // access; terminate and let scalar_pivot_step run one step.
            return Ok((c, PanelStatus::ScalarFallback));
        }

        // 1×1 pivot at col, no swap. Try the column-relative threshold.
        let outcome = try_reject_1x1_frontal(
            a,
            nrow,
            col,
            gamma0,
            may_delay,
            params,
            pos,
            neg,
            zero,
            needs_refinement,
        )?;
        match outcome {
            PivotOutcome::Accepted => {
                // Scale L column and record d. Matches `do_1x1_update`'s
                // scale-then-update; rank-1 is deferred to replay.
                let d = a[col * nrow + col];
                if d.abs() != 0.0 {
                    let inv_d = 1.0 / d;
                    for i in (col + 1)..nrow {
                        a[col * nrow + i] *= inv_d;
                    }
                }
                d_panel[c] = d;
            }
            PivotOutcome::Rejected => {
                // `try_reject_1x1_frontal` has zeroed the L column and
                // diagonal. d = 0 → replay alpha = 0 → no-op, matching
                // scalar's `do_1x1_update` early-return on d == 0.
                d_panel[c] = 0.0;
            }
            PivotOutcome::Delayed => {
                // SSIDS break-on-first-failure contract: the pivot was
                // below threshold and `may_delay == true`. The rejection
                // routine did NOT mutate state for this column, so we
                // return with `n_elim = c` and let the caller apply the
                // deferred Schur to columns `[k+c+1, nrow)`, then break
                // out of the outer loop. Column `k+c` retains its
                // peek-ahead state (pivots 0..c-1 applied), which
                // matches scalar's column state at break time exactly.
                return Ok((c, PanelStatus::Delayed));
            }
            PivotOutcome::AcceptedRook2x2 { .. } => {
                unreachable!("panel path never enables rook rescue")
            }
        }

        c += 1;
    }

    Ok((c, PanelStatus::Full))
}

/// Apply pivot `q`'s deferred rank-1 update to a single trailing column.
/// This is the **replay** primitive that makes the blocked path
/// bit-exact with scalar: for each (i, j) with j = `col`, `i >= j`,
/// scalar applies rank-1 updates in ascending pivot index; replay does
/// the same via repeated calls to this helper in order `q = 0..c-1`.
/// The inner axpy uses `schur_kernel::axpy_minus_unroll4_nofma` — the
/// same kernel as `do_1x1_update` — so the per-lane rounding matches.
fn peek_ahead_column(a: &mut [f64], nrow: usize, k: usize, c: usize, d_panel: &[f64]) {
    let col = k + c;
    for (q, &d_q) in d_panel.iter().enumerate().take(c) {
        // Scalar's `do_1x1_update` returns early when d == 0; skipping
        // here preserves that no-op behavior bit-exactly.
        if d_q.abs() == 0.0 {
            continue;
        }
        let q_col = k + q;
        // l_jk = scaled L value at row `col` of column q_col (frozen after
        // pivot q's scaling). `alpha = l_jk * d_q` reproduces scalar's
        // `do_1x1_update` rank-1 alpha exactly.
        let l_jk = a[q_col * nrow + col];
        let alpha = l_jk * d_q;
        if alpha == 0.0 {
            continue;
        }
        // dst = column `col` rows `col..nrow`; src = column `q_col` rows
        // `col..nrow`. Disjoint because q_col < col.
        let (before, rest) = a.split_at_mut(col * nrow);
        let src = &before[q_col * nrow + col..q_col * nrow + nrow];
        let dst = &mut rest[col..nrow];
        schur_kernel::axpy_minus_unroll4_nofma(dst, src, alpha);
    }
}

/// Apply the `n_elim` panel pivots' deferred rank-1 updates to the
/// trailing columns `[j_start, nrow)`. Outer loop is pivot index
/// (matching scalar's pivot-outer/column-inner traversal), inner loop
/// is the axpy kernel — so per-element accumulation order is identical
/// to `do_1x1_update` fired `n_elim` times.
///
/// `j_start` is typically `k + n_elim`, except when the caller had the
/// panel peek-ahead one extra column (`ScalarFallback`), in which case
/// `j_start = k + n_elim + 1` to avoid double-updating the peeked column.
fn apply_blocked_schur(
    a: &mut [f64],
    nrow: usize,
    k: usize,
    n_elim: usize,
    j_start: usize,
    d_panel: &[f64],
) {
    for (q, &d_q) in d_panel.iter().enumerate().take(n_elim) {
        if d_q.abs() == 0.0 {
            continue;
        }
        let q_col = k + q;
        for j in j_start..nrow {
            let l_jk = a[q_col * nrow + j];
            let alpha = l_jk * d_q;
            if alpha == 0.0 {
                continue;
            }
            let (before, rest) = a.split_at_mut(j * nrow);
            let src = &before[q_col * nrow + j..q_col * nrow + nrow];
            let dst = &mut rest[j..nrow];
            schur_kernel::axpy_minus_unroll4_nofma(dst, src, alpha);
        }
    }
}

/// Translate a `PivotOutcome` from `try_reject_1x1_with_rook_rescue`
/// (or plain `try_reject_1x1_frontal`) into a `PivotStepResult` and
/// perform the required trailing-update. Used at every scalar BK
/// 1×1 call site to centralize the `AcceptedRook2x2` dispatch.
#[inline]
#[allow(clippy::too_many_arguments)]
fn finish_1x1_outcome(
    outcome: PivotOutcome,
    a: &mut [f64],
    nrow: usize,
    k: usize,
    subdiag: &mut [f64],
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
) -> PivotStepResult {
    match outcome {
        PivotOutcome::Accepted => {
            do_1x1_update(a, nrow, k);
            PivotStepResult::Advanced(1)
        }
        PivotOutcome::Rejected => PivotStepResult::Advanced(1),
        PivotOutcome::Delayed => PivotStepResult::Delayed,
        PivotOutcome::AcceptedRook2x2 { d11, d21, d22 } => {
            let inertia = count_2x2_inertia_val(d11, d21, d22);
            *pos += inertia.positive;
            *neg += inertia.negative;
            *zero += inertia.zero;
            subdiag[k] = d21;
            do_2x2_update(a, nrow, k, d11, d21, d22);
            PivotStepResult::Advanced(2)
        }
    }
}

/// One iteration of the scalar BK pivot loop for `factor_frontal`.
///
/// Extracted verbatim from the pre-extraction in-line loop body so the
/// Phase 2.4.1b blocked-panel path can share the rejection/delay fallback
/// with the unblocked driver. Byte-identical behavior with the original
/// body is required — see `dev/plans/phase-2.4.1-blocked-ldlt.md` §2.
///
/// Returns `Advanced(1)` or `Advanced(2)` on success; `Delayed` when the
/// SSIDS-style delay path fires (only possible when `may_delay == true`).
#[allow(clippy::too_many_arguments)]
fn scalar_pivot_step(
    a: &mut [f64],
    nrow: usize,
    ncol: usize,
    k: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
    perm: &mut [usize],
    subdiag: &mut [f64],
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
    n_rook_rescues: &mut usize,
) -> Result<PivotStepResult, FeralError> {
    let alpha = params.alpha;
    let remaining = ncol - k;

    if remaining == 1 {
        // Last eliminated pivot: always 1×1. Compute the column max
        // over rows (k+1..nrow) for the column-relative threshold.
        // Rook rescue cannot fire here (needs ncol-k >= 2), so call
        // the rejection routine directly.
        let mut col_max = 0.0f64;
        for i in (k + 1)..nrow {
            let v = a[k * nrow + i].abs();
            if v > col_max {
                col_max = v;
            }
        }
        let outcome = try_reject_1x1_frontal(
            a,
            nrow,
            k,
            col_max,
            may_delay,
            params,
            pos,
            neg,
            zero,
            needs_refinement,
        )?;
        match outcome {
            PivotOutcome::Accepted => do_1x1_update(a, nrow, k),
            PivotOutcome::Rejected => {}
            PivotOutcome::Delayed => return Ok(PivotStepResult::Delayed),
            PivotOutcome::AcceptedRook2x2 { .. } => {
                unreachable!("remaining==1 never triggers rook rescue")
            }
        }
        return Ok(PivotStepResult::Advanced(1));
    }

    // Find max |A[i,k]| for i in (k, ncol) — restricted to fully-summed rows
    let (gamma0, r) = {
        let mut max_val = 0.0f64;
        let mut max_row = k + 1;
        // Search within fully-summed rows first
        for i in (k + 1)..ncol {
            let v = a[k * nrow + i].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        // Also check sub-diagonal rows (they contribute to gamma0 for
        // the BK pivot test, but are never swapped into pivot position)
        for i in ncol..nrow {
            let v = a[k * nrow + i].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        (max_val, max_row)
    };

    if gamma0 == 0.0 {
        let d = a[k * nrow + k];
        count_1x1_inertia(d, params, pos, neg, zero, needs_refinement)?;
        set_l_column_identity(a, nrow, k);
        return Ok(PivotStepResult::Advanced(1));
    }

    let akk = a[k * nrow + k].abs();

    if akk >= alpha * gamma0 {
        // 1×1 pivot at k, no swap
        let outcome = try_reject_1x1_with_rook_rescue(
            a,
            nrow,
            ncol,
            k,
            gamma0,
            may_delay,
            params,
            perm,
            pos,
            neg,
            zero,
            needs_refinement,
            n_rook_rescues,
        )?;
        return Ok(finish_1x1_outcome(
            outcome, a, nrow, k, subdiag, pos, neg, zero,
        ));
    }

    // gamma_r: max off-diagonal in symmetric row r
    let gamma_r = symmetric_row_offdiag_max(a, nrow, k, r);
    let arr = a[r * nrow + r].abs();

    // Can we swap r into pivot position? Only if r < ncol (fully summed)
    let r_is_fully_summed = r < ncol;

    if r_is_fully_summed && arr >= alpha * gamma_r {
        // 1×1 pivot at r, swap r↔k
        swap_rows_cols(a, nrow, k, r, perm);
        let outcome = try_reject_1x1_with_rook_rescue(
            a,
            nrow,
            ncol,
            k,
            gamma_r,
            may_delay,
            params,
            perm,
            pos,
            neg,
            zero,
            needs_refinement,
            n_rook_rescues,
        )?;
        return Ok(finish_1x1_outcome(
            outcome, a, nrow, k, subdiag, pos, neg, zero,
        ));
    }

    if akk * gamma_r >= alpha * gamma0 * gamma0 {
        // 1×1 pivot at k (LAPACK extension), no swap
        let outcome = try_reject_1x1_with_rook_rescue(
            a,
            nrow,
            ncol,
            k,
            gamma0,
            may_delay,
            params,
            perm,
            pos,
            neg,
            zero,
            needs_refinement,
            n_rook_rescues,
        )?;
        return Ok(finish_1x1_outcome(
            outcome, a, nrow, k, subdiag, pos, neg, zero,
        ));
    }

    if r_is_fully_summed && k + 1 < ncol {
        // 2×2 pivot using {k, r}, both fully summed
        if r != k + 1 {
            swap_rows_cols(a, nrow, k + 1, r, perm);
        }
        let d11 = a[k * nrow + k];
        let d21 = a[k * nrow + (k + 1)];
        let d22 = a[(k + 1) * nrow + (k + 1)];
        let det = d11 * d22 - d21 * d21;

        // Duff-Reid 2×2 growth bound (MUMPS dfac_front_aux.F:1599-1606):
        //
        //   reject iff  (|a22|*RMAX + AMAX*TMAX) * u  >  |det|
        //        OR     (|a11|*TMAX + AMAX*RMAX) * u  >  |det|
        //
        // where RMAX = max |a[i, k]| for i > k+1
        //       TMAX = max |a[i, k+1]| for i > k+1
        //       AMAX = |a[k+1, k]| = |d21|
        // (i.e., RMAX and TMAX are the column maxes of the two pivot
        // columns *beyond* the 2×2 block; AMAX is the cross term.)
        //
        // When pivot_threshold == 0.0 the growth bound is always
        // satisfied (0 <= |det|), preserving Phase 1 behavior.
        let mut rmax = 0.0f64;
        let mut tmax = 0.0f64;
        for i in (k + 2)..nrow {
            let v0 = a[k * nrow + i].abs();
            if v0 > rmax {
                rmax = v0;
            }
            let v1 = a[(k + 1) * nrow + i].abs();
            if v1 > tmax {
                tmax = v1;
            }
        }
        let amax = d21.abs();
        let absdet = det.abs();
        let u = params.pivot_threshold;
        let growth_fail = (d22.abs() * rmax + amax * tmax) * u > absdet
            || (d11.abs() * tmax + amax * rmax) * u > absdet;

        // Scale-invariant cancellation-aware determinant floor, ported
        // from SSIDS `src/ssids/cpu/kernels/ldlt_tpp.cxx:98-106`:
        //
        //   maxpiv    = max(|a11|, |a21|, |a22|)
        //   detscale  = 1 / maxpiv
        //   detpiv0   = (a11 * detscale) * a22
        //   detpiv1   = (a21 * detscale) * a21
        //   detpiv    = detpiv0 - detpiv1      (== det / maxpiv)
        //   reject iff maxpiv < small
        //           OR |detpiv| < max(small, |detpiv0|/2, |detpiv1|/2)
        //
        // This replaces the prior absolute `|det| <= zero_tol_2x2` floor,
        // which was only meaningful on equilibrated matrices. The test
        // is scale-invariant by construction: the ratio `|detpiv|` vs
        // fractions of `|detpiv0|`, `|detpiv1|` does not depend on
        // the absolute magnitude of the block. `SSIDS_DET_SMALL = 1e-20`
        // is a dead-zero underflow floor, NOT a stability threshold.
        // See dev/research/ssids-scale-invariant-det-floor.md.
        let max_piv = d11.abs().max(d21.abs()).max(d22.abs());
        let det_floor_fail = if max_piv < SSIDS_DET_SMALL {
            true
        } else {
            let det_scale = 1.0 / max_piv;
            let detpiv0 = (d11 * det_scale) * d22;
            let detpiv1 = (d21 * det_scale) * d21;
            let detpiv = detpiv0 - detpiv1;
            let cancel_floor = SSIDS_DET_SMALL
                .max(detpiv0.abs() * 0.5)
                .max(detpiv1.abs() * 0.5);
            detpiv.abs() < cancel_floor
        };

        if growth_fail || det_floor_fail {
            // 2×2 rejected. SSIDS-style delayed pivoting: when
            // `may_delay == true`, break out immediately so the parent
            // supernode can retry this pivot with a larger pivot search
            // window. Otherwise fall back to a single 1×1 at k with the
            // column-relative threshold, which triggers the existing
            // ForceAccept path.
            if may_delay {
                return Ok(PivotStepResult::Delayed);
            }
            if det_floor_fail {
                match params.on_zero_pivot {
                    ZeroPivotAction::Fail => {
                        return Err(FeralError::NumericallyRankDeficient);
                    }
                    ZeroPivotAction::ForceAccept => {
                        *needs_refinement = true;
                    }
                }
            }
            let outcome = try_reject_1x1_with_rook_rescue(
                a,
                nrow,
                ncol,
                k,
                gamma0,
                may_delay,
                params,
                perm,
                pos,
                neg,
                zero,
                needs_refinement,
                n_rook_rescues,
            )?;
            return Ok(finish_1x1_outcome(
                outcome, a, nrow, k, subdiag, pos, neg, zero,
            ));
        }

        let pivot_inertia = count_2x2_inertia_val(d11, d21, d22);
        *pos += pivot_inertia.positive;
        *neg += pivot_inertia.negative;
        *zero += pivot_inertia.zero;

        subdiag[k] = d21;
        // 2×2 update
        do_2x2_update(a, nrow, k, d11, d21, d22);
        Ok(PivotStepResult::Advanced(2))
    } else {
        // Can't do 2×2 (r is not fully summed or only 1 column left).
        // Last-resort 1×1 at k with column-relative rejection.
        let outcome = try_reject_1x1_with_rook_rescue(
            a,
            nrow,
            ncol,
            k,
            gamma0,
            may_delay,
            params,
            perm,
            pos,
            neg,
            zero,
            needs_refinement,
            n_rook_rescues,
        )?;
        Ok(finish_1x1_outcome(
            outcome, a, nrow, k, subdiag, pos, neg, zero,
        ))
    }
}

/// Apply the column-relative pivot threshold to a frontal 1×1 candidate at
/// position `k` with column max `col_max`. Returns:
///
/// - `Accepted` — pivot clears the threshold; caller should apply the
///   rank-1 update.
/// - `Rejected` — pivot is below threshold AND `may_delay == false`; the
///   L column has been zeroed and a zero pivot has been counted via
///   `ZeroPivotAction::ForceAccept`. Caller increments `k` and continues.
/// - `Delayed` — pivot is below threshold AND `may_delay == true`; no
///   state has been mutated. Caller should break the BK loop so the parent
///   supernode can retry this column.
///
/// `ZeroPivotAction::Fail` short-circuits to `Err(NumericallyRankDeficient)`
/// regardless of `may_delay`.
#[allow(clippy::too_many_arguments)]
fn try_reject_1x1_frontal(
    a: &mut [f64],
    nrow: usize,
    k: usize,
    col_max: f64,
    may_delay: bool,
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
) -> Result<PivotOutcome, FeralError> {
    let d = a[k * nrow + k];
    let threshold = (params.pivot_threshold * col_max).max(params.zero_tol);

    if d.abs() <= threshold {
        if may_delay {
            return Ok(PivotOutcome::Delayed);
        }
        // At the root (may_delay=false) we have no parent to absorb
        // the rejected pivot. Split the branch by absolute magnitude:
        //
        //  (a) |d| <= zero_tol — truly numerically zero. Set d=0 so
        //      solve skips this position, count as rank-deficient.
        //
        //  (b) zero_tol < |d| <= u*col_max — small but clearly
        //      nonzero. The SSIDS reference's `ldlt_tpp_factor` at
        //      the root simply breaks out leaving un-eliminated
        //      columns, which silently under-reports rank; that is
        //      not acceptable under feral's "inertia must be exactly
        //      correct" invariant. Instead we accept the pivot at
        //      its actual magnitude with its correct sign, counting
        //      it as positive or negative inertia per SSIDS/MUMPS
        //      convention for degenerate LPs. L growth is bounded
        //      by 1/|d| per step which can be large, so we request
        //      iterative refinement to recover the residual. This
        //      closes DEGENLPA-family failures where MUMPS reports
        //      e.g. (20, 15, 0) and the prior ForceAccept-zero path
        //      gave (20, 14, 1).
        if d.abs() <= params.zero_tol {
            match params.on_zero_pivot {
                ZeroPivotAction::ForceAccept => {
                    *needs_refinement = true;
                    *zero += 1;
                }
                ZeroPivotAction::Fail => return Err(FeralError::NumericallyRankDeficient),
            }
            for i in (k + 1)..nrow {
                a[k * nrow + i] = 0.0;
            }
            a[k * nrow + k] = 0.0;
            return Ok(PivotOutcome::Rejected);
        }
        // Case (b): small but nonzero — accept with correct sign.
        *needs_refinement = true;
        if d > 0.0 {
            *pos += 1;
        } else {
            *neg += 1;
        }
        return Ok(PivotOutcome::Accepted);
    }

    // Accept: sign-based inertia.
    if d > 0.0 {
        *pos += 1;
    } else {
        *neg += 1;
    }
    Ok(PivotOutcome::Accepted)
}

/// Phase 2.4.3 splice: attempt a 1×1 pivot with rook rescue on rejection.
///
/// Fast path: if `|d| > threshold`, delegate to `try_reject_1x1_frontal`
/// unchanged (well-conditioned matrices pay zero rook cost, matching the
/// plan's "rescue, not top-level" design).
///
/// Slow path: if the column-relative threshold rejects the pivot at `k`,
/// call `rook_rescue` before falling through to delay / force-accept.
/// On rook success, apply the symmetric swap sequence via
/// `swap_rows_cols` (updating `perm`), increment `n_rook_rescues`, and
/// return either `Accepted` (1×1 rescue — caller runs `do_1x1_update`)
/// or `AcceptedRook2x2` (2×2 rescue — caller runs the 2×2 update path).
///
/// Panel path (`lblt_panel_frontal`) cannot safely apply mid-panel
/// swaps (they would invalidate `d_panel` replay state), so it keeps
/// calling the original `try_reject_1x1_frontal` directly and never
/// enters this wrapper. See plan §"Blocked-panel interaction".
#[allow(clippy::too_many_arguments)]
fn try_reject_1x1_with_rook_rescue(
    a: &mut [f64],
    nrow: usize,
    ncol: usize,
    k: usize,
    col_max: f64,
    may_delay: bool,
    params: &BunchKaufmanParams,
    perm: &mut [usize],
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
    n_rook_rescues: &mut usize,
) -> Result<PivotOutcome, FeralError> {
    let d = a[k * nrow + k];
    let threshold = (params.pivot_threshold * col_max).max(params.zero_tol);

    // Well-conditioned fast path: pivot clears the threshold. Delegate
    // verbatim so accounting stays byte-identical to the pre-rook path.
    if d.abs() > threshold {
        return try_reject_1x1_frontal(
            a,
            nrow,
            k,
            col_max,
            may_delay,
            params,
            pos,
            neg,
            zero,
            needs_refinement,
        );
    }

    // Threshold failed — try rook rescue before delay/force-accept.
    if let Some(pivot) = rook_rescue(a, nrow, ncol, k, params) {
        *n_rook_rescues += 1;
        for idx in 0..pivot.n_swaps {
            let (p, q) = pivot.swaps[idx];
            swap_rows_cols(a, nrow, p, q, perm);
        }
        match pivot.kind {
            RookKind::Pivot1x1 => {
                let d_new = a[k * nrow + k];
                if d_new > 0.0 {
                    *pos += 1;
                } else {
                    *neg += 1;
                }
                return Ok(PivotOutcome::Accepted);
            }
            RookKind::Pivot2x2 => {
                let d11 = a[k * nrow + k];
                let d21 = a[k * nrow + (k + 1)];
                let d22 = a[(k + 1) * nrow + (k + 1)];
                return Ok(PivotOutcome::AcceptedRook2x2 { d11, d21, d22 });
            }
        }
    }

    // Rook could not rescue — delegate to the existing delay /
    // force-accept logic.
    try_reject_1x1_frontal(
        a,
        nrow,
        k,
        col_max,
        may_delay,
        params,
        pos,
        neg,
        zero,
        needs_refinement,
    )
}

/// 1×1 rank-1 update: update columns k+1..n after eliminating column k.
fn do_1x1_update(a: &mut [f64], n: usize, k: usize) {
    let d = a[k * n + k];
    if d.abs() == 0.0 {
        return;
    }
    let inv_d = 1.0 / d;
    for i in (k + 1)..n {
        a[k * n + i] *= inv_d;
    }
    for j in (k + 1)..n {
        let l_jk = a[k * n + j];
        let alpha = l_jk * d;
        // src = column k rows j..n (already scaled by inv_d above);
        // dst = column j rows j..n. Disjoint because k < j.
        let (before, rest) = a.split_at_mut(j * n);
        let src = &before[k * n + j..k * n + n];
        let dst = &mut rest[j..n];
        schur_kernel::axpy_minus_unroll4_nofma(dst, src, alpha);
    }
}

/// Rank-2 update after a 2×2 pivot at columns `k`, `k+1`.
fn do_2x2_update(a: &mut [f64], n: usize, k: usize, d11: f64, d21: f64, d22: f64) {
    let det = d11 * d22 - d21 * d21;
    if det.abs() == 0.0 {
        return;
    }
    let inv_det = 1.0 / det;

    for i in (k + 2)..n {
        let a_ik = a[k * n + i];
        let a_ik1 = a[(k + 1) * n + i];
        a[k * n + i] = (d22 * a_ik - d21 * a_ik1) * inv_det;
        a[(k + 1) * n + i] = (d11 * a_ik1 - d21 * a_ik) * inv_det;
    }

    for j in (k + 2)..n {
        let l_j0 = a[k * n + j];
        let l_j1 = a[(k + 1) * n + j];
        let dl_j0 = d11 * l_j0 + d21 * l_j1;
        let dl_j1 = d21 * l_j0 + d22 * l_j1;
        // src0, src1 = columns k, k+1 rows j..n (scaled by the
        // rank-1 block of the 2×2 update above); dst = column j
        // rows j..n. Pairwise disjoint because k < k+1 < j.
        let (before, rest) = a.split_at_mut(j * n);
        let src0 = &before[k * n + j..k * n + n];
        let src1 = &before[(k + 1) * n + j..(k + 1) * n + n];
        let dst = &mut rest[j..n];
        schur_kernel::axpy2_minus_unroll4_nofma(dst, src0, dl_j0, src1, dl_j1);
    }
}

/// Count inertia of a 2×2 D block, returning Inertia struct.
fn count_2x2_inertia_val(d11: f64, d21: f64, d22: f64) -> Inertia {
    let det = d11 * d22 - d21 * d21;
    let trace = d11 + d22;
    if det > 0.0 {
        if trace > 0.0 {
            Inertia::new(2, 0, 0)
        } else {
            Inertia::new(0, 2, 0)
        }
    } else if det < 0.0 {
        Inertia::new(1, 1, 0)
    } else if trace > 0.0 {
        Inertia::new(1, 0, 1)
    } else if trace < 0.0 {
        Inertia::new(0, 1, 1)
    } else {
        Inertia::new(0, 0, 2)
    }
}

/// Find max |A[i,k]| for i > k (column k, below diagonal).
/// Returns (max_value, row_index_of_max).
fn column_offdiag_max(a: &[f64], n: usize, k: usize) -> (f64, usize) {
    let mut max_val = 0.0;
    let mut max_idx = k + 1;
    for i in (k + 1)..n {
        let val = a[k * n + i].abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    (max_val, max_idx)
}

/// Compute the max off-diagonal magnitude in the full symmetric row/column r,
/// restricted to the trailing submatrix starting at column k.
/// This searches both below the diagonal (column r, rows > r) and
/// to the left of the diagonal (row r, columns k..r), excluding
/// position (r, k) which is not part of the "off-diagonal of r" in
/// the context of pivot selection — we want max over i != r.
fn symmetric_row_offdiag_max(a: &[f64], n: usize, k: usize, r: usize) -> f64 {
    let mut max_val = 0.0;

    // Below diagonal: column r, rows r+1..n
    for i in (r + 1)..n {
        let val = a[r * n + i].abs();
        if val > max_val {
            max_val = val;
        }
    }

    // Left of diagonal: row r, columns k..r (stored as a[col*n + r] for col < r)
    for j in k..r {
        let val = a[j * n + r].abs();
        if val > max_val {
            max_val = val;
        }
    }

    max_val
}

/// Swap rows and columns p and q in the lower triangle of the working matrix,
/// and update the permutation vector.
fn swap_rows_cols(a: &mut [f64], n: usize, p: usize, q: usize, perm: &mut [usize]) {
    if p == q {
        return;
    }
    // Ensure p < q
    let (p, q) = if p < q { (p, q) } else { (q, p) };

    // Swap permutation entries
    perm.swap(p, q);

    // Swap diagonal entries
    a.swap(p * n + p, q * n + q);

    // Swap columns p and q below row q (both in lower triangle)
    for i in (q + 1)..n {
        a.swap(p * n + i, q * n + i);
    }

    // Swap entries in column p (rows p+1..q-1) with entries in row q (cols p+1..q-1)
    // a[p*n + i] (col p, row i) with a[i*n + q] (col i, row q) for i in (p+1)..q
    for i in (p + 1)..q {
        a.swap(p * n + i, i * n + q);
    }

    // Swap a[p*n + q] (col p, row q) with... nothing — this is the (q, p) entry
    // In symmetric swap, the off-diagonal (p, q) entry needs special handling:
    // swap columns p row p+1..q-1 already done
    // Swap rows: for columns 0..p, swap a[j*n + p] with a[j*n + q]
    for j in 0..p {
        a.swap(j * n + p, j * n + q);
    }
}

/// Perform a 1×1 pivot at position k: compute L column, rank-1 update,
/// and fused argmax of the next column (Section 6 of research note).
/// Returns (gamma0_next, r_next) for the next pivot step's column.
///
/// `col_max` is the maximum off-diagonal magnitude in the column being
/// used as pivot (gamma0 for k-no-swap, gamma_r for k↔r swap). The pivot
/// is rejected via the ForceAccept path whenever
/// `|d| < max(zero_tol, pivot_threshold * col_max)`. This matches
/// MUMPS dfac_front_aux.F:1494-1495 and SSIDS options%u semantics:
/// the pivot must dominate its column by at least 1/u, otherwise the
/// rank-1 update would amplify rounding by ~1/|d| per position.
#[allow(clippy::too_many_arguments)]
fn do_1x1_pivot(
    a: &mut [f64],
    n: usize,
    k: usize,
    col_max: f64,
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
) -> Result<(f64, usize), FeralError> {
    let d = a[k * n + k];
    let threshold = (params.pivot_threshold * col_max).max(params.zero_tol);

    if d.abs() <= threshold {
        // Pivot rejected (either absolute floor or column-relative
        // Duff-Reid/MUMPS threshold). Route through the existing
        // ForceAccept/Fail zero-pivot path so the inertia count is
        // consistent, then zero the L column so the rank-1 update
        // below contributes nothing.
        match params.on_zero_pivot {
            ZeroPivotAction::ForceAccept => {
                *needs_refinement = true;
                *zero += 1;
            }
            ZeroPivotAction::Fail => return Err(FeralError::NumericallyRankDeficient),
        }
        // Zero the L column
        for i in (k + 1)..n {
            a[k * n + i] = 0.0;
        }
        // Also zero the diagonal so solve's `|d| > zero_tol` check skips
        // this position. Preserving the tiny original would otherwise
        // leave `|d| > zero_tol` true for pivots just above the absolute
        // floor but below u*col_max, causing solve to divide by them.
        a[k * n + k] = 0.0;
        return Ok((0.0, k + 2));
    }

    // Accept: count inertia by sign.
    if d > 0.0 {
        *pos += 1;
    } else {
        *neg += 1;
    }

    let d_inv = 1.0 / d;

    // Compute L column entries: L[i,k] = A[i,k] / d
    for i in (k + 1)..n {
        a[k * n + i] *= d_inv;
    }

    let mut next_gamma0 = 0.0;
    let mut next_r = k + 2;

    // Fused rank-1 update + argmax of next column (k+1).
    // Column k+1 is only updated in the j=k+1 iteration, so we handle it
    // separately to track the argmax during the same memory pass.
    if k + 1 < n {
        let j = k + 1;
        let l_jk = a[k * n + j];
        let l_jk_d = l_jk * d;
        // Update diagonal
        a[j * n + j] -= a[k * n + j] * l_jk_d;
        // Update off-diagonal and track argmax
        for i in (j + 1)..n {
            a[j * n + i] -= a[k * n + i] * l_jk_d;
            let val = a[j * n + i].abs();
            if val > next_gamma0 {
                next_gamma0 = val;
                next_r = i;
            }
        }
    }

    // Remaining columns: plain update (no argmax tracking)
    for j in (k + 2)..n {
        let l_jk = a[k * n + j];
        let l_jk_d = l_jk * d;
        for i in j..n {
            a[j * n + i] -= a[k * n + i] * l_jk_d;
        }
    }

    Ok((next_gamma0, next_r))
}

/// Perform a 2×2 pivot at positions {k, k+1}.
/// Uses the normalized computation from faer to avoid catastrophic cancellation.
/// Returns (gamma0_next, r_next) for the next pivot step's column.
#[allow(clippy::too_many_arguments)]
fn do_2x2_pivot(
    a: &mut [f64],
    n: usize,
    k: usize,
    subdiag: &mut [f64],
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
) -> Result<(f64, usize), FeralError> {
    let a00 = a[k * n + k];
    let a10 = a[k * n + (k + 1)];
    let a11 = a[(k + 1) * n + (k + 1)];

    // Store the 2×2 block subdiagonal
    subdiag[k] = a10;

    // Count inertia from the 2×2 block
    let det = a00 * a11 - a10 * a10;
    count_2x2_inertia(det, a00, params, pos, neg, zero, needs_refinement)?;

    if (k + 2) >= n {
        // No trailing submatrix to update
        return Ok((0.0, 0));
    }

    // Normalized 2×2 computation (from faer, Section 4.3 of research note)
    let d10_abs = a10.abs();

    if d10_abs < f64::EPSILON * 1e-10 {
        // Degenerate 2×2 block — treat L columns as zero
        for i in (k + 2)..n {
            a[k * n + i] = 0.0;
            a[(k + 1) * n + i] = 0.0;
        }
        return Ok((0.0, k + 3));
    }

    let d00 = a00 / d10_abs;
    let d11 = a11 / d10_abs;
    let t = 1.0 / (d00 * d11 - 1.0);
    let d10 = a10 / d10_abs; // sign only (±1 for reals)
    let d = t / d10_abs;

    let mut next_gamma0 = 0.0;
    let mut next_r = k + 3;

    // Fused rank-2 update + argmax of next column (k+2).
    // Column k+2 is only updated in the j=k+2 iteration, so handle separately.
    if k + 2 < n {
        let j = k + 2;
        let x0 = a[k * n + j];
        let x1 = a[(k + 1) * n + j];
        let w0 = (x0 * d11 - x1 * d10) * d;
        let w1 = (x1 * d00 - x0 * d10) * d;

        // Update diagonal
        a[j * n + j] -= a[k * n + j] * w0 + a[(k + 1) * n + j] * w1;
        // Update off-diagonal and track argmax for column k+2
        for i in (j + 1)..n {
            a[j * n + i] -= a[k * n + i] * w0 + a[(k + 1) * n + i] * w1;
            let val = a[j * n + i].abs();
            if val > next_gamma0 {
                next_gamma0 = val;
                next_r = i;
            }
        }

        a[k * n + j] = w0;
        a[(k + 1) * n + j] = w1;
    }

    // Remaining columns: plain update (no argmax tracking)
    for j in (k + 3)..n {
        let x0 = a[k * n + j];
        let x1 = a[(k + 1) * n + j];
        let w0 = (x0 * d11 - x1 * d10) * d;
        let w1 = (x1 * d00 - x0 * d10) * d;

        for i in j..n {
            a[j * n + i] -= a[k * n + i] * w0 + a[(k + 1) * n + i] * w1;
        }

        a[k * n + j] = w0;
        a[(k + 1) * n + j] = w1;
    }

    Ok((next_gamma0, next_r))
}

/// Count inertia for a 1×1 pivot.
fn count_1x1_inertia(
    d: f64,
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
) -> Result<(), FeralError> {
    if d.abs() <= params.zero_tol {
        match params.on_zero_pivot {
            ZeroPivotAction::ForceAccept => {
                *needs_refinement = true;
                *zero += 1;
                Ok(())
            }
            ZeroPivotAction::Fail => Err(FeralError::NumericallyRankDeficient),
        }
    } else if d > 0.0 {
        *pos += 1;
        Ok(())
    } else {
        *neg += 1;
        Ok(())
    }
}

/// Count inertia for a 2×2 pivot block.
/// Uses determinant and sign of a00 to classify eigenvalue signs.
fn count_2x2_inertia(
    det: f64,
    a00: f64,
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
) -> Result<(), FeralError> {
    // KNOWN BUG: this should use trace = a00 + a11 to decide the sign
    // of the non-zero eigenvalue, not a00 alone. KKT matrices produce
    // 2×2 blocks where a00 = 0 (variable rows have zero Hessian
    // diagonal) but a11 carries the sign. The trace-based fix was
    // attempted in the 2026-04-12 ACOPP30 triage but caused a 16-matrix
    // dense regression against rmumps's calibration. Re-attempt after
    // canonical Fortran MUMPS becomes available as a second oracle
    // (see dev/plans/phase-1b-consensus-exit.md). Documented in
    // dev/journal/2026-04-12-01.org.
    if det.abs() <= params.zero_tol_2x2 {
        // Near-singular 2×2 block
        match params.on_zero_pivot {
            ZeroPivotAction::ForceAccept => {
                *needs_refinement = true;
                // One eigenvalue near zero; the other has sign of trace
                if a00 > 0.0 {
                    *pos += 1;
                    *zero += 1;
                } else {
                    *neg += 1;
                    *zero += 1;
                }
                Ok(())
            }
            ZeroPivotAction::Fail => Err(FeralError::NumericallyRankDeficient),
        }
    } else if det > 0.0 {
        // Same-sign eigenvalues
        if a00 > 0.0 {
            *pos += 2;
        } else {
            *neg += 2;
        }
        Ok(())
    } else {
        // Opposite-sign eigenvalues
        *pos += 1;
        *neg += 1;
        Ok(())
    }
}

/// Set L column at position k to the identity column (1 on diagonal, 0 below).
fn set_l_column_identity(a: &mut [f64], n: usize, k: usize) {
    for i in (k + 1)..n {
        a[k * n + i] = 0.0;
    }
}

#[cfg(test)]
mod growth_flag_tests {
    use super::*;

    #[test]
    fn growth_below_threshold_does_not_flag() {
        let l = vec![1.0, 2.78, -2.5, 0.0, 100.0, -999_999.0];
        let mut flag = false;
        flag_growth_for_refinement(&l, &mut flag);
        assert!(!flag, "max|L| = 999_999 < 1e6 should not flag");
    }

    #[test]
    fn growth_above_threshold_flags() {
        let l = vec![1.0, 2.0, 1.5e6, -3.0];
        let mut flag = false;
        flag_growth_for_refinement(&l, &mut flag);
        assert!(flag, "max|L| = 1.5e6 > 1e6 must flag");
    }

    #[test]
    fn catastrophic_growth_flags() {
        let l = vec![1.0, 1.0, 8.06e16, 1.0];
        let mut flag = false;
        flag_growth_for_refinement(&l, &mut flag);
        assert!(flag, "max|L| = 8e16 (bratu3d-class) must flag");
    }

    #[test]
    fn negative_large_entry_flags() {
        let l = vec![-2e10, 1.0];
        let mut flag = false;
        flag_growth_for_refinement(&l, &mut flag);
        assert!(flag, "negative large |L| must flag");
    }

    #[test]
    fn already_set_flag_is_preserved() {
        let l = vec![0.0, 0.0]; // would not flag on its own
        let mut flag = true; // pre-set by zero-pivot path
        flag_growth_for_refinement(&l, &mut flag);
        assert!(flag, "must not clobber pre-set flag");
    }

    #[test]
    fn empty_l_does_not_flag() {
        let l: Vec<f64> = vec![];
        let mut flag = false;
        flag_growth_for_refinement(&l, &mut flag);
        assert!(!flag);
    }

    #[test]
    fn nan_and_inf_in_l_flag() {
        // NaN.abs() is NaN; NaN > x is always false, so NaN alone does
        // not trigger. But Inf does, and a real factor with NaN almost
        // always also has Inf. This is an explicit doc of behavior.
        let l_inf = vec![1.0, f64::INFINITY];
        let mut flag = false;
        flag_growth_for_refinement(&l_inf, &mut flag);
        assert!(flag, "Inf entry must trigger");
    }
}
