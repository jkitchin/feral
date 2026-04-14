use crate::error::FeralError;
use crate::inertia::Inertia;

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

    let front = factor_frontal(&eq_matrix, n, false, params)?;

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
    /// 1×1 pivot threshold from BunchKaufmanParams (see Factors::zero_tol).
    pub zero_tol: f64,
    /// 2×2 pivot threshold from BunchKaufmanParams.
    pub zero_tol_2x2: f64,
}

/// Outcome of an attempt to accept a 1×1 pivot via `try_reject_1x1_frontal`.
/// The caller uses this to decide whether to continue, force-accept, or
/// break out of the BK loop (SSIDS-style delayed pivoting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub fn factor_frontal(
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
            zero_tol: params.zero_tol,
            zero_tol_2x2: params.zero_tol_2x2,
        });
    }

    // Copy lower triangle into work array (no equilibration)
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

    let alpha = params.alpha;
    let mut k = 0;

    // Factor only the first ncol columns. Pivot search restricted to [k, ncol).
    while k < ncol {
        let remaining = ncol - k;

        if remaining == 1 {
            // Last eliminated pivot: always 1×1. Compute the column max
            // over rows (k+1..nrow) for the column-relative threshold.
            let mut col_max = 0.0f64;
            for i in (k + 1)..nrow {
                let v = a[k * nrow + i].abs();
                if v > col_max {
                    col_max = v;
                }
            }
            let outcome = try_reject_1x1_frontal(
                &mut a,
                nrow,
                k,
                col_max,
                may_delay,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            match outcome {
                PivotOutcome::Accepted => {
                    let d = a[k * nrow + k];
                    // Scale column k below diagonal (including rows ncol..nrow)
                    let inv_d = 1.0 / d;
                    for i in (k + 1)..nrow {
                        a[k * nrow + i] *= inv_d;
                    }
                    // Rank-1 update on trailing matrix
                    for j in (k + 1)..nrow {
                        let l_jk = a[k * nrow + j];
                        for i in j..nrow {
                            a[j * nrow + i] -= l_jk * d * a[k * nrow + i];
                        }
                    }
                    // Keep D on diagonal
                    a[k * nrow + k] = d;
                }
                PivotOutcome::Rejected => {}
                PivotOutcome::Delayed => break,
            }
            k += 1;
            continue;
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
            count_1x1_inertia(
                d,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            set_l_column_identity(&mut a, nrow, k);
            k += 1;
            continue;
        }

        let akk = a[k * nrow + k].abs();

        if akk >= alpha * gamma0 {
            // 1×1 pivot at k, no swap
            let outcome = try_reject_1x1_frontal(
                &mut a,
                nrow,
                k,
                gamma0,
                may_delay,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            match outcome {
                PivotOutcome::Accepted => do_1x1_update(&mut a, nrow, k),
                PivotOutcome::Rejected => {}
                PivotOutcome::Delayed => break,
            }
            k += 1;
            continue;
        }

        // gamma_r: max off-diagonal in symmetric row r
        let gamma_r = symmetric_row_offdiag_max(&a, nrow, k, r);
        let arr = a[r * nrow + r].abs();

        // Can we swap r into pivot position? Only if r < ncol (fully summed)
        let r_is_fully_summed = r < ncol;

        if r_is_fully_summed && arr >= alpha * gamma_r {
            // 1×1 pivot at r, swap r↔k
            swap_rows_cols(&mut a, nrow, k, r, &mut perm);
            let outcome = try_reject_1x1_frontal(
                &mut a,
                nrow,
                k,
                gamma_r,
                may_delay,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            match outcome {
                PivotOutcome::Accepted => do_1x1_update(&mut a, nrow, k),
                PivotOutcome::Rejected => {}
                PivotOutcome::Delayed => break,
            }
            k += 1;
            continue;
        }

        if akk * gamma_r >= alpha * gamma0 * gamma0 {
            // 1×1 pivot at k (LAPACK extension), no swap
            let outcome = try_reject_1x1_frontal(
                &mut a,
                nrow,
                k,
                gamma0,
                may_delay,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            match outcome {
                PivotOutcome::Accepted => do_1x1_update(&mut a, nrow, k),
                PivotOutcome::Rejected => {}
                PivotOutcome::Delayed => break,
            }
            k += 1;
            continue;
        }

        if r_is_fully_summed && k + 1 < ncol {
            // 2×2 pivot using {k, r}, both fully summed
            if r != k + 1 {
                swap_rows_cols(&mut a, nrow, k + 1, r, &mut perm);
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

            let det_floor_fail = absdet <= params.zero_tol_2x2;

            if growth_fail || det_floor_fail {
                // 2×2 rejected. SSIDS-style delayed pivoting: when
                // `may_delay == true`, break out immediately so the parent
                // supernode can retry this pivot with a larger pivot search
                // window. Otherwise fall back to a single 1×1 at k with the
                // column-relative threshold, which triggers the existing
                // ForceAccept path.
                if may_delay {
                    break;
                }
                if det_floor_fail {
                    match params.on_zero_pivot {
                        ZeroPivotAction::Fail => {
                            return Err(FeralError::NumericallyRankDeficient);
                        }
                        ZeroPivotAction::ForceAccept => {
                            needs_refinement = true;
                        }
                    }
                }
                let outcome = try_reject_1x1_frontal(
                    &mut a,
                    nrow,
                    k,
                    gamma0,
                    may_delay,
                    params,
                    &mut pos,
                    &mut neg,
                    &mut zero,
                    &mut needs_refinement,
                )?;
                match outcome {
                    PivotOutcome::Accepted => do_1x1_update(&mut a, nrow, k),
                    PivotOutcome::Rejected => {}
                    PivotOutcome::Delayed => break,
                }
                k += 1;
                continue;
            }

            let pivot_inertia = count_2x2_inertia_val(d11, d21, d22);
            pos += pivot_inertia.positive;
            neg += pivot_inertia.negative;
            zero += pivot_inertia.zero;

            subdiag[k] = d21;
            // 2×2 update
            do_2x2_update(&mut a, nrow, k, d11, d21, d22);
            k += 2;
        } else {
            // Can't do 2×2 (r is not fully summed or only 1 column left).
            // Last-resort 1×1 at k with column-relative rejection.
            let outcome = try_reject_1x1_frontal(
                &mut a,
                nrow,
                k,
                gamma0,
                may_delay,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            match outcome {
                PivotOutcome::Accepted => do_1x1_update(&mut a, nrow, k),
                PivotOutcome::Rejected => {}
                PivotOutcome::Delayed => break,
            }
            k += 1;
        }
    }

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
        zero_tol: params.zero_tol,
        zero_tol_2x2: params.zero_tol_2x2,
    })
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

/// 1×1 rank-1 update: update columns k+1..nrow after eliminating column k.
fn do_1x1_update(a: &mut [f64], n: usize, k: usize) {
    let d = a[k * n + k];
    if d.abs() == 0.0 {
        return;
    }
    let inv_d = 1.0 / d;
    // Scale L column
    for i in (k + 1)..n {
        a[k * n + i] *= inv_d;
    }
    // Rank-1 update
    for j in (k + 1)..n {
        let l_jk = a[k * n + j];
        for i in j..n {
            a[j * n + i] -= l_jk * d * a[k * n + i];
        }
    }
}

/// 2×2 rank-2 update: update columns k+2..nrow after eliminating columns k, k+1.
fn do_2x2_update(a: &mut [f64], n: usize, k: usize, d11: f64, d21: f64, d22: f64) {
    let det = d11 * d22 - d21 * d21;
    if det.abs() == 0.0 {
        return;
    }
    let inv_det = 1.0 / det;

    // Compute L columns k and k+1: L = A * D^{-1}
    for i in (k + 2)..n {
        let a_ik = a[k * n + i];
        let a_ik1 = a[(k + 1) * n + i];
        a[k * n + i] = (d22 * a_ik - d21 * a_ik1) * inv_det;
        a[(k + 1) * n + i] = (d11 * a_ik1 - d21 * a_ik) * inv_det;
    }

    // Rank-2 update: A_trailing -= L * D * L^T
    for j in (k + 2)..n {
        let l_j0 = a[k * n + j];
        let l_j1 = a[(k + 1) * n + j];
        // D * L^T row j = [d11*l_j0 + d21*l_j1, d21*l_j0 + d22*l_j1]
        let dl_j0 = d11 * l_j0 + d21 * l_j1;
        let dl_j1 = d21 * l_j0 + d22 * l_j1;
        for i in j..n {
            let l_i0 = a[k * n + i];
            let l_i1 = a[(k + 1) * n + i];
            a[j * n + i] -= l_i0 * dl_j0 + l_i1 * dl_j1;
        }
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
