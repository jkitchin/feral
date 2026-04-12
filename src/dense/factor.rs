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

/// Result of partial frontal factorization for the multifrontal solver.
#[derive(Debug)]
pub struct FrontalFactors {
    /// Number of rows in the frontal (nrow).
    pub nrow: usize,
    /// Number of eliminated columns (ncol <= nrow).
    pub ncol: usize,
    /// L factor: nrow × ncol column-major. Unit diagonal (implicit).
    /// L[j*nrow + i] for i in [0, nrow), j in [0, ncol).
    pub l: Vec<f64>,
    /// D block diagonal (length ncol).
    pub d_diag: Vec<f64>,
    /// D block subdiagonal for 2×2 pivots (length ncol).
    pub d_subdiag: Vec<f64>,
    /// BK pivot permutation within the first ncol rows.
    /// perm[i] = j means original row j was moved to pivot position i.
    /// Only indices 0..ncol are permuted; ncol..nrow are identity.
    pub perm: Vec<usize>,
    /// Inverse permutation.
    pub perm_inv: Vec<usize>,
    /// Schur complement (contribution block): cdim × cdim column-major
    /// where cdim = nrow - ncol. Lower triangle only.
    /// S = A22 - L21 * D * L21^T
    pub contrib: Vec<f64>,
    /// Dimension of the contribution block.
    pub contrib_dim: usize,
    /// Inertia of the ncol eliminated pivots.
    pub inertia: Inertia,
    /// Whether ForceAccept fired during factorization.
    pub needs_refinement: bool,
    /// 1×1 pivot threshold from BunchKaufmanParams (see Factors::zero_tol).
    pub zero_tol: f64,
    /// 2×2 pivot threshold from BunchKaufmanParams.
    pub zero_tol_2x2: f64,
}

/// Factor a frontal matrix, eliminating only the first `ncol` columns.
///
/// This is the key dense kernel for the multifrontal solver. Unlike `factor()`,
/// pivot search is RESTRICTED to the first `ncol` rows/columns. Rows ncol..nrow
/// are never swapped into pivot positions, preserving their ordering for the
/// contribution block.
///
/// After eliminating ncol pivots, the Schur complement S = A22 - L21*D*L21^T
/// is computed and returned. No equilibration is applied.
pub fn factor_frontal(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    ncol: usize,
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
            l: Vec::new(),
            d_diag: Vec::new(),
            d_subdiag: Vec::new(),
            perm: (0..nrow).collect(),
            perm_inv: (0..nrow).collect(),
            contrib: matrix.data.clone(),
            contrib_dim: nrow,
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
            // Last eliminated pivot: always 1×1
            let d = a[k * nrow + k];
            count_1x1_inertia(
                d,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            // Scale column k below diagonal (including rows ncol..nrow)
            if d.abs() > 0.0 {
                let inv_d = 1.0 / d;
                for i in (k + 1)..nrow {
                    a[k * nrow + i] *= inv_d;
                }
            }
            // Rank-1 update on trailing matrix
            for j in (k + 1)..nrow {
                let l_jk = a[k * nrow + j];
                for i in j..nrow {
                    a[j * nrow + i] -= l_jk * d * a[k * nrow + i];
                }
            }
            // Set L column: diagonal = 1, above diagonal = 0
            a[k * nrow + k] = d; // keep D on diagonal
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
            let d = a[k * nrow + k];
            count_1x1_inertia(
                d,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            do_1x1_update(&mut a, nrow, k);
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
            let d = a[k * nrow + k];
            count_1x1_inertia(
                d,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            do_1x1_update(&mut a, nrow, k);
            k += 1;
            continue;
        }

        if akk * gamma_r >= alpha * gamma0 * gamma0 {
            // 1×1 pivot at k (LAPACK extension), no swap
            let d = a[k * nrow + k];
            count_1x1_inertia(
                d,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            do_1x1_update(&mut a, nrow, k);
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

            if det.abs() <= params.zero_tol_2x2 {
                match params.on_zero_pivot {
                    ZeroPivotAction::Fail => return Err(FeralError::NumericallyRankDeficient),
                    ZeroPivotAction::ForceAccept => {
                        needs_refinement = true;
                    }
                }
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
            // Can't do 2×2 (r is not fully summed or only 1 column left)
            // Fall back to 1×1 with ForceAccept
            let d = a[k * nrow + k];
            count_1x1_inertia(
                d,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            do_1x1_update(&mut a, nrow, k);
            k += 1;
        }
    }

    // Extract L (nrow × ncol), D diagonal, and contribution block
    let mut l = vec![0.0; nrow * ncol];
    let mut d_diag = vec![0.0; ncol];

    let mut j = 0;
    while j < ncol {
        d_diag[j] = a[j * nrow + j];
        l[j * nrow + j] = 1.0; // unit diagonal

        if j + 1 < ncol && subdiag[j] != 0.0 {
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

    // Extract contribution block: trailing (nrow-ncol) × (nrow-ncol) of a
    let cdim = nrow - ncol;
    let mut contrib = vec![0.0; cdim * cdim];
    for cj in 0..cdim {
        for ci in cj..cdim {
            contrib[cj * cdim + ci] = a[(ncol + cj) * nrow + (ncol + ci)];
        }
    }

    let mut perm_inv = vec![0usize; nrow];
    for (i, &p) in perm.iter().enumerate() {
        perm_inv[p] = i;
    }

    Ok(FrontalFactors {
        nrow,
        ncol,
        l,
        d_diag,
        d_subdiag: subdiag[..ncol].to_vec(),
        perm,
        perm_inv,
        contrib,
        contrib_dim: cdim,
        inertia: Inertia::new(pos, neg, zero),
        needs_refinement,
        zero_tol: params.zero_tol,
        zero_tol_2x2: params.zero_tol_2x2,
    })
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
#[allow(clippy::too_many_arguments)]
fn do_1x1_pivot(
    a: &mut [f64],
    n: usize,
    k: usize,
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
) -> Result<(f64, usize), FeralError> {
    let d = a[k * n + k];
    count_1x1_inertia(d, params, pos, neg, zero, needs_refinement)?;

    if d.abs() <= params.zero_tol {
        // Near-zero pivot accepted (ForceAccept) — set L column to zero
        for i in (k + 1)..n {
            a[k * n + i] = 0.0;
        }
        return Ok((0.0, k + 2));
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
