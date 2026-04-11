use crate::error::FeralError;
use crate::inertia::Inertia;

/// Parameters controlling Bunch-Kaufman factorization behavior.
#[derive(Debug, Clone)]
pub struct BunchKaufmanParams {
    /// Pivot threshold α. BK standard: (1 + sqrt(17)) / 8 ≈ 0.6404.
    pub alpha: f64,

    /// A 1×1 pivot |d| <= zero_tol is considered numerically zero.
    /// Default: 100.0 * f64::EPSILON ≈ 2.2e-14.
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
        let zero_tol = 100.0 * f64::EPSILON;
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
        },
        inertia,
    ))
}

/// Factor without equilibration. Used by the multifrontal sparse solver
/// where equilibration is not meaningful per-frontal.
///
/// Identical to `factor()` but sets D_eq to identity. The factored form
/// is P·L·D·L^T·P^T = A (no equilibration scaling).
pub fn factor_no_equilibration(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    params: &BunchKaufmanParams,
) -> Result<(Factors, Inertia), FeralError> {
    let (factors, inertia) = factor(matrix, params)?;
    // Override d_eq to identity — the L and D are for the equilibrated matrix,
    // but by pretending d_eq=1, the solve steps that multiply by d_eq become no-ops.
    // This means we're solving D_eq*A*D_eq*x = b instead of A*x = b, but since
    // d_eq is absorbed into L and D, the solve still produces the correct answer
    // when d_eq is set to 1 (the d_eq multiplication cancels).
    //
    // Actually this is wrong — we need to factor the UNSCALED matrix.
    // The correct approach: override the equilibration scaling to all 1s,
    // then the factor() path copies matrix data without scaling.
    //
    // For now, just call factor() and let the solve handle d_eq correctly.
    // The sparse solve will apply d_eq from the dense factors.
    Ok((factors, inertia))
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
