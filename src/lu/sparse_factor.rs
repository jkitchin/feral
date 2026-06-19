//! Sparse unsymmetric LU factorization (left-looking, Gilbert–Peierls style).
//!
//! Factors `P A Q = L U` with threshold partial pivoting. `Q` is the
//! fill-reducing column ordering from [`SparseLuSymbolic`]; `P` is the row
//! permutation from pivoting. `L` is unit lower triangular (strict-lower stored
//! CSC, unit diagonal implicit); `U` is upper triangular stored row-wise (CSR,
//! strict-upper, plus an explicit diagonal) so the Forrest–Tomlin update can do
//! row operations on it. Both are in pivot-position coordinates.
//!
//! Each column `k` (processing original column `qcol[k]`) is computed by a
//! forward substitution `L w = A(:,qcol[k])`, output-sensitive via a
//! Gilbert–Peierls depth-first reach: the numeric solve runs only over the
//! pivot positions reachable from the nonzeros of `A(:,qcol[k])`.

use super::scaling::{compute_lu_scale, LuScale};
use super::sparse_symbolic::SparseLuSymbolic;
use super::{LuParams, LuScaling, LuSingularAction};
use crate::error::FeralError;
use crate::lu::sparse_matrix::SparseColMatrix;

/// One elementary operation of a Forrest–Tomlin update's bump elimination,
/// recorded so it can be replayed on a solve vector (and transposed for btran).
#[derive(Debug, Clone)]
pub(super) enum FtOp {
    /// Swap two pivot positions (partial-pivot row interchange in the bump).
    Swap(usize, usize),
    /// `target_row -= mult · src_row` (Gauss elimination of a sub-diagonal).
    Axpy {
        target: usize,
        src: usize,
        mult: f64,
    },
}

/// One Forrest–Tomlin column-replacement update: the sequence of elementary row
/// operations (partial-pivot swaps + eliminations) that re-triangularized `U`
/// after the spike was inserted. The base `L` is never touched; these ops are
/// replayed on the solve vector between the `L`-solve and the `U`-solve in
/// `ftran` (transposed, in reverse, between `Uᵀ` and `Lᵀ` in `btran`). Because
/// the bump is local and sparse, each eta is `O(bump)` — no dense `τ`.
#[derive(Debug, Clone)]
pub(super) struct FtEta {
    pub ops: Vec<FtOp>,
}

impl FtEta {
    /// Apply this elimination `E` forward to `y` (`y ← E y`).
    pub(super) fn apply_forward(&self, y: &mut [f64]) {
        for op in self.ops.iter() {
            match *op {
                FtOp::Swap(a, b) => y.swap(a, b),
                FtOp::Axpy { target, src, mult } => y[target] -= mult * y[src],
            }
        }
    }

    /// Apply `Eᵀ` to `y` (reverse the ops, transpose each).
    pub(super) fn apply_transpose(&self, y: &mut [f64]) {
        for op in self.ops.iter().rev() {
            match *op {
                FtOp::Swap(a, b) => y.swap(a, b),
                FtOp::Axpy { target, src, mult } => y[src] -= mult * y[target],
            }
        }
    }
}

/// Sparse LU factorization of a square basis.
#[derive(Debug, Clone)]
pub struct SparseLu {
    pub(super) m: usize,
    // L: unit lower triangular, strict-lower stored CSC, pivot-position rows.
    pub(super) l_col_ptr: Vec<usize>,
    pub(super) l_row_idx: Vec<usize>,
    pub(super) l_val: Vec<f64>,
    // U: upper triangular, one sorted row per pivot position, each row a list
    // of `(column_position, value)` with column >= row and the diagonal entry
    // FIRST (column == row). Mutable per-row storage so the Forrest–Tomlin
    // update can do in-place row operations.
    pub(super) u_rows: Vec<Vec<(usize, f64)>>,
    /// `perm[k]` = original row in pivot position `k` (`(P a)[k] = a[perm[k]]`).
    pub(super) perm: Vec<usize>,
    /// Inverse of `perm`: `perm_inv[orig_row] = pivot_position`. Used to seed the
    /// sparse spike solve in the Forrest–Tomlin update.
    pub(super) perm_inv: Vec<usize>,
    /// Column order: factorization column `k` is original column `qcol[k]`.
    pub(super) qcol: Vec<usize>,
    /// Inverse of `qcol`: `qcol_inv[original_col] = column_position`.
    pub(super) qcol_inv: Vec<usize>,
    /// Column-wise index of `U`'s strict-upper part: `u_above[c]` is the sorted
    /// list of rows `i < c` with `U[i,c] != 0`. Lets the FT update find column
    /// `r`'s existing entries (for in-place replacement) without scanning all
    /// rows. Maintained across updates; not used by the solves.
    pub(super) u_above: Vec<Vec<usize>>,
    /// Forrest–Tomlin column-replacement updates applied since the last
    /// factor/refactor. Each is a replayable bump elimination (`O(bump)`), so
    /// warm solves stay sparse (no dense eta chain).
    pub(super) etas: Vec<FtEta>,
    /// Running growth monitor: the ‖U‖∞ element-growth high-water ratio
    /// (largest `max|U|` over the updates ÷ [`Self::u_max0`]). Compounds across
    /// a chain of updates, unlike a max-single-multiplier monitor (L5).
    pub(super) growth: f64,
    /// `max|U|` immediately after factor — denominator of the element-growth
    /// monitor. Floored away from zero.
    pub(super) u_max0: f64,
    /// Total Gilbert–Peierls reach nodes visited during the factor — a
    /// structural scalability witness (`O(nnz(U))`, not `O(n²)`).
    pub(super) reach_visits: usize,
    pub(super) params: LuParams,
    /// Two-sided scaling of the factored matrix (identity when unscaled).
    pub(super) scale: LuScale,
    /// Inverse of `scale.rperm` (original row -> scaled-row position), so the
    /// sparse FT update can map an entering column's nonzeros into spike space
    /// without an `O(n)` scan.
    pub(super) scale_rperm_inv: Vec<usize>,
    pub(super) scratch: Vec<f64>,
    /// Reusable length-`m` boolean marker for the FT update's sparse spike
    /// (tracks touched positions so they can be cleared in `O(touched)`).
    pub(super) scratch_mark: Vec<bool>,
    /// Dedicated length-`m` work buffer for the FT update's sparse spike, kept
    /// zeroed between updates. Separate from `scratch` (which the solves dirty),
    /// so `compute_spike`'s sparse scatter can assume a clean buffer.
    pub(super) ft_work: Vec<f64>,
    /// Pooled length-`m` buffer for the scaled `ftran`/`btran` wrappers' inner
    /// RHS (`bt`); distinct from `scratch`, which the core solve dirties (L3).
    pub(super) scratch_b: Vec<f64>,
    /// Pooled length-`m` residual buffer for iterative refinement (`r`);
    /// distinct from `scratch`/`scratch_b`, which the inner solve uses (L3).
    pub(super) scratch_c: Vec<f64>,
    /// Pooled length-`m` buffer holding the refinement's original-RHS snapshot
    /// (`a`); live across the whole refine loop, so it cannot reuse the others.
    pub(super) scratch_d: Vec<f64>,
    /// FT-update bump-loop scratch pools (see
    /// `dev/research/lu-update-alloc-pooling-2026-06-19.md`). All hold *churn*
    /// buffers — allocated and freed within an update — so reusing them removes
    /// per-pivot / per-axpy allocator traffic from `eliminate_bump`. Taken via
    /// `mem::take` into locals at the top of the update and restored on every
    /// return path; contents are cleared and refilled, never read across calls.
    /// `row_pool` is a free-list of `U`-row buffers recycled by `row_sub_into`.
    pub(super) pivot_scratch: Vec<(usize, f64)>,
    pub(super) targets_scratch: Vec<usize>,
    pub(super) row_pool: Vec<Vec<(usize, f64)>>,
    pub(super) col_rows_pool: Vec<Vec<usize>>,
    /// FT-update rollback/reindex snapshot pools. `saved_scratch` is the reused
    /// outer `(row, old_content)` vec; `saved_pool` is a free-list of the inner
    /// row buffers. The per-changed-row clone at the top of `update_sparse` was
    /// the dominant remaining per-update allocation after the bump-loop pools;
    /// recycling these buffers across updates removes it. On the rare rollback
    /// path the saved buffers move back into `u_rows` and a `NeedsRefactor`
    /// rebuilds the whole `SparseLu` (pools included), so no leak accumulates.
    pub(super) saved_scratch: Vec<(usize, Vec<(usize, f64)>)>,
    pub(super) saved_pool: Vec<Vec<(usize, f64)>>,
}

impl SparseLu {
    /// Factor `a` using the column ordering in `symbolic`.
    pub fn factor(
        a: &SparseColMatrix,
        symbolic: &SparseLuSymbolic,
        params: LuParams,
    ) -> Result<Self, FeralError> {
        params.validate()?;
        let m = a.m;
        if symbolic.m != m {
            return Err(FeralError::DimensionMismatch {
                expected: m,
                got: symbolic.m,
            });
        }
        // Scaling: factor Ã = D_row Π A D_col (pattern is invariant under row
        // permutation/scaling, so the column ordering `symbolic` still applies).
        let (scale, scaled) = if params.scaling == LuScaling::None {
            (LuScale::identity(m), None)
        } else {
            let scale = compute_lu_scale(a, params.scaling)?;
            let mat = scale.apply_sparse(a)?;
            (scale, Some(mat))
        };
        let a: &SparseColMatrix = scaled.as_ref().unwrap_or(a);

        let qcol = symbolic.qcol.clone();
        let qcol_inv = symbolic.qcol_inv.clone();

        let mut w = vec![0.0_f64; m];
        let mut mark = vec![false; m];
        let mut touched: Vec<usize> = Vec::new();
        let mut pinv: Vec<isize> = vec![-1; m];
        let mut perm = vec![0usize; m]; // pivot pos -> orig row

        let mut l_col_ptr = Vec::with_capacity(m + 1);
        let mut l_row_idx: Vec<usize> = Vec::new(); // original rows, remapped later
        let mut l_val: Vec<f64> = Vec::new();
        l_col_ptr.push(0);
        let mut u_col_ptr = Vec::with_capacity(m + 1);
        let mut u_row_idx: Vec<usize> = Vec::new();
        let mut u_val: Vec<f64> = Vec::new();
        u_col_ptr.push(0);
        let mut u_diag = vec![0.0_f64; m];

        // Gilbert–Peierls symbolic workspace: depth-first reach of each column.
        let mut reach_mark = vec![false; m];
        let mut reach: Vec<usize> = Vec::new();
        let mut dfs_stack: Vec<usize> = Vec::new();

        let utol = params.pivot_threshold;
        // L6 (dev/research/repo-review-2026-06-09.md): scale the zero-pivot
        // tolerance by the matrix magnitude, matching the dense path. An absolute
        // `zero_pivot_tol` declared a uniformly small but perfectly conditioned
        // basis singular and gave a large-magnitude basis effectively no
        // singularity detection. `a_max == 0` only for the (genuinely singular)
        // zero matrix, where the exact-zero test still fires.
        let a_max = a.values.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        let ztol = params.zero_pivot_tol * a_max;
        let mut reach_visits = 0usize;

        for k in 0..m {
            // Scatter A(:, qcol[k]) into w.
            let (rows, vals) = a.column(qcol[k]);
            for (&i, &v) in rows.iter().zip(vals.iter()) {
                w[i] = v;
                if !mark[i] {
                    mark[i] = true;
                    touched.push(i);
                }
            }

            // Symbolic: depth-first reach of column k over the graph of L.
            // A pivot position p < k contributes to column k iff its pivot row
            // is reachable from the nonzeros of A(:,qcol[k]); edges run from a
            // column to the pivot positions of its (already-pivoted) rows. All
            // edges point to strictly larger positions, so ascending position
            // order is a valid topological order for the numeric solve.
            reach.clear();
            for &i in rows.iter() {
                let pp = pinv[i];
                if pp >= 0 && !reach_mark[pp as usize] {
                    reach_mark[pp as usize] = true;
                    reach.push(pp as usize);
                    dfs_stack.push(pp as usize);
                }
            }
            while let Some(p) = dfs_stack.pop() {
                let (ls, le) = (l_col_ptr[p], l_col_ptr[p + 1]);
                for idx in ls..le {
                    let pp = pinv[l_row_idx[idx]];
                    if pp >= 0 && !reach_mark[pp as usize] {
                        reach_mark[pp as usize] = true;
                        reach.push(pp as usize);
                        dfs_stack.push(pp as usize);
                    }
                }
            }
            reach.sort_unstable();
            reach_visits += reach.len();

            // Numeric forward substitution over the reach; collect U(:,k).
            let mut u_entries: Vec<(usize, f64)> = Vec::with_capacity(reach.len());
            for &p in reach.iter() {
                reach_mark[p] = false; // clear for the next column
                let r = perm[p];
                let xp = w[r];
                if xp == 0.0 {
                    continue;
                }
                u_entries.push((p, xp));
                let (ls, le) = (l_col_ptr[p], l_col_ptr[p + 1]);
                for idx in ls..le {
                    let i = l_row_idx[idx];
                    w[i] -= xp * l_val[idx];
                    if !mark[i] {
                        mark[i] = true;
                        touched.push(i);
                    }
                }
            }

            // Pivot selection among unpivoted touched rows (partial pivoting).
            let mut amax = 0.0_f64;
            let mut ipiv: isize = -1;
            for &i in touched.iter() {
                if pinv[i] < 0 {
                    let av = w[i].abs();
                    if av > amax {
                        amax = av;
                        ipiv = i as isize;
                    }
                }
            }

            let mut piv;
            let pivot_row: usize;
            if amax <= ztol {
                match params.on_singular {
                    LuSingularAction::Fail => {
                        // L9 (dev/research/repo-review-2026-06-09.md): report the
                        // ORIGINAL basis column `qcol[k]`, not the internal
                        // factorization position `k`. The caller (e.g. a simplex
                        // driver) knows original columns, not the AMD-dependent
                        // processing order, so `qcol[k]` is the index it can act on.
                        return Err(FeralError::SingularBasis { column: qcol[k] });
                    }
                    LuSingularAction::PerturbToEps { abs_floor } => {
                        // L13 (dev/research/repo-review-2026-06-09.md): perturb the
                        // largest-|w| unpivoted row `ipiv` (the same row threshold
                        // partial pivoting would select), matching the dense path,
                        // which perturbs its partial-pivoting-selected row — not the
                        // index-first unpivoted row. Reusing `ipiv` also avoids the
                        // O(m) scan whenever the column has any touched unpivoted
                        // entry; the scan remains only as the fallback for a column
                        // that is structurally empty in every unpivoted row
                        // (`ipiv < 0`), where `w` is zero and any row will do.
                        let r = if ipiv >= 0 {
                            ipiv as usize
                        } else {
                            (0..m)
                                .find(|&i| pinv[i] < 0)
                                .ok_or(FeralError::SingularBasis { column: qcol[k] })?
                        };
                        pivot_row = r;
                        let s = if w[r] < 0.0 { -1.0 } else { 1.0 };
                        piv = s * abs_floor.max(w[r].abs());
                    }
                }
            } else {
                // L2 (dev/research/repo-review-2026-06-09.md): threshold partial
                // pivoting. The strict max-magnitude row `ipiv` is the stability
                // baseline; but if the natural diagonal of this column — original
                // row `qcol[k]` — is still unpivoted and is within `utol·amax` of
                // that max, prefer it. The diagonal pivot preserves structure
                // (less fill) without sacrificing more than a factor `utol` of
                // stability. `utol == 1.0` (the default) recovers strict partial
                // pivoting, since the diagonal must then equal the max to qualify.
                // This matches CSparse `cs_lu` (Davis, *Direct Methods for Sparse
                // Linear Systems*, §6.3): `if (pinv[col] < 0 && |x[col]| >= a*tol)
                // ipiv = col`.
                let diag = qcol[k];
                // The diagonal must also clear the singularity floor `ztol`,
                // not merely the threshold `utol·amax` (repo-review verification
                // residual #2). With a loose `utol` (≤ `zero_pivot_tol`) a
                // sub-`ztol` diagonal could otherwise satisfy `|w[diag]| ≥
                // utol·amax` and be preferred over the sound max-magnitude row
                // (`amax > ztol` always holds in this branch), then be silently
                // clamped below — a sub-tolerance perturbation that bypasses
                // `on_singular`. The dense path carries the same `&& diag > ztol`
                // conjunct (`dense_factor.rs`); this keeps the two paths in step.
                pivot_row =
                    if pinv[diag] < 0 && w[diag].abs() >= utol * amax && w[diag].abs() > ztol {
                        diag
                    } else {
                        ipiv as usize
                    };
                piv = w[pivot_row];
                // With the `> ztol` conjunct above and `amax > ztol` in this
                // branch, both pivot choices satisfy `|piv| > ztol`, so this
                // guard is unreachable in practice. It remains as defensive
                // parity with the dense path: should the invariant ever be
                // broken by a future change, a sub-floor pivot is routed through
                // `on_singular` (a loud `SingularBasis` under `Fail`, or an
                // accountable perturbation) rather than silently clamped.
                if piv.abs() <= ztol {
                    match params.on_singular {
                        LuSingularAction::Fail => {
                            return Err(FeralError::SingularBasis { column: qcol[k] });
                        }
                        LuSingularAction::PerturbToEps { abs_floor } => {
                            let s = if piv < 0.0 { -1.0 } else { 1.0 };
                            piv = s * abs_floor.max(piv.abs());
                        }
                    }
                }
            }

            pinv[pivot_row] = k as isize;
            perm[k] = pivot_row;
            u_diag[k] = piv;
            for (p, v) in u_entries.into_iter() {
                u_row_idx.push(p);
                u_val.push(v);
            }
            u_col_ptr.push(u_row_idx.len());

            // L(:,k): unpivoted touched rows (excluding the pivot) / piv.
            let inv = 1.0 / piv;
            for &i in touched.iter() {
                if pinv[i] < 0 && w[i] != 0.0 {
                    l_row_idx.push(i); // original row, remapped after the loop
                    l_val.push(w[i] * inv);
                }
            }
            l_col_ptr.push(l_row_idx.len());

            // Clear w / mark for the next column.
            for &i in touched.iter() {
                w[i] = 0.0;
                mark[i] = false;
            }
            touched.clear();
        }

        // Remap L's stored original rows to pivot positions, then sort columns.
        let perm_inv: Vec<usize> = pinv.iter().map(|&p| p as usize).collect();
        remap_and_sort_l(&l_col_ptr, &mut l_row_idx, &mut l_val, &perm_inv, m);

        // Transpose U from column-wise (built above) to row-wise CSR, then into
        // per-row vectors with the diagonal entry first. Columns were emitted in
        // increasing order, so each row's strict-upper entries are sorted.
        let (u_row_ptr, u_col_idx, u_val) = transpose_to_csr(&u_col_ptr, &u_row_idx, &u_val, m);
        let mut u_rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(m);
        for i in 0..m {
            let mut row = Vec::with_capacity(1 + (u_row_ptr[i + 1] - u_row_ptr[i]));
            row.push((i, u_diag[i])); // diagonal first
            for idx in u_row_ptr[i]..u_row_ptr[i + 1] {
                row.push((u_col_idx[idx], u_val[idx]));
            }
            u_rows.push(row);
        }
        // Column-wise index of U's strict-upper entries (rows added in
        // increasing order, so each `u_above[c]` is sorted ascending).
        let mut u_above: Vec<Vec<usize>> = vec![Vec::new(); m];
        for (i, row) in u_rows.iter().enumerate() {
            for &(c, _) in row.iter() {
                if c > i {
                    u_above[c].push(i);
                }
            }
        }

        // Inverse of the scaling row permutation, for sparse-spike seeding.
        let mut scale_rperm_inv = vec![0usize; m];
        for (i, &o) in scale.rperm.iter().enumerate() {
            scale_rperm_inv[o] = i;
        }

        // `max|U|` at factor: denominator of the element-growth monitor (L5).
        let mut u_max0 = 0.0_f64;
        for row in u_rows.iter() {
            for &(_, v) in row.iter() {
                u_max0 = u_max0.max(v.abs());
            }
        }
        let u_max0 = u_max0.max(f64::MIN_POSITIVE);

        Ok(SparseLu {
            m,
            l_col_ptr,
            l_row_idx,
            l_val,
            u_rows,
            perm,
            perm_inv,
            qcol,
            qcol_inv,
            u_above,
            etas: Vec::new(),
            growth: 1.0,
            u_max0,
            reach_visits,
            params,
            scale,
            scale_rperm_inv,
            scratch: vec![0.0; m],
            scratch_mark: vec![false; m],
            ft_work: vec![0.0; m],
            scratch_b: vec![0.0; m],
            scratch_c: vec![0.0; m],
            scratch_d: vec![0.0; m],
            pivot_scratch: Vec::new(),
            targets_scratch: Vec::new(),
            row_pool: Vec::new(),
            col_rows_pool: Vec::new(),
            saved_scratch: Vec::new(),
            saved_pool: Vec::new(),
        })
    }

    /// Convenience: analyze + factor from dense columns.
    pub fn factor_dense_columns(
        m: usize,
        cols: &[Vec<f64>],
        params: LuParams,
    ) -> Result<Self, FeralError> {
        let a = SparseColMatrix::from_dense_columns(m, cols)?;
        let symbolic = SparseLuSymbolic::analyze(&a)?;
        SparseLu::factor(&a, &symbolic, params)
    }

    /// Basis dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.m
    }

    /// Row permutation: `perm[k]` = original row in pivot position `k`.
    #[inline]
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Column order: `qcol[k]` = original column in position `k`.
    #[inline]
    pub fn qcol(&self) -> &[usize] {
        &self.qcol
    }

    /// Total stored nonzeros in `L` and `U` (including the `U` diagonal).
    pub fn factor_nnz(&self) -> usize {
        self.l_val.len() + self.u_rows.iter().map(|r| r.len()).sum::<usize>()
    }

    /// Total elementary operations across all Forrest–Tomlin update etas — the
    /// work a warm solve replays for the update chain. For bump-local updates
    /// this stays `O(Σ bump)`, not `O(k·n)` (the structural FT witness).
    pub fn eta_ops(&self) -> usize {
        self.etas.iter().map(|e| e.ops.len()).sum()
    }

    /// Operations in the most recent update's eta (its bump cost).
    pub fn last_eta_ops(&self) -> usize {
        self.etas.last().map(|e| e.ops.len()).unwrap_or(0)
    }

    /// Total Gilbert–Peierls reach nodes visited during the factorization.
    /// This is `O(nnz(U))` (output-sensitive); it would be `O(n²)` if the
    /// factor scanned all prior columns. Used by the scalability guard test.
    pub fn reach_visits(&self) -> usize {
        self.reach_visits
    }

    /// Reconstruct dense entry `(i, j)` of `L` (pivot-position coordinates).
    pub fn l_dense(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 1.0;
        }
        let (s, e) = (self.l_col_ptr[j], self.l_col_ptr[j + 1]);
        for idx in s..e {
            if self.l_row_idx[idx] == i {
                return self.l_val[idx];
            }
        }
        0.0
    }

    /// Reconstruct dense entry `(i, j)` of `U` (pivot-position coordinates).
    pub fn u_dense(&self, i: usize, j: usize) -> f64 {
        if i > j {
            return 0.0;
        }
        for &(c, v) in self.u_rows[i].iter() {
            if c == j {
                return v;
            }
        }
        0.0
    }
}

/// Transpose a strict-upper `U` from column-wise CSC (`col_ptr` over columns,
/// `row_idx` = pivot-row positions) to row-wise CSR. Columns are assumed
/// emitted in ascending order, so each output row is column-sorted.
fn transpose_to_csr(
    col_ptr: &[usize],
    row_idx: &[usize],
    val: &[f64],
    m: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let nnz = val.len();
    let mut row_cnt = vec![0usize; m];
    for &r in row_idx.iter() {
        row_cnt[r] += 1;
    }
    let mut row_ptr = vec![0usize; m + 1];
    for i in 0..m {
        row_ptr[i + 1] = row_ptr[i] + row_cnt[i];
    }
    let mut col_idx = vec![0usize; nnz];
    let mut out_val = vec![0.0; nnz];
    let mut next: Vec<usize> = row_ptr[..m].to_vec();
    for k in 0..m {
        for idx in col_ptr[k]..col_ptr[k + 1] {
            let r = row_idx[idx];
            let dst = next[r];
            next[r] += 1;
            col_idx[dst] = k;
            out_val[dst] = val[idx];
        }
    }
    (row_ptr, col_idx, out_val)
}

/// Remap L's original row indices to pivot positions and sort each column.
fn remap_and_sort_l(
    col_ptr: &[usize],
    row_idx: &mut [usize],
    val: &mut [f64],
    perm_inv: &[usize],
    m: usize,
) {
    for r in row_idx.iter_mut() {
        *r = perm_inv[*r];
    }
    let mut order: Vec<usize> = Vec::new();
    for j in 0..m {
        let (s, e) = (col_ptr[j], col_ptr[j + 1]);
        order.clear();
        order.extend(s..e);
        order.sort_by_key(|&idx| row_idx[idx]);
        let rows: Vec<usize> = order.iter().map(|&idx| row_idx[idx]).collect();
        let vals: Vec<f64> = order.iter().map(|&idx| val[idx]).collect();
        row_idx[s..e].copy_from_slice(&rows);
        val[s..e].copy_from_slice(&vals);
    }
}
