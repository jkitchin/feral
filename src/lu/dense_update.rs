//! Dense rank-1 column-replacement update (Bartels–Golub).
//!
//! Replacing basis slot `leaving_slot` by a new column whose spike is
//! `s = G⁻¹ L⁻¹ P aₙₑw` (from [`DenseLu::ftran_partial`]) maintains the
//! invariant `P B Q = L G U` in three steps.
//!
//! Step one overwrites column `q = qcol_inv[leaving_slot]` of `U` with the
//! spike `s`; `U` is then upper triangular except column `q`, which has a spike
//! below the diagonal. Step two cyclically shifts columns `q..m-1` of `U` left
//! by one and moves the spike column to position `m-1` (updating `Q`), turning
//! `U` into upper-Hessenberg with a single subdiagonal on positions `q..m-2`.
//! Step three reduces that Hessenberg back to upper triangular by **Bartels–
//! Golub elimination with partial pivoting**: at each subdiagonal it pivots on
//! the larger of the diagonal and the subdiagonal, interchanging the two rows
//! when the subdiagonal wins.
//!
//! Each elimination (and interchange) is recorded as a [`DenseFtOp`] eta rather
//! than folded into `L`: a row interchange is an *unsymmetric* row permutation
//! of `U`, and absorbing it into `L` would require a column swap that destroys
//! `L`'s unit-lower-triangular structure. So — exactly as on the sparse path —
//! the base `L`, `P`, and prior etas stay fixed, and the solves replay the etas
//! between the `L`-solve and the `U`-solve. Partial pivoting is what makes this
//! robust: it dodges the zero-superdiagonal landmine of the old fixed-order
//! sweep (which pivoted on the shifted-in old superdiagonal — structurally zero
//! for slack/triangular bases, so it refactored on trivially valid updates,
//! issue #115) and bounds every multiplier by 1, so element growth stays `O(m)`
//! on the Hessenberg and no compensated accumulation is needed.
//!
//! The work is done on clones of `U` and `Q` plus a local eta list, committed
//! only on success, so a `NeedsRefactor` return leaves `self` unchanged.

use super::dense_factor::{DenseFtOp, DenseLu};
use super::RefactorCause;
use crate::error::FeralError;

impl DenseLu {
    /// Replace basis slot `leaving_slot` with `entering_col` (the new basis
    /// column `aₙₑw`). The spike `L⁻¹ P aₙₑw` is computed internally, then folded
    /// into the factorization. On success the factors reflect the new basis.
    ///
    /// Returns [`FeralError::NeedsRefactor`] (leaving `self` unchanged) when the
    /// update budget (`max_updates`) or growth budget (`max_growth`) is
    /// exceeded, or when a bump pivot vanishes (a singular replacement basis).
    /// In every failure mode the update signals NeedsRefactor rather than
    /// SingularBasis — the authoritative singularity verdict comes from a fresh
    /// factorization, not the incremental update. The sparse update path
    /// ([`super::SparseLu::update`]) follows the identical contract.
    pub fn update(&mut self, leaving_slot: usize, entering_col: &[f64]) -> Result<(), FeralError> {
        let m = self.m;
        if entering_col.len() != m {
            return Err(FeralError::DimensionMismatch {
                expected: m,
                got: entering_col.len(),
            });
        }
        if leaving_slot >= m {
            return Err(FeralError::InvalidInput(format!(
                "leaving_slot {} out of range for basis dimension {}",
                leaving_slot, m
            )));
        }
        // Reject non-finite entries up front, matching the factor path
        // (`dense_factor.rs`): a NaN pivot passes `piv.abs() <= ztol` (NaN
        // comparisons are false) and the `umax` fold ignores it, committing a
        // corrupted factor after `Ok` (issue #114).
        if entering_col.iter().any(|x| !x.is_finite()) {
            return Err(FeralError::InvalidInput(
                "LU entering column contains non-finite entries".to_string(),
            ));
        }
        // Update-count budget (checked before doing any work).
        if self.updates_since_refactor + 1 > self.params.max_updates {
            self.last_refactor =
                Some((RefactorCause::UpdateBudget, self.params.max_updates as f64));
            return Err(FeralError::NeedsRefactor);
        }

        // Scale the entering column into the factored frame Ã, then form the
        // spike L⁻¹ P ãₙₑw (no factor mutation). With identity scaling this is
        // just `entering_col`.
        let mut spike = vec![0.0; m];
        for (i, si) in spike.iter_mut().enumerate() {
            *si = self.scale.d_row[i]
                * entering_col[self.scale.rperm[i]]
                * self.scale.d_col[leaving_slot];
        }
        self.ftran_partial(&mut spike)?;

        let q = self.qcol_inv[leaving_slot];
        // L6 (dev/research/repo-review-2026-06-09.md): the bump-pivot and
        // final-pivot zero tests must be relative to the basis magnitude, not an
        // absolute 1e-13. The reference is `max|A|` — matching the factor path's
        // `zero_pivot_tol · max|A|`. Anchoring to `u_max0` instead (as before)
        // rejected healthy `O(a_max)` pivots on high-growth bases where
        // `u_max0 ≫ a_max`, livelocking update→refactor→retry (issue #118).
        let ztol = self.params.zero_pivot_tol * self.a_max;
        let max_growth = self.params.max_growth;

        // Work on clones of U and Q, and accumulate new eliminations in a local
        // eta list; the base L, P, and the prior etas are never touched, so
        // rollback is simply *not committing*. (Unlike the old scheme, L is not
        // modified: a Bartels–Golub row interchange is an unsymmetric row
        // permutation of U that cannot be absorbed into an explicit
        // unit-lower-triangular L — see the module header.)
        let mut u = self.u.clone();
        let mut qcol = self.qcol.clone();
        let mut new_etas: Vec<DenseFtOp> = Vec::new();

        // 1. Overwrite column q of U with the spike (`G⁻¹ L⁻¹ P aₙₑw`).
        for i in 0..m {
            u[i + q * m] = spike[i];
        }

        // 2. Cyclic column shift q..m-1: spike column moves to position m-1,
        //    turning U into upper-Hessenberg with one subdiagonal on q..m-2.
        cyclic_shift_columns(&mut u, m, q);
        let leaving = qcol[q];
        for j in q..m - 1 {
            qcol[j] = qcol[j + 1];
        }
        qcol[m - 1] = leaving;

        // 3. Reduce the Hessenberg U to triangular by Bartels–Golub elimination
        //    with partial pivoting: pivot on the larger of the diagonal and the
        //    subdiagonal, interchanging the two rows when the subdiagonal wins.
        //    This both dodges the zero-superdiagonal landmine (the old
        //    fixed-order sweep pivoted on the shifted-in old superdiagonal,
        //    structurally zero for slack/triangular bases → spurious
        //    TinyPivot(0.0)) and bounds every multiplier by 1, keeping element
        //    growth O(m) on the Hessenberg (issue #115). Each elimination is
        //    recorded as an eta; the base L is untouched.
        for k in q..m.saturating_sub(1) {
            let mut piv = u[k + k * m];
            let sub = u[k + 1 + k * m];
            if sub.abs() > piv.abs() {
                // Interchange rows k and k+1 of U (all columns; both rows are
                // zero left of column k, so this only reorders columns ≥ k) and
                // record the swap so the base L/P stay fixed.
                for j in 0..m {
                    u.swap(k + j * m, k + 1 + j * m);
                }
                new_etas.push(DenseFtOp::Swap { a: k, b: k + 1 });
                piv = u[k + k * m];
            }
            if piv.abs() <= ztol {
                // Both diagonal and subdiagonal vanish ⇒ column k is negligible
                // ⇒ singular replacement (reported as TinyPivot, as on the
                // sparse path; the authoritative verdict is a fresh factor).
                self.last_refactor = Some((RefactorCause::TinyPivot, piv.abs()));
                return Err(FeralError::NeedsRefactor);
            }
            let sub = u[k + 1 + k * m];
            if sub == 0.0 {
                continue; // subdiagonal already zero (no elimination needed)
            }
            let mult = sub / piv; // |mult| ≤ 1 by the interchange
            for j in k..m {
                u[k + 1 + j * m] -= mult * u[k + j * m];
            }
            u[k + 1 + k * m] = 0.0; // enforce exact zero
            new_etas.push(DenseFtOp::Axpy {
                target: k + 1,
                src: k,
                mult,
            });
        }

        // L5 (dev/research/repo-review-2026-06-09.md): monitor element growth,
        // not the largest single multiplier. The high-water `max|U|/u_max0`
        // compounds across a chain of updates, where a per-multiplier max sat
        // at the largest step and missed the accumulation. Measured on the
        // clone (uncommitted), so an over-budget update leaves `self` intact.
        let umax = u.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
        let growth = self.growth.max(umax / self.u_max0);
        if growth > max_growth {
            self.last_refactor = Some((RefactorCause::Growth, growth));
            return Err(FeralError::NeedsRefactor);
        }

        // L1 (dev/research/repo-review-2026-06-09.md): the loop above validates
        // pivots `q..m-2`; the final diagonal `u[m-1,m-1]` — the new last pivot,
        // or (when `q == m-1`) the spike's last entry with no loop iteration at
        // all — was never checked. Committing a vanishing final pivot would let
        // the next `ftran`/`btran` divide by ~0 and emit a silent `Inf`/`NaN`.
        // Reject before commit (consistent with the in-bump check above).
        let last = m - 1;
        if u[last + last * m].abs() <= ztol {
            self.last_refactor = Some((RefactorCause::TinyPivot, u[last + last * m].abs()));
            return Err(FeralError::NeedsRefactor);
        }

        // Commit. The base L is unchanged; the new eliminations extend G.
        self.u = u;
        self.qcol = qcol;
        for (k, &slot) in self.qcol.iter().enumerate() {
            self.qcol_inv[slot] = k;
        }
        self.etas.extend(new_etas);
        self.growth = growth;
        self.updates_since_refactor += 1;
        Ok(())
    }
}

/// Cyclically shift columns `q..m-1` of a column-major `m`×`m` buffer left by
/// one, moving column `q` to position `m-1`.
fn cyclic_shift_columns(buf: &mut [f64], m: usize, q: usize) {
    if q + 1 >= m {
        return;
    }
    let mut saved = vec![0.0; m];
    saved.copy_from_slice(&buf[q * m..q * m + m]);
    for j in q..m - 1 {
        let (dst, src) = (j * m, (j + 1) * m);
        buf.copy_within(src..src + m, dst);
    }
    let last = (m - 1) * m;
    buf[last..last + m].copy_from_slice(&saved);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lu::dense_factor::DenseLu;
    use crate::lu::LuParams;

    /// L1 (dev/research/repo-review-2026-06-09.md): the bump-elimination loop
    /// only validates pivots `q..m-2`; the final diagonal `u[m-1,m-1]` is never
    /// checked, and when the leaving slot is the last column (`q == m-1`) the
    /// loop body never runs at all. Replacing the last slot of the 2×2 identity
    /// basis with `e_0` makes the new basis `[e_0, e_0]` singular and drives the
    /// new last `U` diagonal to exactly 0. Pre-fix this committed `Ok(())` and
    /// the next `ftran` divided by 0; the fix rejects it before commit.
    #[test]
    fn update_singular_last_pivot_does_not_commit() {
        let cols = vec![vec![1.0, 0.0], vec![0.0, 1.0]]; // identity basis
        let mut lu = DenseLu::factor(&cols, 2, LuParams::default()).expect("factor");

        // Replace slot 1 (the last column) with e_0 → q == m-1 == 1.
        let err = lu.update(1, &[1.0, 0.0]);
        assert!(
            matches!(err, Err(FeralError::NeedsRefactor)),
            "singular replacement basis must be rejected, got {err:?}"
        );

        // And `self` is left unchanged/usable: a solve against the original
        // (still-committed) basis stays finite.
        let mut rhs = vec![1.0, 1.0];
        lu.ftran(&mut rhs).expect("ftran after rejected update");
        assert!(rhs.iter().all(|x| x.is_finite()));
    }

    /// L5 (dev/research/repo-review-2026-06-09.md): the growth monitor recorded
    /// only the largest single elimination multiplier, so compounded element
    /// growth in `U` across a chain of updates went unmonitored. After the fix
    /// `growth` is the ‖U‖∞ high-water ratio (max|U| over update history ÷
    /// max|U| at factor), which compounds. This pins that semantics: the
    /// monitor must equal the element-growth ratio recomputed independently
    /// from the committed `U`. Oracle is the independent recomputation, not the
    /// monitor's own bookkeeping. Pre-fix `growth` is the max single multiplier
    /// and does not match.
    #[test]
    fn growth_monitor_tracks_compounded_element_growth() {
        let cols = vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0, 5.0],
        ];
        let m = 4;
        let params = LuParams {
            max_updates: 20,
            max_growth: 1e12, // large: updates commit on both old and new code
            ..LuParams::default()
        };
        let mut lu = DenseLu::factor(&cols, m, params).expect("factor");

        let umax = |lu: &DenseLu| {
            let mut mx = 0.0_f64;
            for j in 0..m {
                for i in 0..m {
                    mx = mx.max(lu.u(i, j).abs());
                }
            }
            mx
        };
        let u_max0 = umax(&lu);
        let mut hw = 1.0_f64; // independent high-water of max|U| / u_max0

        // Replace the LAST slot each time: no Hessenberg bump forms (q == m-1),
        // so every update commits cleanly while raising max|U| in that column,
        // making the element-growth ratio compound across the chain.
        let updates = [
            (3usize, vec![0.0, 0.0, 1.0, 20.0]),
            (3usize, vec![0.0, 0.0, 1.0, 60.0]),
            (3usize, vec![0.0, 0.0, 1.0, 180.0]),
        ];
        for (i, (slot, col)) in updates.iter().enumerate() {
            lu.update(*slot, col)
                .unwrap_or_else(|e| panic!("update {i} should commit: {e:?}"));
            hw = hw.max(umax(&lu) / u_max0);
            assert!(
                (lu.growth - hw).abs() <= 1e-9 * hw,
                "growth monitor {} must equal element-growth high-water {}",
                lu.growth,
                hw
            );
        }
        assert!(hw > 1.0, "test must exercise genuine element growth");
    }

    /// Issue #93: the public `growth`/`u_max0` getters must expose exactly the
    /// internal fields (no divergence). Fresh factor reports `growth == 1.0`;
    /// after a committed update the getter tracks the internal high-water field.
    #[test]
    fn growth_getters_expose_internal_fields() {
        let cols = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let params = LuParams {
            max_updates: 20,
            max_growth: 1e12,
            ..LuParams::default()
        };
        let mut lu = DenseLu::factor(&cols, 2, params).expect("factor");

        assert_eq!(lu.growth(), 1.0, "fresh factor growth is 1.0");
        assert_eq!(lu.growth(), lu.growth, "getter mirrors internal field");
        assert_eq!(lu.u_max0(), lu.u_max0, "getter mirrors internal field");
        assert!(
            lu.u_max0() > 0.0,
            "reference max|U| is floored away from zero"
        );

        lu.update(1, &[0.0, 40.0]).expect("update commits");
        assert_eq!(
            lu.growth(),
            lu.growth,
            "getter mirrors internal after update"
        );
        assert_eq!(lu.u_max0(), lu.u_max0, "u_max0 unchanged by update");
    }

    /// Issue #95: `last_refactor()` is `None` on a fresh factor and each
    /// `NeedsRefactor` return records the cause + a magnitude. Update-count trip:
    /// with `max_updates = 1` the second update fails as `UpdateBudget`, and the
    /// magnitude is the cap that was hit.
    #[test]
    fn last_refactor_reports_update_budget() {
        // Tridiagonal base; last-slot replacements commit (no Hessenberg bump).
        let cols = vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0, 5.0],
        ];
        let params = LuParams {
            max_updates: 1,
            ..LuParams::default()
        };
        let mut lu = DenseLu::factor(&cols, 4, params).expect("factor");
        assert_eq!(lu.last_refactor(), None, "fresh factor: no cause");

        lu.update(3, &[0.0, 0.0, 1.0, 7.0])
            .expect("first update within budget");
        assert_eq!(lu.last_refactor(), None, "successful update leaves it None");

        let err = lu.update(3, &[0.0, 0.0, 1.0, 8.0]);
        assert!(matches!(err, Err(FeralError::NeedsRefactor)));
        assert_eq!(
            lu.last_refactor(),
            Some((RefactorCause::UpdateBudget, 1.0)),
            "second update trips the count cap (max_updates = 1)"
        );
    }

    /// Issue #95: a dependent replacement drives the final `U` diagonal to ~0 and
    /// is reported as `TinyPivot` (the dense path has no distinct `Singular`), with
    /// the offending `|pivot|` as magnitude. Reuses the `[e_0, e_0]` scenario.
    #[test]
    fn last_refactor_reports_tiny_pivot() {
        let cols = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut lu = DenseLu::factor(&cols, 2, LuParams::default()).expect("factor");
        let err = lu.update(1, &[1.0, 0.0]); // new basis [e0, e0] singular
        assert!(matches!(err, Err(FeralError::NeedsRefactor)));
        let (cause, mag) = lu.last_refactor().expect("cause recorded");
        assert_eq!(cause, RefactorCause::TinyPivot);
        let ztol = lu.params.zero_pivot_tol * lu.u_max0;
        assert!(mag <= ztol, "magnitude {mag} is the ~0 offending pivot");
    }

    /// Issue #95: a growth trip records `Growth` with the ratio that exceeded the
    /// cap as magnitude. Uses the same compounding last-slot updates as
    /// `growth_monitor_tracks_compounded_element_growth`, but with a small
    /// `max_growth` so one commits then the next trips.
    #[test]
    fn last_refactor_reports_growth() {
        let cols = vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0, 5.0],
        ];
        let params = LuParams {
            max_updates: 20,
            max_growth: 5.0, // small: a compounding update soon exceeds it
            ..LuParams::default()
        };
        let mut lu = DenseLu::factor(&cols, 4, params).expect("factor");
        let updates = [
            vec![0.0, 0.0, 1.0, 20.0],
            vec![0.0, 0.0, 1.0, 60.0],
            vec![0.0, 0.0, 1.0, 180.0],
        ];
        let mut tripped = None;
        for col in updates.iter() {
            if let Err(FeralError::NeedsRefactor) = lu.update(3, col) {
                tripped = lu.last_refactor();
                break;
            }
        }
        let (cause, mag) = tripped.expect("some update trips the growth cap");
        assert_eq!(cause, RefactorCause::Growth);
        assert!(mag > 5.0, "growth magnitude {mag} exceeds max_growth = 5.0");
    }

    /// Issue #95 parity: dense `should_refactor()` (cost-based, `>= m` updates)
    /// and `should_refactor_growth()` (pre-empts a growth trip). Fresh factor
    /// recommends neither.
    #[test]
    fn dense_refactor_recommendations() {
        let cols = vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0, 5.0],
        ];
        let m = 4;
        let params = LuParams {
            max_updates: 100,
            max_growth: 1e8,
            ..LuParams::default()
        };
        let mut lu = DenseLu::factor(&cols, m, params).expect("factor");
        assert!(!lu.should_refactor(), "fresh: no cost-based recommendation");
        assert!(!lu.should_refactor_growth(), "fresh: growth == 1");

        // Repeated identical last-slot replacements commit with bounded growth;
        // the cost-based recommendation fires only once the count reaches m.
        for k in 1..=m {
            lu.update(3, &[0.0, 0.0, 1.0, 7.0])
                .unwrap_or_else(|e| panic!("update {k} should commit: {e:?}"));
            assert_eq!(
                lu.should_refactor(),
                k >= m,
                "should_refactor must fire exactly at updates_since_refactor >= m (k = {k})"
            );
        }
        assert!(
            !lu.should_refactor_growth(),
            "bounded growth: no growth-based recommendation"
        );
    }
}
