//! C ABI for embedding feral as Ipopt's `linear_solver=feral`.
//!
//! Designed to be the minimum surface Ipopt's
//! `SparseSymLinearSolverInterface` plug-in shape requires. The C++
//! shim at `feral-ipopt-shim/` is the only intended consumer.
//!
//! Matrix format: matches Ipopt's `CSR_Format_0_Offset` — upper-
//! triangle CSR with 0-based indices. For a symmetric matrix this is
//! byte-identical to feral's `CscMatrix` (lower-triangle CSC); the
//! shim hands us Ipopt's `ia`/`ja` arrays unchanged. See
//! `dev/research/feral-ipopt-c-shim.md` §"Matrix format".
//!
//! Status codes mirror Ipopt's `ESymSolverStatus` enum at
//! `ref/Ipopt/src/Algorithm/LinearSolvers/IpSymLinearSolver.hpp:19-33`.

use crate::{CscMatrix, FactorStatus, Solver};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub const FERAL_SUCCESS: i32 = 0;
pub const FERAL_SINGULAR: i32 = 1;
pub const FERAL_WRONG_INERTIA: i32 = 2;
pub const FERAL_FATAL: i32 = 3;

/// Opaque handle.
pub struct FeralSolver {
    solver: Solver,
    matrix: Option<CscMatrix>,
    neg_evals: i32,
}

/// Create a new solver handle. Returns null on panic.
///
/// Cascade-break is off by default after 585d739. Set
/// `FERAL_CASCADE_BREAK=on` (or `1`/`true`) in the environment to
/// opt back into the legacy `ratio=0.5, eps=1e-10` configuration —
/// required for ipopt-feral parity with pounce-feral, which opts
/// into the same defaults at its own construction site.
#[no_mangle]
pub extern "C" fn feral_new() -> *mut FeralSolver {
    catch_unwind(|| {
        let cb_on = matches!(
            std::env::var("FERAL_CASCADE_BREAK").as_deref(),
            Ok("1") | Ok("on") | Ok("true") | Ok("yes"),
        );
        let solver = if cb_on {
            Solver::new()
                .with_cascade_break(0.5)
                .with_cascade_break_eps(1e-10)
        } else {
            Solver::new()
        };
        Box::into_raw(Box::new(FeralSolver {
            solver,
            matrix: None,
            neg_evals: 0,
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Free a solver handle. Null pointer is a no-op.
///
/// # Safety
/// `s` must be a pointer previously returned by `feral_new` and not
/// already freed.
#[no_mangle]
pub unsafe extern "C" fn feral_free(s: *mut FeralSolver) {
    if s.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller contract — pointer came from `feral_new` and
        // has not been freed. Box reclaims and drops the owned data.
        drop(Box::from_raw(s));
    }));
}

/// Store the matrix structure and allocate the internal values
/// buffer. Returns `FERAL_SUCCESS` (0) or `FERAL_FATAL` (3).
///
/// `ia` must have length `n+1`; `ja` must have length `nnz`.
/// The arrays follow Ipopt's `CSR_Format_0_Offset` convention:
/// 0-based, upper triangle, sorted and deduplicated within each row.
///
/// # Safety
/// `s` must come from `feral_new`. `ia` must point to at least `n+1`
/// `i32`s; `ja` must point to at least `nnz` `i32`s.
#[no_mangle]
pub unsafe extern "C" fn feral_set_structure(
    s: *mut FeralSolver,
    n: i32,
    nnz: i32,
    ia: *const i32,
    ja: *const i32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || ia.is_null() || ja.is_null() || n < 0 || nnz < 0 {
            return FERAL_FATAL;
        }
        // SAFETY: caller contract — `s` came from `feral_new`.
        let s = &mut *s;
        let n_usize = n as usize;
        let nnz_usize = nnz as usize;
        // SAFETY: caller contract — `ia` has at least n+1 entries.
        let ia_slice = std::slice::from_raw_parts(ia, n_usize + 1);
        // SAFETY: caller contract — `ja` has at least nnz entries.
        let ja_slice = std::slice::from_raw_parts(ja, nnz_usize);

        let col_ptr: Vec<usize> = ia_slice.iter().map(|&x| x as usize).collect();
        let row_idx: Vec<usize> = ja_slice.iter().map(|&x| x as usize).collect();

        let matrix = CscMatrix {
            n: n_usize,
            col_ptr,
            row_idx,
            values: vec![0.0; nnz_usize],
        };

        // Basic structural sanity check on what Ipopt handed us.
        // Catches dimension mismatches and out-of-range indices.
        if matrix.validate().is_err() {
            return FERAL_FATAL;
        }

        s.matrix = Some(matrix);
        s.neg_evals = 0;
        FERAL_SUCCESS
    }))
    .unwrap_or(FERAL_FATAL)
}

/// Return a pointer to the internal values buffer (length `nnz`).
/// Caller writes A's nonzero values here in the same order as `ja`
/// from the most recent `feral_set_structure`, then calls
/// `feral_factor`. Returns null on error.
///
/// # Safety
/// `s` must come from `feral_new` and have had `feral_set_structure`
/// called successfully. The returned pointer is valid until the
/// next `feral_set_structure` or `feral_free`.
#[no_mangle]
pub unsafe extern "C" fn feral_values_ptr(s: *mut FeralSolver) -> *mut f64 {
    if s.is_null() {
        return std::ptr::null_mut();
    }
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller contract.
        let s = &mut *s;
        match &mut s.matrix {
            Some(m) => m.values.as_mut_ptr(),
            None => std::ptr::null_mut(),
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}

/// Factor the matrix currently in the values buffer.
///
/// If `check_neg != 0`, returns `FERAL_WRONG_INERTIA` when the
/// number of negative eigenvalues disagrees with `expected_neg`.
/// The factor is still stored — `feral_solve` may be called.
///
/// # Safety
/// `s` must come from `feral_new` and have had
/// `feral_set_structure` called successfully.
#[no_mangle]
pub unsafe extern "C" fn feral_factor(
    s: *mut FeralSolver,
    check_neg: i32,
    expected_neg: i32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() {
            return FERAL_FATAL;
        }
        // SAFETY: caller contract.
        let s = &mut *s;
        let matrix = match &s.matrix {
            Some(m) => m.clone(),
            None => return FERAL_FATAL,
        };
        // Pass None for check_inertia — Ipopt only constrains
        // negative-eigenvalue count, not positive. We do the
        // negative-count check ourselves below.
        let status = s.solver.factor(&matrix, None);
        match status {
            FactorStatus::Success => {
                let inertia = match s.solver.inertia() {
                    Some(i) => i.clone(),
                    None => return FERAL_FATAL,
                };
                s.neg_evals = inertia.negative as i32;
                if check_neg != 0 && s.neg_evals != expected_neg {
                    FERAL_WRONG_INERTIA
                } else {
                    FERAL_SUCCESS
                }
            }
            FactorStatus::Singular => FERAL_SINGULAR,
            FactorStatus::WrongInertia { actual, .. } => {
                // Solver::factor returns WrongInertia only when we
                // pass check_inertia=Some, which we don't above.
                // Defensive branch in case the implementation
                // changes.
                s.neg_evals = actual.negative as i32;
                FERAL_WRONG_INERTIA
            }
            FactorStatus::FatalError(_) => FERAL_FATAL,
        }
    }))
    .unwrap_or(FERAL_FATAL)
}

/// Solve `A X = B` in place. `rhs` is column-major, length
/// `n * nrhs`. On success the buffer holds X.
///
/// By default this routes through `Solver::solve_many_refined`: one
/// round of iterative refinement against the original matrix held
/// in `s.matrix`. This closes the residual floor that caused
/// ipopt-feral to stall in the final-tail convergence on
/// `robot_1600` (feral#17) and `NARX_CFy` (feral#18) — feral's
/// inertia agrees with MA57 on those problems, but cascade-break's
/// L-factor perturbation produces a per-pivot residual the IPM
/// can't drive below the duality gap without refinement.
///
/// Opt out by setting `FERAL_REFINE=0` in the environment. With
/// refinement disabled this is equivalent to `Solver::solve_many`.
///
/// # Safety
/// `s` must come from `feral_new` and a successful `feral_factor`
/// must have run since the most recent `feral_set_structure` /
/// values fill. The values buffer must not have been modified
/// between `feral_factor` and `feral_solve` (Ipopt's protocol).
/// `rhs` must point to at least `n * nrhs` `f64`s.
#[no_mangle]
pub unsafe extern "C" fn feral_solve(s: *mut FeralSolver, nrhs: i32, rhs: *mut f64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || rhs.is_null() || nrhs <= 0 {
            return FERAL_FATAL;
        }
        // SAFETY: caller contract.
        let s = &*s;
        let matrix = match &s.matrix {
            Some(m) => m,
            None => return FERAL_FATAL,
        };
        let n = matrix.n;
        let nrhs_usize = nrhs as usize;
        // SAFETY: caller contract — `rhs` has at least n*nrhs entries.
        let rhs_slice = std::slice::from_raw_parts_mut(rhs, n * nrhs_usize);

        let refined = !matches!(
            std::env::var("FERAL_REFINE").as_deref(),
            Ok("0") | Ok("false") | Ok("off") | Ok("no"),
        );
        let solved = if refined {
            s.solver.solve_many_refined(matrix, rhs_slice, nrhs_usize)
        } else {
            s.solver.solve_many(rhs_slice, nrhs_usize)
        };
        match solved {
            Ok(x) => {
                rhs_slice.copy_from_slice(&x);
                FERAL_SUCCESS
            }
            Err(_) => FERAL_FATAL,
        }
    }))
    .unwrap_or(FERAL_FATAL)
}

/// Number of negative eigenvalues of the most recently factored
/// matrix. Returns -1 if no factor is available.
///
/// # Safety
/// `s` must come from `feral_new`.
#[no_mangle]
pub unsafe extern "C" fn feral_num_neg(s: *const FeralSolver) -> i32 {
    if s.is_null() {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller contract.
        let s = &*s;
        s.neg_evals
    }))
    .unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2x2 indefinite `[[1,2],[2,1]]` (eigenvalues 3, -1) — RHS (3,3),
    /// expected `x = (1, 1)`. CSR upper-triangle with 0-based indices:
    /// row 0: cols (0,1); row 1: col (1). `ia = [0, 2, 3]`,
    /// `ja = [0, 1, 1]`, `values = [1, 2, 1]`. Exercises factor →
    /// refined solve via the C ABI surface.
    #[test]
    fn capi_factor_and_refined_solve() {
        unsafe {
            let s = feral_new();
            assert!(!s.is_null());

            let ia: [i32; 3] = [0, 2, 3];
            let ja: [i32; 3] = [0, 1, 1];
            assert_eq!(
                feral_set_structure(s, 2, 3, ia.as_ptr(), ja.as_ptr()),
                FERAL_SUCCESS
            );
            let vp = feral_values_ptr(s);
            assert!(!vp.is_null());
            std::ptr::copy_nonoverlapping([1.0_f64, 2.0, 1.0].as_ptr(), vp, 3);

            assert_eq!(feral_factor(s, 1, 1), FERAL_SUCCESS);
            assert_eq!(feral_num_neg(s), 1);

            let mut rhs = [3.0_f64, 3.0];
            // Default path (refined).
            assert_eq!(feral_solve(s, 1, rhs.as_mut_ptr()), FERAL_SUCCESS);
            assert!((rhs[0] - 1.0).abs() < 1e-12, "x0 = {}", rhs[0]);
            assert!((rhs[1] - 1.0).abs() < 1e-12, "x1 = {}", rhs[1]);

            feral_free(s);
        }
    }

    /// Same matrix as above, but with `FERAL_REFINE=0` set. Verifies
    /// the opt-out path still solves correctly.
    #[test]
    fn capi_solve_unrefined_opt_out() {
        // Process-wide env var — fine because cargo serialises this
        // test (single mod, single thread per env mutation) and we
        // restore it before exit. If the test panics with REFINE off
        // we leak the override; that only affects this test binary.
        let prior = std::env::var("FERAL_REFINE").ok();
        // SAFETY: single-threaded test sets a process-wide env var. No
        // other thread observes the transition.
        unsafe {
            std::env::set_var("FERAL_REFINE", "0");

            let s = feral_new();
            let ia: [i32; 3] = [0, 2, 3];
            let ja: [i32; 3] = [0, 1, 1];
            assert_eq!(
                feral_set_structure(s, 2, 3, ia.as_ptr(), ja.as_ptr()),
                FERAL_SUCCESS
            );
            let vp = feral_values_ptr(s);
            std::ptr::copy_nonoverlapping([1.0_f64, 2.0, 1.0].as_ptr(), vp, 3);
            assert_eq!(feral_factor(s, 0, 0), FERAL_SUCCESS);

            let mut rhs = [3.0_f64, 3.0];
            assert_eq!(feral_solve(s, 1, rhs.as_mut_ptr()), FERAL_SUCCESS);
            assert!((rhs[0] - 1.0).abs() < 1e-12);
            assert!((rhs[1] - 1.0).abs() < 1e-12);

            feral_free(s);

            match prior {
                Some(v) => std::env::set_var("FERAL_REFINE", v),
                None => std::env::remove_var("FERAL_REFINE"),
            }
        }
    }
}
