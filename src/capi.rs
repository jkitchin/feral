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
#[no_mangle]
pub extern "C" fn feral_new() -> *mut FeralSolver {
    catch_unwind(|| {
        Box::into_raw(Box::new(FeralSolver {
            solver: Solver::new(),
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
/// # Safety
/// `s` must come from `feral_new` and a successful `feral_factor`
/// must have run since the most recent `feral_set_structure` /
/// values fill. `rhs` must point to at least `n * nrhs` `f64`s.
#[no_mangle]
pub unsafe extern "C" fn feral_solve(s: *mut FeralSolver, nrhs: i32, rhs: *mut f64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || rhs.is_null() || nrhs <= 0 {
            return FERAL_FATAL;
        }
        // SAFETY: caller contract.
        let s = &*s;
        let n = match &s.matrix {
            Some(m) => m.n,
            None => return FERAL_FATAL,
        };
        let nrhs_usize = nrhs as usize;
        // SAFETY: caller contract — `rhs` has at least n*nrhs entries.
        let rhs_slice = std::slice::from_raw_parts_mut(rhs, n * nrhs_usize);
        match s.solver.solve_many(rhs_slice, nrhs_usize) {
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
