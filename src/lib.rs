// Deny `.unwrap()` and `.expect()` in production code, but allow them in
// test modules (inside `#[cfg(test)]` blocks) where panics are acceptable.
// This is a structural enforcement of the CLAUDE.md hard rule against
// unwrap in `src/`, replacing the ad-hoc grep check in CI.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
// Style lints that fire only in test scaffolding — relaxed under cfg(test).
// The lib build keeps default clippy strictness.
#![cfg_attr(test, allow(clippy::needless_range_loop))]

pub mod dense;
pub mod error;
pub mod inertia;
pub mod io;
pub mod numeric;
pub mod ordering;
pub mod scaling;
pub mod sparse;
pub mod symbolic;

// Flat public API re-exported at crate root:
pub use dense::factor::{
    factor, factor_single_front, BunchKaufmanParams, Factors, ZeroPivotAction,
};
pub use dense::matrix::SymmetricMatrix;
pub use dense::solve::{solve, solve_refined};
pub use error::FeralError;
pub use inertia::Inertia;
pub use io::mtx::{parse_mtx, read_mtx, MtxMatrix};
pub use io::sidecar::{read_sidecar, KktSidecar, SidecarInertia};
pub use numeric::condition::{estimate_condition_1norm, estimate_inverse_norm_1, matrix_norm_1};
pub use numeric::factorize::{factorize_multifrontal_with_schur, NumericParams, SchurBlock};
pub use numeric::solve::{solve_sparse, solve_sparse_refined};
pub use numeric::solver::{FactorStatus, QualityLevel, Solver};
pub use sparse::csc::{CscMatrix, CscPattern};
