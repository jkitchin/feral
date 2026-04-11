pub mod dense;
pub mod error;
pub mod inertia;

// Flat public API re-exported at crate root:
pub use dense::factor::{factor, BunchKaufmanParams, Factors, ZeroPivotAction};
pub use dense::matrix::SymmetricMatrix;
pub use dense::solve::{solve, solve_refined};
pub use error::FeralError;
pub use inertia::Inertia;
