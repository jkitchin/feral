pub mod dense;
pub mod error;
pub mod inertia;
pub mod io;
pub mod numeric;
pub mod ordering;
pub mod sparse;
pub mod symbolic;

// Flat public API re-exported at crate root:
pub use dense::factor::{factor, BunchKaufmanParams, Factors, ZeroPivotAction};
pub use dense::matrix::SymmetricMatrix;
pub use dense::solve::{solve, solve_refined};
pub use error::FeralError;
pub use inertia::Inertia;
pub use io::mtx::{parse_mtx, read_mtx, MtxMatrix};
pub use io::sidecar::{read_sidecar, KktSidecar, SidecarInertia};
pub use sparse::csc::{CscMatrix, CscPattern};
