/// Errors returned by FERAL's public API.
#[derive(Debug)]
pub enum FeralError {
    /// The matrix is numerically rank-deficient: a pivot was exactly or
    /// near-zero and `ZeroPivotAction::Fail` was specified. The factorization
    /// is incomplete.
    NumericallyRankDeficient,

    /// Input matrix dimensions are inconsistent or the matrix is not square.
    InvalidInput(String),

    /// The RHS vector length does not match the factored matrix dimension.
    DimensionMismatch { expected: usize, got: usize },

    /// An I/O or parse error occurred (e.g. reading a Matrix Market file).
    IoError(String),

    /// `Solver::solve` (or `solve_refined`) was called before any
    /// successful factorization. Call `factor()` first.
    NoFactor,

    /// The SQD fast-path (`Solver::with_sqd_mode(true)`) refused a
    /// diagonal pivot. Either the pivot magnitude fell at or below
    /// `BunchKaufmanParams::zero_tol` (so `|d_kk| ≈ 0`), or the
    /// implied L-column growth `||l_col||_∞ / sqrt(|d_kk|)` would
    /// exceed `1 / sqrt(EPS) ≈ 6.7e7`, breaking the
    /// Gill-Saunders-Shinnerl 1996 stability bound for diagonal
    /// LDL^T on SQD matrices. The factorization aborts immediately —
    /// SQD never falls back silently to BK 1x1-vs-2x2. Caller
    /// must either re-factor with `with_sqd_mode(false)` (BK
    /// fallback) or investigate the input (Vanderbei 1995's
    /// SQD contract is not met at the reported column). See
    /// `dev/research/sqd-fast-path.md` and issue #34.
    SqdContractViolated { column: usize, pivot: f64 },
}

impl std::fmt::Display for FeralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeralError::NumericallyRankDeficient => {
                write!(f, "matrix is numerically rank-deficient")
            }
            FeralError::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
            FeralError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            FeralError::IoError(msg) => write!(f, "I/O error: {}", msg),
            FeralError::NoFactor => {
                write!(f, "no factorization available; call Solver::factor() first")
            }
            FeralError::SqdContractViolated { column, pivot } => {
                write!(
                    f,
                    "SQD contract violated at column {}: pivot = {:e} fails \
                     the diagonal-LDL^T stability bound (near-zero pivot or \
                     L-column growth above 1/sqrt(EPS))",
                    column, pivot
                )
            }
        }
    }
}

impl std::error::Error for FeralError {}
