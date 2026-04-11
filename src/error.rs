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
        }
    }
}

impl std::error::Error for FeralError {}
