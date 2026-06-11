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

    /// A supernode received more delayed pivots from its children at
    /// numeric time than the symbolic-analysis phase budgeted for.
    /// Mirrors MUMPS's `INFO(2)` workspace-overflow path: a predictable,
    /// recoverable failure that bounds worst-case front growth.
    /// See issue #55 and `dev/research/symbolic-delay-budget-2026-05-27.md`.
    DelayBudgetExceeded {
        supernode: usize,
        required: usize,
        capacity: usize,
    },

    /// An unsymmetric LU basis is numerically singular: the basis column
    /// `column` had no candidate pivot above `LuParams::zero_pivot_tol`
    /// and `LuSingularAction::Fail` was specified. At *factor* time
    /// `column` is the original basis column index the caller supplied
    /// (`qcol[k]`, L9), so a simplex driver can repair the basis instead
    /// of receiving a garbage solve. NOTE: the solve and update paths
    /// (`sparse_solve.rs` and the column-replacement update) raise this
    /// with the internal pivot *position* rather than the original column;
    /// callers needing the original index in those paths must map it back
    /// through the column permutation. See issue #81 and
    /// `dev/research/unsymmetric-lu.md`.
    SingularBasis { column: usize },

    /// A rank-1 LU basis update (column replacement) could not be applied
    /// within the stability / update-count budget (`LuParams::max_updates`
    /// or `max_growth`), a stability monitor tripped, or the replacement
    /// produced a vanishing bump pivot (a singular update — the incoming
    /// column is linearly dependent on the retained basis; L8, see the
    /// `dense_update.rs` method docs). The factorization is left
    /// unchanged; the caller must call `refactor()` with the current basic
    /// columns. The recoverable analogue of MUMPS's delayed-pivot
    /// overflow. See issue #81.
    NeedsRefactor,
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
            FeralError::DelayBudgetExceeded {
                supernode,
                required,
                capacity,
            } => {
                write!(
                    f,
                    "delayed-pivot budget exceeded at supernode {}: \
                     required {} delayed columns, capacity {} (issue #55)",
                    supernode, required, capacity
                )
            }
            FeralError::SingularBasis { column } => {
                write!(f, "LU basis is numerically singular at column {}", column)
            }
            FeralError::NeedsRefactor => {
                write!(
                    f,
                    "LU basis update budget exceeded; refactor required (issue #81)"
                )
            }
        }
    }
}

impl std::error::Error for FeralError {}
