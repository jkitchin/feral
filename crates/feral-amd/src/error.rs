//! Error type for AMD ordering.

use core::fmt;

/// Errors that can prevent AMD from completing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AmdError {
    /// Workspace size exceeded `i32::MAX` during AAT or element
    /// expansion. Indicates a matrix too large for the signed-i32
    /// internal index space.
    IndexOverflow,
    /// Debug-only: input pattern was not structurally symmetric.
    /// Release builds trust the caller.
    NonSymmetric,
    /// `col_ptr.len() != n+1`, `row_idx.len() != col_ptr[n]`, or a
    /// row index was out of bounds.
    MalformedInput,
}

impl fmt::Display for AmdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOverflow => f.write_str("AMD workspace exceeded i32::MAX"),
            Self::NonSymmetric => f.write_str("AMD input pattern was not structurally symmetric"),
            Self::MalformedInput => f.write_str("AMD input pattern failed structural validation"),
        }
    }
}

impl std::error::Error for AmdError {}
