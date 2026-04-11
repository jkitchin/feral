use crate::dense::matrix::SymmetricMatrix;
use crate::error::FeralError;
use crate::sparse::csc::CscMatrix;
use std::path::Path;

/// A sparse symmetric matrix in coordinate (COO) format, as read from a Matrix Market file.
/// Entries are 0-indexed, lower triangle only (i >= j).
#[derive(Debug)]
pub struct MtxMatrix {
    pub n: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl MtxMatrix {
    /// Convert to a dense symmetric matrix.
    pub fn to_dense(&self) -> SymmetricMatrix {
        SymmetricMatrix::from_lower_triangle(self.n, &self.entries)
    }

    /// Convert to a CSC sparse matrix (lower triangle).
    pub fn to_csc(&self) -> Result<CscMatrix, FeralError> {
        let rows: Vec<usize> = self.entries.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = self.entries.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<f64> = self.entries.iter().map(|&(_, _, v)| v).collect();
        CscMatrix::from_triplets(self.n, &rows, &cols, &vals)
    }
}

/// Read a Matrix Market file containing a symmetric real coordinate matrix.
///
/// Accepts only `%%MatrixMarket matrix coordinate real symmetric` format.
/// Indices are converted from 1-based (MTX) to 0-based. Upper-triangle entries
/// (i < j) are silently transposed to lower-triangle.
pub fn read_mtx(path: &Path) -> Result<MtxMatrix, FeralError> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| FeralError::IoError(format!("{}: {}", path.display(), e)))?;
    parse_mtx(&contents, path.to_string_lossy().as_ref())
}

/// Parse Matrix Market content from a string. `source` is used in error messages.
pub fn parse_mtx(contents: &str, source: &str) -> Result<MtxMatrix, FeralError> {
    let mut lines = contents.lines().enumerate();

    // Header line
    let (_, header) = lines
        .next()
        .ok_or_else(|| FeralError::IoError(format!("{}: empty file", source)))?;
    let header_lower = header.trim().to_ascii_lowercase();
    if header_lower != "%%matrixmarket matrix coordinate real symmetric" {
        return Err(FeralError::IoError(format!(
            "{}: unsupported header '{}' (expected: %%MatrixMarket matrix coordinate real symmetric)",
            source, header.trim()
        )));
    }

    // Skip comment lines (start with %)
    let mut size_line: Option<(usize, String)> = None;
    for (line_no, line) in &mut lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        size_line = Some((line_no, trimmed.to_string()));
        break;
    }

    let (size_line_no, size_text) = size_line
        .ok_or_else(|| FeralError::IoError(format!("{}: missing size line", source)))?;

    // Parse "rows cols nnz"
    let parts: Vec<&str> = size_text.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(FeralError::IoError(format!(
            "{}: line {}: expected 'rows cols nnz', got '{}'",
            source,
            size_line_no + 1,
            size_text
        )));
    }
    let rows: usize = parts[0].parse().map_err(|_| {
        FeralError::IoError(format!(
            "{}: line {}: invalid row count '{}'",
            source,
            size_line_no + 1,
            parts[0]
        ))
    })?;
    let cols: usize = parts[1].parse().map_err(|_| {
        FeralError::IoError(format!(
            "{}: line {}: invalid col count '{}'",
            source,
            size_line_no + 1,
            parts[1]
        ))
    })?;
    let nnz: usize = parts[2].parse().map_err(|_| {
        FeralError::IoError(format!(
            "{}: line {}: invalid nnz '{}'",
            source,
            size_line_no + 1,
            parts[2]
        ))
    })?;

    if rows != cols {
        return Err(FeralError::IoError(format!(
            "{}: symmetric matrix must be square, got {}x{}",
            source, rows, cols
        )));
    }
    let n = rows;

    // Parse entries
    let mut entries = Vec::with_capacity(nnz);
    for (line_no, line) in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() != 3 {
            return Err(FeralError::IoError(format!(
                "{}: line {}: expected 'i j value', got '{}'",
                source,
                line_no + 1,
                trimmed
            )));
        }
        let i: usize = parts[0].parse().map_err(|_| {
            FeralError::IoError(format!(
                "{}: line {}: invalid row index '{}'",
                source,
                line_no + 1,
                parts[0]
            ))
        })?;
        let j: usize = parts[1].parse().map_err(|_| {
            FeralError::IoError(format!(
                "{}: line {}: invalid col index '{}'",
                source,
                line_no + 1,
                parts[1]
            ))
        })?;
        let v: f64 = parts[2].parse().map_err(|_| {
            FeralError::IoError(format!(
                "{}: line {}: invalid value '{}'",
                source,
                line_no + 1,
                parts[2]
            ))
        })?;

        // Validate bounds (1-indexed in MTX)
        if i == 0 || j == 0 || i > n || j > n {
            return Err(FeralError::IoError(format!(
                "{}: line {}: index ({}, {}) out of bounds for {}x{} matrix",
                source,
                line_no + 1,
                i,
                j,
                n,
                n
            )));
        }

        // Convert to 0-indexed, normalize to lower triangle (i >= j)
        let (i0, j0) = (i - 1, j - 1);
        if i0 >= j0 {
            entries.push((i0, j0, v));
        } else {
            entries.push((j0, i0, v));
        }
    }

    Ok(MtxMatrix { n, entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_symmetric_3x3() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 4
1 1 2.0
2 1 -1.0
2 2 3.0
3 3 1.5
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.n, 3);
        assert_eq!(m.entries.len(), 4);

        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), 2.0);
        assert_eq!(dense.get(1, 0), -1.0);
        assert_eq!(dense.get(0, 1), -1.0); // symmetric
        assert_eq!(dense.get(1, 1), 3.0);
        assert_eq!(dense.get(2, 2), 1.5);
        assert_eq!(dense.get(2, 0), 0.0); // not set
    }

    #[test]
    fn test_parse_with_comments() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
% This is a comment
% Another comment
2 2 1
1 1 5.0
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.n, 2);
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0], (0, 0, 5.0));
    }

    #[test]
    fn test_parse_scientific_notation() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 2
1 1 1.23456789012345678e+02
2 1 -9.87654321098765432e-03
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.entries.len(), 2);
        assert!((m.entries[0].2 - 123.456789012345678).abs() < 1e-10);
        assert!((m.entries[1].2 - (-0.00987654321098765432)).abs() < 1e-16);
    }

    #[test]
    fn test_upper_triangle_normalized() {
        // Entry (1,2) with 1 < 2 should be flipped to (1,0) in 0-indexed
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
1 2 7.0
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0], (1, 0, 7.0)); // normalized to lower triangle
    }

    #[test]
    fn test_reject_general_format() {
        let mtx = "\
%%MatrixMarket matrix coordinate real general
2 2 1
1 1 1.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported header"), "got: {}", msg);
    }

    #[test]
    fn test_reject_complex() {
        let mtx = "\
%%MatrixMarket matrix coordinate complex symmetric
2 2 1
1 1 1.0 0.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported header"), "got: {}", msg);
    }

    #[test]
    fn test_reject_array_format() {
        let mtx = "\
%%MatrixMarket matrix array real symmetric
2 2
1.0
2.0
3.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported header"), "got: {}", msg);
    }

    #[test]
    fn test_reject_nonsquare() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 4 1
1 1 1.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("square"), "got: {}", msg);
    }

    #[test]
    fn test_reject_out_of_bounds() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
3 1 1.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("out of bounds"), "got: {}", msg);
    }

    #[test]
    fn test_empty_matrix() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 0
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.n, 3);
        assert_eq!(m.entries.len(), 0);

        let dense = m.to_dense();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(dense.get(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_diagonal_only() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 3
1 1 1.0
2 2 2.0
3 3 3.0
";
        let m = parse_mtx(mtx, "test").unwrap();
        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), 1.0);
        assert_eq!(dense.get(1, 1), 2.0);
        assert_eq!(dense.get(2, 2), 3.0);
        assert_eq!(dense.get(1, 0), 0.0);
    }

    #[test]
    fn test_negative_values() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 3
1 1 -1.0
2 1 -0.0
2 2 -3.5
";
        let m = parse_mtx(mtx, "test").unwrap();
        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), -1.0);
        assert_eq!(dense.get(1, 1), -3.5);
    }
}
