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
    ///
    /// X2 (REG-4): duplicate coordinates are **summed**, matching `to_csc`
    /// (which sums via `CscMatrix::from_triplets`) and the Matrix Market /
    /// COO convention used by scipy and MATLAB. The previous
    /// `from_lower_triangle` path *overwrote* duplicates (last-wins), so the
    /// same file produced two different matrices depending on the conversion
    /// path. See `dev/decisions.md` (2026-06-11) for the rationale.
    pub fn to_dense(&self) -> SymmetricMatrix {
        let mut mat = SymmetricMatrix::zeros(self.n);
        for &(i, j, v) in &self.entries {
            let prev = mat.get(i, j);
            mat.set(i, j, prev + v);
        }
        mat
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
    // X11: compare the banner token by token (case-insensitive) rather than
    // against one exact single-space string. The Matrix Market spec and the
    // reference NIST `mmio` reader tokenize the banner on arbitrary
    // whitespace, so a legal banner whose fields are separated by multiple
    // spaces or tabs must be accepted, not rejected as "unsupported header".
    const BANNER: [&str; 5] = [
        "%%matrixmarket",
        "matrix",
        "coordinate",
        "real",
        "symmetric",
    ];
    let mut header_tokens = header.split_whitespace();
    let banner_ok = BANNER.iter().all(|expected| {
        header_tokens
            .next()
            .is_some_and(|tok| tok.eq_ignore_ascii_case(expected))
    }) && header_tokens.next().is_none();
    if !banner_ok {
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

    let (size_line_no, size_text) =
        size_line.ok_or_else(|| FeralError::IoError(format!("{}: missing size line", source)))?;

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

    // Parse entries.
    //
    // X10: do not trust the header's `nnz` for the up-front allocation. A
    // corrupt header (e.g. nnz = 10^17) would make this a multi-exabyte
    // request; the allocator returns null and `handle_alloc_error` aborts
    // the process instead of returning an `Err`. `nnz` is only a hint here
    // (never validated against the actual entry count), so clamp the
    // reservation to the source byte length — each entry occupies at least
    // one byte in the file, so that is a hard upper bound on the true
    // count. Valid files (where `nnz <= contents.len()` always holds) get
    // the same exact reservation as before.
    let mut entries = Vec::with_capacity(nnz.min(contents.len()));
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
        // X11: f64::from_str silently accepts "nan", "inf", "-inf". Such a
        // value would build an MtxMatrix carrying a non-finite entry that
        // poisons any downstream factorization; reject it here so library
        // callers (parse_mtx / read_mtx) get a clean error like the bench
        // harness already enforces.
        if !v.is_finite() {
            return Err(FeralError::IoError(format!(
                "{}: line {}: non-finite value '{}'",
                source,
                line_no + 1,
                parts[2]
            )));
        }

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

    // X2 (REG-4, repo-review-2026-06-09-verification.md): the size line's
    // `nnz` field was parsed but used only as an allocation hint (clamped
    // by X10) and never validated against the actual entry count. The
    // Matrix Market spec defines that field as the number of entries that
    // follow, so a body with more or fewer data lines than the header
    // declares is a malformed file — previously it parsed silently into a
    // matrix that did not match its own declaration (a truncated file read
    // as a valid smaller matrix). Validate it now. This also turns the
    // bogus-huge-nnz case (X10) from an `Ok` carrying the unvalidated
    // entries into a recoverable count-mismatch `Err`; the no-abort
    // property X10 protects still holds (the reservation above stays
    // clamped to the source byte length, so no multi-exabyte allocation).
    if entries.len() != nnz {
        return Err(FeralError::IoError(format!(
            "{}: declared nnz {} does not match the {} entries in the file",
            source,
            nnz,
            entries.len()
        )));
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

    /// X10 (dev/research/repo-review-2026-06-09.md): the entries Vec was
    /// reserved with `Vec::with_capacity(nnz)` straight from the untrusted
    /// MTX size line. A corrupt header declaring an enormous nnz turns that
    /// into a multi-exabyte allocation request; the allocator returns null
    /// and Rust's `handle_alloc_error` ABORTS the process — a hard crash,
    /// not a recoverable `FeralError`, on malformed input a library caller
    /// cannot guard against. The reservation is clamped to the source byte
    /// length — a hard upper bound on the true entry count — so the parse
    /// runs to completion instead of aborting.
    ///
    /// Revised for X2 (REG-4): the declared nnz is now also *validated*
    /// against the real entry count, so this file (10^17 declared, two
    /// real) returns a recoverable count-mismatch `Err` rather than an `Ok`
    /// carrying the unvalidated entries. The property X10 protects — a
    /// bogus huge nnz must not *abort the process* — still holds: the call
    /// RETURNS an `Err`, which it could only do by surviving the clamped
    /// `with_capacity` (an unclamped reservation would abort here before any
    /// count check could run).
    #[test]
    fn huge_nnz_header_does_not_abort() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 100000000000000000
1 1 2.0
2 1 -1.0
";
        let err = parse_mtx(mtx, "test")
            .expect_err("a bogus huge nnz must return a recoverable Err, not abort the process");
        match err {
            FeralError::IoError(msg) => assert!(
                msg.contains("declared nnz") && msg.contains("does not match"),
                "expected an nnz-count-mismatch error, got: {msg}"
            ),
            other => panic!("expected FeralError::IoError, got {other:?}"),
        }
    }

    /// X2 / REG-4 (repo-review-2026-06-09-verification.md): the size line's
    /// declared nnz was parsed but never checked against the actual number
    /// of data lines, so a truncated or corrupt file parsed silently into a
    /// matrix that did not match its own declaration. The count is now
    /// validated. Oracle: the Matrix Market spec — the size line's third
    /// field is the number of entries that follow. Here the header declares
    /// 4 entries but the body has 2; pre-fix this returned `Ok` with
    /// `entries.len() == 2`, post-fix it is rejected.
    #[test]
    fn declared_nnz_must_match_entry_count() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 4
1 1 2.0
2 2 3.0
";
        let err = parse_mtx(mtx, "test").expect_err(
            "a declared nnz that disagrees with the actual entry count must be rejected (X2)",
        );
        match err {
            FeralError::IoError(msg) => assert!(
                msg.contains("declared nnz") && msg.contains("does not match"),
                "expected an nnz-count-mismatch error, got: {msg}"
            ),
            other => panic!("expected FeralError::IoError, got {other:?}"),
        }
    }

    /// X2 / REG-4: duplicate coordinates must be summed by BOTH conversion
    /// paths. `to_csc` summed them (via `from_triplets`) while `to_dense`
    /// overwrote them (last-wins), so the same file produced two different
    /// matrices. Oracle: the Matrix Market / COO duplicate-summing
    /// convention (scipy, MATLAB). Here (1,1) is listed as 2.0 then 1.0;
    /// pre-fix `to_dense.get(0,0)` was 1.0 while `to_csc` summed to 3.0,
    /// post-fix both are 3.0.
    #[test]
    fn duplicate_entries_summed_consistently_by_both_paths() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 3
1 1 2.0
1 1 1.0
2 2 5.0
";
        let m = parse_mtx(mtx, "test").expect("valid file with a duplicate coordinate");

        let dense = m.to_dense();
        assert_eq!(
            dense.get(0, 0),
            3.0,
            "to_dense must sum duplicate (1,1) = 2.0 + 1.0"
        );
        assert_eq!(dense.get(1, 1), 5.0);

        // to_csc collapses the duplicate to a single summed entry too.
        let csc = m.to_csc().expect("to_csc");
        // Column 0 holds (row 0, col 0); find its value.
        let mut v00 = None;
        for k in csc.col_ptr[0]..csc.col_ptr[1] {
            if csc.row_idx[k] == 0 {
                v00 = Some(csc.values[k]);
            }
        }
        assert_eq!(
            v00,
            Some(3.0),
            "to_csc must sum duplicate (1,1); both paths must agree"
        );
    }

    /// X11 (dev/research/repo-review-2026-06-09.md): the banner was compared
    /// against the exact single-space string. A legal MTX banner separates
    /// its five fields with arbitrary whitespace; the NIST `mmio` reference
    /// tokenizes it. Pre-fix this multi-space banner failed the exact-string
    /// compare and returned "unsupported header"; post-fix the token-by-token
    /// compare accepts it.
    #[test]
    fn multispace_banner_is_accepted() {
        let mtx = "\
%%MatrixMarket   matrix    coordinate real   symmetric
2 2 2
1 1 4.0
2 2 5.0
";
        let m = parse_mtx(mtx, "test")
            .expect("a banner separated by multiple spaces is legal and must parse (X11)");
        assert_eq!(m.n, 2);
        assert_eq!(m.entries.len(), 2);
    }

    /// X11: a banner whose fields are separated by tabs is equally legal.
    /// Pre-fix the exact-string compare rejected it; post-fix it parses.
    #[test]
    fn tab_separated_banner_is_accepted() {
        let mtx = "%%MatrixMarket\tmatrix\tcoordinate\treal\tsymmetric\n2 2 1\n1 1 7.0\n";
        let m =
            parse_mtx(mtx, "test").expect("a tab-separated banner is legal and must parse (X11)");
        assert_eq!(m.n, 2);
        assert_eq!(m.entries.len(), 1);
    }

    /// X11: `f64::from_str` accepts "nan", so pre-fix this file parsed to an
    /// `Ok(MtxMatrix)` carrying a NaN entry that silently poisons any
    /// downstream factorization. Post-fix a non-finite value is rejected.
    #[test]
    fn nan_value_is_rejected() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
1 1 nan
";
        let err = parse_mtx(mtx, "test")
            .expect_err("a NaN entry value must be rejected, not read through (X11)");
        match err {
            FeralError::IoError(msg) => assert!(
                msg.contains("non-finite"),
                "expected a non-finite error, got: {msg}"
            ),
            other => panic!("expected FeralError::IoError, got {other:?}"),
        }
    }

    /// X11: likewise "inf" is accepted by `f64::from_str`. Pre-fix it built an
    /// `MtxMatrix` carrying +inf; post-fix it is rejected.
    #[test]
    fn inf_value_is_rejected() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
1 1 inf
";
        let err = parse_mtx(mtx, "test")
            .expect_err("an inf entry value must be rejected, not read through (X11)");
        match err {
            FeralError::IoError(msg) => assert!(
                msg.contains("non-finite"),
                "expected a non-finite error, got: {msg}"
            ),
            other => panic!("expected FeralError::IoError, got {other:?}"),
        }
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
        assert!((m.entries[0].2 - 123.456_789_012_345_68).abs() < 1e-10);
        assert!((m.entries[1].2 - (-0.009_876_543_210_987_654)).abs() < 1e-16);
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
