use crate::error::FeralError;
use std::path::Path;

/// Inertia recorded in the KKT sidecar JSON.
#[derive(Debug, serde::Deserialize)]
pub struct SidecarInertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

/// Metadata sidecar for a KKT matrix dumped by ripopt's `collect_kkt`.
///
/// Fields match the JSON produced by `dump_kkt_matrix` in ripopt/src/ipm.rs.
/// `n` is the number of primal variables, `m` the number of constraints;
/// the matrix dimension is `n + m`.
///
/// Several fields use `Option<f64>` because NaN/Inf values in the IPM
/// serialize as JSON `null`.
#[derive(Debug, serde::Deserialize)]
pub struct KktSidecar {
    pub problem_name: String,
    pub iteration: usize,
    pub n: usize,
    pub m: usize,
    pub rhs: Vec<Option<f64>>,
    pub inertia: SidecarInertia,
    #[serde(default)]
    pub delta_w: Option<f64>,
    #[serde(default)]
    pub delta_c: Option<f64>,
}

impl KktSidecar {
    /// Returns the RHS as a Vec<f64> if all entries are finite, or None if
    /// any entry is null (NaN/Inf from the IPM).
    pub fn finite_rhs(&self) -> Option<Vec<f64>> {
        self.rhs.iter().copied().collect()
    }
}

/// Read a KKT sidecar JSON file.
pub fn read_sidecar(path: &Path) -> Result<KktSidecar, FeralError> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| FeralError::IoError(format!("{}: {}", path.display(), e)))?;
    serde_json::from_str(&contents)
        .map_err(|e| FeralError::IoError(format!("{}: {}", path.display(), e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_sidecar() {
        let json = r#"{
            "problem_name": "HS071",
            "iteration": 3,
            "n": 4,
            "m": 2,
            "rhs": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "inertia": {"positive": 4, "negative": 2, "zero": 0},
            "delta_w": 1e-8,
            "delta_c": 0.0
        }"#;
        let sc: KktSidecar = serde_json::from_str(json).unwrap();
        assert_eq!(sc.problem_name, "HS071");
        assert_eq!(sc.iteration, 3);
        assert_eq!(sc.n, 4);
        assert_eq!(sc.m, 2);
        assert_eq!(sc.rhs.len(), 6);
        let rhs = sc.finite_rhs().unwrap();
        assert_eq!(rhs, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(sc.inertia.positive, 4);
        assert_eq!(sc.inertia.negative, 2);
        assert_eq!(sc.inertia.zero, 0);
        assert!((sc.delta_w.unwrap() - 1e-8).abs() < 1e-20);
        assert_eq!(sc.delta_c.unwrap(), 0.0);
    }

    #[test]
    fn test_rhs_with_null() {
        let json = r#"{
            "problem_name": "TEST",
            "iteration": 1,
            "n": 2,
            "m": 0,
            "rhs": [1.0, null],
            "inertia": {"positive": 2, "negative": 0, "zero": 0},
            "delta_w": 0.0,
            "delta_c": 0.0
        }"#;
        let sc: KktSidecar = serde_json::from_str(json).unwrap();
        assert!(sc.finite_rhs().is_none());
    }

    #[test]
    fn test_rhs_precision() {
        let json = r#"{
            "problem_name": "TEST",
            "iteration": 1,
            "n": 1,
            "m": 0,
            "rhs": [1.23456789012345678e+02],
            "inertia": {"positive": 1, "negative": 0, "zero": 0},
            "delta_w": 0.0,
            "delta_c": 0.0
        }"#;
        let sc: KktSidecar = serde_json::from_str(json).unwrap();
        let rhs = sc.finite_rhs().unwrap();
        assert!((rhs[0] - 123.456_789_012_345_68).abs() < 1e-10);
    }
}
