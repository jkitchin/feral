use crate::dense::matrix::SymmetricMatrix;

/// Compute iterative infinity-norm equilibration scaling.
/// Returns a diagonal scaling vector d such that each row i of D·A·D
/// has ||row i||∞ ≈ 1.
///
/// Algorithm: Knight-Ruiz iterative equilibration (Section 2.8 of spec).
pub fn equilibrate_scaling(matrix: &SymmetricMatrix) -> Vec<f64> {
    let n = matrix.n;
    let mut d = vec![1.0; n];

    let max_iter = 10;
    let tol = 1e-8;

    for _ in 0..max_iter {
        let mut max_deviation = 0.0f64;

        for i in 0..n {
            // Compute max |d[i] * A[i,j] * d[j]| over j, using symmetry
            let mut max_entry = 0.0f64;
            for j in 0..n {
                let a_ij = matrix.get(i, j);
                let val = (d[i] * a_ij * d[j]).abs();
                if val > max_entry {
                    max_entry = val;
                }
            }

            if max_entry > 0.0 {
                d[i] /= max_entry.sqrt();
                let deviation = (1.0 - max_entry).abs();
                if deviation > max_deviation {
                    max_deviation = deviation;
                }
            }
            // If max_entry == 0, row is all zeros: d[i] stays at 1.0 (spec guard)
        }

        if max_deviation < tol {
            break;
        }
    }

    d
}
