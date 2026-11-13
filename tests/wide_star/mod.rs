//! Wide-star KKT fixture: `groups` independent blocks of `local`
//! variables, every block coupled to the same `shared` variables. Each
//! block is a leaf front whose contribution block is the dense
//! `shared × shared` update, and all of them land in the one small
//! front of the shared variables — the elimination-tree shape of a
//! many-row QP's KKT (each row a leaf, the shared variables the root).

use feral::CscMatrix;

/// Sizes of a wide star.
#[derive(Clone, Copy, Debug)]
pub struct Star {
    pub groups: usize,
    pub local: usize,
    pub shared: usize,
    /// Zero local diagonal: a local column Bunch-Kaufman cannot pair
    /// with another (`local == 1`) is delayed into the shared front.
    pub zero_local_diagonal: bool,
}

/// Lower triangle of the star's KKT. Shared variables come first, then
/// the groups; each local variable couples to every `local`-th shared
/// variable, so a group's columns together reach all of them.
pub fn star_kkt(star: Star) -> CscMatrix {
    let Star {
        groups,
        local,
        shared,
        zero_local_diagonal,
    } = star;
    let n = shared + groups * local;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for s in 0..shared {
        rows.push(s);
        cols.push(s);
        vals.push(-(1.0 + 0.01 * (s % 5) as f64));
    }
    for g in 0..groups {
        let base = shared + g * local;
        for t in 0..local {
            let v = base + t;
            rows.push(v);
            cols.push(v);
            vals.push(if zero_local_diagonal {
                0.0
            } else {
                4.0 + 0.1 * t as f64
            });
            if t > 0 {
                rows.push(v);
                cols.push(v - 1);
                vals.push(0.5);
            }
            for s in (t..shared).step_by(local) {
                rows.push(v);
                cols.push(s);
                vals.push(0.3 + 0.001 * ((g * 31 + s * 7) % 97) as f64);
            }
        }
    }
    match CscMatrix::from_triplets(n, &rows, &cols, &vals) {
        Ok(m) => m,
        Err(e) => panic!("star KKT: {e}"),
    }
}
