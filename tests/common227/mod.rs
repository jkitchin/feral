//! Shared fixtures for the issue #227 tests (`ReanalysisPolicy`).
//!
//! The synthetic KKT path `[[D_k, Jᵀ], [J, −δI]]`: `D_k` positive
//! diagonal whose entries drift over 12 orders of magnitude along the
//! path, `J` full row rank (its columns `3i` form a nonsingular diagonal
//! block), `δ > 0`. By Sylvester's law of inertia applied to the block
//! LDLᵀ `[[I,0],[J D⁻¹, I]]·diag(D, −δI − J D⁻¹ Jᵀ)·[[I, D⁻¹Jᵀ],[0, I]]`
//! (Haynsworth), the inertia is `(n_x, m, 0)`: `D > 0` and the Schur
//! complement `−δI − J D⁻¹ Jᵀ` is negative definite. That count is an
//! external oracle — it does not come from feral.
#![allow(dead_code)]

use feral::numeric::factorize::SparseFactors;
use feral::{CscMatrix, Inertia};

/// SplitMix64 — the generator the golden-bits test uses.
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, k: usize) -> usize {
        (self.next_u64() % k as u64) as usize
    }
}

pub const N_X: usize = 300;
pub const M: usize = 100;
pub const N_ITER: usize = 8;
pub const DELTA: f64 = 1e-8;

/// Fixed pattern of `J`: row `i` has its pivot column `3i` plus two
/// other columns not divisible by 3 (so `J[:, {3i}]` stays diagonal and
/// `J` has full row rank). Returns `(row, col, value)` triplets.
fn jacobian(seed: u64) -> Vec<(usize, usize, f64)> {
    let mut rng = Rng(seed);
    let mut t = Vec::new();
    for i in 0..M {
        t.push((i, 3 * i, 1.0 + rng.unit()));
        let mut extra = Vec::new();
        while extra.len() < 2 {
            let c = rng.below(N_X);
            if c % 3 != 0 && !extra.contains(&c) {
                extra.push(c);
            }
        }
        extra.sort_unstable();
        for c in extra {
            t.push((i, c, 2.0 * rng.unit() - 1.0));
        }
    }
    t
}

/// `10^e` for small integer `e`, exactly rounded (10^|e| is exact for
/// |e| <= 22; the negative case is one correctly rounded division).
fn pow10(e: i32) -> f64 {
    let mut p = 1.0f64;
    for _ in 0..e.unsigned_abs() {
        p *= 10.0;
    }
    if e < 0 {
        1.0 / p
    } else {
        p
    }
}

/// The `k`-th iterate (`0 <= k < N_ITER`) of the synthetic KKT path.
pub fn synthetic_kkt(k: usize) -> CscMatrix {
    let jt = jacobian(0x227);
    let mut exps = Rng(0x2270);
    let e: Vec<f64> = (0..N_X).map(|_| 12.0 * exps.unit() - 6.0).collect();
    let mut rng = Rng(0x2271 + k as u64);
    let n = N_X + M;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for (j, ej) in e.iter().enumerate() {
        r.push(j);
        c.push(j);
        // Integer exponent drifting from 0 (k = 0) to ±6 (k = N_ITER−1):
        // 10^e by exact multiplication/division, never `powf` (LLVM may
        // rewrite `pow(10, x)` differently in debug and release builds,
        // which would make the fixture build-dependent).
        let ex = (ej * k as f64 / (N_ITER - 1) as f64).round() as i32;
        v.push(pow10(ex) * (1.0 + 0.5 * rng.unit()));
    }
    for &(i, j, val) in &jt {
        r.push(N_X + i);
        c.push(j);
        v.push(val * (1.0 + 0.01 * rng.unit()));
    }
    for i in 0..M {
        r.push(N_X + i);
        c.push(N_X + i);
        v.push(-DELTA);
    }
    match CscMatrix::from_triplets(n, &r, &c, &v) {
        Ok(a) => a,
        Err(e) => panic!("synthetic_kkt({k}): {e}"),
    }
}

/// The Sylvester/Haynsworth inertia of every synthetic iterate.
pub fn synthetic_inertia() -> Inertia {
    Inertia {
        positive: N_X,
        negative: M,
        zero: 0,
    }
}

/// FNV-1a accumulator over u64 words.
pub struct Fnv(pub u64);
impl Fnv {
    pub fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    pub fn word(&mut self, w: u64) {
        for b in w.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    pub fn f64s(&mut self, xs: &[f64]) {
        self.word(xs.len() as u64);
        for x in xs {
            self.word(x.to_bits());
        }
    }
    pub fn usizes(&mut self, xs: &[usize]) {
        self.word(xs.len() as u64);
        for x in xs {
            self.word(*x as u64);
        }
    }
}

/// Digest of a factor: its global `perm`, every front's shape, local
/// `perm`, `l`, `d_diag`, `d_subdiag` bits, and the inertia.
pub fn factor_digest(f: &SparseFactors, inertia: &Inertia) -> u64 {
    let mut h = Fnv::new();
    h.usizes(&f.perm);
    h.word(f.node_factors.len() as u64);
    for nf in &f.node_factors {
        h.word(nf.nelim as u64);
        h.word(nf.nrow as u64);
        let ff = &nf.frontal_factors;
        h.usizes(&ff.perm);
        h.f64s(&ff.l);
        h.f64s(&ff.d_diag);
        h.f64s(&ff.d_subdiag);
    }
    h.word(inertia.positive as u64);
    h.word(inertia.negative as u64);
    h.word(inertia.zero as u64);
    h.0
}
