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
            if !c.is_multiple_of(3) && !extra.contains(&c) {
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

/// Sizes of the qcqp-like KKT fixture.
pub const Q_NX: usize = 300;
pub const Q_MD: usize = 150;
pub const Q_MC: usize = 150;

/// A qcqp-like IPM KKT in pounce's `StdAugSystem` layout, unknowns
/// `(x, s, c, d)`:
///
/// ```text
/// [ W + D_x   0     J_cᵀ   J_dᵀ ]
/// [ 0         D_s   0      −I   ]
/// [ J_c       0     −D_c   0    ]
/// [ J_d       −I    0      −D_d ]
/// ```
///
/// The slack, `c` and `d` columns have stored degree ≤ 2, so
/// `pick_ordering_preprocess` recommends `LdltCompress`, and whether the
/// `Auto` fill race keeps it depends on the values (the MC64 matching).
///
/// `iterate = None` is pounce's least-squares-multiplier init matrix:
/// `W` all explicit zeros, `D_x = D_s = 1`, `D_c = D_d = 0`; its `Auto`
/// analysis resolves `LdltCompress`. `iterate = Some(k)` has a small
/// random `W` (entries in ±1e-3), `D_x = Σ_i |W_ij| + 10^e`, `D_s = 10^e`
/// (`e ∈ [−6, 6]`), `D_c = D_d = 1e-8`; its `Auto` analysis resolves
/// `None`. The pattern is the same for every member.
///
/// Inertia oracle (Sylvester/Haynsworth): `H = blockdiag(W + D_x, D_s)` is
/// positive definite (`W + D_x` strictly diagonally dominant with positive
/// diagonal, Gershgorin; `W = 0` and `D = I` at the init), and
/// `A = [[J_c, 0], [J_d, −I]]` has full row rank (`J_c` has pivot
/// columns `2i` and its other entries sit in odd columns, and the `d`
/// rows carry `−I` on the slacks), so `−D − A H⁻¹ Aᵀ` is negative
/// definite for `D ⪰ 0`: inertia `(n_x + m_d, m_c + m_d, 0)`.
pub fn qcqp_like(iterate: Option<u64>) -> CscMatrix {
    let init = iterate.is_none();
    let mut rng = Rng(1);
    let mut vr = Rng(0x0abc ^ iterate.map_or(0, |k| 0x9e37_79b9 * (k + 1)));
    let (nx, md, mc) = (Q_NX, Q_MD, Q_MC);
    let n = nx + md + mc + md;
    let (ss, cs, ds) = (nx, nx + md, nx + md + mc);
    let mut t: Vec<(usize, usize, f64)> = Vec::new();
    let mut rowsum = vec![0.0f64; nx];
    let mut wt = Vec::new();
    for j in 0..nx {
        for _ in 0..3 {
            let i = rng.below(nx);
            if i > j {
                let v = if init {
                    0.0
                } else {
                    1e-3 * (2.0 * vr.unit() - 1.0)
                };
                rowsum[i] += v.abs();
                rowsum[j] += v.abs();
                wt.push((i, j, v));
            }
        }
    }
    let expo = |r: &mut Rng| pow10(r.below(13) as i32 - 6);
    for (j, rs) in rowsum.iter().enumerate() {
        let d = if init { 1.0 } else { rs + expo(&mut vr) };
        t.push((j, j, d));
    }
    t.extend(wt);
    for i in 0..md {
        let d_s = if init { 1.0 } else { expo(&mut vr) };
        t.push((ss + i, ss + i, d_s));
        t.push((ds + i, ss + i, -1.0));
        t.push((ds + i, ds + i, if init { 0.0 } else { -1e-8 }));
        for _ in 0..4 {
            let j = rng.below(nx);
            t.push((ds + i, j, 2.0 * vr.unit() - 1.0));
        }
    }
    for i in 0..mc {
        t.push((cs + i, cs + i, if init { 0.0 } else { -1e-8 }));
        t.push((cs + i, 2 * i, 1.0 + vr.unit()));
        for _ in 0..3 {
            let j = rng.below(nx);
            if j % 2 == 1 {
                t.push((cs + i, j, 2.0 * vr.unit() - 1.0));
            }
        }
    }
    let r: Vec<usize> = t.iter().map(|x| x.0).collect();
    let c: Vec<usize> = t.iter().map(|x| x.1).collect();
    let v: Vec<f64> = t.iter().map(|x| x.2).collect();
    match CscMatrix::from_triplets(n, &r, &c, &v) {
        Ok(a) => a,
        Err(e) => panic!("qcqp_like({iterate:?}): {e}"),
    }
}

/// Sylvester/Haynsworth inertia of every [`qcqp_like`] member.
pub fn qcqp_like_inertia() -> Inertia {
    Inertia {
        positive: Q_NX + Q_MD,
        negative: Q_MC + Q_MD,
        zero: 0,
    }
}
