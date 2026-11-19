//! Issue #237 / #202: opt-in reuse of the `InfNorm` scaling vector
//! across same-pattern factorizations (`Solver::with_infnorm_reuse`).
//!
//! Oracles:
//! - "fresh scaling" is `scaling::compute_scaling(.., InfNorm)`, the
//!   pre-existing computation, compared bitwise;
//! - inertia of the chain KKT is analytic: `t·nx` positive (strictly
//!   diagonally dominant positive H block) and `(t−1)·nx` negative.

use feral::scaling::{compute_scaling, ScalingStrategy};
use feral::{CscMatrix, FactorStatus, Inertia, InfNormReuse, NumericParams, Solver};

/// Chain-structured saddle-point KKT (same family as
/// `tests/issue203_ordering_race.rs`): `t` dense blocks of `nx`
/// variables, `nx` coupling constraints between consecutive blocks.
fn chain_kkt(t: usize, nx: usize) -> CscMatrix {
    let n_vars = t * nx;
    let n = n_vars + (t - 1) * nx;
    let mut ent: Vec<(usize, usize, f64)> = Vec::new();
    for b in 0..t {
        for i in 0..nx {
            for j in 0..i {
                ent.push((
                    b * nx + i,
                    b * nx + j,
                    0.25 + 0.01 * ((i + j + b) % 7) as f64,
                ));
            }
        }
    }
    for b in 0..(t - 1) {
        for i in 0..nx {
            let row = n_vars + b * nx + i;
            for j in 0..nx {
                let v = 0.5 + 0.01 * ((b + i + j) % 5) as f64;
                ent.push((row, b * nx + j, v));
                ent.push((row, (b + 1) * nx + j, -v));
            }
        }
    }
    let mut absrow = vec![0.0f64; n];
    for &(r, c, v) in &ent {
        absrow[r] += v.abs();
        absrow[c] += v.abs();
    }
    for (k, &a) in absrow.iter().enumerate() {
        ent.push((k, k, if k < n_vars { a + 1.0 } else { -1e-2 }));
    }
    let rows: Vec<usize> = ent.iter().map(|e| e.0).collect();
    let cols: Vec<usize> = ent.iter().map(|e| e.1).collect();
    let vals: Vec<f64> = ent.iter().map(|e| e.2).collect();
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("valid CSC")
}

/// Multiplicative ±`amp` drift of every value; keeps signs (and so the
/// analytic inertia, at the amplitudes used here).
fn drift(m: &CscMatrix, step: usize, amp: f64) -> CscMatrix {
    let mut out = m.clone();
    for (k, v) in out.values.iter_mut().enumerate() {
        let u = ((k.wrapping_mul(2654435761) ^ step.wrapping_mul(40503)) % 1000) as f64;
        *v *= 1.0 + amp * (u / 1000.0 - 0.5);
    }
    out
}

fn infnorm_solver(reuse: Option<InfNormReuse>) -> Solver {
    let np = NumericParams {
        scaling: ScalingStrategy::InfNorm,
        ..NumericParams::default()
    };
    Solver::with_params(np, Default::default())
        .with_parallel(false)
        .with_infnorm_reuse(reuse)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn stored_scaling(s: &Solver) -> Vec<f64> {
    s.factors().expect("factor stored").scaling.clone()
}

fn fresh(m: &CscMatrix) -> Vec<f64> {
    compute_scaling(m, &ScalingStrategy::InfNorm)
        .expect("infnorm")
        .0
}

const T: usize = 20;
const NX: usize = 6;

fn expected_inertia() -> Inertia {
    Inertia::new(T * NX, (T - 1) * NX, 0)
}

#[test]
fn default_recomputes_every_call() {
    let a = chain_kkt(T, NX);
    let mut s = infnorm_solver(None);
    for step in 0..4 {
        let m = drift(&a, step, 0.02);
        assert!(matches!(s.factor(&m, None), FactorStatus::Success));
        assert_eq!(bits(&stored_scaling(&s)), bits(&fresh(&m)), "step {step}");
        assert_eq!(*s.inertia().expect("inertia"), expected_inertia());
    }
    assert_eq!(s.infnorm_reuse_hit_count(), 0);
}

#[test]
fn small_drift_reuses_previous_vector_and_keeps_inertia() {
    let a = chain_kkt(T, NX);
    let mut s = infnorm_solver(Some(InfNormReuse::default()));
    let m0 = drift(&a, 0, 0.02);
    assert!(matches!(s.factor(&m0, None), FactorStatus::Success));
    let d0 = stored_scaling(&s);
    assert_eq!(bits(&d0), bits(&fresh(&m0)), "first call computes fresh");
    for step in 1..4 {
        let m = drift(&a, step, 0.02);
        assert!(matches!(s.factor(&m, None), FactorStatus::Success));
        assert_eq!(
            bits(&stored_scaling(&s)),
            bits(&d0),
            "step {step} reuses d0"
        );
        assert_eq!(*s.inertia().expect("inertia"), expected_inertia());
        // The reused factor still solves the current system.
        let n = m.n;
        let x_true: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
        let b = symv(&m, &x_true);
        let x = s.solve_refined(&m, &b).expect("solve");
        let err = x
            .iter()
            .zip(&x_true)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-8, "step {step}: max error {err:e}");
    }
    assert_eq!(s.infnorm_reuse_hit_count(), 3);
}

#[test]
fn guard_rejects_a_rescaled_row_and_recomputes() {
    let a = chain_kkt(T, NX);
    let mut s = infnorm_solver(Some(InfNormReuse::default()));
    assert!(matches!(s.factor(&a, None), FactorStatus::Success));
    // Scale every entry of row/column 7 by 100: that row of D·A·D
    // (old D) grows ~100x, far outside the ratio-2 guard.
    let mut m = a.clone();
    for j in 0..m.n {
        for k in m.col_ptr[j]..m.col_ptr[j + 1] {
            let i = m.row_idx[k];
            if i == 7 || j == 7 {
                m.values[k] *= 100.0;
            }
        }
    }
    assert!(matches!(s.factor(&m, None), FactorStatus::Success));
    assert_eq!(s.infnorm_reuse_hit_count(), 0, "guard must reject");
    assert_eq!(bits(&stored_scaling(&s)), bits(&fresh(&m)));
    assert_eq!(*s.inertia().expect("inertia"), expected_inertia());
}

#[test]
fn max_age_forces_a_fresh_computation() {
    let a = chain_kkt(T, NX);
    let policy = InfNormReuse {
        max_age: 2,
        max_row_ratio: f64::INFINITY,
    };
    let mut s = infnorm_solver(Some(policy));
    let mut hits = Vec::new();
    for step in 0..7 {
        let m = drift(&a, step, 0.02);
        let before = s.infnorm_reuse_hit_count();
        assert!(matches!(s.factor(&m, None), FactorStatus::Success));
        let hit = s.infnorm_reuse_hit_count() > before;
        if !hit {
            assert_eq!(bits(&stored_scaling(&s)), bits(&fresh(&m)), "step {step}");
        }
        hits.push(hit);
    }
    // fresh, reuse, reuse, fresh, reuse, reuse, fresh
    assert_eq!(hits, vec![false, true, true, false, true, true, false]);
}

#[test]
fn max_age_zero_never_reuses() {
    let a = chain_kkt(T, NX);
    let policy = InfNormReuse {
        max_age: 0,
        max_row_ratio: f64::INFINITY,
    };
    let mut s = infnorm_solver(Some(policy));
    for step in 0..3 {
        let m = drift(&a, step, 0.02);
        assert!(matches!(s.factor(&m, None), FactorStatus::Success));
        assert_eq!(bits(&stored_scaling(&s)), bits(&fresh(&m)));
    }
    assert_eq!(s.infnorm_reuse_hit_count(), 0);
}

#[test]
fn pattern_change_drops_the_cache() {
    let a = chain_kkt(T, NX);
    let b = chain_kkt(T + 1, NX);
    let mut s = infnorm_solver(Some(InfNormReuse::default()));
    assert!(matches!(s.factor(&a, None), FactorStatus::Success));
    assert!(matches!(s.factor(&b, None), FactorStatus::Success));
    assert_eq!(s.infnorm_reuse_hit_count(), 0);
    assert_eq!(bits(&stored_scaling(&s)), bits(&fresh(&b)));
    assert_eq!(
        *s.inertia().expect("inertia"),
        Inertia::new((T + 1) * NX, T * NX, 0)
    );
    // Back on b, a reuse is possible again.
    assert!(matches!(
        s.factor(&drift(&b, 1, 0.02), None),
        FactorStatus::Success
    ));
    assert_eq!(s.infnorm_reuse_hit_count(), 1);
}

#[test]
fn auto_pinned_to_infnorm_reuses_after_the_first_call() {
    // `tridiag(6, 10, 1)` routes Auto -> InfNorm (issue #49 fixture).
    let n = 400usize;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(10.0);
        if i + 1 < n {
            rows.push(i + 1);
            cols.push(i);
            vals.push(if i % 2 == 0 { 6.0 } else { 1.0 });
        }
    }
    let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("valid CSC");
    let mut s = Solver::new()
        .with_parallel(false)
        .with_infnorm_reuse(Some(InfNormReuse::default()));
    for step in 0..3 {
        assert!(matches!(
            s.factor(&drift(&a, step, 0.02), None),
            FactorStatus::Success
        ));
    }
    assert_eq!(s.infnorm_reuse_hit_count(), 2);
    assert_eq!(s.mc64_cache_hit_count(), 0);
}

/// `y = A·x` for lower-triangle CSC storage.
fn symv(a: &CscMatrix, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.n];
    for j in 0..a.n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            y[i] += a.values[k] * x[j];
            if i != j {
                y[j] += a.values[k] * x[i];
            }
        }
    }
    y
}
