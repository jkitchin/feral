//! Finding D4 (`dev/research/repo-review-2026-06-09.md`): the solve-time
//! 2×2 D-block gate in `d_block_solve` (`src/dense/solve.rs`) decides
//! whether to invert a stored 2×2 block with `det = a*c - b*b` (naive,
//! cancellation-prone) tested against the **absolute** floor
//! `zero_tol_2x2 ≈ EPS² ≈ 4.9e-32`. The factor side accepts a 2×2 block
//! with the cancellation-free `det_sym2x2` and the **scale-invariant**
//! SSIDS det floor (`factor.rs`). So a block the factorization validly
//! accepted and stored can be silently *skipped* at solve time — the
//! corresponding solution components are left untouched (wrong solution,
//! no error, no flag). Two independent trigger modes:
//!
//! (b) scale: a well-conditioned block at small absolute scale whose
//!     true `|det|` sits below the absolute `zero_tol_2x2` floor even
//!     though the scale-invariant factor floor accepts it.
//!
//! Facet (a) of the finding — a nonsingular block whose *naive*
//! `a*c - b*b` rounds to exactly `0.0` — is **not** independently
//! reproducible as a factor-accepted-then-skipped bug: a naive
//! cancellation to `0.0` requires `|det| ≲ ULP(a*c) ≈ a*c·2⁻⁵²`, i.e.
//! condition `≳ 2⁵²`, and the *same* SSIDS scale-invariant floor the
//! factor uses for acceptance rejects any such block (verified:
//! `D = [[2⁵³+1, 2⁵³], [2⁵³, 2⁵³]]` has true `det = 2⁵³` yet
//! `detpiv = 0` ⇒ floor rejects). So the factor side never stores it
//! and the solve never sees it. The fix below routes the solve gate
//! through that identical predicate, so solve and factor agree on this
//! axis by construction. See dev/tried-and-rejected.md (D4, facet a).
//!
//! ## Oracle (external, hand-constructed)
//!
//! For each case we hand-build a `Factors` with `L = I`, identity
//! permutation, unit equilibration, and a single 2×2 `D` block, then set
//! `rhs = D · x_true` for a chosen `x_true`. With `L = I` the solve
//! reduces to `x = D⁻¹ · rhs = x_true`. The oracle `x_true` comes from
//! the forward product `D · x_true` (pure linear algebra, computed here),
//! never from the solver — so this is an independent reference. Pre-fix
//! the gate skips the block and returns `x ≈ rhs` (off by orders of
//! magnitude); post-fix it returns `x_true`.

use feral::dense::factor::Factors;
use feral::solve;

/// Build a 2×2 `Factors`: `L = I`, identity perm, unit `d_eq`, one 2×2
/// block `[[a, b], [b, c]]`. `zero_tol`/`zero_tol_2x2` are the library
/// defaults (`EPS`, `EPS²`) so the test exercises the production gate.
fn factors_2x2(a: f64, b: f64, c: f64) -> Factors {
    let eps = f64::EPSILON;
    Factors {
        n: 2,
        // column-major 2×2 identity (unit lower triangular, explicit diag)
        l: vec![1.0, 0.0, 0.0, 1.0],
        d_diag: vec![a, c],
        d_subdiag: vec![b, 0.0],
        perm: vec![0, 1],
        perm_inv: vec![0, 1],
        d_eq: vec![1.0, 1.0],
        needs_refinement: false,
        zero_tol: eps,
        zero_tol_2x2: eps * eps,
    }
}

/// Symmetric 2×2 matrix-vector product `[[a,b],[b,c]] · x`.
fn matvec_2x2(a: f64, b: f64, c: f64, x: &[f64; 2]) -> [f64; 2] {
    [a * x[0] + b * x[1], b * x[0] + c * x[1]]
}

/// D4(b): well-conditioned block at small scale. `D = [[1e-16, 1e-17],
/// [1e-17, 1e-16]]` has `det = 9.9e-33 < zero_tol_2x2 ≈ 4.9e-32`, so the
/// pre-fix absolute gate skips it. The scale-invariant SSIDS floor the
/// factor side uses accepts it (`max_piv = 1e-16`, `detpiv ≈ 9.9e-17`
/// vs `cancel_floor = 5e-17`), so the block is one the factorization
/// would store. Post-fix the solve must invert it.
#[test]
fn d4_small_scale_block_is_solved_not_skipped() {
    let (a, b, c) = (1e-16, 1e-17, 1e-16);
    let x_true = [1.0, 1.0];
    let rhs = matvec_2x2(a, b, c, &x_true);

    let factors = factors_2x2(a, b, c);
    let x = solve(&factors, &rhs).expect("solve");

    // Pre-fix: x ≈ rhs ≈ [1.1e-16, 1.1e-16] (block skipped). Post-fix:
    // x ≈ [1, 1]. A loose relative bound separates the two by 16 orders.
    assert!(
        (x[0] - x_true[0]).abs() < 1e-6 && (x[1] - x_true[1]).abs() < 1e-6,
        "D4(b): small-scale 2×2 must be solved, not skipped; \
         got x = {x:?}, expected ≈ {x_true:?}"
    );
}

/// D4 consistency guard: a block the factor-side SSIDS floor *rejects*
/// (ill-conditioned, `detpiv = 0`) must be *skipped* by the solve, just
/// as the factor side would never store it as an invertible 2×2. This
/// pins the other half of "solve agrees with factor acceptance":
/// `D = [[2^53+1, 2^53], [2^53, 2^53]]` has true `det = 2^53 > 0` but
/// condition `~2^53`, so the shared predicate rejects it and the solve
/// leaves `w` untouched (the pre-fix naive gate also skipped it, via a
/// different — accidental — route: `fl(a*c) = fl(b*b)` so naive
/// `det = 0.0`). Asserting the *skip* documents that the fix did not
/// start inverting rejected blocks.
#[test]
fn d4_rejected_block_is_skipped_like_factor() {
    let p = (1u64 << 53) as f64; // 2^53, exactly representable
    let (a, b, c) = (p + 1.0, p, p);

    // With L = I and identity perm/equilibration the skip leaves w = rhs.
    let x_true = [1.0, 1.0];
    let rhs = matvec_2x2(a, b, c, &x_true);

    let factors = factors_2x2(a, b, c);
    let x = solve(&factors, &rhs).expect("solve");

    // Skipped ⇒ x == rhs (block not inverted), NOT x_true.
    assert_eq!(
        x,
        rhs.to_vec(),
        "D4: a factor-rejected (ill-conditioned, detpiv=0) 2×2 must be \
         skipped by solve, matching factor-side acceptance; got x = {x:?}"
    );
}
