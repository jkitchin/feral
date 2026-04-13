# Implementation plan: scaling-aware pivot rejection (Phase 2.2.2)

## Goal

Land the minimum pivot-rejection machinery that closes the ACOPP30
regression (`2.27e+46` under MC64, commit `8a95825`) and recovers
at least two of the four `#[ignore]`d `mc64_regression.rs` tests,
without regressing any of the 141 currently-passing tests. This is
**Option A** of the research note: a column-relative `zero_tol` on
the 1×1 path plus a Duff-Reid growth bound on the 2×2 path, both
wired through the existing `ZeroPivotAction::ForceAccept` pathway.
**Delayed pivoting is deferred to Phase 2.3.**

**Design document:** `dev/research/scaling-aware-pivot-rejection.md`
(commit `c51709b`). This plan assumes the reader has read the
research note's §2 (canonical solver survey), §3 (option A/B/C/D
tradeoffs), and §6 (risk register). It does not re-derive the
algorithm.

**Diagnostic:** `dev/debugging/2026-04-12-acopp30-regression.md`
(commit `3d0716b`). The root cause is that MC64 shrinks the worst
ACOPP30 pivots from `~1e-8` to `~3.6e-10`, which is above `zero_tol
= f64::EPSILON` so `ForceAccept` does not fire; the solve then
divides by these and amplifies rounding by `~1e9` per affected
position.

**Target matrices:**
- **Primary**: `ACOPP30_0000` (n=209). Must drop from `2.27e+46` to
  at most `2.84e+16` (the Identity-path baseline); ideal landing
  zone `[1e-8, 1e-4]`.
- **Secondary**: `CHWIRUT1_0000` (n=645, post-2.2.1 `8.50e+02`) and
  `CRESC100_0000` (n=806, post-2.2.1 `1.43e+02`). Most likely to
  cross the `< 1e-4` bar with column-relative rejection alone.
- **Stretch**: `CRESC132_0000` (n=5314, post-2.2.1 `1.37e+05`).
  Expected to improve but probably not reach `< 1e-6` without full
  delayed pivoting (Phase 2.3).

## Scope

**In scope:**

- A single new `pivot_threshold: f64` field on `BunchKaufmanParams`
  (`src/dense/factor.rs:6–29`).
- A column-relative rejection clause in the dense 1×1 acceptance
  path (`src/dense/factor.rs` around lines 494, 517, 534 for
  `factor_frontal`, and lines 179, 203, 224 for the non-frontal
  `factor`).
- A Duff-Reid growth bound helper for the 2×2 acceptance path,
  added alongside the existing `det.abs() <= params.zero_tol_2x2`
  test at `src/dense/factor.rs:560`. Mirrors MUMPS
  `dfac_front_aux.F:1599-1606` and SSIDS `block_ldlt.hxx:89-119`.
- Threading `pivot_threshold` from `BunchKaufmanParams` into the
  multifrontal kernel at `src/numeric/factorize.rs:192` (no new
  plumbing required — it flows through `params: &BunchKaufmanParams`
  which is already in the signature).
- Optional telemetry counter `n_pivots_rejected: usize` on
  `FrontalFactors` and aggregated on `SparseFactors` for Step 8
  validation assertions.
- Hand-computable unit tests in `tests/pivot_rejection.rs` (new
  file) with oracles derived from paper/first-principles.
- Un-ignore the four `mc64_regression.rs` tests if the residual
  targets are met. If not, document the measured numbers in
  `dev/validation/phase-2.2.2-pivot-rejection.md` and decide
  case-by-case whether to relax a target (with an explicit
  `decisions.md` entry) or leave the test `#[ignore]`d with an
  updated baseline comment.

**Out of scope (deferred):**

- Full Duff-Reid delayed pivoting (Option C in the research note).
  Phase 2.3.
- SSIDS's determinant cancellation guard (`ldlt_tpp.cxx:181-183`,
  the `fabs(detpiv) >= max(detpiv0/2, detpiv1/2)` clause). Phase
  2.2.2 uses MUMPS's simpler growth bound only; the cancellation
  guard is Phase 2.2.3 work if it turns out to be needed.
- Reworking the BK77 decision tree (Option B). The BK77 `alpha ≈
  0.6404` branch selection is kept untouched; `pivot_threshold` is
  an **additional** check layered on top of BK77's existing
  acceptance, not a replacement.
- The deferred 2×2 trace-vs-`a00` classification fix (tracked in
  `dev/tried-and-rejected.md`). Phase 2.2.3. See Risk R6.
- Regularization on rejected pivots (Option D). Phase 2.3+ if ever.
- Changing `ScalingStrategy` defaults, the MC64 implementation, or
  the `SupernodeParams` surface beyond what the new
  `pivot_threshold` flag needs.

## Dependencies

- `dev/research/scaling-aware-pivot-rejection.md` — design document.
- `dev/debugging/2026-04-12-acopp30-regression.md` — diagnostic.
- `ref/mumps/src/dfac_front_aux.F:1494-1606` — 1×1 and 2×2
  threshold tests (algorithmic cross-check for the new code).
- `ref/spral/src/ssids/cpu/kernels/block_ldlt.hxx:89-119` — SSIDS
  `test_2x2` helper (algorithmic cross-check for the 2×2 path).
- `ref/spral/src/ssids/cpu/kernels/ldlt_tpp.cxx:89-119,164-270` —
  SSIDS scalar `ldlt_tpp_factor` (algorithmic cross-check for the
  1×1 path).
- `dev/validation/phase-2.2.1-mc64-sweep.md` — the 7-matrix sanity
  panel baseline numbers that Step 8 will compare against.
- Existing feral code: `src/dense/factor.rs`, `src/numeric/factorize.rs`,
  `src/symbolic/supernode.rs`, `tests/mc64_regression.rs`,
  `tests/threshold_consistency.rs`.

## Test-first order

Per CLAUDE.md, tests come before implementation. Oracles must come
from external sources (paper hand derivations, reference solver
outputs) rather than from feral itself in the same session.

### Test oracles available before implementation starts

1. **Hand-computed 3×3 and 4×4 matrices.** Pivot magnitudes,
   column maxes, and accept/reject decisions are derivable
   analytically from first principles and from the research note
   §5.1 "Hand-computable miniature reproductions".
2. **Bunch-Kaufman Example 2** (citet:bunch1977stable). A 5×5
   matrix with a known exact pivot sequence under BK77. All of
   its pivots are `O(1)`, so `pivot_threshold = 0.01` must be
   inactive on this matrix — this is the no-regression sanity
   check.
3. **Existing 139-ish-matrix corpus via the ignored regression
   tests.** The four `mc64_regression.rs` tests supply residual
   oracles derived from canonical MUMPS and SSIDS runs captured in
   the Phase 2.1.2 decisions log (`dev/decisions.md` 2026-04-12
   entry).
4. **MUMPS growth bound formula** (`dfac_front_aux.F:1602-1604`,
   verbatim Fortran). The exact form
   `(|a22|·RMAX + AMAX·TMAX)·u ≤ |det|` is the algebraic oracle
   for the 2×2 rejection test. A hand-computed 2×2 case can be
   constructed to sit just above and just below the boundary at
   `u = 0.01` and `u = 0.1`.

### Tests written before implementation

- `tests/pivot_rejection.rs` (new). Contents per Step 2 below.
- Un-ignore of `tests/mc64_regression.rs` is deferred to Step 7
  — do **not** flip them in Step 2, because they need the
  implementation to be working before they are meaningful.

### Gate point

Before modifying any acceptance clause in `src/dense/factor.rs`,
confirm `tests/pivot_rejection.rs` compiles and the `#[ignore]`d
cases fail with the current (Phase 2.2.1) behavior. The `#[ignore]`
flag exists specifically so the suite stays green until Step 6
flips the tests on.

## Data model changes

### `BunchKaufmanParams` (`src/dense/factor.rs:6–29`)

Add one field:

```rust
pub struct BunchKaufmanParams {
    pub alpha: f64,
    pub zero_tol: f64,
    pub zero_tol_2x2: f64,
    pub on_zero_pivot: ZeroPivotAction,

    /// Column-relative pivot threshold `u` (SSIDS `options%u`,
    /// MUMPS `CNTL(1)`). A 1×1 candidate pivot is rejected when
    /// `|a[k,k]| < u * max_{i>k} |a[i,k]|`, in addition to the
    /// existing absolute `zero_tol` check. A 2×2 candidate is
    /// rejected when the Duff-Reid growth bound
    /// `(|a22|·col_max_k + |a21|·col_max_{k+1}) · u > |det|`
    /// or its symmetric partner fails.
    ///
    /// Default: `0.0` (disabled). The dense BK77 validation tests
    /// and every direct use of `BunchKaufmanParams::default()` see
    /// the Phase 1 behavior unchanged. The sparse path opts in by
    /// constructing params with `pivot_threshold: 0.01` (SSIDS and
    /// MUMPS default). See dev/plans/scaling-aware-pivot-rejection.md.
    pub pivot_threshold: f64,
}
```

**Default value rationale.** `pivot_threshold = 0.0` in
`BunchKaufmanParams::default()` preserves backward compatibility
for the 14 dense BK77 validation tests in `tests/dense_ldlt.rs`
and the `BunchKaufmanParams::default()` call-sites in
`tests/threshold_consistency.rs`. The opt-in for the sparse path
happens at the Step 5 integration point, where the caller
explicitly constructs a params struct with `pivot_threshold: 0.01`
when scaling is active.

**Alternative considered and rejected:** a
`PivotThresholdStrategy` enum. Rejected as over-engineering for
Phase 2.2.2 — a single `f64` expresses "off (0.0)", "SSIDS/MUMPS
default (0.01)", and "custom for experimentation" with no extra
type-system cost.

### `FrontalFactors` / `SparseFactors` telemetry (optional)

Add one counter to `FrontalFactors`
(`src/dense/factor.rs:340–352`):

```rust
pub struct FrontalFactors {
    // ... existing fields ...
    /// Number of pivots rejected by the `pivot_threshold` test
    /// (either the 1×1 column-relative or the 2×2 growth bound).
    /// Always `0` when `pivot_threshold == 0.0`. Always a subset
    /// of `inertia.zero` because rejected pivots flow through
    /// ForceAccept and land as zero pivots.
    pub n_threshold_rejected: usize,
}
```

Aggregate analogously on `SparseFactors` (sum across node factors).
Not load-bearing for correctness — useful only for Step 8
validation assertions of the form "ACOPP30 rejected exactly N
pivots under `u=0.01`".

**If telemetry complicates Step 1, drop it.** It is labeled
"optional" deliberately; the correctness fix does not depend on
it. Revisit in Step 9 cleanup if time allows.

### Construction site inventory

All places that construct `BunchKaufmanParams` explicitly:

- `tests/mc64_regression.rs:32-37` — `ldlt_params()`, uses
  `..BunchKaufmanParams::default()`. Will need `pivot_threshold:
  0.01` to exercise the new path.
- `tests/threshold_consistency.rs:23-28` — `ldlt_params()`, same
  pattern. Keep `pivot_threshold: 0.0` (the default) to preserve
  the rank-deficient test's exact behavior.
- `src/numeric/factorize.rs:421-428` (test-only helper inside
  `#[cfg(test)] mod tests`). Keep `pivot_threshold: 0.0` unless a
  new test there specifically needs the feature.
- `examples/debug_acopp30_mc64.rs` — diagnostic binary. Add a new
  `--pivot-threshold <u>` option so Step 8 validation can sweep
  values without rebuilding.
- Benchmark harness (`src/bin/bench.rs` or `src/bin/`): find and
  verify whether it uses `BunchKaufmanParams::default()` directly
  or constructs with explicit fields. Since default is `0.0`, if
  it uses default the benchmark does NOT opt in. Decide whether to
  flip the bench default to `0.01` (recommended: yes, because the
  bench is the validation vehicle and must exercise the new path).

All other sites use `..BunchKaufmanParams::default()` spread so
they pick up the new field automatically.

## Implementation steps (ordered)

### Step 1 — Data model changes (~30 min)

File: `src/dense/factor.rs`.

- Add `pivot_threshold: f64` to `BunchKaufmanParams`.
- Set `Default::default()` to `0.0` (disabled) for backward
  compatibility with the dense BK77 tests.
- Add `n_threshold_rejected: usize` to `FrontalFactors` and to
  `Factors` (if consistent to add on both). Initialize to `0` at
  every construction site inside `factor_frontal` and `factor`.
- Thread the counter through the `count_1x1_inertia` /
  `count_2x2_inertia` helpers (`src/dense/factor.rs:965-993` and
  `993-1032`) so a rejection increments it exactly once per
  rejected pivot.
- Copy the counter onto `SparseFactors` in
  `src/numeric/factorize.rs:229-249` as a sum over `node_factors`.
- `cargo check` (or `cargo test --no-run`) must pass. This is a
  pure-additive change and every current test still compiles.

### Step 2 — Failing tests first (~1 hour)

File: `tests/pivot_rejection.rs` (new).

Write four hand-computed test cases, all `#[ignore]`d until Step 6:

**Test A: `col_relative_rejects_tiny_pivot_under_scaled_regime`.**
A 5×5 matrix mimicking the ACOPP30 pattern — a 3×3 SPD block in
the upper-left plus a 2×2 block with one `1e-9` diagonal entry
coupled through a `1e-3` off-diagonal. With `pivot_threshold =
0.0`, feral accepts the `1e-9` pivot and the solve error blows up
(`||x||∞ > 1e6`). With `pivot_threshold = 0.01` and `ForceAccept`,
the `1e-9` pivot is rejected (forced to zero), and the solve is
clean (`||x||∞ < 1e2`). Assert on residual and on
`factors.n_threshold_rejected == 1`. Oracle: hand-computed trace
of the BK decision tree plus the algebraic column-max test.

```rust
#[test]
#[ignore = "Step 6 of dev/plans/scaling-aware-pivot-rejection.md"]
fn col_relative_rejects_tiny_pivot_under_scaled_regime() {
    // 5×5 matrix with a 1e-9 diagonal entry at [3,3] and an off-
    // diagonal 1e-3 at [3,4] that couples it to an O(1) block.
    let mut mat = SymmetricMatrix::zeros(5);
    mat.set(0, 0, 2.0);
    mat.set(1, 0, 1.0);
    mat.set(1, 1, 2.0);
    mat.set(2, 1, 1.0);
    mat.set(2, 2, 2.0);
    mat.set(3, 3, 1e-9);
    mat.set(4, 3, 1e-3);
    mat.set(4, 4, 1.0);

    // Case 1: no threshold, expect blowup.
    let params0 = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.0,
        ..BunchKaufmanParams::default()
    };
    let (f0, _) = factor(&mat, &params0).expect("factor u=0");
    let rhs = vec![5.0, 5.0, 5.0, 1.0, 1.0];
    let x0 = solve(&f0, &rhs).expect("solve u=0");
    assert!(x0.iter().any(|xi| xi.abs() > 1e6),
        "baseline should blow up without threshold, got x = {:?}", x0);

    // Case 2: threshold = 0.01, expect clean solve.
    let params1 = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    let (f1, _) = factor(&mat, &params1).expect("factor u=0.01");
    assert!(f1.n_threshold_rejected >= 1,
        "expected >= 1 rejected pivot, got {}",
        f1.n_threshold_rejected);
    let x1 = solve(&f1, &rhs).expect("solve u=0.01");
    for (i, xi) in x1.iter().enumerate() {
        assert!(xi.abs() < 1e2,
            "with threshold, x[{}]={} should not amplify", i, xi);
    }
}
```

**Test B: `threshold_zero_reproduces_phase_1_behavior`.** A
well-conditioned 4×4 SPD matrix. With `pivot_threshold = 0.0` and
`pivot_threshold = 0.01`, feral must produce identical factors
(up to floating-point). Oracle: the inertia and D diagonal are
identical. This is the "threshold is inactive on healthy
problems" sanity check.

**Test C: `duff_reid_2x2_growth_bound_accepts_at_u_001_rejects_at_u_01`.**
A 2×2 indefinite block constructed so that `|det|` sits exactly
at the Duff-Reid boundary for `u = 0.01` and `u = 0.1`. At `u =
0.01` the block is accepted; at `u = 0.1` it is rejected.
Oracle: algebraic hand derivation against the MUMPS formula
`(|a22|·RMAX + AMAX·TMAX)·u ≤ |det|`.

Construction: pick `a11 = 1, a22 = -1, a21 = 2`, trailing column
maxes `RMAX = TMAX = 1`, `AMAX = 0`. Then `|det| = 1·(-1) - 4 =
5`, and the RHS for the first growth inequality is
`(|-1|·1 + 0·1) · u = u`. At `u = 0.01`, RHS = 0.01 and 0.01 ≤ 5
so accept. At `u = 0.1`, RHS = 0.1 and 0.1 ≤ 5 so still accept —
this case is too easy. Sharpen by shrinking `|det|`: pick `a11 =
0.01, a22 = -0.01, a21 = 0.1`, so `|det| = 0.0001 + 0.01 =
0.0101`. With column maxes `1`, the growth RHS is `(0.01·1 +
0.1·1)·u = 0.11u`. Accept iff `0.11u ≤ 0.0101`, i.e. `u ≤
0.0918`. So `u = 0.01` accepts, `u = 0.1` rejects. Wrap this 2×2
into a larger frontal where the column-max context can be set
explicitly via additional rows. Hand-verify the trace on paper
before coding.

**Test D: `pivot_threshold_field_is_plumbed`.** Compile-time
smoke test that `BunchKaufmanParams { pivot_threshold: 0.01, ..
Default::default() }` compiles, that the field is readable on the
resulting `Factors`, and that it survives through
`factorize_multifrontal` onto `SparseFactors`. Mirrors the existing
`factors_carry_zero_tol_from_params` test in
`tests/threshold_consistency.rs:67-76`.

All four are gated `#[ignore = "Step 6 of dev/plans/scaling-aware-
pivot-rejection.md — fails without threshold implementation"]`.

`cargo test` stays green because none of them run by default.

### Step 3 — 1×1 column-relative rejection (~2 hours)

File: `src/dense/factor.rs`.

Locate the three 1×1 acceptance sites in `factor_frontal`:

- Line 494: `if akk >= alpha * gamma0` — standard BK 1×1 at `k`.
- Line 517: `if r_is_fully_summed && arr >= alpha * gamma_r` — BK
  1×1 at `r` after swap.
- Line 534: `if akk * gamma_r >= alpha * gamma0 * gamma0` — LAPACK
  extension 1×1 at `k`.

And the three mirror sites in `factor` (the non-frontal version):

- Line 179, 203, 224 — same decision tree, same pattern.

For each site, add the column-relative rejection **before**
`count_1x1_inertia` is called. Exact sketch for the first site:

```rust
// Current (src/dense/factor.rs:494-507):
if akk >= alpha * gamma0 {
    let d = a[k * nrow + k];
    count_1x1_inertia(
        d, params,
        &mut pos, &mut neg, &mut zero, &mut needs_refinement,
    )?;
    do_1x1_update(&mut a, nrow, k);
    k += 1;
    continue;
}

// New:
if akk >= alpha * gamma0 {
    let d = a[k * nrow + k];

    // Phase 2.2.2: column-relative rejection. `gamma0` is already
    // the max |a[i,k]| for i > k (see the block at lines 454-475),
    // which is exactly `max_{i>k} |a[i,k]|`. Column-max including
    // the pivot itself is `max(d.abs(), gamma0)`. Reject when the
    // pivot is below `u · col_max`.
    let col_max = d.abs().max(gamma0);
    if params.pivot_threshold > 0.0
        && d.abs() < params.pivot_threshold * col_max
    {
        // Rejection: force to zero via the same codepath as
        // `count_1x1_inertia` does for |d| <= zero_tol.
        match params.on_zero_pivot {
            ZeroPivotAction::ForceAccept => {
                zero += 1;
                needs_refinement = true;
                n_threshold_rejected += 1;
                // Zero the column like count_1x1_inertia does.
                a[k * nrow + k] = 0.0;
                set_l_column_identity(&mut a, nrow, k);
                k += 1;
                continue;
            }
            ZeroPivotAction::Fail => {
                return Err(FeralError::NumericallyRankDeficient);
            }
        }
    }

    count_1x1_inertia(
        d, params,
        &mut pos, &mut neg, &mut zero, &mut needs_refinement,
    )?;
    do_1x1_update(&mut a, nrow, k);
    k += 1;
    continue;
}
```

Repeat for the other two 1×1 sites (at line 517 and 534), using
`gamma_r` and `gamma0` respectively as the `col_max` source per
the BK77 tree semantics. The `r`-swap case uses `gamma_r` because
after the swap, the pivot column is the permuted `r` column and
its max-below-diagonal is `gamma_r`.

Repeat for the corresponding three sites in `factor`.

Decision point to verify via Test A: is `col_max` the
max-below-diagonal (`gamma0`) or the max-including-diagonal
(`max(d.abs(), gamma0)`)? Research note §3 Option A specifies
"including the pivot itself", matching SSIDS `ldlt_tpp.cxx:226`
which compares `fabs(a[p*lda+p]) >= u*maxp` where `maxp` is
computed by `find_rc_abs_max_exclude` — **excluding** the
diagonal. So SSIDS uses col_max = max-below-diagonal. Feral
should follow SSIDS: `col_max = gamma0` (not
`max(d.abs(), gamma0)`). Update the sketch accordingly during
implementation; this is a subtle call that must match the
reference exactly.

**Corrected sketch line:**

```rust
let col_max = gamma0;  // max_{i>k} |a[i,k]|, matching SSIDS
if params.pivot_threshold > 0.0 && d.abs() < params.pivot_threshold * col_max {
```

Watch for:

- When `gamma0 == 0` the column below the diagonal is already
  empty (the matrix is structurally already eliminated at this
  position); the existing early-return at line 477 handles this
  case and the new rejection is unreachable. Good.
- When `params.pivot_threshold == 0.0`, the guarded branch
  short-circuits and the kernel is identical to Phase 2.2.1. This
  is the backward-compat guarantee that the dense BK77 tests rely
  on.
- `needs_refinement = true` on rejection, matching the
  `ForceAccept` on the absolute `zero_tol` path.
- The `n_threshold_rejected` counter is incremented exactly once
  per rejected pivot, not once per rejection branch evaluation.
- The `set_l_column_identity` call must match what
  `count_1x1_inertia` does on the `zero_tol` path (verify by
  reading lines 975-988).

`cargo test` after Step 3 should still pass 141/141 — no test
currently enables `pivot_threshold > 0.0` so the new code is
dead.

### Step 4 — 2×2 Duff-Reid growth bound (~2 hours)

File: `src/dense/factor.rs:550-577`.

Locate the 2×2 acceptance site at line 550 (`if r_is_fully_summed
&& k + 1 < ncol`). The existing code computes `d11, d21, d22,
det` and checks `det.abs() <= params.zero_tol_2x2`. Add a
parallel Duff-Reid bound check:

```rust
// Current (src/dense/factor.rs:555-567):
let d11 = a[k * nrow + k];
let d21 = a[k * nrow + (k + 1)];
let d22 = a[(k + 1) * nrow + (k + 1)];
let det = d11 * d22 - d21 * d21;

if det.abs() <= params.zero_tol_2x2 {
    match params.on_zero_pivot {
        ZeroPivotAction::Fail => return Err(FeralError::NumericallyRankDeficient),
        ZeroPivotAction::ForceAccept => {
            needs_refinement = true;
        }
    }
}
```

Insert after the absolute `det` check, before
`count_2x2_inertia_val`:

```rust
// Phase 2.2.2: Duff-Reid 2×2 growth bound, mirroring MUMPS
// dfac_front_aux.F:1602-1604. `col_max_k` and `col_max_k1` are
// the max magnitudes below the 2×2 block in columns k and k+1.
if params.pivot_threshold > 0.0 {
    let col_max_k = col_max_below(&a, nrow, k + 2, k);
    let col_max_k1 = col_max_below(&a, nrow, k + 2, k + 1);
    let abs_det = det.abs();
    // Growth bound pair (MUMPS): reject if either
    //   (|a22|·RMAX + |a21|·TMAX) · u > |det|
    //   (|a11|·TMAX + |a21|·RMAX) · u > |det|
    // where RMAX=col_max_k, TMAX=col_max_k1, AMAX=|a21|.
    let growth_1 = (d22.abs() * col_max_k + d21.abs() * col_max_k1)
        * params.pivot_threshold;
    let growth_2 = (d11.abs() * col_max_k1 + d21.abs() * col_max_k)
        * params.pivot_threshold;
    if growth_1 > abs_det || growth_2 > abs_det || abs_det == 0.0 {
        match params.on_zero_pivot {
            ZeroPivotAction::Fail => {
                return Err(FeralError::NumericallyRankDeficient);
            }
            ZeroPivotAction::ForceAccept => {
                // Reject the 2×2: zero both pivot columns below
                // the diagonal and count both as zero pivots.
                zero += 2;
                needs_refinement = true;
                n_threshold_rejected += 2;
                a[k * nrow + k] = 0.0;
                a[(k + 1) * nrow + (k + 1)] = 0.0;
                a[k * nrow + (k + 1)] = 0.0;
                set_l_column_identity(&mut a, nrow, k);
                set_l_column_identity(&mut a, nrow, k + 1);
                subdiag[k] = 0.0;
                k += 2;
                continue;
            }
        }
    }
}
```

Add the helper `col_max_below` next to `column_offdiag_max`
(`src/dense/factor.rs:726`):

```rust
/// Returns `max_{i >= start} |a[i, col]|`, zero when the range is
/// empty. Used by the Phase 2.2.2 Duff-Reid growth bound on 2×2
/// pivot acceptance.
fn col_max_below(a: &[f64], nrow: usize, start: usize, col: usize) -> f64 {
    let mut m = 0.0f64;
    for i in start..nrow {
        let v = a[col * nrow + i].abs();
        if v > m {
            m = v;
        }
    }
    m
}
```

**Transcription-error risk.** The MUMPS formula is the canonical
source; cite `ref/mumps/src/dfac_front_aux.F:1602-1604` in a
comment on the growth-bound block. Test C will catch sign/index
errors by construction. If Test C fails on first run, re-read
the MUMPS Fortran line-by-line before touching the sketch.

`cargo test` after Step 4 should still pass 141/141.

### Step 5 — Integration: where does `pivot_threshold` flow from? (~30 min)

`factor_frontal` is called at `src/numeric/factorize.rs:192` with
a `params: &BunchKaufmanParams` argument. That argument is
threaded from `factorize_multifrontal`'s own
`params: &BunchKaufmanParams` at line 76. So there is no new
plumbing required; the caller of `factorize_multifrontal`
controls the threshold by constructing the struct with
`pivot_threshold: 0.01`.

**Caller changes:**

1. **`tests/mc64_regression.rs:32-37`.** Update `ldlt_params()`:

   ```rust
   fn ldlt_params() -> BunchKaufmanParams {
       BunchKaufmanParams {
           on_zero_pivot: ZeroPivotAction::ForceAccept,
           pivot_threshold: 0.01,
           ..BunchKaufmanParams::default()
       }
   }
   ```

2. **`src/bin/bench.rs`** (or wherever the bench harness lives).
   Flip its `BunchKaufmanParams` construction to set
   `pivot_threshold: 0.01` so Step 8 bench runs exercise the new
   path. Verify with a grep after Step 1.

3. **`tests/threshold_consistency.rs:23-28`.** Do **not**
   change. Keep `pivot_threshold: 0.0` (the default) so the
   existing `dense_solve_skips_zero_pivots_rank_deficient` and
   `sparse_solve_skips_zero_pivots_rank_deficient` tests run on
   the unchanged Phase 1 codepath. They rely on the exact
   inertia of the 3×3 `[[2,1,0],[1,1,1],[0,1,2]]` matrix and
   raising the threshold could flip their decisions.

4. **Other `tests/*.rs` files.** Scan for
   `BunchKaufmanParams` construction sites during Step 1 and make
   sure each one explicitly leaves `pivot_threshold` at the
   default (`0.0`) unless it intends to exercise the new path.

5. **`examples/debug_acopp30_mc64.rs`.** Add a new command-line
   flag `--pivot-threshold <f64>` so Step 8 can sweep the threshold
   value without rebuilding. The diagnostic binary is already
   structured for this kind of sweep (it already has Identity vs
   MC64 columns); add a fourth column "MC64 + u=0.01".

### Step 6 — Un-ignore Step 2 tests (~30 min)

File: `tests/pivot_rejection.rs`.

Remove the `#[ignore]` attribute from Tests A, B, C, D. Run:

```bash
cargo test --test pivot_rejection
```

All four tests must pass. If any fail:

- Test A (rejects tiny pivot): verify `gamma0` is the right
  variable to use as `col_max` — the BK sequential variant
  stores this slightly differently from SSIDS. Read
  `src/dense/factor.rs:730-750` (`column_offdiag_max`) to
  confirm it returns `max_{i>k} |a[i,k]|` and not
  `max_{i>=k} |a[i,k]|`.
- Test B (threshold inactive on SPD): if this fails,
  `pivot_threshold = 0.01` is rejecting a legitimate pivot — the
  `col_max` formula is likely over-aggressive. Reduce to
  `pivot_threshold = 0.001` and re-run; if that also fails, the
  sign or index calculation is wrong.
- Test C (Duff-Reid boundary): if this fails, the MUMPS formula
  transcription is wrong. Re-read `dfac_front_aux.F:1602-1604`.
- Test D (plumbing): if this fails, `n_threshold_rejected` is not
  being aggregated onto `SparseFactors` (or was dropped for time;
  that is acceptable — remove Test D).

After all four pass, `cargo test` must still pass 141 + 4 = 145
(143 if Test D is dropped).

### Step 7 — Un-ignore `mc64_regression.rs` and measure (~1 hour)

File: `tests/mc64_regression.rs`.

**Do not flip the `#[ignore]` yet.** Instead, run:

```bash
cargo test --test mc64_regression --release -- --ignored --nocapture
```

Capture the four residuals into
`dev/validation/phase-2.2.2-pivot-rejection.md`. Expected outcomes
(from research note §5.2):

| Matrix          | Pre-2.2.1  | Post-2.2.1 | Target  | Realistic post-2.2.2 |
|-----------------|-----------:|-----------:|--------:|----------------------:|
| ACOPP30_0000    |  3.15e-2   | 2.27e+46   | < 1e-8  | 1e-4 to 1e-8         |
| CRESC132_0000   |  2.39e+08  | 1.37e+05   | < 1e-6  | 1e+0 to 1e-4         |
| CHWIRUT1_0000   |  1.41e+09  | 8.50e+02   | < 1e-8  | 1e-4 to 1e-8         |
| CRESC100_0000   |  2.54e+04  | 1.43e+02   | < 1e-8  | 1e-4 to 1e-8         |

**Decision gate.** After measuring:

- **If a matrix is ≥ 1e-4 residual**, do **not** un-ignore it.
  Leave it `#[ignore]`d, update the header comment with the new
  baseline, and log the deficit in `dev/decisions.md` as
  "Phase 2.2.2 landing — N of 4 `mc64_regression.rs` tests
  recovered, M deferred to Phase 2.3 pending delayed pivoting".
- **If a matrix is in `[1e-4, 1e-8)`**, un-ignore it **only if**
  the current `< 1e-8` target is relaxed with an explicit
  `decisions.md` entry citing the measured number. Do not
  silently lower the threshold. The CLAUDE.md hard rule
  "NEVER loosen a test tolerance without human approval" applies
  here — in practice, for Phase 2.2.2, this means pausing and
  reporting the numbers to the user before flipping the test.
- **If a matrix is `< 1e-8`**, un-ignore it and cheer.

**Acceptance bar (from success criteria):** at least **2 of 4**
matrices must reach `< 1e-4` residual post-Option-A. If fewer than
2 reach that bar, **stop and report** — the column-relative
threshold implementation is likely buggy and delayed pivoting
(Phase 2.3) may be necessary sooner than planned.

**Minimum acceptable for ACOPP30:** residual ≤ `2.84e+16`
(the Identity-path baseline). Anything worse than that means
Phase 2.2.2 has regressed on the primary target matrix and must
be rolled back.

### Step 8 — Validation sweep (~1 hour)

Re-run the 7-matrix sanity panel from Phase 2.2.1:

```bash
cargo run --release --example triage_large_cresc132
cargo run --release --example debug_acopp30_mc64
```

Capture residuals for `CHWIRUT1, HAHN1, GAUSS2, CRESC100,
MUONSINE, VESUVIO, CRESC132` alongside ACOPP30.

Produce `dev/validation/phase-2.2.2-pivot-rejection.md` with the
same structure as `dev/validation/phase-2.2.1-mc64-sweep.md`.
Include these columns:

| Matrix | Pre-Phase-2.2.1 | Post-Phase-2.2.1 | Post-Phase-2.2.2 | Target | Status |

Flag any matrix that got **worse** in Phase 2.2.2 vs Phase 2.2.1.
A worsening is a bug signal — investigate before committing.

Also re-run the full test suite in release mode:

```bash
cargo test --release
```

Must show `141 + N passed, 0 failed` where N is the number of
Step 2 tests that were un-ignored (expected: 3 or 4) plus any
`mc64_regression.rs` tests that cleared the bar in Step 7.

Bench harness:

```bash
cargo run --release --bin bench
```

Record the aggregate residual-pass count and compare to the
pre-2.2.2 baseline. If fewer tests pass than before, this is a
regression and must be reconciled before landing.

If any existing test's inertia changed (indicated by a test
failure on an inertia assertion), record which test in the
validation report and check whether it is a correctness
regression (rolled-back pivot classification) or an
improvement (closer to MUMPS's counts). Correctness regressions
block landing; improvements are noted and possibly enable
tightening other tests.

### Step 9 — Commit-ready cleanup (~30 min)

- `cargo fmt --all`
- `cargo clippy --all-targets -- -D warnings` — must be clean
- `pre-commit run --all-files` — must be clean
- Remove any debug `eprintln!` calls added during Step 3/4/7
  debugging
- Verify no `unwrap()` or `expect()` was introduced in `src/`
  during Step 1-4 (grep for `unwrap()` under `src/`)
- Verify no `unsafe` was introduced
- Journal entries appended real-time throughout Steps 1-8
  (reminder: CLAUDE.md requires this, not retroactive writing)
- Sequence of commits (one per step, matching the Phase 2.2.1
  convention):
  1. `Phase 2.2.2 Step 1: add pivot_threshold to BunchKaufmanParams`
  2. `Phase 2.2.2 Step 2: failing tests for scaling-aware pivot rejection`
  3. `Phase 2.2.2 Step 3: 1×1 column-relative rejection in dense BK`
  4. `Phase 2.2.2 Step 4: Duff-Reid 2×2 growth bound`
  5. `Phase 2.2.2 Step 5-6: wire sparse caller and un-ignore pivot tests`
  6. `Phase 2.2.2 Step 7-8: validation sweep and regression un-ignore`

  Each with a body explaining what/why/evidence. No `--no-verify`.

### Step 10 — Narrative update (~1 hour, only if Step 7/8 succeeds)

Only if at least 2 of 4 regression tests recovered:

- Update `CHANGELOG.md` `[Unreleased]` → `### Added`: "Scaling-aware
  pivot rejection (MUMPS/SSIDS threshold partial pivoting, `u =
  0.01` default) in the sparse multifrontal kernel."
- Update `CHANGELOG.md` `### Fixed`: "ACOPP30 regression under MC64
  scaling (from `2.27e+46` to `<residual>`)."
- Append to `dev/decisions.md`: "2026-04-12 (Phase 2.2.2)" entry
  with the measured residuals, the choice of `pivot_threshold =
  0.01`, and the rationale from research note §3.
- Update `README.md` Status section **only if** the mc64 gap
  meaningfully narrowed. Phase 2.2.2 alone likely does not close
  the n>500 gap; hold the README update for Phase 2.3.
- Append session checkpoint to `dev/sessions/2026-04-12-NN.md`
  using `dev/templates/session.md`.

**Total estimated effort:** 8–10 hours, one focused session.

| Step | Hours |
|------|------:|
| 1. Data model             | 0.5 |
| 2. Failing tests          | 1.0 |
| 3. 1×1 column-relative    | 2.0 |
| 4. 2×2 Duff-Reid          | 2.0 |
| 5. Integration wiring     | 0.5 |
| 6. Un-ignore Step 2 tests | 0.5 |
| 7. Regression measurement | 1.0 |
| 8. Validation sweep       | 1.0 |
| 9. Cleanup + commits      | 0.5 |
| 10. Narrative update      | 1.0 |
| **Total**                 | **10.0** |

## Acceptance criteria

The implementation is complete and Phase 2.2.2 can be closed when
**all** of the following are true:

1. `cargo test --all-targets` passes — all 141 existing tests plus
   at least 3 of the 4 new `tests/pivot_rejection.rs` tests (Test D
   is optional).
2. `cargo clippy --all-targets -- -D warnings` clean.
3. `pre-commit run --all-files` clean.
4. `cargo test --release --test mc64_regression -- --ignored`
   reports **at least 2 of 4** matrices under a `< 1e-4` residual
   bar. The Phase 2.2.1 `< 1e-8` targets may be relaxed with
   documented measurements in `decisions.md`, but the `< 1e-4`
   floor is the minimum Phase 2.2.2 accomplishment.
5. **ACOPP30_0000 residual ≤ `2.84e+16`** (the Identity-path
   baseline — the absolute minimum regression bar). The landing
   zone target is `[1e-4, 1e-8]`; anything in that range is a
   clear win over Phase 2.2.1.
6. No inertia regression on the 141-test corpus. Inertia changes
   on the 7-matrix sanity panel are allowed and should be
   reported in the validation doc — they may be closer to MUMPS's
   counts or further, but **no matrix moves from "inertia MATCH"
   to "inertia MISMATCH"** vs Phase 2.2.1.
7. `dev/validation/phase-2.2.2-pivot-rejection.md` exists and
   reports the 7-matrix panel plus the 4 regression matrices.
8. No `unwrap()` / `expect()` introduced in `src/`, no `unsafe`
   introduced, no test tolerance silently loosened, no hook
   skipped.
9. Commit history: 4–6 atomic commits, each with a body
   explaining what/why/evidence, all signed off by the
   `pre-commit` hook.

## Rollback plan

If the implementation reveals a fundamental issue that Phase 2.2.2
cannot address in this session:

- Changes are contained to `src/dense/factor.rs` (new field,
  new clauses at 6 sites, new helper function), a handful of
  aggregated counter fields in `src/numeric/factorize.rs`, and
  `tests/pivot_rejection.rs`.
- `git revert` on Steps 3-4 commits takes feral back to Phase
  2.2.1 state cleanly. The new `BunchKaufmanParams` field and
  telemetry are additive and can be kept or reverted
  independently.
- If Test C (Duff-Reid boundary) cannot be made to pass because
  the MUMPS formula transcription is wrong, fall back to "1×1
  rejection only, defer 2×2 to Phase 2.2.3". The ACOPP30
  regression is driven by 1×1 pivots exclusively per the
  diagnostic report's §"Bugs and quirks uncovered"; 1×1-only
  still gets most of the fix.
- If the sanity panel shows a new regression on a previously-
  passing matrix, roll back Step 3/4 and investigate in isolation
  on a smaller test case before re-attempting.

## Risk register

**R1. `pivot_threshold = 0.01` is too loose, ACOPP30 still blows up.**
*Symptom:* Step 7 reports ACOPP30 still worse than `2.84e+16`.
*Probability:* low. The ACOPP30 offenders are at `~3.6e-10`, the
column max is `~1` post-MC64, so `u · col_max = 0.01 >> 3.6e-10`
decisively rejects them.
*Mitigation:* sweep `u ∈ {0.01, 0.001, 0.1}` in Step 7 via the
`debug_acopp30_mc64` command-line flag (see Step 5 item 5) and
pick the best. If **no** `u` works, the bug is in the rejection
codepath rather than the threshold value, and Steps 3/4 need
revisiting.

**R2. `pivot_threshold = 0.01` is too strict, `sparse_postorder.rs`
or `threshold_consistency.rs` regresses.**
*Symptom:* a currently-passing test fails after Step 5 because
the scaling default flipped `SupernodeParams` to use MC64, and
the new threshold now rejects a pivot on a matrix that Phase 1
accepted.
*Probability:* medium. The
`sparse_solve_skips_zero_pivots_rank_deficient` test at
`tests/threshold_consistency.rs:132-168` is the leading risk —
it constructs a 3×3 structurally rank-deficient matrix with a
pivot that Phase 1 force-accepts, and the new threshold may
reject it instead.
*Mitigation:* keep `pivot_threshold = 0.0` in the threshold test
by not modifying its `ldlt_params()` (Step 5 item 3). The test
continues to exercise the Phase 1 path. If even that fails, the
test's factor-time assertions are being broken by the Step 5
bench default flip — in which case also keep the bench default
at `0.0` until Phase 2.3.

**R3. Duff-Reid growth bound formula is wrong.**
*Symptom:* Test C fails and/or post-2.2.2 residuals on 2×2-heavy
matrices are worse than post-2.2.1.
*Probability:* medium. The MUMPS formula is subtle and has
multiple symmetric variants; a transcription error is easy.
*Mitigation:* Test C is constructed specifically to catch this.
If it fails, re-read `dfac_front_aux.F:1594-1606` verbatim and
compare the Rust transcription line-by-line. If the formula is
right but the test case construction is wrong, fix the test
oracle (it is hand-computed). If neither yields a passing test,
**fall back to 1×1-only rejection** (Rollback plan item 3) and
defer 2×2 to Phase 2.2.3.

**R4. Rejecting more pivots inflates `n_forced_zero` and changes
inertia counts, breaking exact-inertia assertions.**
*Symptom:* existing tests fail on `assert_eq!(inertia.zero, N)`
or similar.
*Probability:* medium-high. Several tests in `kkt_matrices.rs`
and `kkt_hardening.rs` assert on exact inertia.
*Mitigation:* search for `inertia.zero == `, `inertia.negative
== ` and `inertia.positive == ` assertions across `tests/` in
Step 1 and catalog each one's expected behavior under the new
threshold. If any are affected, add `scaling_strategy:
ScalingStrategy::Identity` (or `pivot_threshold: 0.0`) to pin
them to the Phase 1 path. Per the hard rule, pinning is not
tolerance loosening — it is scoping the test to the code regime
it was written for.

**R5. Forced-zero pivots inside rejected 2×2 blocks interact
badly with the solve.**
*Symptom:* even with 2×2 rejection working, solve still produces
garbage on a rejected 2×2 block because the solve's D-inverse
step does not handle "both pivots zero" correctly.
*Probability:* low. The existing `solve` already checks
`|d_diag[k]| <= zero_tol` before dividing and skips the position
cleanly. The 2×2 rejection path sets both `d11 = d22 = 0`, which
should route through the same skip logic.
*Mitigation:* Test A (and Test C with a larger matrix) is the
end-to-end check. If solve misbehaves, add an explicit guard in
`src/numeric/solve.rs` on the 2×2 D-solve path.

**R6. The deferred Phase 2.1.2 2×2 trace-vs-a00 classification
bug is still lurking and masquerades as a 2×2 threshold issue.**
*Symptom:* 2×2-heavy matrices' inertia changes in unexpected
ways after Step 4, even though the 1×1 path is fine.
*Probability:* low-medium. The trace-vs-`a00` bug is latent in
`count_2x2_inertia_val` (`src/dense/factor.rs:704-726`) and
only manifests in specific sign arrangements.
*Mitigation:* Phase 2.2.2 explicitly keeps the
classification logic untouched. If R6 fires, track it in the
Phase 2.2.2 checkpoint and fold the trace fix into Phase 2.2.3
(research note §7 Open Question 4 recommendation). Do **not**
conflate rejection with classification.

**R7. Bench harness uses `BunchKaufmanParams::default()` and the
validation sweep does not exercise the new code path.**
*Symptom:* Step 8 sees no change in the sanity-panel residuals
because the bench is still on `pivot_threshold = 0.0`.
*Probability:* medium. Must verify during Step 1 construction
site inventory.
*Mitigation:* Step 5 item 2 explicitly flips the bench default.
If the bench uses a separate construction path, update that.

## Open questions

1. **Default `pivot_threshold` value.** Research note recommends
   `0.01` (MUMPS/SSIDS default). Sweep `{0.001, 0.01, 0.1}` on
   ACOPP30 in Step 7 before committing. If `0.01` does not
   maximize the combined residual/inertia improvement across all
   7 sanity panel matrices, document the winning value in
   `decisions.md`. **Plan default recommendation: `0.01`**.

2. **Should the threshold be per-params or a global constant?**
   Per-params. Matches MUMPS's `CNTL(1)` and SSIDS's `options%u`
   style; trivially overridable for experiments.

3. **Does `ScalingStrategy::Identity` use a different default
   threshold than `Mc64Symmetric`?** MUMPS/SSIDS use the same
   `0.01` regardless. Feral should too, for consistency. The
   `threshold_consistency.rs` rank-deficient tests will stay
   pinned to `pivot_threshold = 0.0` via their own
   `ldlt_params()` helper, not via scaling strategy.

4. **Interaction with `ZeroPivotAction::ForceAccept` vs
   `::Fail`.** The Step 3/4 sketches route rejections through
   the `on_zero_pivot` match, so `::Fail` sees threshold-rejected
   pivots as a hard error (returns `NumericallyRankDeficient`).
   This is consistent with `::Fail` semantics: "any non-accepted
   pivot is fatal". If a test needs to distinguish threshold-reject
   from absolute-zero-reject under `::Fail`, the `FeralError`
   variant needs extending — defer to Phase 2.3 unless Step 7/8
   surfaces a real need.

5. **Is `FrontalFactors::n_threshold_rejected` worth the
   plumbing?** Optional per Step 1. Keeps the PR smaller if
   dropped; keeps Step 8 validation more assertable if kept.
   Decision: keep unless Step 1 exceeds its time budget.

6. **Do we touch `solve_sparse_refined` at all?** No. The solve
   already handles zero pivots correctly via the existing
   `factors.zero_tol` check. Phase 2.2.2's rejection path lands
   values as zero, which is already handled.

## Literature citations

All citations referenced already exist in `dev/references.bib`:

- `bunch1977stable` — BK77 paper, pivot selection α ≈ 0.6404.
- `duff1983multifrontal` — delayed pivoting in multifrontal.
- `hogg2013pivoting` — threshold pivoting in SSIDS, `u = 0.01`
  recommendation.
- `duff2001mc64` — MC64 scaling (Phase 2.2.1).
- `duff2005symmetric` — symmetric MC64 averaging (Phase 2.2.1).

No new additions required.

## What happens next (not part of this plan)

- **Phase 2.2.3** — fold in the deferred 2×2 trace-vs-`a00`
  classification fix, plus the SSIDS cancellation guard
  (`ldlt_tpp.cxx:181-183`) if Step 8 shows it is needed.
- **Phase 2.3** — full Duff-Reid delayed pivoting (Option C in the
  research note), the real answer. Closes the remaining
  CRESC132/CRESC100 gap that Option A is expected to leave behind.
- **Phase 2.3.x** — re-run full corpus consensus and publish the
  delta against the Phase 2 baseline.

bibliography:../references.bib
