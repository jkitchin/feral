# D.3 plan — dense fast-path for small-dense matrices

**Authorized by:** `dev/research/sparse-tail-d3-d4-2026-04-19.md`.
**Date opened:** 2026-04-19 (session 04 continuation).

## Goal

Route matrices that satisfy `n ≤ N_max && density ≥ ρ_min` directly
to the dense BK kernel, skipping symbolic analysis + supernodal
assembly. The gate fires at `factorize_multifrontal` entry so both
direct callers (bench, tests) and `Solver` benefit uniformly.

**Measurable win target:**

- TRO3X3_0013: 83 µs → ≤ 25 µs.
- Corpus factor/MUMPS geomean: 0.46 → ≤ 0.44.
- No regression on any matrix outside the gate (the out-of-gate
  branch is bit-identical to the pre-D.3 code).

If TRO3X3 does not drop to ≤ 30 µs on the first authorized
measurement, the gate is wrong or `CscMatrix::to_dense` is itself
the bottleneck — diagnose before widening.

## Non-goals

- Tiny-n fast path (D.4). The synthesis function from this plan
  is its prerequisite; D.4 is a separate plan.
- Phase 2.4 (blocked BK + SIMD on dense root frontal). Arrow-KKT
  tail (CRESC50, HAHN1_*, NET1) stays in its existing lane.
- Changing the `factor_frontal` kernel itself. D.3 is routing
  only — the dense path calls the same kernel the multifrontal
  path already uses.
- Multi-supernode synthesis. The fast path is a single-node
  `SparseFactors`; matrices that *would* benefit from two or
  three supernodes stay on the multifrontal path.

## Gate

```rust
// src/numeric/factorize.rs — inside factorize_multifrontal_with_workspace,
// before the symbolic-driven path.

#[inline]
fn should_use_dense_fast_path(n: usize, nnz_lower: usize) -> bool {
    const N_MAX: usize = 128;
    const RHO_MIN_NUM: usize = 1;  // density ratio as nnz_lower * 4 >= n*(n+1)
    const RHO_MIN_DEN: usize = 4;  // i.e. ρ >= 0.25
    if n == 0 || n > N_MAX { return false; }
    // Integer-only density test: nnz_lower * den >= n * (n+1) / 2 * num.
    let lower_cells = n * (n + 1) / 2;
    nnz_lower * RHO_MIN_DEN >= lower_cells * RHO_MIN_NUM
}
```

`N_MAX = 128` chosen so a dense `n*n = 16 384 f64 = 128 KB`
workspace fits in L1/L2 comfortably. First-pass floor —
measurement stage may widen.

`ρ_min = 0.25` is the research-note estimate; confirm with a
profiled sweep on 5 at-boundary matrices before committing the
threshold to code.

Gate inputs come from `CscMatrix::n` and `row_idx.len()`, both
available pre-symbolic — the gate costs ~5 ns.

## `dense_fast_factor` API

```rust
// src/numeric/factorize.rs

/// Fast-path factorization for small-and-dense matrices.
///
/// Skips symbolic analysis entirely: densifies the CSC into a
/// `SymmetricMatrix`, applies the usual global symmetric scaling,
/// runs the dense BK kernel on all `n` columns, and wraps the
/// `FrontalFactors` in a single-supernode `SparseFactors` that is
/// shape-compatible with `solve_sparse`.
///
/// Called from `factorize_multifrontal_with_workspace` under the
/// gate in `should_use_dense_fast_path`. Callers must not invoke
/// this directly — the gate is the authoritative entry point.
fn dense_fast_factor(
    matrix: &CscMatrix,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError>;
```

### Body sketch

1. `let (scaling_user, scaling_pivot, scaling_info) =
   compute_scaling(matrix, &params.scaling)?;` — same scaling
   contract as the multifrontal path. Pivot-order == user-order
   because perm is identity here.
2. Densify: `let mut sym = matrix.to_dense();` (exists at
   `src/sparse/csc.rs:261`), then scale in place as `D · A · D`.
3. `let ff = factor_frontal(&sym, n, /*may_delay=*/ false,
   &params.bk)?;` — `may_delay = false` matches multifrontal's
   single-root behavior (`ZeroPivotAction::ForceAccept` absorbs
   instability).
4. Synthesize:
   - `perm = (0..n).collect()`; `perm_inv = perm.clone()`.
   - One `NodeFactors` with `first_col=0`, `ncol=n`,
     `nelim=ff.nelim`, `n_delayed_in=0`, `nrow=n`,
     `row_indices=(0..n).collect()`, `frontal_factors=ff`,
     `inertia=ff.inertia.clone()`.
   - `SparseFactors { n, perm, perm_inv, node_factors: vec![..],
     needs_refinement: ff.needs_refinement, scaling: scaling_user,
     scaling_info }`.
5. Return `(factors, ff.inertia)` — inertia comes from the
   single node.

### Shape compatibility

`solve_sparse` iterates `node_factors`, applies each node's
`FrontalFactors` to its slice, and permutes in/out using
`perm`/`perm_inv`. A single node covering `0..n` with identity
perm reduces exactly to the dense solve. Parity test (§Tests)
is the forcing function.

## Interaction with FactorWorkspace

The dense path allocates its own `sym.data` Vec (length `n*n ≤
16 384` when `N_MAX = 128`). It does **not** consume `ws` — the
multifrontal scratch fields are not touched. The `Solver`
still holds `workspace: FactorWorkspace`, but on a gate-hit
iteration the workspace is pass-through.

Follow-up (not in this plan): widen the workspace to pool the
dense-path `sym.data` too, once the gate is validated. The first
cut keeps the dense path allocator-owned to minimize the blast
radius.

## Tests (before implementation)

All tests go in `tests/dense_fast_path.rs`, a new integration
test file.

1. **Gate off** (n = 512): produces a `SparseFactors` via the
   multifrontal path. Assert the current-path behavior is
   unchanged — same `SparseFactors` fields modulo `.n`, same
   inertia — by byte-equal comparison against a snapshot taken
   before this commit on e.g. `LAKES_1199`.
2. **Gate on, solve parity**: densify an in-gate matrix
   (TRO3X3_0013 or a synthetic n=64, density 0.5 KKT) on the
   dense path AND force-route it through the multifrontal path
   via a test-only feature flag / direct call. Solve both
   factorizations against the same random RHS. Assert
   `‖x_dense - x_multifrontal‖∞ / ‖x_multifrontal‖∞ ≤ 1e-10`.
   Assert inertia is byte-equal.
3. **Boundary**: n = 128, density = 0.25 (on the gate). Gate
   should fire; solve round-trip residual passes tolerance.
4. **Just-outside gate**: n = 129 (one above `N_MAX`) or
   density = 0.249: gate should NOT fire. Multifrontal path
   still produces a valid factor.
5. **Zero-pattern column**: gate-in matrix with an all-zero
   column triggers the kernel's zero-pivot handling the same
   way the multifrontal root frontal does. Must pass under
   `ZeroPivotAction::ForceAccept`.
6. **Cross-path determinism**: factor the same in-gate matrix
   twice via the dense path. Assert bit-equal `SparseFactors`
   (guards against scratch-state leakage once the dense path
   starts touching the workspace in a follow-up).

Test (2) is the primary correctness oracle. Inertia parity is
non-negotiable (hard rule). Solve parity uses a tight
relative-ℓ∞ tolerance because both paths run the same BK kernel
on the same scaled matrix — the only differences are in
synthesis bookkeeping.

## Measurement plan

### Stage 1 — TRO3X3_0013 micro

Extend `alloc_probe` (or write `bin/d3_probe.rs`) to time
TRO3X3_0013 and report dense-path vs multifrontal fac(µs) +
allocs. Expected: dense-path 20–30 µs, multifrontal 80 µs.

### Stage 2 — gate-boundary sweep

For each `n ∈ {32, 64, 96, 128, 160, 192}` and density
`∈ {0.1, 0.25, 0.5, 0.75}`, build a synthetic symmetric KKT and
time both paths. Tabulate µs ratio. Pick final `N_MAX` and
`ρ_min` from the crossover. Commit the tuned thresholds only
after this sweep.

### Stage 3 — corpus bench

Full 154 588-matrix `cargo run --release --bin bench`. Acceptance
gate:

- factor/MUMPS geomean 0.46 → ≤ 0.44.
- No matrix slower than its pre-D.3 time by more than 20%.
  (The 20% tolerance accounts for measurement noise on tiny
  matrices; anything larger is an unintended regression.)

Top-10-by-ratio list should show TRO3X3_0013 dropping out.

## Rollout

1. Plan (this file) — commit.
2. Tests (red) — commit. Failing because `dense_fast_factor`
   doesn't exist yet.
3. `dense_fast_factor` + gate + synthesis, with placeholder
   thresholds from the research note (`N_MAX=128, ρ_min=0.25`)
   — commit. Tests green.
4. Stage 1 measurement + threshold sweep (stage 2) — commit the
   tuned thresholds (and the sweep results under
   `dev/results/lever-d3/`).
5. Stage 3 corpus bench — commit results + decision (D.3 done
   vs widen vs pivot to D.4).
6. Session checkpoint.

## Risks

- **Threshold misses the sweet spot.** Mitigation: stage-2 sweep
  picks thresholds from data, not intuition. Commit only after
  data is in hand.
- **Synthesized `SparseFactors` trips a `solve_sparse`
  assumption.** Mitigation: test (2) is the oracle — if solve
  parity fails, the synthesis is wrong, not the kernel.
- **`CscMatrix::to_dense` allocates each call.** Acceptable for
  the first cut (the dense path saves much more than the
  densify costs). If stage-1 shows densify dominating, pool the
  buffer in a follow-up commit.
- **MC64 scaling on dense matrices.** `compute_scaling` already
  supports the dense case; no new failure mode expected, but
  the stage-1 measurement will catch it if scaling adds
  surprising cost at this size.
- **Inertia mismatch under ForceAccept.** Any zero pivot gets
  accepted into the `zero` count; the multifrontal path behaves
  the same way on a single-root supernode, so inertia should
  agree. Test (5) forces this case.

## What this plan does not do

- Authorize D.4. After D.3 lands and we see which
  LEWISPOL/METHANL8LS-class matrices remain in the top-10, D.4
  gets its own scoping note.
- Pool the dense-path scratch. `dense_fast_factor` allocates
  `sym.data` each call; follow-up only.
- Touch `src/dense/factor.rs`. The kernel is unchanged; the
  fast path is pure routing.
