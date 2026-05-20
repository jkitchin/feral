# Plan: near-singularity signal (`min|λ(D)|`)

Research note: `dev/research/near-singularity-signal.md`. Read it first.

Additive change. No factorization / pivoting / solve behavior changes. Two
aggregation methods over already-stored D-block data, plus C-ABI exposure.
Mirrors the existing `SparseFactors::min_diagonal()` precedent
(`src/numeric/factorize.rs:882`) line for line.

## Steps (tests-first, atomic commits)

### Step 1 — `SparseFactors` accessors + tests

Commit 1: tests.
- `tests/pounce_interface.rs`: add `min_pivot_magnitude_*` / `max_pivot_*`
  tests next to the `min_diagonal_*` block (~line 443). Cases 1–4 from
  research note §5. These fail to compile until step 2 — commit them
  together with step 2 so `cargo test` stays green per protocol, OR commit
  the tests `#[ignore]`-free in the same commit as the impl. **Decision:
  one commit for `SparseFactors` impl + its tests** (the oracle is external
  — hand calculation — so the same-session rule in CLAUDE.md is satisfied).

Commit 1 (combined): in `src/numeric/factorize.rs`, immediately after
`min_diagonal()` (line 914):
- `pub fn min_pivot_magnitude(&self) -> Option<f64>` — walk `node_factors`,
  for each pivot compute the per-pivot smaller-magnitude eigenvalue:
  - 1×1: `ff.d_diag[k].abs()`
  - 2×2: eigenvalues `λ± = (t ± √(t²−4Δ))/2`; smaller magnitude is
    `min(|λ₊|, |λ₋|)`. Compute via `|Δ| / max(|λ₊|, |λ₋|)` to avoid
    cancellation when one eigenvalue is tiny; guard `max == 0`.
  - reduce with `min`, `None` when nothing eliminated.
- `pub fn max_pivot_magnitude(&self) -> Option<f64>` — same walk, per-pivot
  larger-magnitude eigenvalue (`d_diag[k].abs()` / `max(|λ₊|,|λ₋|)`),
  reduce with `max`.
- 2×2 detection: `k + 1 < nelim && ff.d_subdiag[k] != 0.0`, identical to
  `min_diagonal` / `summary`.
- Doc comments: cite the research note, contrast with `min_diagonal`
  (signed-min vs. magnitude-min), state scaled-space domain.
- Tests in `tests/pounce_interface.rs`: cases 1, 2, 3, 4 from research §5.

`cargo test && cargo clippy --all-targets -- -D warnings` before commit.

### Step 2 — `Solver` accessors + tests

Commit 2: in `src/numeric/solver.rs`, after `min_diagonal()` (line 896):
- `pub fn min_pivot_magnitude(&self) -> Option<f64>` —
  `self.last_factors.as_ref().and_then(|f| f.min_pivot_magnitude())`.
- `pub fn max_pivot_magnitude(&self) -> Option<f64>` — analogous.
- Doc comments mirror `min_diagonal`'s, cross-link `SparseFactors`.
- The `min_diagonal_before_factor_is_none`-style `None` test is already
  covered by case 3 in step 1 (Solver-level). Keep step-1 case 3 at the
  `Solver` level so it exercises this delegation.

### Step 3 — C ABI + tests

Commit 3: in `src/capi.rs`, after `feral_num_neg` (line 397):
- `pub unsafe extern "C" fn feral_min_pivot(s: *const FeralSolver) -> f64`
  — null → `-1.0`; else `s.solver.min_pivot_magnitude().unwrap_or(-1.0)`.
  Wrap in `catch_unwind`, `.unwrap_or(-1.0)`.
- `feral_max_pivot` — analogous.
- `#[no_mangle]`, doc comment with `# Safety`, negative-sentinel contract.
- Test in `capi.rs` `mod tests`: on the existing 2×2 indefinite
  `[[1,2],[2,1]]` fixture — sentinel `< 0.0` before factor, expected
  magnitudes after. `[[1,2],[2,1]]` has eigenvalues 3 and −1; under
  identity-ish handling confirm the values land as the BK D-block predicts
  (compute the oracle from the actual D blocks if scaling perturbs them —
  prefer a diagonal fixture if the 2×2 oracle is fragile under default
  scaling; a `diag(...)` matrix gives an exact hand oracle).

### Step 4 — header / FFI surface

- If a generated C header or `.pyi`/`ctypes` shim lists the C ABI, add the
  two new symbols. Check `grep -rln "feral_num_neg" --include=*.h
  --include=*.py --include=*.pyi .` and update consistently. If none, skip.

### Step 5 — CHANGELOG + checkpoint

- `CHANGELOG.md` Unreleased: "Added `Solver::min_pivot_magnitude` /
  `max_pivot_magnitude` and C ABI `feral_min_pivot` / `feral_max_pivot` —
  near-singularity signal for IPM perturbation handlers (MA57 `CNTL(2)`
  analog)."
- Journal entries in real time.
- Session checkpoint `dev/sessions/2026-05-19-NN.md`, `assemble-context.sh`,
  bench, final `dev/` commit — per CLAUDE.md At-Session-End.

## Non-goals

- No `FactorStatus` variant, no behavior change in factor/solve.
- No pounce-side change (different repo) — documented in research §4.
- No dense-direct `Factors` accessor (optional follow-up).

## Risks

- 2×2 oracle under default scaling: the D blocks live in scaled space, so a
  non-diagonal fixture's eigenvalues are rescaled. Mitigation: force
  `ScalingStrategy::Identity` in tests (the `solver_identity_scaling()`
  helper already exists in `tests/pounce_interface.rs:450`), exactly as the
  `min_diagonal` tests do.
- Cancellation in the 2×2 smaller-magnitude eigenvalue: `(t−√(t²−4Δ))/2`
  loses precision when the block is near-singular. Mitigation: compute the
  smaller magnitude as `|Δ| / larger_magnitude` (product of magnitudes is
  `|Δ|`). The larger magnitude `(|t|+√(t²−4Δ))/2` is cancellation-free.
