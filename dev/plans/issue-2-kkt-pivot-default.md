# Plan: Promote `NumericParams::default()` `pivot_threshold` to 0.01

**Date:** 2026-05-02
**Issue:** https://github.com/jkitchin/feral/issues/2
**Research:** `dev/research/issue-2-kkt-pivot-default.md`
**Status:** In progress

## Goal

Make the sparse multifrontal default activate the rescue infra
(rook + delayed pivoting) on rank-deficient KKT-augmented systems,
so consumers like ripopt that build `NumericParams::default()` get
the same configuration that feral's own in-tree sparse benchmarks
have used since 2026-04-13.

Concretely: change `NumericParams::default()` so
`bk.pivot_threshold = 1e-8` (matching Ipopt's `ma27_pivtol` /
MA27's `cntl[1]` reference default). Leave
`BunchKaufmanParams::default()` (the dense entry point) at `0.0`.

## API shape

Replace `#[derive(Default)]` on `NumericParams`
(`src/numeric/factorize.rs:25`) with a manual impl:

```rust
impl Default for NumericParams {
    fn default() -> Self {
        Self {
            bk: BunchKaufmanParams {
                pivot_threshold: 1e-8,
                ..BunchKaufmanParams::default()
            },
            scaling: ScalingStrategy::default(),
            small_leaf: SmallLeafBatch::default(),
            profiler: None,
        }
    }
}
```

`with_bk` (line 225) is unchanged — callers passing an explicit
`BunchKaufmanParams` get exactly that. This keeps the dense `factor()`
entry point and every test that constructs `BunchKaufmanParams`
directly bit-for-bit identical.

## Touched call-sites — audit

Existing in-tree consumers of `NumericParams::default()`:

| Site | Effect of new default |
|------|------|
| `src/numeric/solver.rs:128` (`Solver::with_params`) | Sparse path. Picks up `0.01` — the value `bench.rs` already uses. No-op for callers that don't customize. |
| `src/numeric/condition.rs:186` (condition estimator) | Internal. Same factor pipeline; switching the threshold is what we want. |
| `src/numeric/factorize.rs:2596,2629,2649,2678,2749,2913` (in-file tests/benches) | Internal sanity tests on small matrices; pivots not near threshold. Verify pass. |
| `src/bin/diag_*.rs` (six diagnostic bins) | Diagnostic, not a correctness gate. |
| `tests/multi_rhs.rs:30`, `tests/ldlt_compress.rs:76,110` | Small KKT/diagonal cases; pivots `O(1)` magnitude, threshold change is a no-op. |
| `examples/triage_bratu3d.rs:202` | Triage script; not in CI. |

In-tree consumers using `with_bk` or explicit `BunchKaufmanParams`
construction (the `pivot_threshold: 0.01` callers) — unchanged.

The `tests/threshold_consistency.rs::sparse_params` builds
`NumericParams::with_bk(ldlt_params())` where `ldlt_params` returns
`BunchKaufmanParams::default()` — that path still gets `0.0`,
matching its documented intent (it tests the dense
threshold-consistency invariant).

## Steps

### Step 1: regression test (TDD, before code change)

New `tests/issue_2_kkt_ls_init.rs`:

1. Construct a synthetic saddle-point CSC matrix:
   ```
   A = [ I_n      J^T   ]
       [ J        D     ]
   ```
   with `n = 4`, `m = 6` (so `m > n`), `J` shaped to have:
   - 4 inequality rows with `nnz = 2`, entries `±1`
   - 1 equality row (zero diagonal in `D`)
   - 1 redundant constraint (linear combination of two others)
   `D` = `diag(0, -1, -1, -1, -1, -1)`.
2. Build the lower-triangle CSC via `CscMatrix::from_triplets`.
3. Factor and solve with RHS `b = [grad_f; v_inq]` matching ripopt's
   LS pattern: top block `[1, 1, 1, 1]`, bottom block
   `[0, 0.5, 0.5, 0.5, 0.5, 0.5]`.
4. Assert: `||A·x - b||_∞ < 1e-8`.
5. Assert: every output index that maps to a non-structurally-zero
   inequality row is non-zero in the solution.
6. Negative control: re-run with
   `NumericParams::with_bk(BunchKaufmanParams { pivot_threshold: 0.0,
   on_zero_pivot: ForceAccept, ..default() })` and confirm at least
   one non-structurally-zero output is exact `0.0` — guards against
   the test passing trivially if rescue infrastructure changes
   later.

This is a unit test, not a Mittelmann-corpus test, to keep CI fast
and self-contained.

### Step 2: implementation

Edit `src/numeric/factorize.rs`:

1. Remove `Default` from the `#[derive(...)]` on `NumericParams`
   (line 25).
2. Add the `impl Default for NumericParams` block as shown above.
3. Add a doc comment on the impl pointing at this plan and the
   2026-04-13 decision so the asymmetry vs `BunchKaufmanParams::
   default()` is explicit.

### Step 3: full test sweep

```
cargo test --release           # all 146+ unit + integration tests
cargo test --release -- --ignored  # opt-in large-matrix smokes
cargo clippy --all-targets -- -D warnings
```

Acceptance: zero failures, zero new warnings.

### Step 4: bench parity check

```
cargo run --bin bench --release
```

The bench already uses `pivot_threshold = 0.01` explicitly
(`src/bin/bench.rs:1226,1236`), so the new default is a no-op for
this binary. Confirm the residual numbers reported in
`CHANGELOG.md` Unreleased / latest session checkpoint do not
regress.

### Step 5: decisions + changelog

Append to `dev/decisions.md` a new section
`## 2026-05-02 — `NumericParams::default()` pivot_threshold = 0.01`
documenting:

- The change.
- Cross-reference to issue #2.
- That `BunchKaufmanParams::default()` (dense) stays at `0.0`.
- The 2026-04-13 split as the precedent.

Append to `CHANGELOG.md` Unreleased section under "Changed":

> `NumericParams::default()` now sets `bk.pivot_threshold = 0.01`,
> matching the SSIDS/MUMPS canonical default and feral's in-tree
> sparse-caller convention. `BunchKaufmanParams::default()` (dense)
> is unchanged at `0.0`. Issue #2.

### Step 6: session checkpoint

Per CLAUDE.md normal protocol: write `dev/sessions/2026-05-02-NN.md`,
commit `dev/` changes as the final commit.

## Risk + rollback

Low risk. The 0.01 default has been the in-tree sparse convention
for ~3 weeks (since 2026-04-13). The change brings the default into
agreement with what every existing sparse caller already writes
explicitly. Rollback is a one-line revert of the manual `Default`
impl back to `#[derive(Default)]`.

## Out of scope

- Dense `BunchKaufmanParams::default()` change.
- Promoting the `inertia.zero` reinterpretation from
  `ripopt/src/linear_solver/feral_direct.rs:163-167` into feral.
- ripopt-side cleanup of `set_pivot_threshold(1e-8)` workaround.
- Wider "pivot defaults for KKT" survey (would need to cross-cut
  scaling, `zero_tol`, `on_zero_pivot`).
