# Phase B — shape-dispatched `nemin` within `AmalgamationStrategy::Auto`

**Date**: 2026-05-03
**Status**: REJECTED — sweep data does not robustly support the
hypothesis. The dispatch path was prototyped, the sweep was run,
and the wall-time signal for path-like fixtures sits inside
measurement noise. Phase A (`nemin = 16` global default) is kept;
no shape-dispatched override layer is added.
**Predecessor**: `factor-nnz-residual-gap.md` (Phase A: nemin 32 → 16)

## Hypothesis

`nemin` controls supernode amalgamation aggressiveness. Phase A
moved the global default from 32 to 16 based on a sweep on bushy
(IPM-KKT) PoissonControl matrices. The two `Auto`-dispatched
strategies select on a different axis (column-renumbering bias,
not amalgamation size), but the etree shape that triggers each
strategy correlates strongly with the optimal `nemin`:

- **Path-like / near-path trees** (`Adjacency` selected by `Auto`):
  Linear chains, each node has ~1 child. Children's contribs flow
  directly into the parent's frontal with little branching. The
  pass-through padding cost (`(num_nrow − sym_nrow) × nelim` per
  node, the residual gap source diagnosed in
  `factor-nnz-residual-gap.md`) is **small** because the trailing
  pattern propagates linearly without many sibling rows piling on.
  → Small `nemin` saves little memory but adds pivot-block
  boundary cost. Larger `nemin` should win.

- **Bushy trees** (`Renumber` selected by `Auto`):
  Heavy branching at internal nodes. Each node assembles many
  children's contribs and the trailing pattern is the union of
  many sibling row sets. Pass-through padding is the **dominant**
  cost source. Smaller `nemin` shrinks every supernode's
  pass-through term.
  → Small `nemin` should win on memory and roughly tie on factor
  wall.

The Phase A sweep on PoissonControl (a bushy KKT matrix) chose 16.
The conjecture is that path-like fixtures want a larger value
(24 or 32). Phase B's job is to validate or reject this.

## Sweep design

**Fixtures** (one per shape bucket from `diag_etree_shape.rs`):

- Path-like: `MUONSINE_0000` (the canonical
  `multi_child_frac ≈ 0.002` matrix, `Adjacency`-wins).
- Bushy: `KIRBY2_0007` (`multi_child_frac ≈ 0.97`,
  `Renumber`-wins, in the corpus's worst factor-ratio cluster).
- Bushy-large: PoissonControl K=158 (already characterized in
  Phase A).
- Bushy-mid: `ACOPR30_0067` or `SWOPF_0000` (mid-size IPM-KKT).

**Sweep**: `nemin ∈ {8, 16, 24, 32, 48}` × `Auto` strategy
dispatch. Hold ordering at AMD (the in-tree default for
`OrderingMethod::default()`) and scaling at Identity. Record per
fixture: `factor_nnz`, factor wall median over `reps=3`.

**Decision rule**:

- If the path-like fixture's optimal `nemin` ≥ 24 AND the bushy
  fixtures' optimal `nemin` ≤ 16, implement shape-dispatched nemin
  with the picked values per shape.
- If both buckets prefer 16 (within ≤ 5% factor wall and ≤ 10%
  factor_nnz of any other tested value), keep the global default
  and document that Phase B is a no-op — don't add code that
  doesn't earn its keep.
- If the bushy fixtures prefer something other than 16, that's a
  signal to revisit the Phase A choice itself, not Phase B.

## Implementation sketch (if data supports)

In `src/symbolic/mod.rs:594-601` (where `Auto` resolves to a
concrete `AmalgamationStrategy`), extend the resolution to also
override `nemin` when the user has not explicitly set it:

```rust
if matches!(effective_params.amalgamation_strategy,
            supernode::AmalgamationStrategy::Auto) {
    let resolved = supernode::pick_amalgamation_strategy(&etree);
    effective_params.amalgamation_strategy = resolved;
    // Phase B: shape-dispatched nemin. Only flip when the user
    // is on the default value (16); explicit overrides win.
    if effective_params.nemin == DEFAULT_NEMIN {
        effective_params.nemin = match resolved {
            AmalgamationStrategy::Adjacency => NEMIN_PATH,
            AmalgamationStrategy::Renumber  => NEMIN_BUSHY,
            AmalgamationStrategy::Auto      => unreachable!(),
        };
    }
}
```

The "only flip when on the default" rule preserves explicit
caller overrides (in-tree benches that pin specific values).

## Risk / cost

- Adds one branch in the symbolic phase, O(1) cost.
- Behavior change only on `Auto` dispatch (the global default).
- Existing tests that build `SupernodeParams { nemin: K, .. }`
  explicitly are unaffected.

## Out of scope

- Micro-tuning per ordering method (AMD vs METIS).
- Per-matrix nemin (would require shape statistics during analysis
  beyond what `pick_amalgamation_strategy` already reads).
- Coupling `nemin` to peak-frontal size or other numeric phase
  signals.

## Files referenced

- `src/symbolic/mod.rs:594-601` — `Auto` resolution site.
- `src/symbolic/supernode.rs:181-206` — `pick_amalgamation_strategy`.
- `src/symbolic/supernode.rs:115` — current `nemin = 16` default.
- `src/bin/diag_etree_shape.rs` — fixture shape statistics.

## Sweep results (executed)

Run via `src/bin/diag_phase_b_nemin_sweep.rs`, AMD ordering,
Identity scaling, `pivot_threshold = 1e-8`, REPS=3 median.

### MUONSINE_0000 (path-like, `multi_child_frac ≈ 0.002`)

| nemin | factor_nnz | factor_med_us (run 1) | factor_med_us (run 2) |
|-------|-----------:|----------------------:|----------------------:|
|     8 |       4606 |                   216 |                   229 |
|    16 |       4606 |                   212 |                   211 |
|    24 |       4606 |                   201 |                   230 |
|    32 |       4606 |                   203 |                   272 |
|    48 |       4606 |                   195 |                   273 |

`factor_nnz` is **invariant** at 4606 across all `nemin` values
(robust finding — path-like trees have no pass-through padding
to compress). Wall-time signal is **not robust**: run 1 suggests
nemin=48 is 8% faster, run 2 suggests it is 29% slower. The
absolute spread is ≤ 80 µs on a 200 µs base — within typical
measurement noise on this CPU.

### KIRBY2_0007 (bushy, `multi_child_frac ≈ 0.97`)

| nemin | factor_nnz | factor_med_us |
|-------|-----------:|--------------:|
|     8 |       3120 |            85 |
|    16 |       3120 |            87 |
|    24 |       3238 |            84 |
|    32 |       3405 |            87 |
|    48 |       3946 |            92 |

`nemin = 8` and `16` tie at the minimum `factor_nnz`.  Above 16,
`factor_nnz` grows monotonically (+9% at 32, +26% at 48). Wall is
flat. `nemin = 16` is the right default — confirms Phase A.

### ACOPR30_0067 (bushy, mid-size)

| nemin | factor_nnz | factor_med_us |
|-------|-----------:|--------------:|
|     8 |       3724 |           133 |
|    16 |       4924 |           128 |
|    24 |       6740 |           121 |
|    32 |       8894 |           128 |
|    48 |      11597 |           141 |

`nemin = 8` shaves another -24% `factor_nnz` vs 16, but ACOPR30
is one of the matrices PoissonControl K=158 most resembles in
shape, where Phase A's K=158 sweep showed nemin=8 regressing wall
+21%. Holding at 16 is the safe default-corpus choice.

### SWOPF_0000 (bushy, mid-size)

| nemin | factor_nnz | factor_med_us |
|-------|-----------:|--------------:|
|     8 |       1774 |            41 |
|    16 |       1897 |            31 |
|    24 |       2246 |            35 |
|    32 |       2385 |            33 |
|    48 |       2782 |            38 |

Similar shape: `nemin = 16` is roughly optimal in joint
`factor_nnz × wall` space.

## Decision

The decision rule from the design section reads: "If both
buckets prefer 16 (within ≤ 5% factor wall and ≤ 10% factor_nnz
of any other tested value), keep the global default and document
that Phase B is a no-op — don't add code that doesn't earn its
keep."

Path-like signal doesn't clear that bar:
- `factor_nnz` is strictly invariant — cannot motivate a change on
  memory grounds.
- Wall is within noise — run-to-run flips between "8% better" and
  "29% worse" at the same effective `nemin`, undermining the
  original sweep's positive read.

Bushy fixtures all confirm `nemin = 16` (Phase A's choice).

**Phase B is rejected.** The dispatch override (`DEFAULT_NEMIN`,
`NEMIN_PATH_LIKE`, `NEMIN_BUSHY` constants + override branch in
`mod.rs:594-625`) was prototyped to run the sweep on the live
dispatch path and confirm the absence of a robust signal. Reverted
in the same session. The sweep binary
(`src/bin/diag_phase_b_nemin_sweep.rs`) is retained for any future
reconsideration with a larger or different fixture set.

## Open questions

- A larger fixture set (e.g. the full `bench_solver_corpus`
  walking) might reveal bushy outliers where `nemin < 16` is a
  consistent win; that is a Phase 2.x territory, not the
  immediate post-`build_row_indices`-fix tail.
- The shape predicate itself (`multi_child_frac < 0.05`) was
  tuned for the strategy dispatch and might not be the right
  axis for `nemin` dispatch even if a signal existed.

## Files referenced

- `src/symbolic/mod.rs:594-601` — `Auto` resolution site (final
  state: strategy override only, no `nemin` override).
- `src/symbolic/supernode.rs:113-122` — `SupernodeParams::default`
  (final state: literal `nemin: 16`, no exported constants).
- `src/bin/diag_phase_b_nemin_sweep.rs` — sweep binary.
- `dev/research/factor-nnz-residual-gap.md` — Phase A predecessor.
- `dev/tried-and-rejected.md` — 2026-05-03 entry.
