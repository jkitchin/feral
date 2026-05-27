# Symbolic-Analysis-Time Delay Budget (Phase B, Issue #55)

Date: 2026-05-27
Tracking: issue #55 ("Bound delayed-pivot accumulation in wide root
supernodes — IPM-KKT cascade overflow")
Phase 0 evidence: `dev/research/cb-on-default-revalidation-2026-05-27.md`
Phase A audit: `dev/research/mumps-perturbation-alignment-2026-05-27.md`
Phase A instrumentation: `dev/research/per-front-delay-instrumentation-2026-05-27.md`

## Problem

FERAL's previous policy was unbounded delayed pivoting: any Bunch-Kaufman
rejection in a subtree leaf could be passed up the elimination tree to
its parent, growing the parent's frontal block in-place at numeric time.
On IPM-KKT matrices with a wide top-level Schur complement (Mittelmann
problems `pinene_3200`, `nql180`), this produced effectively-dense
root supernodes and >100 GB allocations on a 64 GB host.

The `cascade_break` heuristic at `src/numeric/factorize.rs:2234-2265`
existed as a numeric-time escape: when the delayed-pivot fraction
`n_delayed_in / expanded_ncol` exceeded a configurable ratio, the
factor would force-accept incoming pivots with a sign-preserving
static perturbation. CB was disarmed by default after issues #17
(robot_1600), #18 (NARX_CFy), and #48 (marine_1600) showed IPM stalls
from CB firing on pivots that delay could have absorbed.

Phase A's MUMPS alignment audit (`mumps-perturbation-alignment-2026-05-27.md`)
established that FERAL's `perturb_to_floor` formula, inertia counting,
and the `n_tiny` diagnostic counter are already MUMPS-aligned. The
single divergence was the *trigger condition* — MUMPS perturbs only
when delay is structurally impossible (per-front delay capacity
exhausted, set at analysis time); FERAL's CB triggered on a
numeric-time ratio heuristic.

## Approach

Mirror MUMPS by giving the symbolic-analysis phase a per-supernode
delay capacity:

```
delayed_capacity(s) = min(subtree_col_count(s) - own_ncol(s),
                          K · own_ncol(s))
```

where `K = DELAY_CAPACITY_MULTIPLIER = 4`. The worst-case bound (left
term) is the count of columns in `s`'s subtree minus `s`'s own
columns: at most one delay per subtree column could ever reach `s`.
The tight bound (right term) is empirically derived from Phase A's
per-front instrumentation across the cascade-victim corpus.

At numeric time:
- If `n_delayed_in <= delayed_capacity`: proceed as before.
- If `n_delayed_in > delayed_capacity` AND CB armed: engage the
  sign-preserving static-perturbation fallback at this supernode
  (`perturb_to_floor`, `sign(d) · max(|d|, eps)`).
- If `n_delayed_in > delayed_capacity` AND CB disarmed AND not root:
  return structured error `FeralError::DelayBudgetExceeded`, mirroring
  MUMPS's `INFO(2)` workspace-overflow recovery path.

The root supernode is exempt from the error path: by the time delays
reach root, the frontal size is committed and there is no further
delay target.

## Capacity multiplier `K`

`K = 4` is conservative. Phase A's per-front telemetry on the
cascade-victim corpus showed `max(observed_n_delayed_in / own_ncol)`
peaks around 1.7 on `pinene_3200` and below 1.0 on the non-pathological
matrices (`robot_1600`, `NARX_CFy`). A 4× multiplier:
- Absorbs all observed non-pathological delays without firing CB
  (so the historical regressions of issues #17 / #18 stay clear),
- Triggers CB at the cascade-victim frontier where MUMPS would have
  exhausted its delay capacity,
- Leaves headroom for problems we haven't yet seen.

If future telemetry suggests `K = 2` is safe, the constant in
`src/symbolic/supernode.rs:DELAY_CAPACITY_MULTIPLIER` is the single
tuning knob. The worst-case `subtree_col_count - own_ncol` bound
always provides the upper limit.

## Defensive root-supernode width cap

Independently of the delay budget, the root supernode's *own* width
is capped at amalgamation time to `min(0.05 * n, 2048)`. This bounds
the worst-case frontal width even when the elimination tree's natural
amalgamation rules would merge a wide tail into the root. Loose
enough not to disturb non-pathological problems, tight enough that
the nql180 / pinene_3200 root cannot grow back to its pre-Phase-B
catastrophic size purely through size-based amalgamation.

## Cascade-break default

CB is re-armed by default (`cascade_break_ratio = Some(0.5)`,
`cascade_break_eps = Some(1e-10)`). On budgeted supernodes (the
default after Phase B) the numeric ratio value is unused — the
trigger is `n_delayed_in > delayed_capacity`. On unbudgeted paths
(`delayed_capacity == usize::MAX`, used by older constructors and as
the escape hatch for test scaffolds), the legacy ratio-based trigger
is retained for backward compatibility.

The Weyl-bound concern that motivated the original disarm
(`dev/research/cascade-break-l-perturbation-2026-05-15.md`) is not
*resolved* by Phase B: per-pivot `||Δ||_∞ ≤ eps` still does not hold
strictly when L is scaled by `1/d_new`. Phase B closes the practical
gap by ensuring CB only fires when delay was structurally impossible
— exactly MUMPS's invariant. Pivots that delay could have absorbed
are now absorbed by delay (and respect the BK growth bound through
the delayed-pivot machinery), so the unbounded `Δ` growth path is
unreachable on the budgeted code path.

## Implementation map

- `src/symbolic/supernode.rs:125-140`: `Supernode::delayed_capacity` field.
- `src/symbolic/supernode.rs:DELAY_CAPACITY_MULTIPLIER`: K = 4 constant.
- `src/symbolic/supernode.rs:assign_delayed_capacities()`: subtree-sum
  bottom-up pass; `O(n)`.
- `src/symbolic/supernode.rs:find_supernodes()` root-cap branch:
  declines amalgamation merges into the elimination-tree root when
  the merged width would exceed `min(0.05 * n, 2048)`.
- `src/symbolic/mod.rs`: two call sites — the standard symbolic path
  and the F3.2b Schur-tail-merge path — both call
  `assign_delayed_capacities` after the supernodes are finalized.
- `src/numeric/factorize.rs:factor_one_supernode`: budget check after
  `n_delayed_in` is computed; CB trigger rewired to `budget_exceeded`.
- `src/error.rs`: `FeralError::DelayBudgetExceeded { supernode,
  required, capacity }` variant.
- `src/numeric/factorize.rs:NumericParams::default()` and `with_bk`:
  CB armed by default with `ratio = 0.5`, `eps = 1e-10`.

## Asymptotic cost

`assign_delayed_capacities` is a single bottom-up sweep over the
supernodes, summing each supernode's `ncol` plus its children's
`subtree_ncol`. Each supernode is touched once with constant work plus
a single iteration over its children list. Total work `O(n_snodes +
∑ |children(s)|) = O(n_snodes)` since each supernode appears once as
a child. Memory: one `usize` per supernode for `subtree_ncol`,
released after the pass. No measurable impact on symbolic time
(symbolic is already O(n + |edges|)).

The numeric-time check is two integer comparisons per supernode call
— amortized free relative to the dense factor work that follows.

## Expected impact

- `nql180` (KKT order 130,080): Phase 0 attempted with CB-on
  unbudgeted; the root supernode still ran away. Phase B caps the
  root width at `min(0.05 * 130080, 2048) = 2048`, bounding the
  worst-case frontal allocation. Acceptance target: IPM iteration 1
  completes within 16 GB peak RSS.
- `pinene_3200_0009`: Phase 0 evidence showed CB-on factor at 33 ms.
  Phase B's budget-based trigger fires on the same supernodes (the
  trigger is correlated with the ratio heuristic at pinene's
  cascade frontier), so factor time should match or beat CB-on.
- `robot_1600`, `NARX_CFy`, `marine_1600`, `rocket_12800`: Phase 0
  evidence shows these matrices' max delay-to-own ratio sits below
  the budget threshold, so CB never engages and the factor follows
  the standard delay path. Historical regressions stay resolved.
- `marine_1600_0017`, `nuffield2_trap_iter1` (Phase 0 holdouts):
  CB previously fired on pivots that MUMPS would delay. With the
  budget-based trigger, CB engages only when delay is structurally
  impossible — these matrices' delay catchment is well within
  `delayed_capacity`, so CB no longer fires and the inertia matches
  MUMPS.

## Acceptance (#55)

- `FeralConfig::default()` ships with bounded delay catchment enabled
  via the symbolic delay budget + CB-armed default.
- The full Phase 0 corpus (41 historical regression matrices) passes
  with default settings.
- pounce can delete its per-problem `.opt` overrides for FERAL
  (`benchmarks/mittelmann/profiles/nql180.opt` and any others).
- Inertia parity preserved against the corpus consensus framework
  (`external_benchmarks/consensus/compute_consensus.py`).
