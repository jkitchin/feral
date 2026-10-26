# Interior-point use

An interior-point method (Ipopt, pounce) factors hundreds of KKT matrices
with the same sparsity pattern and different values. Keep one `Solver` for
the whole solve. It caches the symbolic analysis per pattern and reuses
its buffers, so each later `factor` call does only the numeric work.

```rust,ignore
use feral::{FactorStatus, Inertia, Solver};

let mut solver = Solver::new();
// A KKT matrix [[W + δ_x I, Jᵀ], [J, -δ_c I]] with n primal and m dual
// rows must have inertia (n, m, 0).
let expected = Inertia { positive: n, negative: m, zero: 0 };
match solver.factor(&kkt, Some(expected)) {
    FactorStatus::Success => { /* solve */ }
    FactorStatus::WrongInertia { actual, .. } => { /* raise δ_x, or ask for quality */ }
    FactorStatus::Singular => { /* raise δ_c */ }
    other => { /* fatal */ }
}
```

## Inertia check

`factor(&a, Some(expected))` compares the certified inertia with the one
you expect, and returns `WrongInertia { actual, expected }` when they
differ. The factor is kept either way, so you can inspect it. On
non-singular matrices the count is exact; see
[Inertia semantics](./inertia.md).

## `increase_quality`

`Solver::increase_quality()` corresponds to Ipopt's `IncreaseQuality`. The
first call switches `Identity` scaling to InfNorm (a no-op if scaling is
already something else). Later calls raise the pivot
threshold `u` (1e-8 → 1e-6 → … → `pivtol_max`), which tightens the
multiplier bound `|L| ≤ 1/u`. It returns `false` when a higher threshold
cannot help, and the caller should then move on to its next remedy (for
Ipopt, `δ_c`). It declines in two cases, both measured on the stored
factor:

- **Already admissible.** No pivot was floored, force-accepted or zeroed,
  and the realized `max |L|` already meets the next rung's bound. The
  current factor is already valid at the higher threshold.
- **The last raise made it worse.** A higher `u` means more delayed
  pivots, and fronts that cannot delay then floor or force-accept them.
  If the factor at the raised threshold gave up on more pivots than the
  factor before it, the threshold returns to where the ladder started
  and the ladder is exhausted.

`FactorStats::max_abs_l` reports the realized `max |L_ij|`, so
`1 / max_abs_l` is the threshold the factor actually meets.

## Re-analysis policy

With the default `Auto` preprocessing, the cached analysis depends on the
values of the matrix it was built from: the MC64 matching that
`LdltCompress` uses, and the ordering computed on the compressed graph.
Under the default `ReanalysisPolicy::Never` it is reused for every later
matrix on the pattern, which is bit-identical to feral before the policy
existed.

`ReanalysisPolicy::OnStale(OnStaleParams::default())` lets the solver
replace that analysis when a trial factorization of the **current** matrix
under another analysis is cheaper:

```rust,ignore
use feral::{OnStaleParams, ReanalysisPolicy, Solver};

let mut solver = Solver::new()
    .with_reanalysis(ReanalysisPolicy::OnStale(OnStaleParams::default()));
```

- **Events.** These three events open a trial:
  - the factor's actual flops exceed `trigger_ratio` (2) times the
    predicted flops;
  - a sparse probe schedule fires (factors 2, 8, 32, …, only on patterns
    where a factor costs more than an analysis);
  - the ordering escalation fires.
- **Adoption.** A candidate is adopted only if it is at least `margin`
  (25 %) cheaper in actual flops, has the same inertia, and has no more
  floored or zero pivots. Up to `max_stored` (2) earlier analyses are
  kept as candidates.
- **Ladder.** Nothing fires while an `increase_quality` ladder is active.
- **Determinism.** Results are deterministic for a given sequence of
  matrices and identical across thread counts.

`OnStale` pays off when one matrix's analysis is badly wrong for later
ones. qcqp1000-2c and qcqp1500-1c are 27–65 % faster end to end in
pounce. On short solves of 10–40 factors, though, an analysis that is
declined is not paid back, and some models run 10–50 % slower. That is
why `Never` is the default.

## Diagnostics

After each `factor`, `Solver::last_factor_stats()` returns `FactorStats`:

| field | meaning |
|---|---|
| `inertia`, `n_tiny`, `n_threshold_fail_accepts` | inertia, floored pivots, pivots force-accepted below the threshold |
| `max_abs_l` | realized multiplier bound |
| `n_cascade_break_nodes` | fronts whose delay budget was exceeded, so they could not delay further |
| `pattern_reused` | the cached analysis was reused |
| `predicted_flops`, `actual_flops` | dense-front work predicted by the analysis, and actually done (delayed pivots inflate it) |
| `reanalysed`, `reanalysis_count` | this factor adopted a new analysis; adoptions on this pattern |

`Solver::last_reanalysis()` reports what the policy did on the last call:
the event, the trials and why it declined. It is always `None` under
`Never`.
