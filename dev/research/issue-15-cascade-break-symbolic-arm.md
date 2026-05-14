# Issue #15: Symbolic-arm gate for cascade-break default

Date: 2026-05-14
Decision owner: jkitchin@andrew.cmu.edu (per session 2026-05-13-04)
Issue: https://github.com/jkitchin/feral/issues/15

## Question

Should the auto-armed default `cascade_break_ratio: Some(0.5)`
(introduced in 0.3.0+, issue #8) fire on every problem, or should we
gate it on a structural property of the symbolic factor so that small
problems where it cannot pay off bypass it entirely?

## Evidence

`src/bin/diag_cascade_ratio_distribution.rs` measures `n_delayed_in /
ncol` on every non-root supernode of every IPM iterate, across three
families:

| family       | iters | non-root snodes | p99   | max   | fires @ 0.5? |
|--------------|------:|----------------:|------:|------:|--------------|
| qcqp1000-1nc |    30 |           5,340 | 0.000 | 0.000 | never        |
| marine_1600  |    18 |         146,930 | 0.846 | 0.999 | yes, cliff   |
| pinene_3200  |    10 |         105,669 | 0.515 | 0.999 | yes, cliff   |

Cliff fronts on the armed families (where breaking saves 60+ s per
factor) reach `expanded_ncol ≈ 14,500–15,400`. On qcqp1000-1nc no
supernode ever sees a delay, so cascade-break is a no-op there.

## What does NOT discriminate

The natural first guess — gate on symbolic `max(supernode.ncol)` —
fails. The symbolic max ncols are:

| family       |  n      | max sym ncol | numeric expanded |
|--------------|--------:|-------------:|-----------------:|
| MSS1         |     163 |           25 |              ~25 |
| qcqp1000-1nc |   1,154 |          391 |             ~391 |
| pinene_3200  | 127,995 |          320 |          ~14,800 |
| marine_1600  |  76,807 |        1,119 |          ~15,400 |

Pinene's pathological cliff front grows 50× via delay propagation;
the symbolic front size at that node is only 320. So symbolic ncol
ordering is qcqp > pinene, but the actual cascade is in pinene, not
qcqp. The symbolic max ncol metric is anti-correlated with the
behavior we care about.

## What does discriminate

Problem size `symbolic.n`:

| family       | n       | needs break? |
|--------------|--------:|--------------|
| MSS1         |     163 | no           |
| qcqp1000-1nc |   1,154 | no           |
| marine_1600  |  76,807 | yes          |
| pinene_3200  | 127,995 | yes          |

There is a ~70× gap between the largest non-armed problem and the
smallest armed problem. Any threshold in `[1500, 70000]` separates
them.

## Principled framing

Cascade-break can save non-trivial time only if some front can grow,
via delay accumulation, to an expanded ncol large enough that its
dense O(ncol³) factor dominates the IPM iteration. An upper bound on
achievable expanded ncol is `n` (you cannot have more columns than
the problem). For dense factor flops to reach the 100ms order — the
scale where cascade-break savings begin to accumulate — `ncol_expanded`
must be ≳ 1000–4000. Therefore `n < N_HEAVY` is a *sufficient*
condition for cascade-break to be irrelevant.

Choosing `N_HEAVY = 4096`:

- Floor: dense LDLᵀ of a 4096×4096 front is O(4096³) ≈ 7×10¹⁰ flops,
  reaching the multi-second scale where cascade-break trade-offs
  matter.
- Above MSS1 (163), arki0003 (4010 — close but probably small dense
  Hessian), qcqp1000-1nc (1154). All four-digit n problems below 4k
  are disarmed.
- Below marine_1600 (76,807) and pinene_3200 (127,995). All known
  cascade-break beneficiaries are armed.
- 4096 has the nice property of being the row-tile dimension used in
  several existing kernels, so it has implementation precedent.

## Decision

Gate the cascade-break trigger inside `factor_one_supernode` on
`symbolic.n >= N_HEAVY` where `N_HEAVY = 4096`. The gate lives at
the trigger site (`src/numeric/factorize.rs:~1821`), not at the
`NumericParams::default()` constructor — that way explicit user
configuration via `cascade_break_ratio: Some(r)` is still subject
to the gate, since the gate's job is "this trigger cannot pay off
on this problem size" regardless of who armed it.

Off-switch / override: callers who specifically want to force
cascade-break on a small synthetic (e.g., test fixtures) can either
construct a problem with `symbolic.n >= 4096`, or — if that becomes
necessary — we add a `cascade_break_min_n: usize` field with default
4096 and let callers set it to 0. For now no such caller exists; all
in-tree call sites either use `None` or rely on the default on
realistically-sized problems.

## Implementation outline

1. Add module-level constant
   `const CASCADE_BREAK_MIN_N: usize = 4096;`
2. In `factor_one_supernode`, change the trigger to:
   ```rust
   let cascade_break = match params.cascade_break_ratio {
       Some(r)
           if !is_root[snode_idx]
               && params.allow_delayed_pivots
               && expanded_ncol > 0
               && symbolic.n >= CASCADE_BREAK_MIN_N =>
       {
           (n_delayed_in as f64) / (expanded_ncol as f64) >= r
       }
       _ => false,
   };
   ```
3. Update the doc comment on `cascade_break_ratio` and the comment
   block above `Default::default()` to mention the gate.
4. Tests:
   - `tests/issue_15_cascade_arm_gate.rs::small_n_disarms_cascade`
     — synthetic indefinite matrix with `n < 4096` engineered to
     produce delays; verify factor outcome is bit-identical with
     `cascade_break_ratio = Some(0.5)` and `= None`.
   - `tests/issue_15_cascade_arm_gate.rs::large_n_keeps_cascade`
     — synthetic `n >= 4096` where the trigger fires; verify the
     `Some(0.5)` configuration produces a different (smaller
     `n_delayed_in`) factor than `= None`.
5. Bench: `cargo run --bin bench --release`; expect no regression
   on marine_1600/pinene_3200 (still armed) and verify qcqp1000-1nc
   factor wallclock under default is bit-identical to None config.

## What we are NOT doing

- Not changing `cascade_break_ratio` from `Some(0.5)`. The data does
  not justify a tightening on the optimal-control side and the
  proposed loosening to `Some(0.85)` was motivated by a regression
  the data refutes.
- Not introducing flop-projection or front-size-projection cost
  models (Options 1 and 2 from the session discussion). The
  symbolic-arm gate is sufficient and far less sensitive to
  per-family tuning.
- Not gating on `nnz/n` density or any pattern-aware heuristic. `n`
  alone separates the corpus cleanly and is the principled bound on
  achievable expanded ncol.
