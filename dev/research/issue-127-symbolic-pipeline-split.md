# Issue #127 — split the symbolic pipeline so race losers skip the tail

## Problem

`symbolic_factorize_with_method` (`src/symbolic/mod.rs`) is a monolithic
pipeline. Two dispatchers race multiple candidates through the **whole**
pipeline but decide on `factor_nnz_estimate` alone:

- `symbolic_factorize_preprocess_auto` (preprocess `Auto`): races `None` vs
  `LdltCompress`, ~2× symbolic on every cache-miss factorization whose
  structural predicate fires (typical slack-heavy IPM KKTs).
- `symbolic_factorize_race` (`OrderingMethod::AutoRace`): races 4 ordering
  candidates → up to 4×. With preprocess `Auto` the two nest → up to ~8×.

Everything after the decision value is wasted for the losers.

## Split point

The decision value is `factor_nnz = total_factor_nnz(&col_counts)` at
`mod.rs:1335`, from which `factor_nnz_estimate = (factor_nnz * factor_slack)
as usize` (slack = 1.2). The stages *before* it (ordering — including the
LdltCompress MC64 matching, which is the dominant cost — permutes, etree,
postorder, col_counts, and the Phase-2.12 Renumber rebuild which itself
feeds col_counts) are all needed to compute the estimate. The stages
*after* it are the wasted tail for losers:

- `find_supernodes` + `assign_delayed_capacities`
- `find_small_leaf_groups`
- `compute_peak_contrib` (contrib_sizes)
- `compute_static_row_indices` (#125)
- struct assembly

So: **prefix** = `mod.rs:1068..=1335` (through `factor_nnz`); **finish** =
`mod.rs:1337..=1407` (tail + build `SymbolicFactorization`).

## Why prefix/finish split, not "estimate then recompute winner"

A cheaper-looking alternative — run an estimate-only prefix per candidate,
then re-run the *full* `symbolic_factorize_with_method` on the winner —
recomputes the winner's ordering. For the preprocess `Auto` case the
LdltCompress arm's MC64 matching is the dominant cost and would run **twice**
when LdltCompress wins, a regression on that path. The prefix/finish split
keeps the winner's already-computed prefix and finishes it exactly once, so
no ordering/MC64 work is ever repeated.

## Invariants (must hold; how tested)

1. **Bit-identical winner selection.** The estimate compared is
   `factor_nnz_estimate` (the post-`* slack` `as usize` truncation), with the
   exact same `<` / `<= none_limit` operators as today, so ties break
   identically. Tested by *self-consistency*, no golden constants:
   - `AutoRace` result byte-matches the result of directly requesting the
     argmin concrete ordering method (`resolve` the winner, compare
     `perm`, `perm_inv`, `supernodes`, `factor_nnz_estimate`,
     `resolved_method`, `col_counts`).
   - preprocess-`Auto` result byte-matches the winning concrete-preprocess
     result on a None-wins matrix and an LdltCompress-wins matrix.
2. **Losers skip the tail.** A `#[cfg(test)]` atomic counter incremented at
   the top of `symbolic_finish`; an in-module unit test asserts it is `1`
   after a default-params (`Auto` preprocess) `symbolic_factorize` and `1`
   after an `AutoRace` — i.e. exactly one finish regardless of candidate
   count.
3. **Profiler still reflects exactly one run.** Preserved by keeping the
   existing "fresh profiler per raced candidate/arm; the winner's tail is
   recorded into that same fresh profiler; snapshot copied into the caller's
   shared profiler once, at the end" discipline. `set_total` is called once,
   in `symbolic_finish`, from a `t_total` captured at prefix start and
   carried in the prefix — so `total_us` spans prefix+finish and
   `accounted_us <= total_us` with no validation warnings. Guarded by the
   existing `tests/symbolic_profiler.rs`.

## Profiler bookkeeping (the fiddly part; diagnostics-only)

`SymbolicPrefix` carries the effective `SupernodeParams` it ran under
(hence the profiler arc used for its prefix stages) and `t_total`.
`symbolic_finish` records the tail stages and `set_total` into
*that same* profiler arc. The caller then copies the arc's snapshot into
the shared profiler iff it is a different `Arc` (`Arc::ptr_eq`) — i.e. the
concrete top-level path records straight into the shared profiler (no copy),
while a raced arm/candidate records into its fresh profiler and is copied
once at the end. This mirrors today's `copy_profiler` behaviour, just moved
after the single finish.

## Blast radius

Only the standard `symbolic_factorize_with_method` path and its two race
dispatchers. The Schur-tail variant (`symbolic_factorize_with_schur_tail`,
a separate function that does not race) is untouched. No public signature
changes; `SymbolicFactorization` is unchanged.

## Validation plan

- Full suite green, especially `tests/auto_strategy.rs`,
  `tests/issue91_preprocess_misfire.rs`, `tests/symbolic_profiler.rs`,
  the delayed-pivoting suites, plus the new parity + finish-count tests.
- Bench before/after in the checkpoint: symbolic-phase wall on an
  Auto-firing slack-heavy KKT should drop toward 1× + counts-only overhead
  (the tail no longer runs per loser). Inertia gate must stay 100%.
