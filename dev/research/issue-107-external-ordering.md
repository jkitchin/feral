# Research Note: External / user-supplied ordering (`OrderingMethod::External`)

**Status:** Pre-implementation
**Date:** 2026-07-02
**Related spec sections:** 5.1 (feature lifecycle), symbolic factorization pipeline
**Key references:** issue #107; Parker, Garcia & Bent, *Exploiting block
triangular submatrices in KKT systems* (arXiv:2602.17968); Baharev et al.,
*An exact method for the minimum feedback arc set problem*, ACM JEA 26 (2021);
`ScalingStrategy::External(Vec<f64>)` (the scaling analogue already in tree).

## Overview

`OrderingMethod` today maps only to *ordering algorithms* (`Amd`, `Amf`,
`MetisND`, `ScotchND`, `KahipND`, `Auto`, `AutoRace`). A caller that has
computed a *better-than-generic fill-reducing permutation from problem
structure* (block-triangular Schur KKT reuse, tearing orderings from
equation-oriented decomposition) has no way to inject it — the ordering is
always recomputed internally.

This mirrors what FERAL already does for **scaling**: `ScalingStrategy` has an
`External(Vec<f64>)` variant carrying a user-supplied vector that cannot be
expressed as a string option. Ordering has no analogous path. This note adds
`OrderingMethod::External(Vec<usize>)`.

## Algorithm

There is no numerical algorithm. The supplied permutation *replaces* the
internal AMD/METIS/etc. pass and is fed to the existing downstream pipeline
unchanged (postorder composition → etree → column counts → supernode
amalgamation → memory plan). The permutation convention is identical to the
one FERAL already returns from `run_external_ordering` and stores in
`SymbolicFactorization::perm`:

- **new-to-old**, 0-based: `perm[k]` is the original column that becomes
  column `k`.
- length exactly `n`.

Injection point: `run_external_ordering` is the single function that produces
the initial ordering (`amd_perm`) for every method. For `External(perm)` it
returns a clone of `perm` directly instead of calling an ordering crate. From
there the pipeline is byte-identical to any other method — the same postorder
is composed on top, so `SymbolicFactorization::perm` is the postorder ∘ user
ordering, exactly as for AMD's output.

### Validation

A user permutation must be a bijection of `0..n`. `validate_external_perm`
checks:
1. `perm.len() == n`,
2. every entry `< n`,
3. no duplicates (seen-bitset).

Invalid input returns `FeralError::InvalidInput` — never a panic, never
`unwrap`. A *valid but poor* ordering only costs fill/time, never correctness
(a factorization with any valid ordering is exact), exactly as the issue
states and as with `ScalingStrategy::External`.

### Interaction with `OrderingPreprocess`

`OrderingPreprocess::LdltCompress` reorders a *compressed super-graph*
(MC64-matched supervariables), whose dimension `ncmp <= n`. A full-length
user permutation cannot be applied to that compressed graph. Therefore
`External` **forces `OrderingPreprocess::None`**, regardless of the requested
preprocess (including the default `Auto`). This is the only semantic
subtlety and is documented on the variant. Scaling is unaffected: it is
computed independently in the numeric phase from `ScalingStrategy`; only the
MC64 *cache reuse* optimization (a symbolic-time perf shortcut) is skipped,
not any correctness-bearing scaling.

### `Copy` removal

`Vec<usize>` is not `Copy`, so `OrderingMethod` can no longer derive `Copy`
(exactly as `ScalingStrategy`, which is `Clone` not `Copy`). This ripples to
by-value uses: `run_external_ordering` takes `&OrderingMethod`; the
`AutoRace` race loop and `Solver::factor` clone the method; the three numeric
constructors clone `symbolic.resolved_method`; `feral-diagnostics` binaries
that reuse a `method` binding get `.clone()`s. A hand-written `Debug` keeps
the diagnostic `summary()` one-liner compact (`External { len: N }`) instead
of dumping the whole permutation.

## Design Decisions

- **Faithful `resolved_method`.** `run_external_ordering` reports
  `OrderingMethod::External(perm.clone())` as the resolved method, matching
  the existing contract that `resolved_method` is a truthful record of what
  ran. The extra length-`n` clone is negligible beside the `perm`/`perm_inv`
  clones the numeric constructor already does.
- **Programmatic-only.** No string parsing (`feral_ordering` / bench
  `--ordering`), matching scaling's `External`. `Solver::with_ordering` and
  `symbolic_factorize_with_method` are the entry points.
- **Not a race/auto candidate.** `External` is concrete; it is never produced
  by `choose_adaptive`, `pick_default_method`, or the `AutoRace` /
  preprocess-`Auto` races, and those paths never receive it (guarded).

## Test Strategy

External-facing test (`tests/issue107_external_ordering.rs`) — oracle is the
identity/known permutation, computed by hand, not by the solver:

1. **Identity permutation** on a small KKT solves correctly and yields the
   oracle inertia and a residual at the same tolerance as the default path.
2. **Non-trivial fixed permutation** (a hand-written reversal / interleave)
   solves to the same solution and inertia — ordering changes fill, not the
   answer.
3. **`resolved_method` reports `External`** and `resolved_preprocess` is
   `None` even when the default `Auto` preprocess was requested.
4. **Validation rejects** wrong length, out-of-range index, and duplicate
   index with `InvalidInput` (no panic).
5. **`Solver::with_ordering(External(..))`** end-to-end parity with the
   default solve on the same matrix.

Unit tests in `src/symbolic/mod.rs`: `External` produces a valid bijection
perm through the pipeline (mirrors the per-method `..._produces_valid_perm`
tests) and forces `resolved_preprocess == None`.
