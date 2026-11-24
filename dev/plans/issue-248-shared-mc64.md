# Plan: share one MC64 matching across AutoRace arms (issue #248)

## Problem
`AutoRace` runs `resolve_prefix` per candidate ordering. With preprocess
`Auto` and a predicate that recommends `LdltCompress`, every arm runs its
own `symbolic_prefix_concrete(.., LdltCompress)`, which calls
`compute_mc64_cache(matrix)`. The matching depends only on the matrix, so
the 4-arm race at n >= 1M computes the same matching four times
(fhnw-binschedule1: 1294 s race vs 476 s for two arms).

## Design
- `Mc64Memo` (a `RefCell<Option<Mc64Cache>>`, private to `src/symbolic`):
  computed on first use, cloned out to every later `LdltCompress` prefix.
- `symbolic_factorize_race_over` creates one memo per race and passes it to
  each arm's `resolve_prefix` → `symbolic_prefix_preprocess_auto` →
  `symbolic_prefix_concrete`. Non-race calls use a fresh memo (no change).
- Only an uninterrupted `Ok` matching is stored. The interrupt check of
  issue #246 (check before reading a possibly partial matching) moves into
  `Mc64Memo::get_or_compute`. Errors are not stored; the next arm recomputes
  and fails the same way, as before.
- Each prefix still owns its `cached_mc64` (a clone, O(n) memory) so the
  winner carries it to the numeric phase unchanged.

## Why selection stays bit-identical
MC64 is deterministic in the matrix, so every arm gets the same matching it
would have computed itself, hence the same super-variable map, compressed
graph, ordering and `factor_nnz_estimate`.

## Tests (oracle external to the race)
1. `autorace_computes_mc64_once_per_race`: a thread-local probe counts
   matchings; one per race for `RACE_CANDIDATES` and `RACE_CANDIDATES_LARGE`.
   Mutation check: a fresh memo per arm must make it fail.
2. `autorace_shared_mc64_selection_matches_independent_arms`: the oracle is
   each candidate run alone through `symbolic_factorize_with_method` (own
   matching = pre-#248 behaviour) and the race's documented
   min-`factor_nnz_estimate`, first-wins rule. Same `perm`.

## Measure
fhnw-binschedule1 four-arm race, `FERAL_MC64_TRACE=1` for MC64 count and
time, winner's `SymbolicProfiler` for the other stages
(`examples/issue248_race_mc64.rs`).
