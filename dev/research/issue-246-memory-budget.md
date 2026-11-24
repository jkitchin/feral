# Issue #246 step 3 — a factor-memory budget in the symbolic phase

Date: 2026-11-23. Journal: `dev/journal/2026-11-23-01.org`.

## Problem

A host cannot bound feral's memory before the damage is done. On
fhnw-binschedule1 (before the AMF fix), `symbolic_finish` itself allocated
past 11 GB, so even a host that reads `SymbolicFactorization::work_estimate()`
and refuses the numeric factor (pounce#1249) is killed first. pounce works
around this by running a second symbolic pass under its own cap, which costs
minutes on fhnw.

## Where the memory goes (code inspection)

The symbolic pipeline has two stages (issue #127):

- **prefix** (`resolve_prefix`): ordering → etree → column counts →
  `factor_nnz_estimate = floor(1.2 · Σ colcount)`. Memory is O(n + nnz(A)):
  fhnw's prefix peaked at 1.3 GB while the predicted factor was 1.4e10
  entries.
- **finish** (`symbolic_finish`): supernodes, small-leaf, peak-contribution
  simulation, static row indices. Memory is O(Σ nrow over supernodes), in
  two `Vec<usize>`s. This is where fhnw aborted.

The races (`AutoRace` in `symbolic_factorize_race`, preprocess-`Auto` in
`symbolic_prefix_preprocess_auto`) compare prefixes and finish only the
winner. So there are exactly two call sites of `finish_and_copy`, and both
have the winning prefix in hand.

## Design

- `SupernodeParams::max_factor_bytes: Option<usize>`, default `None` (no
  behaviour change). It lives on `SupernodeParams` because that is what
  every symbolic entry point and every `Solver` path (`factor`, the
  reanalysis trial, race probes via `race_probe`) already threads through.
- **Check 1, before the finish:** `factor_nnz_estimate · size_of::<f64>()`.
  This is exactly `WorkEstimate::factor_bytes` of the eventual analysis
  (same slacked count, same width), and it is a lower bound on `peak_bytes`.
  It is the check that would have stopped fhnw at 1.3 GB instead of 11 GB.
- **Check 2, after the finish:** `work_estimate().peak_bytes` (factor plus
  simulated contribution peak). This is the figure a host compares against
  RAM before the numeric phase. The finish has already run at that point,
  but it fit, and check 1 bounds how large it can be.
- New error `FeralError::MemoryBudgetExceeded { predicted_bytes,
  budget_bytes, stage }`, with `stage` being `"factor"` (check 1) or
  `"peak"` (check 2). This is an additive enum variant, following the
  precedent of `SingularBasis` / `NeedsRefactor` (#81, #95). Python maps it
  to a new `MemoryBudgetExceeded` exception under `FactorError`. The C API
  already maps every `FatalError` to `FERAL_FATAL`.
- `Solver::with_max_factor_bytes(Option<usize>)`. Through `Solver::factor`
  the error surfaces as `FactorStatus::FatalError`. Race arms that exceed
  the budget fail their probe and are skipped, which is existing behaviour
  for failed arms. The reanalysis trial ignores a failed fresh analysis
  (`if let Ok`).

## Not in scope

- Bounding the finish's own transient memory more tightly than check 1
  does: Σ nrow ≤ nnz(L), so the finish needs at most about 2·8·nnz(L)
  bytes, a small multiple of the factor bytes check 1 already bounds.
- An environment-variable override. Hosts set it through the API, and
  pounce already reads its own `POUNCE_FERAL_MAX_FACTOR_BYTES`.
